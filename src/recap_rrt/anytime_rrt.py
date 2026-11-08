"""Anytime / budget-adaptive relaxed recursive cross network for CTR.

This module is intentionally additive: the existing RecursiveFCNv2 and its
checkpoints are left unchanged.  The new model targets *inference-time compute
scaling* directly by removing absolute-depth-specific cross parameters and
learning reusable recurrent control:

  * monotonic state-conditioned ECN<->LCN phase progression;
  * state-conditioned bounded residual update strength;
  * a reusable bank of low-rank cross adapters with state routing;
  * variable-depth / anytime supervision and progressive-refinement losses;
  * a model-level halting head trained from a label-derived oracle depth.

The result can be evaluated at arbitrary recurrent depths without inventing new
LoRA parameters or reusing depth-indexed slots.
"""
from __future__ import annotations

import math
import random
from pathlib import Path
from typing import Iterable

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import base_model as bm
from . import common_layers as cl
from . import tracking, utils


def _logit(p: float) -> float:
    p = min(max(float(p), 1e-6), 1.0 - 1e-6)
    return math.log(p / (1.0 - p))


def _as_depth_list(value, *, default: Iterable[int] = ()) -> list[int]:
    if value is None:
        return [int(x) for x in default]
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        return [int(x.strip()) for x in text.split(",") if x.strip()]
    return [int(x) for x in value]


class AnytimeRecursiveCrossTower(nn.Module):
    """Depth-agnostic recursive cross tower.

    The large cross matrices are shared by semantic phase (ECN/LCN), while a
    small bank of LoRA experts is reused at every step.  There are no
    depth-indexed LoRA/bias/LN/step parameters, so running beyond training depth
    is a native operation rather than a slot-reuse heuristic.

    ``phase_direction``:
      e2l: start ECN-like (z=1) and monotonically move toward LCN-like (z=0)
      l2e: start LCN-like (z=0) and monotonically move toward ECN-like (z=1)
    """

    def __init__(
        self,
        input_dim: int,
        *,
        phase_direction: str,
        nominal_switch_fraction: float,
        phase_gate_mode: str = "monotonic_state",
        phase_hidden_dim: int = 64,
        phase_time_scale_init: float = 12.0,
        lora_rank: int = 32,
        lora_bank_size: int = 4,
        lora_router_hidden_dim: int = 64,
        lora_router_top_k: int = 2,
        lora_router_temperature: float = 1.0,
        budget_condition_phase: bool = False,
        budget_condition_router: bool = False,
        budget_condition_update: bool = False,
        budget_max_hint: int = 32,
        use_update_gate: bool = True,
        update_hidden_dim: int = 64,
        update_eta_max: float = 1.0,
        update_eta_init: float = 0.98,
        layer_norm: bool = True,
        net_dropout: float = 0.0,
        cross_norm: bool = True,
        cross_norm_cap: float = 1e4,
        name: str = "tower",
    ):
        super().__init__()
        if input_dim % 2 != 0:
            raise ValueError("input_dim must be even")
        if phase_direction not in {"e2l", "l2e"}:
            raise ValueError("phase_direction must be e2l or l2e")
        if phase_gate_mode not in {"fixed_schedule", "monotonic_state"}:
            raise ValueError("phase_gate_mode must be fixed_schedule or monotonic_state")
        if lora_bank_size < 0:
            raise ValueError("lora_bank_size must be >= 0")
        if lora_bank_size > 0 and lora_rank <= 0:
            raise ValueError("lora_rank must be > 0 when lora_bank_size > 0")

        self.d = int(input_dim)
        self.h = self.d // 2
        self.phase_direction = phase_direction
        self.nominal_switch_fraction = float(min(max(nominal_switch_fraction, 0.0), 1.0))
        self.phase_gate_mode = phase_gate_mode
        self.phase_time_scale_init = float(phase_time_scale_init)
        self.name = name
        self.cross_norm = bool(cross_norm)
        self.cross_norm_cap = float(cross_norm_cap)
        self.budget_condition_phase = bool(budget_condition_phase)
        self.budget_condition_router = bool(budget_condition_router)
        self.budget_condition_update = bool(budget_condition_update)
        self.budget_max_hint = max(int(budget_max_hint), 1)
        self._budget_log_denom = math.log1p(float(self.budget_max_hint))

        # Phase-shared cross core.  Index 0 = ECN, 1 = LCN.
        self.W_phase = nn.Parameter(torch.empty(2, self.h, self.d))
        self.bias_phase = nn.Parameter(torch.empty(2, self.d))

        # Monotonic transition controller.  State network starts as a zero
        # residual over the schedule-shaped time prior; it can learn deviations
        # from that prior based on the current representation.
        phase_in = self.d + 4 + (2 if self.budget_condition_phase else 0)
        # pooled state + q + delta_norm + state_norm + t/B
        # optionally + remaining-budget fraction + log-budget
        self.phase_state_net = nn.Sequential(
            nn.Linear(phase_in, phase_hidden_dim),
            nn.SiLU(),
            nn.Linear(phase_hidden_dim, 1),
        )
        self.phase_time_scale = nn.Parameter(torch.tensor(float(phase_time_scale_init)))
        self.phase_time_bias = nn.Parameter(torch.tensor(0.0))

        # Reusable LoRA bank.  B=0 preserves the shared-core model at init.
        self.lora_rank = int(lora_rank)
        self.lora_bank_size = int(lora_bank_size)
        self.lora_router_top_k = int(max(1, lora_router_top_k)) if lora_bank_size else 0
        self.lora_router_temperature = max(float(lora_router_temperature), 1e-6)
        if self.lora_bank_size > 0:
            self.lora_A_bank = nn.Parameter(
                torch.randn(self.lora_bank_size, self.lora_rank, self.d) / math.sqrt(self.d)
            )
            self.lora_B_bank = nn.Parameter(
                torch.zeros(self.lora_bank_size, self.h, self.lora_rank)
            )
            router_in = self.d + 4 + (2 if self.budget_condition_router else 0)
            # pooled state + z + q + delta_norm + t/B
            # optionally + remaining-budget fraction + log-budget
            self.lora_router = nn.Sequential(
                nn.Linear(router_in, lora_router_hidden_dim),
                nn.SiLU(),
                nn.Linear(lora_router_hidden_dim, self.lora_bank_size),
            )
        else:
            self.register_parameter("lora_A_bank", None)
            self.register_parameter("lora_B_bank", None)
            self.lora_router = None

        # State-conditioned damped residual update.
        self.use_update_gate = bool(use_update_gate)
        self.update_eta_max = float(update_eta_max)
        self.update_eta_init = float(update_eta_init)
        update_in = self.d + 5 + (2 if self.budget_condition_update else 0)
        # pooled state + z + q + delta_norm + state_norm + t/B
        # optionally + remaining-budget fraction + log-budget
        self.update_gate = nn.Sequential(
            nn.Linear(update_in, update_hidden_dim),
            nn.SiLU(),
            nn.Linear(update_hidden_dim, 1),
        )
        if self.phase_gate_mode == "fixed_schedule":
            for p in self.phase_state_net.parameters():
                p.requires_grad_(False)
            self.phase_time_scale.requires_grad_(False)
            self.phase_time_bias.requires_grad_(False)
        if not self.use_update_gate:
            for p in self.update_gate.parameters():
                p.requires_grad_(False)

        self.ln = nn.LayerNorm(self.h) if layer_norm else None
        self.gamma = nn.Parameter(torch.ones(self.h)) if not layer_norm else None
        self.beta = nn.Parameter(torch.zeros(self.h)) if not layer_norm else None
        self.dropout = nn.Dropout(net_dropout) if net_dropout > 0 else None
        self.head = nn.Linear(self.d, 1)

        self._special_init()

    def _special_init(self) -> None:
        for k in range(self.W_phase.size(0)):
            nn.init.xavier_normal_(self.W_phase[k])
        nn.init.uniform_(self.bias_phase)
        if self.lora_B_bank is not None:
            nn.init.zeros_(self.lora_B_bank)

        # Keep state corrections initially zero so the monotonic gate starts from
        # a familiar one-switch schedule prior.
        last = self.phase_state_net[-1]
        nn.init.zeros_(last.weight)
        nn.init.zeros_(last.bias)
        self.phase_time_scale.data.fill_(self.phase_time_scale_init)
        self.phase_time_bias.data.zero_()

        # Make eta approximately the requested initial residual step.
        last_u = self.update_gate[-1]
        nn.init.zeros_(last_u.weight)
        if self.update_eta_max <= 0:
            nn.init.zeros_(last_u.bias)
        else:
            p = min(max(self.update_eta_init / self.update_eta_max, 1e-5), 1.0 - 1e-5)
            nn.init.constant_(last_u.bias, _logit(p))

    def reset_after_global_linear_init(self) -> None:
        """Restore special initial conditions after BaseModel.reset_parameters()."""
        self._special_init()

    def _norm_cap(self, x: torch.Tensor) -> torch.Tensor:
        if not self.cross_norm:
            return x
        rms = x.pow(2).mean(dim=-1, keepdim=True).clamp_min(1e-12).sqrt()
        return x * (self.cross_norm_cap / rms).clamp(max=1.0)

    @staticmethod
    def _scalar_broadcast(v: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        # [B,1] -> [B,1,1] for B x heads x channels tensors.
        return v.view(v.size(0), *([1] * (x.dim() - 1)))

    def _budget_features(
        self, q: torch.Tensor, step_idx: int, requested_budget: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        budget = max(int(requested_budget), 1)
        t_frac_value = float(step_idx + 1) / float(budget)
        remaining_value = max(float(budget - (step_idx + 1)), 0.0) / float(budget)
        log_budget_value = math.log1p(float(budget)) / max(self._budget_log_denom, 1e-6)
        t_frac = q.new_full((q.size(0), 1), t_frac_value)
        remaining = q.new_full((q.size(0), 1), remaining_value)
        log_budget = q.new_full((q.size(0), 1), log_budget_value)
        return t_frac, remaining, log_budget

    def _fixed_z(self, step_idx: int, reference_steps: int, batch: int, device, dtype) -> torch.Tensor:
        switch = int(math.floor(self.nominal_switch_fraction * max(reference_steps, 1) + 0.5))
        switch = min(max(switch, 0), max(reference_steps, 1))
        if self.phase_direction == "e2l":
            value = 1.0 if step_idx < switch else 0.0
        else:
            value = 0.0 if step_idx < switch else 1.0
        return torch.full((batch, 1), value, device=device, dtype=dtype)

    def _phase_z(self, q: torch.Tensor) -> torch.Tensor:
        return 1.0 - q if self.phase_direction == "e2l" else q

    def _advance_phase(
        self,
        state: torch.Tensor,
        q: torch.Tensor,
        delta_norm: torch.Tensor,
        step_idx: int,
        reference_steps: int,
        requested_budget: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.phase_gate_mode == "fixed_schedule":
            return q, q.new_zeros(q.shape)
        pooled = state.mean(dim=1)
        state_norm = pooled.norm(dim=-1, keepdim=True) / math.sqrt(max(self.d, 1))
        t_frac, remaining_frac, log_budget = self._budget_features(q, step_idx, requested_budget)
        pieces = [pooled, q, delta_norm.unsqueeze(-1), state_norm, t_frac]
        if self.budget_condition_phase:
            pieces.extend([remaining_frac, log_budget])
        inp = torch.cat(pieces, dim=-1)
        state_logit = self.phase_state_net(inp)
        time_logit = self.phase_time_scale * (t_frac - self.nominal_switch_fraction) + self.phase_time_bias
        hazard = torch.sigmoid(state_logit + time_logit)
        q_next = q + (1.0 - q) * hazard
        return q_next.clamp(0.0, 1.0), hazard

    def _router_probs(
        self,
        state: torch.Tensor,
        z: torch.Tensor,
        q: torch.Tensor,
        delta_norm: torch.Tensor,
        step_idx: int,
        reference_steps: int,
        requested_budget: int,
    ) -> torch.Tensor | None:
        if self.lora_router is None:
            return None
        pooled = state.mean(dim=1)
        t_frac, remaining_frac, log_budget = self._budget_features(q, step_idx, requested_budget)
        pieces = [pooled, z, q, delta_norm.unsqueeze(-1), t_frac]
        if self.budget_condition_router:
            pieces.extend([remaining_frac, log_budget])
        inp = torch.cat(pieces, dim=-1)
        logits = self.lora_router(inp) / self.lora_router_temperature
        logits, top_k = self._apply_route(logits)
        if top_k < self.lora_bank_size:
            vals, idx = torch.topk(logits, k=top_k, dim=-1)
            masked = torch.full_like(logits, -torch.inf)
            masked.scatter_(1, idx, vals)
            logits = masked
        return torch.softmax(logits, dim=-1)

    def _inject_noise(self, state: torch.Tensor) -> torch.Tensor:
        """Evaluation-time stochastic recursion (PTRM-style): ``self.noise_override`` = {"sigma": s, "seed": i}
        adds N(0, (s * rms(state))^2) noise to the state after every step (deterministic per seed; the
        generator is created on first use so consecutive batches draw consecutive noise)."""
        spec = self.noise_override
        gen = getattr(self, "_noise_gen", None)
        if gen is None or gen[0] != (int(spec["seed"]), str(state.device)):
            g = torch.Generator(device=state.device)
            g.manual_seed(10_007 * (int(spec["seed"]) + 1))
            gen = ((int(spec["seed"]), str(state.device)), g)
            self._noise_gen = gen
        rms = state.pow(2).mean(dim=-1, keepdim=True).sqrt()
        eps = torch.randn(state.shape, generator=gen[1], device=state.device, dtype=state.dtype)
        return self._norm_cap(state + float(spec["sigma"]) * rms * eps)

    def _apply_route(self, logits: torch.Tensor) -> tuple[torch.Tensor, int]:
        """Evaluation-time routing override (``self.route_override``, default None = the trained router).

        route = {"allowed": [expert ids] or None, "top_k": int or None}: the router is restricted to
        the allowed experts (logits of the others -inf) and mixes the top_k of them.  Branches that
        differ only in their route share every trained parameter but use different experts.
        """
        route = getattr(self, "route_override", None)
        top_k = self.lora_router_top_k
        if not route:
            return logits, top_k
        allowed = route.get("allowed")
        n_allowed = self.lora_bank_size
        if allowed is not None:
            keep = torch.zeros(self.lora_bank_size, dtype=torch.bool, device=logits.device)
            keep[list(allowed)] = True
            logits = logits.masked_fill(~keep.unsqueeze(0), -torch.inf)
            n_allowed = len(allowed)
        if route.get("top_k"):
            top_k = int(route["top_k"])
        if route.get("gumbel"):                        # stochastic routing at test time: Gumbel-perturbed logits before the top-k
            u = torch.rand_like(logits).clamp_(1e-9, 1 - 1e-9)
            logits = logits + float(route["gumbel"]) * (-torch.log(-torch.log(u)))
        return logits, max(1, min(top_k, n_allowed))

    def _lora_bank_update(self, x: torch.Tensor, probs: torch.Tensor | None) -> torch.Tensor:
        if probs is None or self.lora_A_bank is None:
            return x.new_zeros(x.shape[:-1] + (self.h,))
        # x: B x H x d, A: K x r x d -> B x H x K x r
        low = torch.einsum("bnd,krd->bnkr", x, self.lora_A_bank)
        rank_mask = getattr(self, "rank_mask", None)          # sampled-branch latent: active LoRA rank dimensions
        if rank_mask is not None:
            low = low * rank_mask.to(low.dtype).view(1, 1, 1, -1)
        if getattr(self, "lora_nonlinear", False):           # wave 21: nonlinear experts B_k silu(A_k x)
            low = torch.nn.functional.silu(low)
        # B x H x K x r, Bbank: K x h x r -> B x H x K x h
        expert = torch.einsum("bnkr,khr->bnkh", low, self.lora_B_bank)
        return (expert * probs[:, None, :, None]).sum(dim=2)

    def _eta(
        self,
        state: torch.Tensor,
        z: torch.Tensor,
        q: torch.Tensor,
        raw_delta_norm: torch.Tensor,
        step_idx: int,
        reference_steps: int,
        requested_budget: int,
    ) -> torch.Tensor:
        if not self.use_update_gate:
            return state.new_ones((state.size(0), 1))
        pooled = state.mean(dim=1)
        state_norm = pooled.norm(dim=-1, keepdim=True) / math.sqrt(max(self.d, 1))
        t_frac, remaining_frac, log_budget = self._budget_features(q, step_idx, requested_budget)
        pieces = [pooled, z, q, raw_delta_norm.unsqueeze(-1), state_norm, t_frac]
        if self.budget_condition_update:
            pieces.extend([remaining_frac, log_budget])
        inp = torch.cat(pieces, dim=-1)
        return self.update_eta_max * torch.sigmoid(self.update_gate(inp))

    def forward(
        self,
        x: torch.Tensor,
        *,
        num_steps: int,
        reference_steps: int | None = None,
        requested_budget: int | None = None,
        return_states: bool = False,
    ) -> tuple[torch.Tensor, dict]:
        num_steps = int(num_steps)
        if num_steps <= 0:
            raise ValueError("num_steps must be >= 1")
        reference_steps = int(reference_steps or num_steps)
        requested_budget = int(requested_budget or reference_steps or num_steps)

        x = self._norm_cap(x)
        x0 = x
        state = x
        batch = x.size(0)
        q = x.new_zeros((batch, 1))
        prev_delta_norm = x.new_zeros((batch,))

        logits: list[torch.Tensor] = []
        zs: list[torch.Tensor] = []
        qs: list[torch.Tensor] = []
        hazards: list[torch.Tensor] = []
        etas: list[torch.Tensor] = []
        delta_norms: list[torch.Tensor] = []
        router_probs: list[torch.Tensor] = []
        states: list[torch.Tensor] = []

        for t in range(num_steps):
            if self.phase_gate_mode == "fixed_schedule":
                z = self._fixed_z(t, reference_steps, batch, x.device, x.dtype)
            else:
                z = self._phase_z(q)
            z_b = self._scalar_broadcast(z, state)
            anchor = (1.0 - z_b) * x0 + z_b * state

            h_e = F.linear(state, self.W_phase[0])
            h_l = F.linear(state, self.W_phase[1])
            h1 = z_b * h_e + (1.0 - z_b) * h_l

            probs = self._router_probs(
                state, z, q, prev_delta_norm, t, reference_steps, requested_budget
            )
            h1 = h1 + self._lora_bank_update(state, probs)

            if self.ln is not None:
                h2 = self.ln(h1)
            else:
                h2 = self.gamma * h1 + self.beta
            h = torch.cat([h1, h2], dim=-1)

            bias = z_b * self.bias_phase[0] + (1.0 - z_b) * self.bias_phase[1]
            raw_delta = anchor * (h + bias)
            raw_delta_norm = raw_delta.reshape(batch, -1).norm(dim=-1) / math.sqrt(raw_delta[0].numel())
            eta = self._eta(
                state, z, q, raw_delta_norm, t, reference_steps, requested_budget
            )
            eta_b = self._scalar_broadcast(eta, state)
            delta = eta_b * raw_delta
            state = self._norm_cap(state + delta)
            if self.dropout is not None:
                state = self.dropout(state)
            if getattr(self, "noise_override", None):
                state = self._inject_noise(state)

            delta_norm = delta.reshape(batch, -1).norm(dim=-1) / math.sqrt(delta[0].numel())
            logit = self.head(state)

            q_next, hazard = self._advance_phase(
                state, q, delta_norm, t, reference_steps, requested_budget
            )

            logits.append(logit)
            zs.append(z)
            qs.append(q)
            hazards.append(hazard)
            etas.append(eta)
            delta_norms.append(delta_norm)
            if probs is not None:
                router_probs.append(probs)
            if return_states:
                states.append(state)

            q = q_next
            prev_delta_norm = delta_norm.detach() if not self.training else delta_norm

        aux = {
            "step_logits": logits,
            "z": zs,
            "q": qs,
            "phase_hazard": hazards,
            "eta": etas,
            "delta_norms": delta_norms,
            "router_probs": router_probs,
            "states": states,
        }
        return logits[-1], aux

    def router_regularizers(self, aux: dict) -> tuple[torch.Tensor, torch.Tensor]:
        probs = aux.get("router_probs", [])
        if not probs:
            zero = self.W_phase.new_zeros(())
            return zero, zero
        p = torch.cat(probs, dim=0)
        mean_p = p.mean(dim=0)
        # K * sum(mean_p^2) is 1 at perfectly uniform utilization.
        balance = self.lora_bank_size * mean_p.pow(2).sum() - 1.0
        entropy = -(p.clamp_min(1e-8) * p.clamp_min(1e-8).log()).sum(dim=-1).mean()
        return balance, entropy


class AnytimeRecursiveFCNv2(bm.BaseModel):
    """Budget-adaptive recursive FCNv2 designed for test-time compute scaling."""

    def __init__(
        self,
        feature_map,
        model_id="AnytimeRecursiveFCNv2",
        gpu=0,
        learning_rate=1e-3,
        embedding_dim=16,
        num_heads=1,
        embedding_regularizer=None,
        net_regularizer=None,
        # Nominal FCN schedule used only as a prior / fixed-gate control.
        exp_num_layers=3,
        lin_num_layers=5,
        phase_gate_mode="monotonic_state",
        phase_hidden_dim=64,
        phase_time_scale_init=12.0,
        # Reusable adapter bank.
        lora_rank=32,
        lora_bank_size=4,
        lora_router_hidden_dim=64,
        lora_router_top_k=2,
        lora_router_temperature=1.0,
        router_balance_weight=0.0,
        router_entropy_weight=0.0,
        # Explicit inference-budget conditioning.  These are independent so
        # H7 can isolate where the budget signal matters.
        budget_condition_phase=False,
        budget_condition_router=False,
        budget_condition_update=False,
        budget_max_hint=32,
        # Stable recurrent update.
        use_update_gate=True,
        update_hidden_dim=64,
        update_eta_max=1.0,
        update_eta_init=0.98,
        stability_ratio_weight=0.0,
        stability_ratio_target=1.0,
        eta_late_weight=0.0,
        # Variable-depth / anytime training.
        train_min_steps=4,
        train_max_steps=12,
        train_depth_choices=None,
        train_depth_mode="random_target",
        eval_steps=8,
        anytime_bce_weight=0.0,
        anytime_supervision_depths=None,
        anytime_depth_decay=1.0,
        progressive_margin_weight=0.0,
        progressive_margin_delta=0.0,
        progressive_uncertain_margin=2.0,
        # Same-example, multi-budget training.  Each requested budget gets its
        # own rollout, which is required when the dynamics themselves are
        # budget-conditioned.
        multi_budget_training=False,
        multi_budget_choices=None,
        multi_budget_sample_count=0,
        multi_budget_depth_decay=1.0,
        cross_budget_progressive_weight=0.0,
        cross_budget_margin_delta=0.0,
        cross_budget_uncertain_margin=2.0,
        # Halting / oracle-depth distillation.
        use_halting_head=False,
        halt_hidden_dim=32,
        halt_oracle_weight=0.0,
        halt_compute_cost=0.0,
        halt_ponder_weight=0.0,
        halt_threshold=0.5,
        halt_min_steps=1,
        halt_eval_mode="fixed",  # fixed | learned
        rarity_table_path=None,
        # Optimization groups.
        embedding_lr_mult=1.0,
        shared_lr_mult=1.0,
        lora_lr_mult=1.0,
        phase_lr_mult=1.0,
        update_lr_mult=1.0,
        halt_lr_mult=1.0,
        other_lr_mult=1.0,
        # Numerical / logging.
        layer_norm=True,
        net_dropout=0.0,
        cross_norm=True,
        cross_norm_cap=1e4,
        track_anytime=True,
        aux_log_every=200,
        **kwargs,
    ):
        super().__init__(
            feature_map,
            model_id=model_id,
            gpu=gpu,
            embedding_regularizer=embedding_regularizer,
            net_regularizer=net_regularizer,
            **kwargs,
        )
        self.exp_num_layers = int(exp_num_layers)
        self.lin_num_layers = int(lin_num_layers)
        nominal_total = max(self.exp_num_layers + self.lin_num_layers, 1)
        self.train_min_steps = int(train_min_steps)
        self.train_max_steps = int(train_max_steps)
        self.eval_steps = int(eval_steps)
        if self.train_min_steps < 1 or self.train_max_steps < self.train_min_steps:
            raise ValueError("require 1 <= train_min_steps <= train_max_steps")
        if self.eval_steps < 1:
            raise ValueError("eval_steps must be >= 1")

        choices = _as_depth_list(train_depth_choices)
        if not choices:
            choices = list(range(self.train_min_steps, self.train_max_steps + 1))
        choices = sorted({x for x in choices if self.train_min_steps <= x <= self.train_max_steps})
        if not choices:
            raise ValueError("train_depth_choices has no valid depths")
        self.train_depth_choices = choices
        if train_depth_mode not in {"fixed", "random_target"}:
            raise ValueError("train_depth_mode must be fixed or random_target")
        self.train_depth_mode = train_depth_mode

        self.anytime_bce_weight = float(anytime_bce_weight)
        self.anytime_supervision_depths = _as_depth_list(anytime_supervision_depths)
        self.anytime_depth_decay = float(anytime_depth_decay)
        self.progressive_margin_weight = float(progressive_margin_weight)
        self.progressive_margin_delta = float(progressive_margin_delta)
        self.progressive_uncertain_margin = float(progressive_uncertain_margin)
        self.stability_ratio_weight = float(stability_ratio_weight)
        self.stability_ratio_target = float(stability_ratio_target)
        self.eta_late_weight = float(eta_late_weight)
        self.router_balance_weight = float(router_balance_weight)
        self.router_entropy_weight = float(router_entropy_weight)
        self.budget_condition_phase = bool(budget_condition_phase)
        self.budget_condition_router = bool(budget_condition_router)
        self.budget_condition_update = bool(budget_condition_update)
        self.budget_max_hint = max(int(budget_max_hint), 1)
        self.multi_budget_training = bool(multi_budget_training)
        mb_choices = _as_depth_list(multi_budget_choices, default=self.train_depth_choices)
        self.multi_budget_choices = sorted({
            int(x) for x in mb_choices if self.train_min_steps <= int(x) <= self.train_max_steps
        })
        if self.multi_budget_training and not self.multi_budget_choices:
            raise ValueError("multi_budget_training requires at least one valid multi_budget_choices depth")
        self.multi_budget_sample_count = max(int(multi_budget_sample_count), 0)
        self.multi_budget_depth_decay = float(multi_budget_depth_decay)
        self.cross_budget_progressive_weight = float(cross_budget_progressive_weight)
        self.cross_budget_margin_delta = float(cross_budget_margin_delta)
        self.cross_budget_uncertain_margin = float(cross_budget_uncertain_margin)
        if self.multi_budget_training and (halt_oracle_weight > 0 or halt_ponder_weight > 0):
            raise ValueError("multi_budget_training is currently intended for backbone experiments; disable halting losses")

        self.use_halting_head = bool(use_halting_head)
        self.halt_oracle_weight = float(halt_oracle_weight)
        self.halt_compute_cost = float(halt_compute_cost)
        self.halt_ponder_weight = float(halt_ponder_weight)
        self.halt_threshold = float(halt_threshold)
        self.halt_min_steps = max(1, int(halt_min_steps))
        if halt_eval_mode not in {"fixed", "learned"}:
            raise ValueError("halt_eval_mode must be fixed or learned")
        self.halt_eval_mode = halt_eval_mode
        self.rarity_table_path = None if rarity_table_path in (None, "", "null") else str(rarity_table_path)
        self._rarity_buffer_specs = []

        self.embedding_lr_mult = float(embedding_lr_mult)
        self.shared_lr_mult = float(shared_lr_mult)
        self.lora_lr_mult = float(lora_lr_mult)
        self.phase_lr_mult = float(phase_lr_mult)
        self.update_lr_mult = float(update_lr_mult)
        self.halt_lr_mult = float(halt_lr_mult)
        self.other_lr_mult = float(other_lr_mult)
        self.track_anytime = bool(track_anytime)
        self.aux_log_every = int(aux_log_every)
        self._eval_steps_override: int | None = None
        self._halt_eval_override: str | None = None

        self.embedding_layer = cl.MultiHeadFeatureEmbedding(
            feature_map, embedding_dim * num_heads, num_heads
        )
        self._init_rarity_tables(feature_map)
        input_dim = feature_map.num_fields * embedding_dim

        common = dict(
            input_dim=input_dim,
            phase_gate_mode=phase_gate_mode,
            phase_hidden_dim=phase_hidden_dim,
            phase_time_scale_init=phase_time_scale_init,
            lora_rank=lora_rank,
            lora_bank_size=lora_bank_size,
            lora_router_hidden_dim=lora_router_hidden_dim,
            lora_router_top_k=lora_router_top_k,
            lora_router_temperature=lora_router_temperature,
            budget_condition_phase=self.budget_condition_phase,
            budget_condition_router=self.budget_condition_router,
            budget_condition_update=self.budget_condition_update,
            budget_max_hint=self.budget_max_hint,
            use_update_gate=use_update_gate,
            update_hidden_dim=update_hidden_dim,
            update_eta_max=update_eta_max,
            update_eta_init=update_eta_init,
            layer_norm=layer_norm,
            net_dropout=net_dropout,
            cross_norm=cross_norm,
            cross_norm_cap=cross_norm_cap,
        )
        self.E2L = AnytimeRecursiveCrossTower(
            phase_direction="e2l",
            nominal_switch_fraction=self.exp_num_layers / float(nominal_total),
            name="E2L",
            **common,
        )
        self.L2E = AnytimeRecursiveCrossTower(
            phase_direction="l2e",
            nominal_switch_fraction=self.lin_num_layers / float(nominal_total),
            name="L2E",
            **common,
        )

        # Difficulty / stopping head uses only inference-available signals:
        # tower disagreement, fused confidence, prediction change, recurrent
        # update magnitude, and static embedding-state norm.
        self.halt_head = None
        if self.use_halting_head:
            self.halt_head = nn.Sequential(
                nn.Linear(6, halt_hidden_dim),
                nn.SiLU(),
                nn.Linear(halt_hidden_dim, 1),
            )

        self.output_activation_fn = self.get_final_activation(kwargs.get("task", "binary_classification"))
        self.reset_parameters()
        self.E2L.reset_after_global_linear_init()
        self.L2E.reset_after_global_linear_init()
        if self.halt_head is not None:
            # Bias toward continuing initially; the last step is always forced stop.
            nn.init.zeros_(self.halt_head[-1].weight)
            nn.init.constant_(self.halt_head[-1].bias, -2.0)
        self.compile(kwargs["optimizer"], kwargs["loss"], learning_rate)
        self.to_device()

    def _init_rarity_tables(self, feature_map):
        """Load optional frequency-derived rarity lookup tables.

        The table file is produced by scripts/build_rarity_tables.py.  Tables
        become checkpoint buffers, so inference does not need the original npz.
        If no table is supplied, the rarity feature is exactly zero.
        """
        if self.rarity_table_path is None:
            return
        path = Path(self.rarity_table_path).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"rarity_table_path not found: {path}")
        data = np.load(path, allow_pickle=False)
        for feature, spec in feature_map.feature_specs.items():
            idx = int(spec["index"])
            key = f"field_{idx}"
            if key not in data:
                continue
            arr = np.asarray(data[key], dtype=np.float32)
            name = f"rarity_table_{idx}"
            self.register_buffer(name, torch.from_numpy(arr), persistent=True)
            self._rarity_buffer_specs.append((idx, name))

    def _rarity_score(self, X: torch.Tensor) -> torch.Tensor:
        if not self._rarity_buffer_specs:
            return X.new_zeros((X.size(0), 1), dtype=torch.float32)
        vals = []
        for idx, name in self._rarity_buffer_specs:
            table = getattr(self, name)
            ids = X[:, idx].long().clamp(min=0, max=max(table.numel() - 1, 0))
            vals.append(table[ids].float().unsqueeze(-1))
        return torch.stack(vals, dim=0).mean(dim=0) if vals else X.new_zeros((X.size(0), 1), dtype=torch.float32)

    def compile(self, optimizer, loss, lr=1e-3):
        self.to(self.device)
        buckets = {
            "embedding": [], "shared": [], "lora": [], "phase": [],
            "update": [], "halt": [], "other": [], "refresh": [],
        }
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if name.startswith(("embedding_layer.", "branch_embedding_layers.")):
                key = "embedding"
            elif any(tok in name for tok in ("W_phase", "bias_phase")):
                key = "shared"
            elif any(tok in name for tok in ("lora_A_bank", "lora_B_bank", "lora_router")):
                key = "lora"
            elif any(tok in name for tok in ("phase_state_net", "phase_time_scale", "phase_time_bias")):
                key = "phase"
            elif "update_gate" in name:
                key = "update"
            elif "step_mlp" in name:
                key = "refresh"
            elif "halt_head" in name:
                key = "halt"
            else:
                key = "other"
            buckets[key].append(param)
        mult = {
            "embedding": self.embedding_lr_mult,
            "shared": self.shared_lr_mult,
            "lora": self.lora_lr_mult,
            "phase": self.phase_lr_mult,
            "update": self.update_lr_mult,
            "refresh": float(getattr(self, "refresh_lr_mult", 1.0)),
            "halt": self.halt_lr_mult,
            "other": self.other_lr_mult,
        }
        groups = [
            {"params": ps, "lr": lr * mult[k], "group_name": k}
            for k, ps in buckets.items() if ps
        ]
        self.optimizer = utils.set_optimizer(optimizer)(groups, lr=lr)
        self.loss_fn = getattr(torch.nn.functional, utils.set_loss(loss))

    def _regularization_loss(self):
        reg_loss = 0.0
        if not (self._embedding_regularizer or self._net_regularizer):
            return reg_loss
        emb_reg = utils.set_regularizer(self._embedding_regularizer)
        net_reg = utils.set_regularizer(self._net_regularizer)
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            pairs = emb_reg if name.startswith(("embedding_layer.", "branch_embedding_layers.")) else net_reg
            for p_norm, weight in pairs:
                reg_loss = reg_loss + (weight / p_norm) * torch.norm(param, p_norm) ** p_norm
        return reg_loss

    def set_inference_steps(self, steps: int, halt_mode: str | None = None):
        steps = int(steps)
        if steps < 1:
            raise ValueError("steps must be >= 1")
        self._eval_steps_override = steps
        if halt_mode is not None:
            if halt_mode not in {"fixed", "learned"}:
                raise ValueError("halt_mode must be fixed or learned")
            self._halt_eval_override = halt_mode
        return self

    def clear_inference_steps(self):
        self._eval_steps_override = None
        self._halt_eval_override = None
        return self

    def _sample_target_depth(self) -> int:
        if self.train_depth_mode == "fixed":
            return min(self.eval_steps, self.train_max_steps)
        return int(random.choice(self.train_depth_choices))

    def _rollout_steps(self) -> tuple[int, int]:
        if self.training:
            target = self._sample_target_depth()
            # Once a controller receives the requested budget, its rollout must
            # actually use that budget.  Padding a B=4 request to 12 steps would
            # give the controller contradictory remaining-compute features.
            if (
                self.budget_condition_phase
                or self.budget_condition_router
                or self.budget_condition_update
            ) and not self.multi_budget_training:
                return target, target
            # Legacy H0-H6 behavior: a single max-horizon trajectory supports
            # intermediate supervision and diagnostics.
            need_full = any([
                self.anytime_bce_weight > 0,
                self.progressive_margin_weight > 0,
                self.stability_ratio_weight > 0,
                self.halt_oracle_weight > 0,
                self.halt_ponder_weight > 0,
                self.track_anytime,
            ])
            return (self.train_max_steps if need_full else target), target
        steps = int(self._eval_steps_override or self.eval_steps)
        return steps, steps

    @staticmethod
    def _mean_logit(step_logit: torch.Tensor) -> torch.Tensor:
        return step_logit.mean(dim=1)

    def _fused_step_logits(self, e_aux: dict, l_aux: dict) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
        e = [self._mean_logit(x) for x in e_aux["step_logits"]]
        l = [self._mean_logit(x) for x in l_aux["step_logits"]]
        fused = [0.5 * (a + b) for a, b in zip(e, l)]
        return fused, e, l

    def _halt_features(
        self,
        fused: list[torch.Tensor],
        e_logits: list[torch.Tensor],
        l_logits: list[torch.Tensor],
        e_aux: dict,
        l_aux: dict,
        feature_emb: torch.Tensor,
        rarity: torch.Tensor,
    ) -> torch.Tensor:
        static_norm = feature_emb.reshape(feature_emb.size(0), -1).norm(dim=-1, keepdim=True)
        static_norm = torch.log1p(static_norm)
        feats = []
        prev = None
        for t, (f, e, l) in enumerate(zip(fused, e_logits, l_logits)):
            pe, pl, pf = torch.sigmoid(e), torch.sigmoid(l), torch.sigmoid(f)
            disagreement = (pe - pl).abs()
            confidence = (2.0 * pf - 1.0).abs()
            pred_change = torch.zeros_like(f) if prev is None else (f - prev).abs()
            update_mag = 0.5 * (
                e_aux["delta_norms"][t].unsqueeze(-1) + l_aux["delta_norms"][t].unsqueeze(-1)
            )
            update_mag = torch.log1p(update_mag)
            feats.append(torch.cat([disagreement, confidence, pred_change, update_mag, static_norm, rarity], dim=-1))
            prev = f
        return torch.stack(feats, dim=1)  # B x T x 5

    def _halt_distribution(self, stop_probs: torch.Tensor) -> torch.Tensor:
        # stop_probs: B x T. Last depth is forced to absorb remaining mass.
        b, t = stop_probs.shape
        surv = stop_probs.new_ones((b,))
        masses = []
        for i in range(t):
            if i == t - 1:
                mass = surv
            else:
                p = stop_probs[:, i]
                mass = surv * p
                surv = surv * (1.0 - p)
            masses.append(mass)
        return torch.stack(masses, dim=1)

    def _halt_outputs(
        self,
        fused: list[torch.Tensor],
        e_logits: list[torch.Tensor],
        l_logits: list[torch.Tensor],
        e_aux: dict,
        l_aux: dict,
        feature_emb: torch.Tensor,
        rarity: torch.Tensor,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
        if self.halt_head is None:
            return None, None, None
        feats = self._halt_features(fused, e_logits, l_logits, e_aux, l_aux, feature_emb, rarity)
        stop_probs = torch.sigmoid(self.halt_head(feats).squeeze(-1))
        if self.halt_min_steps > 1:
            stop_probs = stop_probs.clone()
            stop_probs[:, : self.halt_min_steps - 1] = 0.0
        dist = self._halt_distribution(stop_probs)
        expected_depth = (dist * torch.arange(1, dist.size(1) + 1, device=dist.device, dtype=dist.dtype)).sum(dim=1)
        return stop_probs, dist, expected_depth

    def _select_learned_halt(self, fused: list[torch.Tensor], stop_probs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        b, t = stop_probs.shape
        eligible = stop_probs >= self.halt_threshold
        if self.halt_min_steps > 1:
            eligible[:, : self.halt_min_steps - 1] = False
        any_stop = eligible.any(dim=1)
        first = eligible.float().argmax(dim=1)
        first = torch.where(any_stop, first, first.new_full(first.shape, t - 1))
        stack = torch.stack([x.squeeze(-1) for x in fused], dim=1)
        chosen = stack.gather(1, first.unsqueeze(1))
        return chosen, first + 1

    def _anytime_bce(self, fused: list[torch.Tensor], y_true: torch.Tensor) -> torch.Tensor:
        if self.anytime_supervision_depths:
            depths = [d for d in self.anytime_supervision_depths if 1 <= d <= len(fused)]
        else:
            depths = list(range(1, len(fused) + 1))
        if not depths:
            return y_true.new_zeros(())
        terms, weights = [], []
        for d in depths:
            terms.append(F.binary_cross_entropy_with_logits(fused[d - 1], y_true))
            weights.append(self.anytime_depth_decay ** (len(fused) - d))
        w = y_true.new_tensor(weights)
        return (torch.stack(terms) * w).sum() / w.sum().clamp_min(1e-12)

    def _progressive_margin(self, fused: list[torch.Tensor], y_true: torch.Tensor) -> torch.Tensor:
        if len(fused) < 2:
            return y_true.new_zeros(())
        sign = 2.0 * y_true - 1.0
        terms = []
        for a, b in zip(fused[:-1], fused[1:]):
            ma, mb = sign * a, sign * b
            mask = ma.abs() < self.progressive_uncertain_margin
            if mask.any():
                terms.append(F.relu(ma + self.progressive_margin_delta - mb)[mask].mean())
        return torch.stack(terms).mean() if terms else y_true.new_zeros(())

    def _cross_budget_progressive(
        self, budget_logits: list[torch.Tensor], y_true: torch.Tensor
    ) -> torch.Tensor:
        if len(budget_logits) < 2:
            return y_true.new_zeros(())
        sign = 2.0 * y_true - 1.0
        terms = []
        for low, high in zip(budget_logits[:-1], budget_logits[1:]):
            m_low, m_high = sign * low, sign * high
            mask = m_low.abs() < self.cross_budget_uncertain_margin
            if mask.any():
                terms.append(
                    F.relu(m_low + self.cross_budget_margin_delta - m_high)[mask].mean()
                )
        return torch.stack(terms).mean() if terms else y_true.new_zeros(())

    def _stability_ratio_loss(self, *auxes: dict) -> torch.Tensor:
        terms = []
        for aux in auxes:
            ds = aux["delta_norms"]
            for prev, cur in zip(ds[:-1], ds[1:]):
                ratio = cur / prev.detach().clamp_min(1e-6)
                terms.append(F.relu(ratio - self.stability_ratio_target).pow(2).mean())
        return torch.stack(terms).mean() if terms else self.E2L.W_phase.new_zeros(())

    def _eta_late_loss(self, *auxes: dict) -> torch.Tensor:
        terms = []
        for aux in auxes:
            etas = aux["eta"]
            if not etas:
                continue
            start = max(0, len(etas) // 2)
            terms.extend([e.mean() for e in etas[start:]])
        return torch.stack(terms).mean() if terms else self.E2L.W_phase.new_zeros(())

    def _oracle_halting_loss(self, fused: list[torch.Tensor], y_true: torch.Tensor, dist: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        per_depth = []
        for d, lg in enumerate(fused, start=1):
            bce = F.binary_cross_entropy_with_logits(lg, y_true, reduction="none").squeeze(-1)
            cost = self.halt_compute_cost * (d / float(len(fused)))
            per_depth.append(bce + cost)
        cost_mat = torch.stack(per_depth, dim=1)
        oracle = cost_mat.argmin(dim=1)
        chosen_prob = dist.gather(1, oracle.unsqueeze(1)).squeeze(1).clamp_min(1e-8)
        return -chosen_prob.log().mean(), oracle + 1

    def _router_regs(self, e_aux: dict, l_aux: dict) -> tuple[torch.Tensor, torch.Tensor]:
        eb, ee = self.E2L.router_regularizers(e_aux)
        lb, le = self.L2E.router_regularizers(l_aux)
        return 0.5 * (eb + lb), 0.5 * (ee + le)

    def _diagnostic_payload(self, e_aux: dict, l_aux: dict, stop_probs, expected_depth, target_depth: int) -> dict:
        payload = {"anytime/target_depth": float(target_depth)}
        for prefix, aux in (("E2L", e_aux), ("L2E", l_aux)):
            for t, z in enumerate(aux["z"], start=1):
                payload[f"anytime/{prefix}/z_{t}"] = float(z.detach().mean())
            for t, eta in enumerate(aux["eta"], start=1):
                payload[f"anytime/{prefix}/eta_{t}"] = float(eta.detach().mean())
            for t, dn in enumerate(aux["delta_norms"], start=1):
                payload[f"anytime/{prefix}/delta_norm_{t}"] = float(dn.detach().mean())
            if aux["router_probs"]:
                p = torch.stack([x.detach().mean(dim=0) for x in aux["router_probs"]]).mean(dim=0)
                for k, v in enumerate(p):
                    payload[f"anytime/{prefix}/router_usage_{k}"] = float(v)
        if stop_probs is not None:
            for t in range(stop_probs.size(1)):
                payload[f"anytime/halt_p_{t+1}"] = float(stop_probs[:, t].detach().mean())
        if expected_depth is not None:
            payload["anytime/expected_depth"] = float(expected_depth.detach().mean())
        return payload

    def _run_rollout(
        self, feature_emb: torch.Tensor, rarity: torch.Tensor, *, steps: int, requested_budget: int | None = None
    ) -> dict:
        budget = int(requested_budget or steps)
        _, e_aux = self.E2L(
            feature_emb, num_steps=int(steps), reference_steps=budget, requested_budget=budget
        )
        _, l_aux = self.L2E(
            feature_emb, num_steps=int(steps), reference_steps=budget, requested_budget=budget
        )
        fused, e_logits, l_logits = self._fused_step_logits(e_aux, l_aux)
        stop_probs, halt_dist, expected_depth = self._halt_outputs(
            fused, e_logits, l_logits, e_aux, l_aux, feature_emb, rarity
        )
        return {
            "fused": fused, "e_logits": e_logits, "l_logits": l_logits,
            "e_aux": e_aux, "l_aux": l_aux,
            "stop_probs": stop_probs, "halt_dist": halt_dist,
            "expected_depth": expected_depth, "budget": budget,
        }

    def analyze_batch(self, inputs, *, steps: int | None = None) -> dict:
        """Return per-depth tensors for diagnostic/evaluation scripts."""
        X, y = self.inputs_to_device(inputs)
        feature_emb = self.embedding_layer(X)
        rarity = self._rarity_score(X)
        steps = int(steps or self._eval_steps_override or self.eval_steps)
        run = self._run_rollout(feature_emb, rarity, steps=steps, requested_budget=steps)
        return {
            "y_true": y,
            "fused_logits": run["fused"],
            "e2l_logits": run["e_logits"],
            "l2e_logits": run["l_logits"],
            "e2l_aux": run["e_aux"],
            "l2e_aux": run["l_aux"],
            "stop_probs": run["stop_probs"],
            "halt_dist": run["halt_dist"],
            "expected_depth": run["expected_depth"],
        }

    def _sample_multi_budgets(self) -> list[int]:
        budgets = list(self.multi_budget_choices)
        count = self.multi_budget_sample_count
        if count <= 0 or count >= len(budgets):
            return budgets
        # A contiguous budget window keeps the progressive loss local (e.g.
        # 4→8, 8→12, 12→16) and bounds memory while still covering all budget
        # transitions over training.
        start = random.randint(0, len(budgets) - count)
        return budgets[start : start + count]

    def _forward_multi_budget(
        self, X: torch.Tensor, y_true_original: torch.Tensor, feature_emb: torch.Tensor, rarity: torch.Tensor
    ) -> dict:
        budgets = self._sample_multi_budgets()
        runs = {
            b: self._run_rollout(feature_emb, rarity, steps=b, requested_budget=b)
            for b in budgets
        }
        primary_budget = min(budgets, key=lambda b: (abs(b - self.eval_steps), b))
        primary = runs[primary_budget]
        logit = primary["fused"][-1]
        y_pred = self.output_activation_fn(logit) if self.output_activation_fn else logit
        out = {"y_pred": y_pred}
        components = {}

        if self.loss_fn is not None:
            y_true = y_true_original.view_as(y_pred)
            budget_losses = []
            budget_weights = []
            for b in budgets:
                run = runs[b]
                fused = run["fused"][-1]
                e_logit = run["e_logits"][-1]
                l_logit = run["l_logits"][-1]
                loss = F.binary_cross_entropy_with_logits(fused, y_true)
                loss_e = F.binary_cross_entropy_with_logits(e_logit, y_true)
                loss_l = F.binary_cross_entropy_with_logits(l_logit, y_true)
                zero_e, zero_l = loss_e.new_zeros(()), loss_l.new_zeros(())
                w_e = torch.where(loss_e > loss, loss_e - loss, zero_e)
                w_l = torch.where(loss_l > loss, loss_l - loss, zero_l)
                budget_losses.append(loss + loss_e * w_e + loss_l * w_l)
                budget_weights.append(self.multi_budget_depth_decay ** (max(budgets) - b))
            w = y_true.new_tensor(budget_weights)
            total = (torch.stack(budget_losses) * w).sum() / w.sum().clamp_min(1e-12)
            components["multi_budget_tri_bce"] = total.detach()

            if self.cross_budget_progressive_weight > 0:
                val = self._cross_budget_progressive(
                    [runs[b]["fused"][-1] for b in budgets], y_true
                )
                total = total + self.cross_budget_progressive_weight * val
                components["cross_budget_progressive"] = val.detach()

            if self.progressive_margin_weight > 0:
                vals = [self._progressive_margin(runs[b]["fused"], y_true) for b in budgets]
                val = torch.stack(vals).mean()
                total = total + self.progressive_margin_weight * val
                components["progressive_margin"] = val.detach()
            if self.stability_ratio_weight > 0:
                vals = [
                    self._stability_ratio_loss(runs[b]["e_aux"], runs[b]["l_aux"])
                    for b in budgets
                ]
                val = torch.stack(vals).mean()
                total = total + self.stability_ratio_weight * val
                components["stability_ratio"] = val.detach()
            if self.router_balance_weight > 0 or self.router_entropy_weight > 0:
                regs = [self._router_regs(runs[b]["e_aux"], runs[b]["l_aux"]) for b in budgets]
                balance = torch.stack([x[0] for x in regs]).mean()
                entropy = torch.stack([x[1] for x in regs]).mean()
                if self.router_balance_weight > 0:
                    total = total + self.router_balance_weight * balance
                    components["router_balance"] = balance.detach()
                if self.router_entropy_weight > 0:
                    total = total + self.router_entropy_weight * entropy
                    components["router_entropy"] = entropy.detach()
            total = total + self._regularization_loss()
            out["loss"] = total

        step = getattr(self, "_total_batches", 0)
        do_log = self.training and self.track_anytime and (step % self.aux_log_every == 0)
        if do_log:
            payload = self._diagnostic_payload(
                primary["e_aux"], primary["l_aux"], primary["stop_probs"],
                primary["expected_depth"], primary_budget
            )
            payload["anytime/multi_budget_min"] = float(min(budgets))
            payload["anytime/multi_budget_max"] = float(max(budgets))
            for k, v in components.items():
                payload[f"loss/{k}"] = float(v)
            tracking.log(payload, step=step)
        return out

    def forward(self, inputs):
        X, y_true_original = self.inputs_to_device(inputs)
        feature_emb = self.embedding_layer(X)
        rarity = self._rarity_score(X)
        if self.training and self.multi_budget_training:
            return self._forward_multi_budget(X, y_true_original, feature_emb, rarity)

        rollout_steps, target_depth = self._rollout_steps()
        has_budget_conditioning = (
            self.budget_condition_phase
            or self.budget_condition_router
            or self.budget_condition_update
        )
        requested_budget = target_depth if has_budget_conditioning else rollout_steps
        run = self._run_rollout(
            feature_emb, rarity, steps=rollout_steps, requested_budget=requested_budget
        )
        e_aux, l_aux = run["e_aux"], run["l_aux"]
        fused, e_logits, l_logits = run["fused"], run["e_logits"], run["l_logits"]
        stop_probs, halt_dist, expected_depth = (
            run["stop_probs"], run["halt_dist"], run["expected_depth"]
        )

        halt_mode = self._halt_eval_override or self.halt_eval_mode
        selected_depths = None
        if (not self.training) and halt_mode == "learned" and stop_probs is not None:
            logit, selected_depths = self._select_learned_halt(fused, stop_probs)
        else:
            logit = fused[target_depth - 1]

        y_pred = self.output_activation_fn(logit) if self.output_activation_fn else logit
        out = {"y_pred": y_pred}
        components = {}

        if self.loss_fn is not None:
            y_true = y_true_original.view_as(y_pred)
            target_idx = target_depth - 1
            bce = F.binary_cross_entropy_with_logits
            loss = bce(fused[target_idx], y_true)
            loss_e = bce(e_logits[target_idx], y_true)
            loss_l = bce(l_logits[target_idx], y_true)
            zero_e, zero_l = loss_e.new_zeros(()), loss_l.new_zeros(())
            w_e = torch.where(loss_e > loss, loss_e - loss, zero_e)
            w_l = torch.where(loss_l > loss, loss_l - loss, zero_l)
            total = loss + loss_e * w_e + loss_l * w_l
            components["tri_bce"] = total.detach()

            if self.anytime_bce_weight > 0:
                val = self._anytime_bce(fused, y_true)
                total = total + self.anytime_bce_weight * val
                components["anytime_bce"] = val.detach()
            if self.progressive_margin_weight > 0:
                val = self._progressive_margin(fused, y_true)
                total = total + self.progressive_margin_weight * val
                components["progressive_margin"] = val.detach()
            if self.stability_ratio_weight > 0:
                val = self._stability_ratio_loss(e_aux, l_aux)
                total = total + self.stability_ratio_weight * val
                components["stability_ratio"] = val.detach()
            if self.eta_late_weight > 0:
                val = self._eta_late_loss(e_aux, l_aux)
                total = total + self.eta_late_weight * val
                components["eta_late"] = val.detach()
            if self.router_balance_weight > 0 or self.router_entropy_weight > 0:
                balance, entropy = self._router_regs(e_aux, l_aux)
                if self.router_balance_weight > 0:
                    total = total + self.router_balance_weight * balance
                    components["router_balance"] = balance.detach()
                if self.router_entropy_weight > 0:
                    total = total + self.router_entropy_weight * entropy
                    components["router_entropy"] = entropy.detach()
            oracle_depth = None
            if self.halt_oracle_weight > 0 and halt_dist is not None:
                val, oracle_depth = self._oracle_halting_loss(fused, y_true, halt_dist)
                total = total + self.halt_oracle_weight * val
                components["halt_oracle"] = val.detach()
                components["oracle_depth"] = oracle_depth.float().mean().detach()
            if self.halt_ponder_weight > 0 and expected_depth is not None:
                val = expected_depth.mean() / float(max(rollout_steps, 1))
                total = total + self.halt_ponder_weight * val
                components["halt_ponder"] = val.detach()

            total = total + self._regularization_loss()
            out["loss"] = total

        step = getattr(self, "_total_batches", 0)
        do_log = self.training and self.track_anytime and (step % self.aux_log_every == 0)
        if do_log:
            payload = self._diagnostic_payload(
                e_aux, l_aux, stop_probs, expected_depth, target_depth
            )
            for k, v in components.items():
                payload[f"loss/{k}"] = float(v)
            tracking.log(payload, step=step)

        if selected_depths is not None:
            out["selected_depth"] = selected_depths
        return out
