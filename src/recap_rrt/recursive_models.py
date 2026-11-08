"""Experimental models for the ctr-campaign campaign (new file; frozen modules untouched).

Everything here composes with the stable rank-96 multi-budget recipe and is
shared across recursive depth, so nothing introduces depth-indexed parameters.

* ``MultiTowerRecursiveFCNv2``: K recursive Bag-of-LoRA towers (alternating
  E2L / L2E phase orders) on one shared embedding, fused by the mean logit and
  trained with the K-tower generalisation of the conditional tri-BCE objective.
  K = 2 is the original two-tower architecture.
* ``MultiTowerFCNv2``: the parameter-matched K-tower untied FCNv2 control.
* ``RecursiveCrossTower``: the stable tower with two optional shared
  operators taken from the new-architecture campaign, global RMSNorm
  pre-normalisation of the core input (wave 18) and an identity-started
  two-scale causal exponential memory (wave 10).
* ``apply_late_step_damping``: the deployment rule eta_t = eta_S * rho^(t-S)
  for steps beyond the training depth S.  With the RMS norm cap on the state
  the per-step update norm is then bounded by a geometric series, so the state
  sequence is Cauchy and the logits converge.
* ``budget_advantage_weight``: GRPO-style group-relative advantage over the
  sampled budgets of the same example, used to re-weight the per-budget BCE.
"""
from __future__ import annotations

import contextlib
import logging
import math
import os
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import anytime_rrt
from . import dcn_layers
from . import tracking


def _inverse_tanh_fraction(value: float, maximum: float) -> float:
    fraction = min(max(float(value) / max(float(maximum), 1e-8), -0.999), 0.999)
    return float(math.atanh(fraction))


class GlobalRMSNorm(nn.Module):
    """RMSNorm over the full flattened state (mean preserved), with affine."""

    def __init__(self, width: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(int(width)))
        self.bias = nn.Parameter(torch.zeros(int(width)))
        self.eps = float(eps)

    def reset_parameters(self) -> None:
        nn.init.ones_(self.weight)
        nn.init.zeros_(self.bias)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        normalized = state * torch.rsqrt(state.square().mean(dim=-1, keepdim=True) + self.eps)
        return normalized * self.weight + self.bias


def masked_tri_bce(tower_logits, y_true, keep: float, mode: str, training: bool):
    """Branch-masked K-tower tri-BCE.

    A random branch mask M (B x K) selects which branches take part in this step's fused prediction:
    mode "step" draws ONE subset for the whole batch (branch dropout; python RNG, identical on all ranks),
    mode "example" draws a subset per example (bagging: every branch trains on a different random
    subset of the data).  At least one branch survives per row.  fused = mean over the surviving
    branches; the conditional per-branch terms are taken over the rows where the branch is active.
    Returns (loss, fused logits of the surviving branches).
    """
    L = torch.stack([t.reshape(-1) for t in tower_logits], dim=1)          # B x K
    y = y_true.reshape(-1)
    K = L.size(1)
    if training and keep < 1.0 and K > 1:
        if mode == "step":
            m = [1.0 if random.random() < keep else 0.0 for _ in range(K)]
            if sum(m) == 0:
                m[random.randrange(K)] = 1.0
            M = L.new_tensor(m).unsqueeze(0).expand_as(L)
        else:
            M = (torch.rand_like(L) < keep).float()
            none = (M.sum(dim=1) == 0).nonzero().reshape(-1)
            if none.numel():
                M = M.clone()
                M[none, torch.randint(0, K, (none.numel(),), device=L.device)] = 1.0
    else:
        M = torch.ones_like(L)
    fused = (L * M).sum(dim=1) / M.sum(dim=1)
    loss = F.binary_cross_entropy_with_logits(fused, y)
    total = loss
    for k in range(K):
        mk = M[:, k]
        if float(mk.sum()) == 0.0:
            continue
        loss_k = (F.binary_cross_entropy_with_logits(L[:, k], y, reduction="none") * mk).sum() / mk.sum()
        total = total + loss_k * torch.where(loss_k > loss, loss_k - loss, loss_k.new_zeros(()))
    return total, fused


class RecursiveCrossTower(anytime_rrt.AnytimeRecursiveCrossTower):
    """Stable recursive tower with optional pre-norm and multiscale memory.

    With ``pre_norm="none"`` and ``memory_scales=0`` the forward pass is
    identical to ``AnytimeRecursiveCrossTower``.
    """

    def __init__(
        self,
        *args,
        pre_norm: str = "none",
        memory_scales: int = 0,
        memory_hidden_dim: int = 128,
        memory_max_timescale: float = 16.0,
        memory_residual_max: float = 0.25,
        budget_agnostic: bool = False,
        reference_depth: int = 8,
        hrm_period: int = 0,
        hrm_readout: str = "l_plus_h",
        hrm_detach_segments: bool = False,
        hrm_one_step_grad: bool = False,
        hrm_eta_init: float = 0.5,
        phase_pattern: str | None = None,
        stars_weight: float = 0.0,
        stars_target: float = 1.0,
        stars_power_iters: int = 2,
        stars_min_step: int = 4,
        anchor_power: int = 1,
        gated_cross: bool = False,
        step_mlp_hidden: int = 0,
        step_mlp_every: int = 1,
        step_mlp_kind: str = "flat",
        step_mlp_fields: int = 0,
        cross_nonlinear_hidden: int = 0,
        lora_nonlinear: bool = False,
        hyper_rank: int = 0,
        state_width: int = 0,
        step_mlp_untied: int = 1,
        step_mlp_cap: float = 0.0,
        clock_mode: str = "depth",
        bank_per_phase: bool = False,
        phase_norm_affine: bool = False,
        head_order_film: bool = False,
        tie_phase_cores: bool = False,
        stochastic_sigma0: float = 0.0,
        stochastic_beta: float = 0.0,
        stochastic_mean: bool = True,
        **kwargs,
    ):
        in_dim = None
        if int(state_width) > 0:                       # wave 21: recursion in a wider latent state (input projected in, operator/readout at state_width)
            in_dim = int(kwargs.pop("input_dim")); kwargs["input_dim"] = int(state_width)
        super().__init__(*args, **kwargs)
        self.in_proj = nn.Linear(in_dim, self.d, bias=False) if in_dim else None
        # tie_phase_cores: ONE cross core (W, bias) serves both the E and the L phase; the phases then
        # differ only through the anchor (state vs input) and, with bank_per_phase, through their
        # experts.  The parameter keeps its leading phase axis (size 1) so init, tying and
        # checkpoint naming are unchanged; index -1 resolves to the single core.
        self.tie_phase_cores = bool(tie_phase_cores)
        if self.tie_phase_cores:
            self.W_phase = nn.Parameter(self.W_phase.data[:1].clone())
            self.bias_phase = nn.Parameter(self.bias_phase.data[:1].clone())
        # Schedule-order generalisation (wave 1i).
        # clock_mode "order": the controller clock is log2 of the interaction order reached so far,
        #   n_E + log2(1 + n_L), divided by the reference depth, instead of the depth t/B; any
        #   E/L order lands on the axis the state statistics actually follow.
        # bank_per_phase: the first half of the LoRA bank is routable only on E steps, the second
        #   half only on L steps (no new parameters).
        # phase_norm_affine: a separate affine (per symbol) after the global RMS pre-norm.
        # head_order_film: the per-step head reads state * (1 + g(n)) + b(n) with n = (n_E, n_L),
        #   zero-initialised (identity at start).
        self.clock_mode = str(clock_mode)
        if self.clock_mode not in ("depth", "order"):
            raise ValueError("clock_mode must be depth or order")
        self.bank_per_phase = bool(bank_per_phase)
        self.phase_norm_affine = bool(phase_norm_affine)
        self.head_order_film = bool(head_order_film)
        self._order_counts = (0, 0)
        # GRAM-style stochastic recursion: after every step z = u + rms(u) * (mu(u) + sigma(u) * n), n ~ N(0, I) while
        # training (or when sample_override is set at evaluation); n = 0 otherwise.  KL(N(mu, sigma^2) || N(0, sigma0^2))
        # per element, averaged, returned in aux["kl"] and weighted by stochastic_beta in the model loss.
        self.stochastic_sigma0 = float(stochastic_sigma0)
        self.stochastic_beta = float(stochastic_beta)
        self.stochastic_mean = bool(stochastic_mean)      # False: learned variance only (mu = 0), no extra mean update
        if self.stochastic_sigma0 > 0:
            self.stoch_head = nn.Linear(self.d, 2 * self.d)
        else:
            self.stoch_head = None
        # Degree of the ECN-like update in the state: 1 = x * (Wx + b) (quadratic update,
        # order doubles per step); p > 1 replaces the anchor x by sign(x)|x|^p rescaled to
        # the RMS of x (degree p+1, order multiplies by p+1 per step, ECN-scale magnitude).
        # The LCN-like update (anchor x0) is unaffected.
        self.anchor_power = max(int(anchor_power), 1)
        # ---- wave 21: nonlinearity inside the recursion.
        # step_mlp_hidden > 0: after every step_mlp_every-th cross step the state takes a residual per-channel nonlinear
        #   step s + rms(s) * MLP(LN(s / rms)) (a transformer block = mixing + FFN; the cross is only the mixing).  kind
        #   "flat" = one MLP on the whole per-head state; "field" = a tiny MLP shared across the F field vectors (token-wise FFN).
        # cross_nonlinear_hidden > 0: the cross core becomes W2 silu(W1 s): the anchor product stays, the order is no longer polynomial.
        # lora_nonlinear: experts are B_k silu(A_k s) instead of B_k A_k s (no new parameters).
        # hyper_rank > 0: instance-conditioned operator W(x0) = W + U diag(tanh(g x0)) V (zero U: identity at start).
        self.step_mlp_hidden = int(step_mlp_hidden); self.step_mlp_every = max(int(step_mlp_every), 1)
        self.step_mlp_kind = str(step_mlp_kind); self.step_mlp_fields = int(step_mlp_fields)
        self.step_mlp = self.step_mlp_norm = None
        if self.step_mlp_hidden > 0:
            if self.step_mlp_kind not in ("flat", "field"):
                raise ValueError("step_mlp_kind must be flat or field")
            if self.step_mlp_kind == "field" and (self.step_mlp_fields <= 0 or self.h % self.step_mlp_fields):
                raise ValueError("field FFN needs the per-head half width to be a multiple of the field count")
            width = self.d if self.step_mlp_kind == "flat" else 2 * (self.h // self.step_mlp_fields)
            self.step_mlp_norm = nn.LayerNorm(width)
            self.step_mlp = nn.Sequential(nn.Linear(width, self.step_mlp_hidden), nn.SiLU(), nn.Linear(self.step_mlp_hidden, width))
        # step_mlp_untied n > 1: the r-th refresh of a rollout uses module r mod n (stage-specific recombination; cycles with depth)
        self.step_mlp_untied = max(int(step_mlp_untied), 1)
        self.step_mlp_cap = float(step_mlp_cap)      # > 0: the refresh delta RMS is capped at step_mlp_cap x the state RMS (row-wise)
        self.step_mlp_extra = nn.ModuleList()
        if self.step_mlp is not None:
            width = self.step_mlp_norm.normalized_shape[0]
            for _ in range(self.step_mlp_untied - 1):
                self.step_mlp_extra.append(nn.Sequential(nn.LayerNorm(width), nn.Linear(width, self.step_mlp_hidden), nn.SiLU(), nn.Linear(self.step_mlp_hidden, width)))
        self.cross_nonlinear_hidden = int(cross_nonlinear_hidden)
        if self.cross_nonlinear_hidden > 0:
            n_phase = self.W_phase.size(0)
            self.W1_phase = nn.Parameter(torch.empty(n_phase, self.cross_nonlinear_hidden, self.d))
            self.W_phase = nn.Parameter(torch.empty(n_phase, self.h, self.cross_nonlinear_hidden))
        self.lora_nonlinear = bool(lora_nonlinear)
        self.hyper_rank = int(hyper_rank)
        if self.hyper_rank > 0:
            self.hyper_g = nn.Linear(self.d, self.hyper_rank)
            self.hyper_V = nn.Linear(self.d, self.hyper_rank, bias=False)
            self.hyper_U = nn.Linear(self.hyper_rank, self.h, bias=False)
        self._reset_new_operator_parameters()
        self.gated_cross = bool(gated_cross)
        if self.gated_cross:      # low-rank gate on the interaction term: 2*sigmoid(V U s + b), identity at init
            self.gate_U = nn.Linear(self.d, 96, bias=False)
            self.gate_V = nn.Linear(96, self.d, bias=True)
            nn.init.zeros_(self.gate_V.weight); nn.init.zeros_(self.gate_V.bias)
        # STARS-style Jacobian spectral-radius regularisation of the fast step.
        self.stars_weight = float(stars_weight)
        self.stars_target = float(stars_target)
        self.stars_power_iters = max(int(stars_power_iters), 1)
        self.stars_min_step = max(int(stars_min_step), 1)
        self.stars_active = False   # set per batch by the model (one tower per batch)
        # Explicit ECN/LCN pattern (e.g. "ELELELEL", "EEELEEEL"); extended cyclically
        # beyond its length.  None keeps the contiguous single-switch schedule.
        self.phase_pattern = None
        if phase_pattern:
            pattern = str(phase_pattern).upper()
            if not pattern or any(c not in "ELNRP" for c in pattern):
                raise ValueError("phase_pattern must be a non-empty string over {E, L, N, R, P}")
            self.phase_pattern = pattern
        # Within-branch HRM (slow state z_H updated every hrm_period fast steps).
        self.hrm_period = int(hrm_period)
        self.hrm_readout = str(hrm_readout)
        self.hrm_detach_segments = bool(hrm_detach_segments)
        self.hrm_one_step_grad = bool(hrm_one_step_grad)
        self.hrm_eta_init = float(hrm_eta_init)
        if self.hrm_readout not in {"l", "l_plus_h"}:
            raise ValueError("hrm_readout must be l or l_plus_h")
        if self.hrm_period > 0:
            self.hrm_core = nn.Linear(self.d, self.h, bias=False)      # slow cross core W_H
            self.hrm_gamma = nn.Parameter(torch.ones(self.h))
            self.hrm_beta = nn.Parameter(torch.zeros(self.h))
            self.hrm_bias = nn.Parameter(torch.empty(self.d))
            self.hrm_eta_raw = nn.Parameter(torch.zeros(()))
            self.hrm_inject = nn.Linear(self.d, self.h, bias=False)    # U_H: slow -> fast conditioning
            self.hrm_head = nn.Linear(self.d, 1) if self.hrm_readout == "l_plus_h" else None
        else:
            self.hrm_core = None
            self.hrm_inject = None
            self.hrm_head = None
            for name in ("hrm_gamma", "hrm_beta", "hrm_bias", "hrm_eta_raw"):
                self.register_parameter(name, None)
        # Budget-agnostic mode: the ECN/LCN switch point and the controller clock
        # t/B are referenced to a fixed depth and the clock is frozen beyond it,
        # so the trajectory does not depend on the requested depth (prefix equality).
        self.budget_agnostic = bool(budget_agnostic)
        self.reference_depth = max(int(reference_depth), 1)
        if self.budget_agnostic and (
            self.budget_condition_phase or self.budget_condition_router or self.budget_condition_update
        ):
            raise ValueError("budget_agnostic requires all budget_condition_* flags to be false")
        if pre_norm not in {"none", "global_rmsnorm"}:
            raise ValueError("pre_norm must be none or global_rmsnorm")
        self.pre_norm_mode = pre_norm
        self.state_pre_norm = GlobalRMSNorm(self.d) if pre_norm == "global_rmsnorm" else None
        if self.phase_norm_affine:
            if self.state_pre_norm is None:
                raise ValueError("phase_norm_affine requires pre_norm: global_rmsnorm")
            self.phase_affine_weight = nn.Parameter(torch.ones(2, self.d))
            self.phase_affine_bias = nn.Parameter(torch.zeros(2, self.d))
        else:
            self.register_parameter("phase_affine_weight", None)
            self.register_parameter("phase_affine_bias", None)
        self.head_film = nn.Linear(2, 2 * self.d) if self.head_order_film else None
        if self.bank_per_phase and (self.lora_bank_size < 2 or self.lora_router_top_k > self.lora_bank_size // 2):
            raise ValueError("bank_per_phase needs an even bank with top_k <= bank_size / 2")

        self.memory_scales = int(memory_scales)
        self.memory_max_timescale = float(memory_max_timescale)
        self.memory_residual_max = float(memory_residual_max)
        if self.memory_scales > 0:
            hidden = max(int(memory_hidden_dim), 16)
            self.memory_norm = nn.LayerNorm(self.d)
            self.memory_query = nn.Linear(self.d, self.memory_scales)
            self.memory_gate = nn.Sequential(
                nn.Linear(2 * self.d, hidden), nn.SiLU(), nn.Linear(hidden, self.d)
            )
            self.memory_scale_raw = nn.Parameter(torch.zeros(()))
            self.memory_decay_raw = nn.Parameter(torch.zeros(self.memory_scales))
        else:
            self.memory_norm = None
            self.memory_query = None
            self.memory_gate = None
            self.register_parameter("memory_scale_raw", None)
            self.register_parameter("memory_decay_raw", None)
        self._reset_recursive_parameters()

    def _reset_new_operator_parameters(self) -> None:
        if self.step_mlp is not None:
            nn.init.zeros_(self.step_mlp[-1].weight); nn.init.zeros_(self.step_mlp[-1].bias)   # identity at start
            for m in getattr(self, "step_mlp_extra", []):
                nn.init.zeros_(m[-1].weight); nn.init.zeros_(m[-1].bias)
        if self.cross_nonlinear_hidden > 0:
            for k in range(self.W_phase.size(0)):
                nn.init.xavier_normal_(self.W1_phase[k]); nn.init.xavier_normal_(self.W_phase[k])
        if self.hyper_rank > 0:
            nn.init.zeros_(self.hyper_U.weight)
        if getattr(self, "in_proj", None) is not None:
            nn.init.orthogonal_(self.in_proj.weight)

    def _core(self, s, k: int):
        if self.cross_nonlinear_hidden > 0:
            return F.linear(F.silu(F.linear(s, self.W1_phase[k])), self.W_phase[k])
        return F.linear(s, self.W_phase[k])

    def _step_mlp(self, s, t: int = 0):
        """Residual per-channel nonlinear step, RMS-matched to the state."""
        rms = s.pow(2).mean(dim=-1, keepdim=True).clamp_min(1e-12).sqrt()
        idx = ((t + 1) // self.step_mlp_every - 1) % self.step_mlp_untied
        fn = (lambda v: self.step_mlp(self.step_mlp_norm(v))) if idx == 0 else self.step_mlp_extra[idx - 1]
        if self.step_mlp_kind == "field":             # token-wise FFN over the F field vectors (layout [F x w | F x w] per head)
            B, H, _ = s.shape; nf = self.step_mlp_fields; w = self.h // nf
            a = (s[..., :self.h] / rms).reshape(B, H, nf, w); b = (s[..., self.h:] / rms).reshape(B, H, nf, w)
            out = fn(torch.cat([a, b], dim=-1))
            delta = torch.cat([out[..., :w].reshape(B, H, self.h), out[..., w:].reshape(B, H, self.h)], dim=-1)
        else:
            delta = fn(s / rms)
        if self.step_mlp_cap > 0:
            drms = delta.pow(2).mean(dim=-1, keepdim=True).clamp_min(1e-12).sqrt()
            delta = delta * (self.step_mlp_cap / drms).clamp(max=1.0)
        return rms * delta

    def _reset_recursive_parameters(self) -> None:
        self._reset_new_operator_parameters()
        if self.stoch_head is not None:
            nn.init.zeros_(self.stoch_head.weight)
            nn.init.zeros_(self.stoch_head.bias)
            # zero head: mu = 0 and sigma = sigma0 at init (the bounded parameterisation adds the log sigma0 offset itself)
        if self.hrm_period > 0:
            nn.init.xavier_normal_(self.hrm_core.weight)
            nn.init.ones_(self.hrm_gamma)
            nn.init.zeros_(self.hrm_beta)
            nn.init.uniform_(self.hrm_bias)
            self.hrm_eta_raw.data.fill_(float(torch.logit(torch.tensor(min(max(self.hrm_eta_init, 1e-4), 1 - 1e-4)))))
            nn.init.zeros_(self.hrm_inject.weight)          # fast step ignores z_H at init
            if self.hrm_head is not None:
                nn.init.zeros_(self.hrm_head.weight)        # readout ignores z_H at init
                nn.init.zeros_(self.hrm_head.bias)
        if self.state_pre_norm is not None:
            self.state_pre_norm.reset_parameters()
        if self.phase_affine_weight is not None:
            nn.init.ones_(self.phase_affine_weight)
            nn.init.zeros_(self.phase_affine_bias)
        if self.head_film is not None:
            nn.init.zeros_(self.head_film.weight)           # identity FiLM at init
            nn.init.zeros_(self.head_film.bias)
        if self.memory_scales > 0:
            nn.init.ones_(self.memory_norm.weight)
            nn.init.zeros_(self.memory_norm.bias)
            nn.init.xavier_uniform_(self.memory_query.weight)
            nn.init.zeros_(self.memory_query.bias)
            nn.init.xavier_uniform_(self.memory_gate[0].weight)
            nn.init.zeros_(self.memory_gate[0].bias)
            nn.init.zeros_(self.memory_gate[-1].weight)   # gate starts uniformly at 0.5
            nn.init.zeros_(self.memory_gate[-1].bias)
            self.memory_scale_raw.data.fill_(_inverse_tanh_fraction(0.0, self.memory_residual_max))
            # Log-spaced time constants from one step to the maximum horizon; learnable.
            timescales = torch.logspace(0.0, math.log10(max(self.memory_max_timescale, 1.0)), self.memory_scales)
            decays = torch.exp(-1.0 / timescales).clamp(1e-5, 1.0 - 1e-5)
            self.memory_decay_raw.data.copy_(torch.logit(decays))

    def reset_after_global_linear_init(self) -> None:
        super().reset_after_global_linear_init()
        self._reset_recursive_parameters()

    def _memory_scale(self) -> torch.Tensor:
        return self.memory_residual_max * torch.tanh(self.memory_scale_raw)

    def _fixed_z(self, step_idx, reference_steps, batch, device, dtype):
        if self.phase_pattern is None:
            return super()._fixed_z(step_idx, reference_steps, batch, device, dtype)
        zo = getattr(self, "z_override", None)            # continuous schedule (sampled-branch latent): per-step anchor mix
        if zo is not None:
            return torch.full((batch, 1), float(zo[step_idx % len(zo)]), device=device, dtype=dtype)
        symbol = self.phase_pattern[step_idx % len(self.phase_pattern)]
        return torch.full((batch, 1), 1.0 if symbol == "E" else 0.0, device=device, dtype=dtype)

    def _symbol_at(self, step_idx: int, reference_steps: int) -> str:
        """Schedule symbol of step ``step_idx`` under the fixed schedule (pattern or contiguous switch)."""
        if self.phase_pattern is not None:
            return self.phase_pattern[step_idx % len(self.phase_pattern)]
        switch = int(math.floor(self.nominal_switch_fraction * max(reference_steps, 1) + 0.5))
        switch = min(max(switch, 0), max(reference_steps, 1))
        first_e = self.phase_direction == "e2l"
        return "E" if (step_idx < switch) == first_e else "L"

    def _order_clock(self, q: torch.Tensor) -> torch.Tensor:
        n_e, n_l = self._order_counts
        ref = float(self.reference_depth if self.budget_agnostic else 8)
        return q.new_full((q.size(0), 1), (n_e + math.log2(1.0 + n_l)) / ref)

    def _order_film(self, state: torch.Tensor) -> torch.Tensor:
        n_e, n_l = self._order_counts
        ref = float(self.reference_depth if self.budget_agnostic else 8)
        feats = state.new_tensor([n_e / ref, n_l / ref]).expand(state.size(0), 2)
        gamma, beta = self.head_film(feats).chunk(2, dim=-1)
        shape = (state.size(0),) + (1,) * (state.dim() - 2) + (state.size(-1),)
        return state * (1.0 + gamma.view(shape)) + beta.view(shape)

    def _budget_features(self, q, step_idx, requested_budget):
        if self.clock_mode == "order":
            zeros = q.new_zeros((q.size(0), 1))
            if self.budget_agnostic:
                return self._order_clock(q), zeros, zeros
            _, remaining, log_budget = super()._budget_features(q, step_idx, requested_budget)
            return self._order_clock(q), remaining, log_budget
        if not self.budget_agnostic:
            return super()._budget_features(q, step_idx, requested_budget)
        ref = self.reference_depth
        t_frac = q.new_full((q.size(0), 1), float(min(step_idx + 1, ref)) / float(ref))  # clock frozen after ref
        zeros = q.new_zeros((q.size(0), 1))
        return t_frac, zeros, zeros

    def _router_probs(self, state, z, q, delta_norm, step_idx, reference_steps, requested_budget):
        if not self.bank_per_phase or self.lora_router is None:
            return super()._router_probs(state, z, q, delta_norm, step_idx, reference_steps, requested_budget)
        pooled = state.mean(dim=1)
        t_frac, remaining_frac, log_budget = self._budget_features(q, step_idx, requested_budget)
        pieces = [pooled, z, q, delta_norm.unsqueeze(-1), t_frac]
        if self.budget_condition_router:
            pieces.extend([remaining_frac, log_budget])
        logits = self.lora_router(torch.cat(pieces, dim=-1)) / self.lora_router_temperature
        half = self.lora_bank_size // 2
        is_e_expert = (torch.arange(self.lora_bank_size, device=logits.device) < half).unsqueeze(0)   # 1 x K
        allowed = torch.where(z > 0.5, is_e_expert, ~is_e_expert)                                     # B x K
        logits = logits.masked_fill(~allowed, -torch.inf)
        routed, top_k = self._apply_route(logits)          # route override on top of the phase mask
        # a route that leaves no eligible expert in this step's phase half (e.g. a single E expert on an L step)
        # falls back to the phase half with the trained router for that step
        dead = torch.isinf(routed).all(dim=-1, keepdim=True)
        logits = torch.where(dead, logits, routed)
        top_k = min(top_k, half) if not bool(dead.all()) else self.lora_router_top_k
        if top_k < half:
            vals, idx = torch.topk(logits, k=top_k, dim=-1)
            masked = torch.full_like(logits, -torch.inf)
            masked.scatter_(1, idx, vals)
            logits = masked
        return torch.softmax(logits, dim=-1)

    def _hrm_update(self, z_h: torch.Tensor, fast_state: torch.Tensor, step_idx: int) -> torch.Tensor:
        """Slow-state cross update anchored on the fast state at the cycle end."""
        h1 = self.hrm_core(z_h)
        h = torch.cat([h1, self.hrm_gamma * h1 + self.hrm_beta], dim=-1)
        delta = fast_state * (h + self.hrm_bias)
        eta = torch.sigmoid(self.hrm_eta_raw)
        damping = getattr(self, "late_step_damping", None)
        if damping is not None and step_idx + 1 > damping["start"]:
            eta = eta * (damping["rho"] ** (step_idx + 1 - damping["start"]))
        return self._norm_cap(z_h + eta * delta)

    def _fast_step(self, state, x0, z, q, prev_delta_norm, t, reference_steps, requested_budget, z_h, memory):
        """One fast cross step without dropout: the inference-time map state -> state.

        Returns the norm-capped new state, the updated memory bank, the router
        probabilities and the gate value.
        """
        batch = state.size(0)
        z_b = self._scalar_broadcast(z, state)
        if self.anchor_power > 1:
            # Degree-p anchor with its RMS matched to the state's RMS, so the update keeps
            # ECN-scale magnitude while the interaction degree rises to p + 1.
            rms = state.pow(2).mean(dim=-1, keepdim=True).clamp_min(1e-12).sqrt()
            powered = torch.sign(state) * (state.abs() / rms).pow(self.anchor_power)
            powered = powered / powered.pow(2).mean(dim=-1, keepdim=True).clamp_min(1e-12).sqrt()
            ecn_anchor = powered * rms
        else:
            ecn_anchor = state
        x0a = x0
        am = getattr(self, "anchor_mask", None)             # per-branch mask on the anchor only (the state path keeps every feature)
        if am is not None:
            x0a = x0a * am.to(x0a.dtype).view(1, 1, -1)
        if getattr(self, "anchor_proj", None) is not None:  # per-branch learned anchor map (identity init)
            x0a = self.anchor_proj(x0a)
        sym = self._symbol_at(t, reference_steps) if self.phase_pattern is not None else None
        if t == 0:
            self._lag_state = x0
        if sym == "N":        # linear step: no interaction; constant anchor matched to the input's RMS
            anchor = x0.pow(2).mean(dim=-1, keepdim=True).clamp_min(1e-12).sqrt().expand_as(x0)
        elif sym == "R":      # residual anchor
            anchor = state - x0
        elif sym == "P":      # lagged-state anchor (x_{t-1})
            anchor = self._lag_state
        else:
            anchor = (1.0 - z_b) * x0a + z_b * ecn_anchor
        self._lag_state = state
        core_state = self.state_pre_norm(state) if self.state_pre_norm is not None else state
        if self.phase_affine_weight is not None:
            core_state = core_state * (z_b * self.phase_affine_weight[0] + (1.0 - z_b) * self.phase_affine_weight[1]) \
                + (z_b * self.phase_affine_bias[0] + (1.0 - z_b) * self.phase_affine_bias[1])
        if self.W_phase.size(0) == 1:                 # tied phase cores
            h1 = self._core(core_state, 0)
        else:
            h1 = z_b * self._core(core_state, 0) + (1.0 - z_b) * self._core(core_state, 1)
        probs = self._router_probs(state, z, q, prev_delta_norm, t, reference_steps, requested_budget)
        h1 = h1 + self._lora_bank_update(core_state, probs)
        if self.hyper_rank > 0:                       # instance-conditioned operator: W(x0) = W + U diag(tanh(g x0)) V
            h1 = h1 + self.hyper_U(torch.tanh(self.hyper_g(x0)) * self.hyper_V(core_state))
        if z_h is not None:
            h1 = h1 + self.hrm_inject(z_h)            # fast step conditioned on the slow state
        h2 = self.ln(h1) if self.ln is not None else self.gamma * h1 + self.beta
        h = torch.cat([h1, h2], dim=-1)
        bias = self.bias_phase[0] if self.bias_phase.size(0) == 1 else z_b * self.bias_phase[0] + (1.0 - z_b) * self.bias_phase[1]
        raw_delta = anchor * (h + bias)
        if getattr(self, "gated_cross", False):
            raw_delta = raw_delta * (2.0 * torch.sigmoid(self.gate_V(self.gate_U(core_state))))
        raw_delta_norm = raw_delta.reshape(batch, -1).norm(dim=-1) / math.sqrt(raw_delta[0].numel())
        eta = self._eta(state, z, q, raw_delta_norm, t, reference_steps, requested_budget)
        candidate = state + self._scalar_broadcast(eta, state) * raw_delta
        new_memory = memory
        if memory is not None:
            current_n = self.memory_norm(candidate)
            read = torch.softmax(self.memory_query(current_n), dim=-1)          # B x H x K
            context = (read.unsqueeze(-1) * memory).sum(dim=-2)                  # B x H x d
            gate = torch.sigmoid(self.memory_gate(torch.cat([current_n, self.memory_norm(context)], dim=-1)))
            candidate = candidate + self._memory_scale() * gate * (context - candidate)
            decay = torch.sigmoid(self.memory_decay_raw).view(*([1] * (memory.ndim - 2)), self.memory_scales, 1)
            new_memory = decay * memory + (1.0 - decay) * candidate.unsqueeze(-2)
        if self.step_mlp is not None and (t + 1) % self.step_mlp_every == 0:   # nonlinear step after every k-th cross step
            candidate = candidate + self._step_mlp(candidate, t)
        return self._norm_cap(candidate), new_memory, probs, eta

    def _stars_penalty(self, state, step_args):
        """Power-iteration estimate of the spectral radius of d(fast step)/d(state), per example."""
        batch = state.size(0)
        x = state.detach().requires_grad_(True)
        out = self._fast_step(x, *step_args)[0]
        u = torch.zeros_like(out, requires_grad=True)
        jt_u = torch.autograd.grad(out, x, grad_outputs=u, create_graph=True)[0]   # J^T u (linear in u)
        v = torch.randn_like(x)
        v = v / v.reshape(batch, -1).norm(dim=-1).clamp_min(1e-12).view(batch, *([1] * (x.dim() - 1)))
        rho = None
        for _ in range(self.stars_power_iters):
            jv = torch.autograd.grad(jt_u, u, grad_outputs=v, create_graph=True, retain_graph=True)[0]  # J v
            rho = jv.reshape(batch, -1).norm(dim=-1)
            v = (jv / rho.clamp_min(1e-12).view(batch, *([1] * (x.dim() - 1)))).detach()
        penalty = F.relu(rho - self.stars_target).pow(2).mean()
        return penalty, rho.detach().mean()

    def forward(self, x, *, num_steps, reference_steps=None, requested_budget=None, return_states=False,
                detach_after_steps=None):
        num_steps = int(num_steps)
        if num_steps <= 0:
            raise ValueError("num_steps must be >= 1")
        if self.budget_agnostic:
            reference_steps = requested_budget = self.reference_depth
        reference_steps = int(reference_steps or num_steps)
        requested_budget = int(requested_budget or reference_steps or num_steps)
        detach_after_steps = set(int(d) for d in (detach_after_steps or ())) if self.hrm_detach_segments else set()

        if getattr(self, "in_proj", None) is not None:
            x = self.in_proj(x)
        x = self._norm_cap(x)
        x0 = x
        state = x
        batch = x.size(0)
        q = x.new_zeros((batch, 1))
        prev_delta_norm = x.new_zeros((batch,))
        memory = None
        if self.memory_scales > 0:
            memory = x.unsqueeze(-2).expand(*x.shape[:-1], self.memory_scales, x.size(-1)).contiguous()
        z_h = x if self.hrm_period > 0 else None
        stars_step = -1
        if self.training and self.stars_weight > 0 and self.stars_active and num_steps > self.stars_min_step:
            stars_step = random.randint(self.stars_min_step, num_steps - 1)   # python RNG: identical on all ranks
        stars_penalty = stars_rho = None

        logits, zs, qs, hazards, etas, delta_norms, router_probs, states = [], [], [], [], [], [], [], []
        kls: list = []
        n_e = n_l = 0
        for t in range(num_steps):
            previous_state = state
            self._order_counts = (n_e, n_l)                  # symbols consumed before this step
            # HRM one-step gradient: only the last fast step of each cycle carries gradient.
            cycle_end = self.hrm_period > 0 and (t + 1) % self.hrm_period == 0
            grad_ctx = torch.no_grad() if (self.training and self.hrm_one_step_grad and self.hrm_period > 0
                                          and not cycle_end) else contextlib.nullcontext()
            with grad_ctx:
                if self.phase_gate_mode == "fixed_schedule":
                    z = self._fixed_z(t, reference_steps, batch, x.device, x.dtype)
                else:
                    z = self._phase_z(q)
                step_args = (x0, z, q, prev_delta_norm, t, reference_steps, requested_budget, z_h, memory)
                if t == stars_step:
                    stars_penalty, stars_rho = self._stars_penalty(state, step_args)
                new_state, memory, probs, eta = self._fast_step(state, *step_args)
                state = self.dropout(new_state) if self.dropout is not None else new_state
                if getattr(self, "noise_override", None):
                    state = self._inject_noise(state)          # evaluation-time stochastic recursion (noise branches)
                if self.stoch_head is not None:
                    state, kl_t = self._stochastic_transition(state)
                    kls.append(kl_t)
                if z_h is not None and cycle_end:
                    z_h = self._hrm_update(z_h, state, t)       # slow update once per cycle
                delta = state - previous_state
                delta_norm = delta.reshape(batch, -1).norm(dim=-1) / math.sqrt(delta[0].numel())
                if self._symbol_at(t, reference_steps) == "E":
                    n_e += 1
                else:
                    n_l += 1
                self._order_counts = (n_e, n_l)              # order reached after this step (readout)
                logit = self.head(self._order_film(state) if self.head_film is not None else state)
                if self.hrm_head is not None:
                    logit = logit + self.hrm_head(z_h)
                q_next, hazard = self._advance_phase(state, q, delta_norm, t, reference_steps, requested_budget)

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
            if (t + 1) in detach_after_steps:                # HRM deep supervision: new segment
                state = state.detach()
                prev_delta_norm = prev_delta_norm.detach()
                if z_h is not None:
                    z_h = z_h.detach()
                if memory is not None:
                    memory = memory.detach()

        aux = {"step_logits": logits, "z": zs, "q": qs, "phase_hazard": hazards, "eta": etas,
               "delta_norms": delta_norms, "router_probs": router_probs, "states": states,
               "stars_penalty": stars_penalty, "stars_rho": stars_rho, "kl": kls}
        return logits[-1], aux

    def _stochastic_transition(self, u: torch.Tensor):
        """Learned Gaussian transition (see __init__).  Returns (new state, mean KL to N(0, sigma0^2))."""
        # bounded parameterisation (v3): the head reads a layer-normalised state and its outputs pass through tanh, so
        # log sigma stays within +-1.5 of log sigma0 and mu within +-sigma0 (an unbounded dense head on the raw state let
        # Adam swing log sigma to its clamp within one epoch and the KL exploded: v1/v2 collapsed)
        raw_mu, raw_ls = self.stoch_head(F.layer_norm(u, (u.size(-1),))).chunk(2, dim=-1)
        mu = self.stochastic_sigma0 * torch.tanh(raw_mu) if self.stochastic_mean else torch.zeros_like(raw_mu)
        log_sigma = math.log(self.stochastic_sigma0) + 1.5 * torch.tanh(raw_ls)
        sigma = log_sigma.exp()
        rms = u.pow(2).mean(dim=-1, keepdim=True).sqrt().detach() + 1e-6
        spec = getattr(self, "sample_override", None)
        if self.training or spec:
            if spec and spec.get("seed") is not None:
                gen = getattr(self, "_sample_gen", None)
                if gen is None or gen[0] != (int(spec["seed"]), str(u.device)):
                    g = torch.Generator(device=u.device)
                    g.manual_seed(20_011 * (int(spec["seed"]) + 1))
                    gen = ((int(spec["seed"]), str(u.device)), g)
                    self._sample_gen = gen
                n = torch.randn(u.shape, generator=gen[1], device=u.device, dtype=u.dtype)
            else:
                n = torch.randn_like(u)
        else:
            n = torch.zeros_like(u)
        z = u + rms * (mu + sigma * n)
        s0 = self.stochastic_sigma0
        kl = (math.log(s0) - log_sigma + (sigma.pow(2) + mu.pow(2)) / (2.0 * s0 * s0) - 0.5).mean()
        return self._norm_cap(z), kl

def _tower_kwargs(old: anytime_rrt.AnytimeRecursiveCrossTower) -> dict:
    return dict(
        input_dim=old.d, nominal_switch_fraction=old.nominal_switch_fraction,
        phase_gate_mode=old.phase_gate_mode, phase_hidden_dim=old.phase_state_net[0].out_features,
        phase_time_scale_init=old.phase_time_scale_init, lora_rank=old.lora_rank,
        lora_bank_size=old.lora_bank_size,
        lora_router_hidden_dim=(old.lora_router[0].out_features if old.lora_router is not None else 64),
        lora_router_top_k=old.lora_router_top_k, lora_router_temperature=old.lora_router_temperature,
        budget_condition_phase=old.budget_condition_phase, budget_condition_router=old.budget_condition_router,
        budget_condition_update=old.budget_condition_update, budget_max_hint=old.budget_max_hint,
        use_update_gate=old.use_update_gate, update_hidden_dim=old.update_gate[0].out_features,
        update_eta_max=old.update_eta_max, update_eta_init=old.update_eta_init,
        layer_norm=old.ln is not None, net_dropout=(old.dropout.p if old.dropout is not None else 0.0),
        cross_norm=old.cross_norm, cross_norm_cap=old.cross_norm_cap,
    )


_BANK_TOKENS = ("lora_A_bank", "lora_B_bank", "lora_router")


def _is_bank_parameter(full_name: str) -> bool:
    return any(tok in full_name for tok in _BANK_TOKENS)


def _tie_parameters(src: nn.Module, dst: nn.Module, scope: str = "all") -> int:
    """Make parameters of ``dst`` the same tensors as the like-named parameters of ``src``.

    scope: "all" ties everything; "core" ties everything except the LoRA bank and its
    router (shared backbone, branch-specific experts); "bank" ties only the LoRA bank
    and router (separate cores, one common expert pool).  The modules keep their own
    non-parameter attributes (phase pattern, anchor degree, ...).
    """
    if scope not in ("all", "core", "bank"):
        raise ValueError(f"unknown tie scope {scope!r}")
    src_modules = dict(src.named_modules())
    tied = 0
    for name, module in dst.named_modules():
        source = src_modules[name]
        for pname, param in list(module._parameters.items()):
            if param is None:
                continue
            full = f"{name}.{pname}" if name else pname
            is_bank = _is_bank_parameter(full)
            if scope == "core" and is_bank:
                continue
            if scope == "bank" and not is_bank:
                continue
            module._parameters[pname] = source._parameters[pname]
            tied += 1
    return tied


def apply_late_step_damping(model: nn.Module, rho: float, start: int) -> None:
    """Deployment rule: for steps t > start, eta_t = eta_start * rho^(t - start).

    Patches ``_eta`` on every recursive tower instance (frozen or recap), so the
    learned gate is untouched up to ``start`` steps and decays geometrically
    afterwards.  ``eta_start`` is the gate value each example received at step
    ``start``; it is cached during the same forward pass.
    """
    rho = float(rho)
    start = int(start)
    if not (0.0 < rho < 1.0) or start < 1:
        raise ValueError("require 0 < rho < 1 and start >= 1")
    towers = [m for m in model.modules() if isinstance(m, anytime_rrt.AnytimeRecursiveCrossTower)]
    for tower in towers:
        base_eta = tower._eta

        def damped_eta(state, z, q, raw_delta_norm, step_idx, reference_steps, requested_budget,
                       _base=base_eta, _tower=tower):
            if step_idx == 0:
                _tower._damping_eta_ref = None
            if step_idx + 1 <= start:
                eta = _base(state, z, q, raw_delta_norm, step_idx, reference_steps, requested_budget)
                if step_idx + 1 == start:
                    _tower._damping_eta_ref = eta.detach()
                return eta
            ref = _tower._damping_eta_ref
            if ref is None:  # rollout started beyond `start`; fall back to the gate value at this step
                ref = _base(state, z, q, raw_delta_norm, step_idx, reference_steps, requested_budget).detach()
                _tower._damping_eta_ref = ref
            return ref * (rho ** (step_idx + 1 - start))

        tower._eta = damped_eta
        tower.late_step_damping = {"rho": rho, "start": start}


class _MLPTower(nn.Module):
    """Heterogeneous branch: plain MLP over the flattened embedding (a different function family from the cross towers)."""

    def __init__(self, in_dim: int, hidden: list, dropout: float = 0.1):
        super().__init__()
        layers, d = [], int(in_dim)
        for h in hidden:
            layers += [nn.Linear(d, int(h)), nn.ReLU(), nn.Dropout(dropout)]
            d = int(h)
        layers.append(nn.Linear(d, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, emb):                       # B x H x L -> B
        return self.net(emb.flatten(start_dim=1)).view(-1, 1)      # B x 1, like the cross towers' step logits


class _HashTailEmbedding(nn.Module):
    """Exact rows for the head of every field (frequency rank < cap) + V independent hashed tables for the tail.

    forward(X, view=None) -> B x H x [F x D/(2H) | F x D/(2H)] (the MultiHeadFeatureEmbedding layout).  view=None averages
    the V tail views; an integer selects one view (a branch).  Head/tail membership comes from {data_dir}/field_freq_rank.npy
    (non-persistent buffers, rebuilt at load time from the same file)."""

    def __init__(self, feature_map, d_total: int, num_heads: int, cap: int, views: int, buckets: int, data_dir: str, init_std: float = 1e-4):
        super().__init__()
        import numpy as _np
        names = [n for n, spec in feature_map.feature_specs.items() if "vocab_size" in spec]
        vocab = [int(feature_map.feature_specs[n]["vocab_size"]) for n in names]
        self.F, self.D, self.H, self.V, self.B = len(vocab), int(d_total), int(num_heads), int(views), int(buckets)
        slots = [min(v, int(cap)) for v in vocab]; offsets = _np.concatenate([[0], _np.cumsum(slots)[:-1]]); self.head_slots = int(sum(slots))
        Vmax = max(vocab)
        remap = _np.full((self.F, Vmax), -1, dtype=_np.int32)
        rank_path = os.path.join(str(data_dir), "field_freq_rank.npy")
        if os.path.exists(rank_path):
            rank = _np.load(rank_path)
            for f in range(self.F):
                r = rank[f, :vocab[f]].astype(_np.int64)
                remap[f, :vocab[f]] = _np.where(r < slots[f], offsets[f] + r, -1)
        else:
            logging.getLogger(__name__).warning("field_freq_rank.npy missing in %s: every id treated as tail", data_dir)
        self.register_buffer("remap", torch.from_numpy(remap), persistent=False)
        self.register_buffer("field_idx", torch.arange(self.F).view(1, -1), persistent=False)
        self.register_buffer("primes", torch.tensor([1000003 + 7919 * f for f in range(self.F)], dtype=torch.int64), persistent=False)
        self.head = nn.Embedding(self.head_slots + 1, self.D, padding_idx=self.head_slots)      # padding row = tail placeholder
        self.tails = nn.Embedding(self.V * self.B, self.D)
        nn.init.normal_(self.head.weight, std=init_std); nn.init.normal_(self.tails.weight, std=init_std)
        with torch.no_grad():
            self.head.weight[self.head_slots].zero_()

    def _tail_index(self, xl, v: int):
        h = (xl * self.primes.view(1, -1) + (v + 1) * 2654435761 + self.field_idx * 104729) & 0x7FFFFFFF
        return v * self.B + (h % self.B)

    def dense(self, X, view=None):
        """B x F x D field-major embeddings."""
        xl = X.long()
        slot = self.remap[self.field_idx, xl]                       # B x F, -1 for tail ids
        is_tail = (slot < 0)
        head = self.head(torch.where(is_tail, torch.full_like(slot, self.head_slots), slot).long())
        if view is None:
            tail = torch.stack([self.tails(self._tail_index(xl, v)) for v in range(self.V)], 0).mean(0)
        else:
            tail = self.tails(self._tail_index(xl, int(view) % self.V))
        return torch.where(is_tail.unsqueeze(-1), tail, head)

    def forward(self, X, view=None):
        t = self.dense(X, view)                                      # B x F x D
        heads = torch.stack(torch.tensor_split(t, self.H, dim=-1), dim=1)     # B x H x F x D/H
        a, b = torch.tensor_split(heads, 2, dim=-1)
        return torch.cat([a.flatten(start_dim=2), b.flatten(start_dim=2)], dim=-1)


class _RecursiveMLPTower(nn.Module):
    """Heterogeneous recursive branch: residual MLP block iterated T steps on the flattened embedding, per-step readout."""

    def __init__(self, in_dim: int, hidden: int = 512):
        super().__init__()
        self.norm = nn.LayerNorm(int(in_dim))
        self.fc1, self.fc2 = nn.Linear(int(in_dim), int(hidden)), nn.Linear(int(hidden), int(in_dim))
        self.alpha = nn.Parameter(torch.tensor(0.1))
        self.head = nn.Linear(int(in_dim), 1)

    def steps_forward(self, emb, steps: int):
        s = emb.flatten(start_dim=1)
        out = []
        for _ in range(int(steps)):
            s = s + self.alpha * self.fc2(F.relu(self.fc1(self.norm(s))))
            out.append(self.head(s))                # B x 1 per step
        return out

    def forward(self, emb, steps: int = 8):
        return self.steps_forward(emb, steps)[-1]


class _BilinearTower(nn.Module):
    """Heterogeneous branch: FiBiNET-style field-pair bilinear interactions (v_i W) * v_j over all field pairs, linear readout."""

    def __init__(self, num_fields: int, half_dim: int, num_heads: int):
        super().__init__()
        self.F, self.half, self.H = int(num_fields), int(half_dim), int(num_heads)
        d = 2 * self.half * self.H                # full per-field embedding width
        self.W = nn.Parameter(torch.empty(d, d))
        nn.init.xavier_uniform_(self.W)
        i, j = torch.triu_indices(self.F, self.F, offset=1)
        self.register_buffer("pair_i", i)
        self.register_buffer("pair_j", j)
        self.out = nn.Linear(int(i.numel()) * d, 1)

    def forward(self, emb):                       # B x H x (2 F half) -> B
        B = emb.size(0)
        v = emb.view(B, self.H, 2, self.F, self.half).permute(0, 3, 1, 2, 4).reshape(B, self.F, -1)   # B x F x d
        p = (v @ self.W.to(v.dtype))[:, self.pair_i] * v[:, self.pair_j]                              # B x P x d
        return self.out(p.flatten(start_dim=1)).view(-1, 1)


class MultiTowerRecursiveFCNv2(anytime_rrt.AnytimeRecursiveFCNv2):
    """K recursive towers on a shared embedding with mean-logit fusion."""

    def __init__(
        self,
        feature_map,
        num_towers: int = 4,
        tower_pre_norm: str = "none",
        memory_scales: int = 0,
        memory_hidden_dim: int = 128,
        memory_max_timescale: float = 16.0,
        memory_residual_max: float = 0.25,
        budget_advantage_weight: float = 0.0,
        budget_advantage_temperature: float = 1.0,
        budget_agnostic: bool = False,
        reference_depth: int = 8,
        hrm_period: int = 0,
        hrm_readout: str = "l_plus_h",
        hrm_detach_segments: bool = False,
        hrm_one_step_grad: bool = False,
        hrm_eta_init: float = 0.5,
        phase_pattern_e2l: str | None = None,
        phase_pattern_l2e: str | None = None,
        stars_weight: float = 0.0,
        stars_target: float = 1.0,
        stars_power_iters: int = 2,
        stars_min_step: int = 4,
        tower_specs: list | None = None,
        tied_towers: bool = False,
        tie_scope: str = "all",
        branch_decorrelation_weight: float = 0.0,
        schedule_mirror_pairs: bool = False,
        pair_tri_bce: bool = False,
        schedule_family: list | None = None,
        schedule_sampler: str = "family",
        schedule_e_prob_range: tuple = (0.2, 0.8),
        schedule_sample_length: int = 24,
        rollout_extension_depth: int = 0,
        rollout_extension_prob: float = 0.0,
        fusion_distill_weight: float = 0.0,
        order_consistency_weight: float = 0.0,
        order_consistency_block: int = 4,
        schedule_advantage_weight: float = 0.0,
        depth_improvement_weight: float = 0.0,
        depth_improvement_delta: float = 0.0005,
        eta_floor: float = 0.0,
        eta_floor_weight: float = 0.0,
        eta_floor_start: int = 4,
        clock_mode: str = "depth",
        bank_per_phase: bool = False,
        phase_norm_affine: bool = False,
        head_order_film: bool = False,
        tie_phase_cores: bool = False,
        tbptt_segment: int = 0,
        branch_mask_keep: float = 1.0,
        branch_mask_mode: str = "step",
        route_dropout_prob: float = 0.0,
        route_dropout_family: str = "halves",
        bank_orth_weight: float = 0.0,
        eval_route: str = "trained",
        deploy_towers: list | None = None,
        indep_loss_weight: float = 0.0,
        bsd_weight: float = 0.0,
        bsd_curriculum_steps: int = 30000,
        distill_mode: str = "none",
        distill_weight: float = 0.0,
        distill_rank_temp: float = 1.0,
        distill_member_fusion_weight: float = 1.0,
        branch_subspace_keep: float = 0.0,
        branch_rotation: str = "none",
        branch_transform_seed: int = 1234,
        extra_towers: list | None = None,
        extra_mlp_hidden: list | None = None,
        residual_branches: bool = False,
        branch_objective: str = "tri",
        objective_mix: float = 0.5,
        private_lowrank_dim: int = 4,
        private_cap: int = 2000,
        branch_bagging: bool = False,
        branch_losses: list | None = None,
        branch_budget_focus: float = 0.0,
        self_distill_weight: float = 0.0,
        sampled_branches: bool = False,
        sb_experts: int = 4,
        sb_rank_keep: float = 0.5,
        sb_delta_groups: int = 4,
        sb_delta_keep: int = 2,
        anchor_mask_keep: float = 0.0,
        anchor_proj: bool = False,
        sb_alpha: bool = False,
        rare_reg_scale: float = 1.0,
        rare_reg_cap: int = 2000,
        fusion_head: str = "mean",
        fusion_rank: int = 64,
        gated_cross: bool = False,
        step_mlp_hidden: int = 0,
        step_mlp_every: int = 1,
        step_mlp_kind: str = "flat",
        cross_nonlinear_hidden: int = 0,
        lora_nonlinear: bool = False,
        hyper_rank: int = 0,
        state_width: int = 0,
        step_mlp_untied: int = 1,
        step_mlp_cap: float = 0.0,
        refresh_lr_mult: float = 1.0,
        residual_step_weight: float = 0.0,
        view_by_route: bool = False,
        partition_member_grads: bool = False,
        tail_cap: int = 2000,
        tail_views: int = 4,
        tail_buckets: int = 131072,
        tower_data_split: bool = False,
        split_fused_weight: float = 0.0,
        anytime_fuse_weight: float = 0.0,
        private_hash_buckets: int = 65536,
        private_hash_init_std: float = 1e-4,
        teacher_logits: str | None = None,
        branch_embeddings: str = "shared",
        branch_gate_reg: float = 0.0,
        branch_gate_temp: float = 0.2,
        branch_gate_init: float = 3.0,
        gate_pool: int = 0,
        gate_tokens: bool = False,
        stochastic_sigma0: float = 0.0,
        stochastic_beta: float = 0.0,
        stochastic_mean: bool = True,
        input_views: int = 0,
        view_keep_prob: float = 0.75,
        views_include_full: bool = True,
        view_resample: bool = True,
        **kwargs,
    ):
        super().__init__(feature_map, **kwargs)
        # Input views: every tower also runs `input_views` trajectories on random field subspaces of the
        # shared embedding (each field kept with prob view_keep_prob; the mask is shared by the batch, so a
        # view is a fixed subspace, not dropout).  Training resamples the masks every batch (python RNG,
        # identical on all ranks) so the operator learns the view distribution; evaluation uses
        # deterministic per-view masks (view_mask(v)), and the breadth evaluator can ask for more views
        # than were trained with.  Trajectory 0 of each tower is the full input when views_include_full.
        # Random branch masking in the loss: keep prob per branch (1.0 = off); "step" = branch dropout
        # (one subset per batch), "example" = bagging (subset per example).  Evaluation uses all branches.
        self.branch_mask_keep = float(branch_mask_keep)
        self.branch_mask_mode = str(branch_mask_mode)
        if self.branch_mask_mode not in ("step", "example"):
            raise ValueError("branch_mask_mode must be step or example")
        # Route dropout: with prob route_dropout_prob a training rollout restricts each tower's router to a
        # random expert subset (family "halves": one of the four fixed half-banks low/high/even/odd; "drop1":
        # all but one expert; "random_half": any bank_size/2 subset).  The subset is fixed for the whole
        # rollout and sampled per tower and per rollout (python RNG, identical on all ranks).  Evaluation
        # through the model uses the full trained router; evaluate_breadth --routes turns the subset routes
        # into test-time branches.  Zero extra parameters.
        self.route_dropout_prob = float(route_dropout_prob)
        self.route_dropout_family = str(route_dropout_family)
        if self.route_dropout_family not in ("halves", "drop1", "random_half", "topk", "mixed", "single"):
            raise ValueError("route_dropout_family must be halves, drop1, random_half, topk, mixed or single")
        # expert orthogonality: penalise the off-diagonal Gram entries of the normalised LoRA A factors of each
        # representative tower, so leave-one-out routes lose something no other expert replaces
        self.bank_orth_weight = float(bank_orth_weight)
        # eval_route "dense": the model's own evaluation forward (validation / test in the trainer) routes over all
        # experts, so checkpoint selection and the reported training-time number use the deployed dense route
        self.eval_route = str(eval_route)
        if self.eval_route not in ("trained", "dense"):
            raise ValueError("eval_route must be trained or dense")
        # auxiliary readers: towers outside deploy_towers take part in training (fused loss over all towers) but the
        # evaluation forward fuses only the deployed ones; the breadth evaluator evaluates only the deployed towers
        self.deploy_towers = [int(i) for i in deploy_towers] if deploy_towers else None
        # interpolation between the joint (fused) and independent objectives: lambda x mean_k BCE(f_k)
        self.indep_loss_weight = float(indep_loss_weight)
        # breadth self-distillation: the deployed subset (student) is pulled toward the all-tower fusion (teacher, detached);
        # lambda goes from ground truth (1) to distillation (0) over bsd_curriculum_steps training forwards
        self.bsd_weight = float(bsd_weight)
        self.bsd_curriculum_steps = int(bsd_curriculum_steps)
        # distillation from a fixed teacher (precomputed logits): red | fusion | rank, see _distill_loss
        self.distill_mode = str(distill_mode)
        if self.distill_mode not in ("none", "red", "fusion", "rank", "fusion_rank", "member", "route_member"):
            raise ValueError("distill_mode must be none, red, fusion, rank, fusion_rank, member or route_member")
        self.distill_weight = float(distill_weight)
        self.distill_rank_temp = float(distill_rank_temp)
        # member-wise distillation: tower k is trained toward teacher member (k mod M); the fusion keeps a KD term to the mean
        self.distill_member_fusion_weight = float(distill_member_fusion_weight)
        # per-tower data streams: tower k receives its own label loss only on rows with index % K == k (disjoint minibatch
        # halves), plus split_fused_weight x the fused BCE on all rows; members reach different minima instead of colluding
        self.tower_data_split = bool(tower_data_split)
        self.split_fused_weight = float(split_fused_weight)
        # anytime fused-depth term: BCE of the mean of the fused logits over the sampled budgets (the deployed depth ensemble)
        self.anytime_fuse_weight = float(anytime_fuse_weight)
        self.private_hash_buckets = int(private_hash_buckets)
        self.private_hash_init_std = float(private_hash_init_std)
        self.teacher_logits = teacher_logits
        if self.distill_mode != "none" and not teacher_logits:
            raise ValueError("distill_mode needs teacher_logits (name of the precomputed teacher, see recap.dump_teacher)")
        self._teacher = None
        self._bsd_step = 0
        # Per-branch embeddings: "shared" (one table, the coupling every breadth grid measured), "private" (tower k > 0
        # gets its own MultiHeadFeatureEmbedding, K x embedding parameters = the seed-ensemble analogue inside one
        # model), "split" (one table; tower k reads only slice k of every field's embedding dims, zero extra params).
        self.branch_embeddings = str(branch_embeddings)
        if self.branch_embeddings not in ("shared", "private", "split", "soft", "hashed", "lowrank", "capped", "hashtail"):
            raise ValueError("branch_embeddings must be shared, private, split, soft, hashed or lowrank")
        # "soft": tower k reads emb * sigmoid(g_k) with a learned scale per embedding dimension (sigma-N-Ens style feature
        # ownership); the ownership regulariser (branch_gate_reg x mean over dims of the entropy of softmax_k(s_k / temp))
        # makes each dimension the property of some towers while the rest stays shared.  Zero-ish extra parameters.
        self.branch_gate_reg = float(branch_gate_reg)
        self.branch_gate_temp = float(branch_gate_temp)
        self.branch_gate_init = float(branch_gate_init)
        # Trajectory pool: gate_pool = M > num_towers gate vectors; every training rollout samples num_towers distinct
        # gates (python RNG, identical on all ranks); evaluation through the model uses gates 0..K-1; the breadth
        # evaluator (--gates M) runs each gate as a branch.  gate_tokens adds a learned additive vector per gate.
        self.gate_pool = int(gate_pool)
        self.gate_tokens = bool(gate_tokens)
        self._gate_assign = None
        self._gate_override = None
        self.input_views = int(input_views)
        self.view_keep_prob = float(view_keep_prob)
        self.views_include_full = bool(views_include_full)
        self.view_resample = bool(view_resample)
        self.num_fields = int(feature_map.num_fields)
        self.view_embedding_dim = int(kwargs.get("embedding_dim", 16))
        self.view_num_heads = int(kwargs.get("num_heads", 1))
        self._eval_mask_cache = {}
        # tbptt_segment k > 0: segment-wise truncated backprop.  The state (and the HRM slow state, if
        # any) is detached every k steps of a training rollout, so gradients never cross a segment
        # boundary; pair it with prefix supervision so every segment has its own loss.
        self.tbptt_segment = int(tbptt_segment)
        # schedule_family: during training each tower view draws a distinct schedule
        # from this list per batch (same draw on every rank), so the shared operator
        # learns to be valid along any schedule in the family; evaluation uses the
        # fixed tower_specs patterns.  With separate-weight towers every tower is trained on
        # the whole family, so at inference each tower can run any schedule (branches =
        # towers x schedules).
        self.schedule_family = [str(s).upper() for s in (schedule_family or [])]
        # Wave 1i training techniques (single-rollout, budget-agnostic path):
        # schedule_sampler "bernoulli": every tower draws a fresh random schedule per batch
        #   (per-step E with probability p ~ U(schedule_e_prob_range), length schedule_sample_length).
        # rollout_extension_*: with the given probability the training rollout runs to
        #   rollout_extension_depth (prefix supervision covers the extra depths).
        # fusion_distill_weight: one tower on a random schedule is distilled towards the detached
        #   fused prediction of the canonical branches at the supervised depths.
        # order_consistency_weight: two random schedules with equal (n_E, n_L) at every block
        #   boundary are penalised for disagreeing at the supervised depths.
        # schedule_advantage_weight: GRPO-style advantage weighting across branches.
        self.schedule_sampler = str(schedule_sampler)
        if self.schedule_sampler not in ("family", "bernoulli"):
            raise ValueError("schedule_sampler must be family or bernoulli")
        # Pair-structured (antithetic) breadth.  Towers are grouped in consecutive pairs (2k, 2k+1).
        # schedule_mirror_pairs: each pair draws ONE schedule per batch (family representative or
        #   Bernoulli) and the partner runs its mirror (E <-> L), so every pair covers the E-first
        #   and the L-first path through the interaction-order lattice; orientation is random.
        # pair_tri_bce: the conditional tri-BCE terms are taken on the pair-mean logit instead of
        #   the single tower, so a pair (not a tower) is the unit trained to stand alone and the
        #   unit added at test time.
        self.schedule_mirror_pairs = bool(schedule_mirror_pairs)
        self.pair_tri_bce = bool(pair_tri_bce)
        self.schedule_e_prob_range = (float(schedule_e_prob_range[0]), float(schedule_e_prob_range[1]))
        self.schedule_sample_length = int(schedule_sample_length)
        self.rollout_extension_depth = int(rollout_extension_depth)
        self.rollout_extension_prob = float(rollout_extension_prob)
        self.fusion_distill_weight = float(fusion_distill_weight)
        self.order_consistency_weight = float(order_consistency_weight)
        self.order_consistency_block = max(int(order_consistency_block), 1)
        self.schedule_advantage_weight = float(schedule_advantage_weight)
        # Gate-collapse fixes for the budget-agnostic recipe (wave 2b).
        # depth_improvement_*: each supervised depth must lower the fused BCE of the previous
        #   supervised depth by at least delta (hinge), i.e. later steps are rewarded for
        #   improving, not only penalised for regressing.
        # eta_floor_*: hinge keeping the update gate above eta_floor from step eta_floor_start on,
        #   so the operator cannot satisfy the prefix losses by switching itself off.
        self.depth_improvement_weight = float(depth_improvement_weight)
        self.depth_improvement_delta = float(depth_improvement_delta)
        self.eta_floor = float(eta_floor)
        self.eta_floor_weight = float(eta_floor_weight)
        self.eta_floor_start = int(eta_floor_start)
        # tied_towers: all towers share one set of recurrent weights and differ only in
        # schedule / anchor degree (trajectory diversity, zero extra recurrent parameters).
        # branch_decorrelation_weight: penalises the mean pairwise Pearson correlation of
        # the towers' error signals (p_k - y) over the batch (negative-correlation learning).
        self.tied_towers = bool(tied_towers)
        # tie_scope: which parameters a tie group shares: all | core (everything but the
        # LoRA bank + router) | bank (only the LoRA bank + router).
        self.tie_scope = str(tie_scope)
        if self.tie_scope not in ("all", "core", "bank"):
            raise ValueError("tie_scope must be all, core or bank")
        self.branch_decorrelation_weight = float(branch_decorrelation_weight)
        # tower_specs: optional explicit list of towers, each {direction: e2l|l2e,
        # pattern: <E/L string or null>, anchor_power: int, tie: <group name or null>};
        # overrides num_towers and the alternating E2L/L2E layout.  Towers with the same
        # `tie` group share one set of recurrent weights (tied_towers = every tower in one group).
        self.tower_specs = [dict(spec) for spec in (tower_specs or [])]
        self.num_towers = len(self.tower_specs) if self.tower_specs else int(num_towers)
        self.stars_weight = float(stars_weight)
        if phase_pattern_e2l and not phase_pattern_l2e:   # mirror: swap E and L
            phase_pattern_l2e = str(phase_pattern_e2l).upper().translate(str.maketrans("EL", "LE"))
        self.phase_patterns = {"e2l": phase_pattern_e2l or None, "l2e": phase_pattern_l2e or None}
        self.hrm_detach_segments = bool(hrm_detach_segments)
        if self.num_towers < 1:
            raise ValueError("num_towers must be >= 1")
        self.budget_advantage_weight = float(budget_advantage_weight)
        self.budget_advantage_temperature = max(float(budget_advantage_temperature), 1e-6)
        extras = dict(pre_norm=tower_pre_norm, memory_scales=int(memory_scales),
                      memory_hidden_dim=int(memory_hidden_dim), memory_max_timescale=float(memory_max_timescale),
                      memory_residual_max=float(memory_residual_max),
                      budget_agnostic=bool(budget_agnostic), reference_depth=int(reference_depth),
                      hrm_period=int(hrm_period), hrm_readout=str(hrm_readout),
                      hrm_detach_segments=bool(hrm_detach_segments), hrm_one_step_grad=bool(hrm_one_step_grad),
                      hrm_eta_init=float(hrm_eta_init),
                      stars_weight=float(stars_weight), stars_target=float(stars_target),
                      stars_power_iters=int(stars_power_iters), stars_min_step=int(stars_min_step),
                      clock_mode=str(clock_mode), bank_per_phase=bool(bank_per_phase),
                      phase_norm_affine=bool(phase_norm_affine), head_order_film=bool(head_order_film),
                      tie_phase_cores=bool(tie_phase_cores),
                      stochastic_sigma0=float(stochastic_sigma0), stochastic_beta=float(stochastic_beta),
                      stochastic_mean=bool(stochastic_mean), gated_cross=bool(gated_cross),
                      step_mlp_hidden=int(step_mlp_hidden), step_mlp_every=int(step_mlp_every), step_mlp_kind=str(step_mlp_kind),
                      step_mlp_fields=int(feature_map.num_fields), cross_nonlinear_hidden=int(cross_nonlinear_hidden),
                      lora_nonlinear=bool(lora_nonlinear), hyper_rank=int(hyper_rank), state_width=int(state_width),
                      step_mlp_untied=int(step_mlp_untied), step_mlp_cap=float(step_mlp_cap))
        self.stochastic_beta = float(stochastic_beta)
        # residual_step_weight > 0: every step increment z_t - z_{t-1} of every tower is trained against the residual of the
        # detached earlier logit, BCE(sg(z_{t-1}) + z_t - z_{t-1}, y): later depth explains what earlier depth missed (functional-gradient
        # refinement); the final tri-BCE stays the principal loss.
        self.residual_step_weight = float(residual_step_weight)
        # wave 23 route-member distillation: a training rollout whose router drops expert j (route drop_j) is distilled toward
        # teacher member j mod M (the trained/dense route toward the member mean), so the K leave-one-out routes of a 2-tower model
        # become K member imitations at test time.  view_by_route: with a hashed-tail table, route drop_j also reads tail view j
        # (the trained route reads the mean view), giving every generated branch its own rare-id embeddings.
        self.view_by_route = bool(view_by_route); self._route_member = {}
        # partition_member_grads: the per-route member terms are returned separately as out["loss_private"] (the trainer applies
        # their gradient to the LoRA bank only); the fused mean-teacher term stays in the main loss.
        self.partition_member_grads = bool(partition_member_grads); self._private_terms = []
        self.refresh_lr_mult = float(refresh_lr_mult)     # optimizer group for the refresh MLPs (step_mlp*), see compile()
        if self.tbptt_segment > 0:
            extras["hrm_detach_segments"] = True      # enables detach_after_steps in the tower forward
        self.budget_agnostic = bool(budget_agnostic)
        templates = {"e2l": self.E2L, "l2e": self.L2E}
        towers = []
        for k in range(self.num_towers):
            if self.tower_specs:
                spec = self.tower_specs[k]
                direction = str(spec.get("direction", "e2l"))
                pattern = spec.get("pattern") or self.phase_patterns[direction]
                power = int(spec.get("anchor_power", 1))
                name = f"T{k}_{spec.get('name', direction)}"
            else:
                direction = "e2l" if k % 2 == 0 else "l2e"
                pattern, power, name = self.phase_patterns[direction], 1, f"T{k}_{direction}"
            towers.append(RecursiveCrossTower(
                phase_direction=direction, name=name, phase_pattern=pattern, anchor_power=power,
                **_tower_kwargs(templates[direction]), **extras,
            ))
        del self.E2L, self.L2E
        self.towers = nn.ModuleList(towers)
        if self.branch_embeddings == "private":
            # fresh tables (feature embeddings initialise themselves at construction, so each copy differs)
            self.branch_embedding_layers = nn.ModuleList([
                anytime_rrt.cl.MultiHeadFeatureEmbedding(feature_map, int(kwargs.get("embedding_dim", 16)) * int(kwargs.get("num_heads", 1)),
                                                         int(kwargs.get("num_heads", 1)))
                for _ in range(self.num_towers - 1)])
        if self.branch_embeddings == "hashed":
            # every tower reads the shared table plus its own small hashed sub-table (a private, independently
            # initialised delta on (field, value) pairs): a cheap private read of the input at ~buckets x dim params
            hdim = int(kwargs.get("embedding_dim", 16)) * int(kwargs.get("num_heads", 1))
            self.branch_hash_tables = nn.ModuleList([nn.Embedding(self.private_hash_buckets, hdim) for _ in range(self.num_towers)])
            for t in self.branch_hash_tables:
                nn.init.normal_(t.weight, std=self.private_hash_init_std)
        self._dim_mask_cache = {}
        # ---- wave 16 diversity mechanisms: fixed parameter-free branch transforms (random subspace mask, orthogonal
        # rotation of the embedding basis), heterogeneous extra towers (mlp | bilinear) and boosting-style residual branches
        self.branch_subspace_keep = float(branch_subspace_keep)
        self.branch_rotation = str(branch_rotation)
        if self.branch_rotation not in ("none", "block", "full"):
            raise ValueError("branch_rotation must be none, block or full")
        self.extra_tower_kinds = [str(x) for x in (extra_towers or [])]
        self.residual_branches = bool(residual_branches)
        self.branch_objective = str(branch_objective)
        if self.branch_objective not in ("tri", "independent", "mixed", "detached"):
            raise ValueError("branch_objective must be tri, independent, mixed or detached")
        self.objective_mix = float(objective_mix)
        self.private_lowrank_dim = int(private_lowrank_dim)
        if self.branch_embeddings == "lowrank":
            # wave 17: per-branch full-vocabulary private tables of width r (init ~0), projected per field to the embedding
            # width and added to the shared read: table-level diversity at r/D of a private table, exact ids (no hashing)
            d_full = int(kwargs.get("embedding_dim", 16)) * int(kwargs.get("num_heads", 1))
            self.branch_embedding_layers = nn.ModuleList([
                anytime_rrt.cl.MultiHeadFeatureEmbedding(feature_map, self.private_lowrank_dim, 1) for _ in range(self.num_towers)])
            self.branch_lowrank_proj = nn.ModuleList([nn.Linear(self.private_lowrank_dim, d_full, bias=False) for _ in range(self.num_towers)])
            for p in self.branch_lowrank_proj:
                nn.init.normal_(p.weight, std=0.1)
        self.private_cap = int(private_cap)
        # ---- wave 17c: zero-parameter diversity levers and the fused self-distillation level lever
        self.branch_bagging = bool(branch_bagging)                       # per-branch Poisson(1) row weights (bagging in one model)
        self.branch_losses = [str(x) for x in (branch_losses or [])]     # per-unit loss kind: bce | rank (independent objective)
        if any(x not in ("bce", "rank") for x in self.branch_losses):
            raise ValueError("branch_losses entries must be bce or rank")
        if self.branch_losses and self.branch_objective != "independent":
            raise ValueError("branch_losses needs branch_objective independent")
        self.branch_budget_focus = float(branch_budget_focus)            # sigma of the budget-specialist weights (0 = off)
        if self.branch_budget_focus > 0 and self.branch_objective != "independent":
            raise ValueError("branch_budget_focus needs branch_objective independent")
        self.self_distill_weight = float(self_distill_weight)
        self._bag_X = None
        self._delta_scale = 1.0          # test-time overrides for capped deltas (evaluator routes dscale / dswap)
        self._delta_src = None
        # ---- wave 17d: sampled-branch training (latent = expert subset + LoRA rank mask + delta dim-group mask per tower)
        self.sampled_branches = bool(sampled_branches)
        self.sb_experts, self.sb_rank_keep = int(sb_experts), float(sb_rank_keep)
        self.sb_delta_groups, self.sb_delta_keep = int(sb_delta_groups), int(sb_delta_keep)
        if self.sampled_branches and self.branch_objective != "independent":
            raise ValueError("sampled_branches needs branch_objective independent (each sampled branch is trained standalone)")
        self.sb_alpha = bool(sb_alpha)
        # ---- wave 20: hash-view tail embeddings replace the full table
        self.tail_cap, self.tail_views, self.tail_buckets = int(tail_cap), int(tail_views), int(tail_buckets)
        self._view_fixed = {}
        if self.branch_embeddings == "hashtail":
            d_total = int(kwargs.get("embedding_dim", 16)) * int(kwargs.get("num_heads", 1))
            self.embedding_layer = _HashTailEmbedding(feature_map, d_total, int(kwargs.get("num_heads", 1)), self.tail_cap, self.tail_views,
                                                      self.tail_buckets, str(kwargs.get("data_dir", "")))
        # ---- wave 18: rare-id shrinkage, bilinear fusion head
        self.rare_reg_scale, self.rare_reg_cap = float(rare_reg_scale), int(rare_reg_cap)
        self._rare_masks = None
        if self.rare_reg_scale > 1.0:
            import numpy as _np
            rank_path = os.path.join(str(kwargs.get("data_dir", "")), "field_freq_rank.npy")
            names = [n for n, spec in feature_map.feature_specs.items() if "vocab_size" in spec]
            rank = _np.load(rank_path) if os.path.exists(rank_path) else None
            masks = {}
            for f, n in enumerate(names):
                v = int(feature_map.feature_specs[n]["vocab_size"])
                if rank is None:
                    masks[n] = torch.zeros(v, dtype=torch.bool)
                else:
                    masks[n] = torch.from_numpy(rank[f, :v] >= self.rare_reg_cap)
            self._rare_masks = masks
            for n, mk in masks.items():
                self.register_buffer(f"rare_mask_{f}_{n}".replace(".", "_"), mk, persistent=False)
        self.fusion_head = str(fusion_head)
        if self.fusion_head not in ("mean", "bilinear"):
            raise ValueError("fusion_head must be mean or bilinear")
        if self.fusion_head == "bilinear":
            d_in = int(kwargs.get("num_heads", 1)) * 2 * (self.view_embedding_dim // (2 * self.view_num_heads)) * self.num_fields
            self.fusion_U = nn.Linear(d_in, int(fusion_rank), bias=False)
            self.fusion_V = nn.Linear(d_in, int(fusion_rank), bias=False)
            nn.init.zeros_(self.fusion_V.weight)          # the correction starts at zero: fused = mean logit at init
        self.anchor_mask_keep = float(anchor_mask_keep)
        self.anchor_proj_enabled = bool(anchor_proj)
        self._bag_seed = {}              # tower index -> latent hash used by the bagging weights
        self._delta_masks = {}           # tower index -> D-dim mask over the private delta (capped models)
        self._latent_fixed = {}          # tower index -> evaluator-applied latent (apply_latent / clear_latent)
        self.register_buffer("bag_primes", torch.tensor([1000003 + 7919 * f for f in range(int(feature_map.num_fields))], dtype=torch.int64), persistent=False)   # not in checkpoints (old runs must load)
        if self.branch_embeddings == "capped":
            # wave 17: per-branch private deltas for the `private_cap` most frequent ids of every field (frequency rank
            # of the training split, {data_dir}/field_freq_rank.npy); the tail shares the table (zero delta, padding row)
            import numpy as _np
            d_full = int(kwargs.get("embedding_dim", 16)) * int(kwargs.get("num_heads", 1))
            vocab = [int(spec["vocab_size"]) for spec in feature_map.feature_specs.values() if "vocab_size" in spec]
            if len(vocab) != self.num_fields:
                raise ValueError(f"capped embeddings need a vocab_size per field ({len(vocab)} of {self.num_fields})")
            slots = [min(v, self.private_cap) for v in vocab]
            offsets = _np.concatenate([[0], _np.cumsum(slots)[:-1]])
            total = int(sum(slots))
            rank_path = os.path.join(str(kwargs.get("data_dir", "")), "field_freq_rank.npy")
            V = max(vocab)
            remap = _np.full((self.num_fields, V), total, dtype=_np.int64)          # default: overflow (padding row)
            if os.path.exists(rank_path):
                rank = _np.load(rank_path)
                for f in range(self.num_fields):
                    r = rank[f, :vocab[f]].astype(_np.int64)
                    remap[f, :vocab[f]] = _np.where(r < slots[f], offsets[f] + r, total)
            else:
                logging.getLogger(__name__).warning("field_freq_rank.npy not found in %s: remap comes from the checkpoint", kwargs.get("data_dir"))
            self.register_buffer("capped_remap", torch.from_numpy(remap))
            self.register_buffer("capped_field_idx", torch.arange(self.num_fields).view(1, -1), persistent=False)
            self.capped_slots = total
            self.branch_embedding_layers = nn.ModuleList([nn.Embedding(total + 1, d_full, padding_idx=total) for _ in range(self.num_towers)])
            for t in self.branch_embedding_layers:
                nn.init.normal_(t.weight, std=1e-4)
                with torch.no_grad():
                    t.weight[total].zero_()
        half_dim = self.view_embedding_dim // (2 * self.view_num_heads)
        emb_len = 2 * half_dim * self.num_fields                       # per-head embedding layout [F x half | F x half]
        total_k = self.num_towers + len(self.extra_tower_kinds)
        gen = torch.Generator().manual_seed(int(branch_transform_seed))
        if self.branch_subspace_keep > 0:
            self.register_buffer("branch_subspace_masks", (torch.rand(total_k, emb_len, generator=gen) < self.branch_subspace_keep).float())
        if self.branch_rotation != "none":
            size = half_dim if self.branch_rotation == "block" else emb_len
            rots = []
            for _ in range(total_k):
                q, r = torch.linalg.qr(torch.randn(size, size, generator=gen))
                rots.append(q * torch.sign(torch.diagonal(r)).view(1, -1))    # Haar-distributed orthogonal matrix
            self.register_buffer("branch_rotations", torch.stack(rots))
        if self.anchor_mask_keep > 0:
            for tower in self.towers:
                tower.register_buffer("anchor_mask", (torch.rand(emb_len, generator=gen) < self.anchor_mask_keep).float())
        if self.anchor_proj_enabled:
            for tower in self.towers:
                proj = nn.Linear(emb_len, emb_len, bias=False)
                with torch.no_grad():
                    proj.weight.copy_(torch.eye(emb_len))
                tower.anchor_proj = proj
        extra = []
        for kind in self.extra_tower_kinds:
            if kind == "mlp":
                extra.append(_MLPTower(self.view_num_heads * emb_len, [int(h) for h in (extra_mlp_hidden or [512, 256, 128])]))
            elif kind == "bilinear":
                extra.append(_BilinearTower(self.num_fields, half_dim, self.view_num_heads))
            elif kind == "rmlp":
                extra.append(_RecursiveMLPTower(self.view_num_heads * emb_len))
            else:
                raise ValueError("extra_towers entries must be mlp, bilinear or rmlp")
        self.extra_towers = nn.ModuleList(extra)
        if self.branch_embeddings == "soft":
            gate_dim = self.num_fields * (self.view_embedding_dim // self.view_num_heads)
            n_gates = max(self.gate_pool, self.num_towers)
            # small random asymmetry at init: at exactly equal gates the ownership entropy is at a saddle (zero gradient)
            self.branch_gate_logits = nn.Parameter(torch.full((n_gates, gate_dim), self.branch_gate_init)
                                                   + 0.5 * torch.randn(n_gates, gate_dim))
            if self.gate_tokens:
                self.branch_gate_tokens = nn.Parameter(0.01 * torch.randn(n_gates, gate_dim))
        self.reset_parameters()
        for tower in self.towers:
            tower.reset_after_global_linear_init()
        if getattr(self, "anchor_proj_enabled", False):        # identity init must survive the generic parameter reset
            for tower in self.towers:
                with torch.no_grad():
                    tower.anchor_proj.weight.copy_(torch.eye(tower.anchor_proj.weight.shape[0]))
        with torch.no_grad():                                    # zero inits that must survive the generic parameter reset
            if getattr(self, "fusion_head", "mean") == "bilinear":
                self.fusion_V.weight.zero_()
            for tower in self.towers:
                if getattr(tower, "gated_cross", False):
                    tower.gate_V.weight.zero_(); tower.gate_V.bias.zero_()
        if self.halt_head is not None:
            # reset_parameters() re-initialises the halting head generically (bias 0 -> stop prob 0.5, so a
            # learned-exit evaluation would stop at step 1 from the first epoch); restore the base-class init
            nn.init.zeros_(self.halt_head[-1].weight)
            nn.init.constant_(self.halt_head[-1].bias, -2.0)
        # weight sharing: group -> index of the tower that owns the parameters
        self.tie_group_of_tower = []
        owners = {}
        for k in range(self.num_towers):
            group = "all" if self.tied_towers else (self.tower_specs[k].get("tie") if self.tower_specs else None)
            group = str(group) if group else None
            self.tie_group_of_tower.append(group)
            if group is None:
                continue
            if group in owners:
                _tie_parameters(self.towers[owners[group]], self.towers[k], scope=self.tie_scope)
            else:
                owners[group] = k
        # towers with a distinct full parameter set (partial tying keeps every tower distinct)
        self.tie_representatives = [k for k in range(self.num_towers)
                                    if self.tie_scope != "all" or self.tie_group_of_tower[k] is None
                                    or owners[self.tie_group_of_tower[k]] == k]
        self._eval_patterns = [tower.phase_pattern for tower in self.towers]
        self.compile(kwargs["optimizer"], kwargs["loss"], float(kwargs.get("learning_rate", 1e-3)))
        self.to_device()

    # ----------------------------------------------------------------- input views
    def _field_mask(self, rng) -> torch.Tensor:
        fields = [1.0 if rng.random() < self.view_keep_prob else 0.0 for _ in range(self.num_fields)]
        if sum(fields) == 0:
            fields[rng.randrange(self.num_fields)] = 1.0
        # MultiHeadFeatureEmbedding layout per head: [F x D/(2H) | F x D/(2H)] (two field-major halves)
        half = self.view_embedding_dim // (2 * self.view_num_heads)
        return torch.tensor(fields).repeat_interleave(half).repeat(2)

    def view_mask(self, v: int):
        """Deterministic evaluation mask of view v (v = 0: the full input, returns None)."""
        v = int(v)
        if v == 0:
            return None
        if v not in self._eval_mask_cache:
            self._eval_mask_cache[v] = self._field_mask(random.Random(10_000 + v))
        return self._eval_mask_cache[v]

    def _rollout_views(self):
        """[(view index, mask or None)] for one rollout; fresh random masks while training."""
        if self.input_views <= 0:
            return [(0, None)]
        views = [(0, None)] if self.views_include_full else []
        for v in range(1, self.input_views + 1):
            mask = self._field_mask(random) if (self.training and self.view_resample) else self.view_mask(v)
            views.append((v, mask))
        return views

    # ----------------------------------------------------------------- per-tower data streams
    def _split_loss(self, fused, tower_logits_last, y_true) -> torch.Tensor:
        """Each tower's label loss on its own disjoint row subset (row index % K == k) + split_fused_weight x fused BCE."""
        k_n = len(tower_logits_last)
        rows = torch.arange(y_true.size(0), device=y_true.device)
        terms = []
        for k, tl in enumerate(tower_logits_last):
            mask = rows % k_n == k
            terms.append(F.binary_cross_entropy_with_logits(tl.view_as(y_true)[mask], y_true[mask]))
        total = torch.stack(terms).mean()
        if self.split_fused_weight > 0:
            total = total + self.split_fused_weight * F.binary_cross_entropy_with_logits(fused.view_as(y_true), y_true)
        return total

    # ----------------------------------------------------------------- teacher distillation
    def inputs_to_device(self, inputs):
        """(X, y) or (X, y, teacher_logit): the teacher logit is stashed for the distillation terms of this forward."""
        teacher = None
        if isinstance(inputs, (tuple, list)) and len(inputs) == 3:
            X, y, teacher = inputs
            inputs = (X, y)
        X, y = super().inputs_to_device(inputs)
        if teacher is None:
            self._teacher = None
        else:
            teacher = teacher.float().to(self.device, non_blocking=self.device.type == "cuda")
            self._teacher = teacher.view(-1, 1) if teacher.dim() == 1 else teacher            # N x 1 (mean) or N x M (members)
        return X, y

    def _distill_loss(self, tower_logits_last, fused_last, y_true, members=None) -> torch.Tensor:
        """tower_logits_last: K last-step branch logits; fused_last: their fusion; teacher logit t from the batch."""
        t_all = self._teacher
        t = (t_all.mean(dim=1, keepdim=True) if t_all.dim() == 2 and t_all.size(1) > 1 else t_all).view_as(y_true)
        fused = fused_last.view_as(y_true)
        if self.distill_mode == "route_member":                            # route drop_j -> member j mod M; trained route -> mean; fusion -> mean of its targets
            m = t_all.size(1) if t_all.dim() == 2 else 1
            targets = []
            for k, tl in enumerate(tower_logits_last):
                j = members[k] if (members is not None and k < len(members)) else None
                targets.append(t if (j is None or m == 1) else t_all[:, int(j) % m].view_as(y_true))
            terms = [F.binary_cross_entropy_with_logits(tl.view_as(y_true), torch.sigmoid(tg)) for tl, tg in zip(tower_logits_last, targets)]
            fused_target = torch.stack(targets, dim=0).mean(dim=0)
            fused_term = self.distill_member_fusion_weight * F.binary_cross_entropy_with_logits(fused, torch.sigmoid(fused_target))
            if self.partition_member_grads:
                self._private_terms.append(torch.stack(terms).mean())
                return fused_term
            return torch.stack(terms).mean() + fused_term
        if self.distill_mode == "member":                                  # tower k -> teacher member k mod M, fusion -> mean
            m = t_all.size(1) if t_all.dim() == 2 else 1
            terms = [F.binary_cross_entropy_with_logits(tl.view_as(y_true), torch.sigmoid(t_all[:, k % m].view_as(y_true)))
                     for k, tl in enumerate(tower_logits_last)]
            return torch.stack(terms).mean() + self.distill_member_fusion_weight * F.binary_cross_entropy_with_logits(fused, torch.sigmoid(t))
        if self.distill_mode == "fusion_rank":
            f, tt = fused.reshape(-1), t.reshape(-1)
            h = f.numel() // 2
            perm = torch.randperm(f.numel(), device=f.device)
            a, b = perm[:h], perm[h:2 * h]
            rank = F.binary_cross_entropy_with_logits((f[a] - f[b]) / self.distill_rank_temp, torch.sigmoid((tt[a] - tt[b]) / self.distill_rank_temp))
            return F.binary_cross_entropy_with_logits(fused, torch.sigmoid(t)) + rank
        if self.distill_mode == "fusion":                                  # fusion-level KD (probability space)
            return F.binary_cross_entropy_with_logits(fused, torch.sigmoid(t))
        if self.distill_mode == "rank":                                    # ranking KD: teacher pairwise preferences
            f, tt = fused.reshape(-1), t.reshape(-1)
            h = f.numel() // 2
            if h == 0:
                return f.new_zeros(())
            perm = torch.randperm(f.numel(), device=f.device)
            a, b = perm[:h], perm[h:2 * h]
            return F.binary_cross_entropy_with_logits((f[a] - f[b]) / self.distill_rank_temp,
                                                      torch.sigmoid((tt[a] - tt[b]) / self.distill_rank_temp))
        # red: each branch regresses the residual the others leave against the teacher (others stop-gradient)
        logits = [tl.view_as(y_true) for tl in tower_logits_last]
        k = len(logits)
        total_sg = torch.stack([l.detach() for l in logits], dim=0).sum(dim=0)
        terms = [((l - (k * t - (total_sg - l.detach()))) ** 2).mean() for l in logits]
        return torch.stack(terms).sum() / float(k * k)        # value and gradient equal the fusion-level logit MSE

    def _distill_active(self) -> bool:
        return self.distill_mode != "none" and self.distill_weight > 0 and self._teacher is not None

    # ----------------------------------------------------------------- breadth self-distillation
    def _bsd_loss(self, tower_logits_last, y_true) -> torch.Tensor:
        """tower_logits_last: list of K last-step logits (all towers).  Student = mean over deploy_towers, teacher = mean over all."""
        if not self.deploy_towers:
            return y_true.new_zeros(())
        teacher = torch.stack(tower_logits_last, dim=0).mean(dim=0).detach().view_as(y_true)
        student = torch.stack([tower_logits_last[i] for i in self.deploy_towers], dim=0).mean(dim=0).view_as(y_true)
        lam = max(0.0, 1.0 - self._bsd_step / max(self.bsd_curriculum_steps, 1))
        gt = F.binary_cross_entropy_with_logits(student, y_true)
        dist = F.binary_cross_entropy_with_logits(student, torch.sigmoid(teacher))
        return lam * gt + (1.0 - lam) * dist

    # ----------------------------------------------------------------- expert orthogonality
    def _bank_orth_loss(self) -> torch.Tensor:
        terms = []
        for k in self.tie_representatives:
            A = getattr(self.towers[k], "lora_A_bank", None)
            if A is None:
                continue
            flat = A.reshape(A.size(0), -1)
            flat = flat / (flat.norm(dim=1, keepdim=True) + 1e-8)
            G = flat @ flat.t()
            off = G - torch.diag(torch.diag(G))
            terms.append(off.pow(2).sum() / (G.size(0) * (G.size(0) - 1)))
        return torch.stack(terms).mean() if terms else self.towers[0].W_phase.new_zeros(())

    # ----------------------------------------------------------------- stochastic recursion
    def _stochastic_kl(self, auxes) -> torch.Tensor:
        terms = [torch.stack(a["kl"]).mean() for a in auxes if a.get("kl")]
        return torch.stack(terms).mean() if terms else self.towers[0].W_phase.new_zeros(())

    # ----------------------------------------------------------------- per-branch embeddings
    def _dim_mask(self, k: int) -> torch.Tensor:
        """Split mask of tower k over the embedding layout [F x D/(2H) | F x D/(2H)] per head (slice k of each field)."""
        if k not in self._dim_mask_cache:
            half = self.view_embedding_dim // (2 * self.view_num_heads)
            K = self.num_towers
            pattern = torch.tensor([1.0 if (j * K) // half == k else 0.0 for j in range(half)])
            self._dim_mask_cache[k] = pattern.repeat(self.num_fields).repeat(2)
        return self._dim_mask_cache[k]

    def tower_embedding(self, k: int, X, shared_emb=None):
        """Input embedding seen by tower k (shared table, its private table, or its slice of the shared table), followed by
        the fixed per-branch transforms (random subspace mask, orthogonal rotation) when enabled.  k >= num_towers = extra tower."""
        own = k < self.num_towers
        if self.branch_embeddings == "private" and own and k > 0:
            emb = self.branch_embedding_layers[k - 1](X)
        else:
            emb = shared_emb if shared_emb is not None else self.embedding_layer(X)
            if self.branch_embeddings == "hashed" and own:
                xl = X.long()
                fields = torch.arange(xl.size(1), device=xl.device).view(1, -1)
                idx = (xl * (1000003 + 7 * k) + fields * 7919 + k * 104729) % self.private_hash_buckets
                delta = self.branch_hash_tables[k](idx)                                  # N x F x hdim
                emb = emb + delta.view_as(emb).to(emb.dtype)
            elif self.branch_embeddings == "lowrank" and own:
                low = self.branch_embedding_layers[k].embedding_layer(X)                # B x F x r (private, full vocabulary)
                delta = self.branch_lowrank_proj[k](low.to(self.branch_lowrank_proj[k].weight.dtype))   # B x F x D
                emb = emb + self._to_head_layout(delta).to(emb.dtype)
            elif self.branch_embeddings == "hashtail":
                emb = self.embedding_layer(X, view=self._view_for(k))
            elif self.branch_embeddings == "capped" and own:
                idx = self.capped_remap[self.capped_field_idx, X.long()]              # B x F slot ids (overflow -> padding row)
                src = k if self._delta_src is None else ((k + 1) % self.num_towers if self._delta_src == "swap" else int(self._delta_src) % self.num_towers)
                delta = self.branch_embedding_layers[src](idx) * float(self._delta_scale)   # B x F x D, zero for tail ids
                dmask = self._delta_masks.get(k)
                if dmask is not None:
                    delta = delta * dmask.to(delta.device, delta.dtype).view(1, 1, -1)
                emb = emb + self._to_head_layout(delta).to(emb.dtype)
            elif self.branch_embeddings == "split" and own:
                emb = emb * self._dim_mask(k).to(emb.device, emb.dtype).view(1, 1, -1)
            elif self.branch_embeddings == "soft" and own:
                g = self._gate_index(k)
                emb = emb * torch.sigmoid(self.branch_gate_logits[g]).to(emb.dtype).view(1, 1, -1)
                if self.gate_tokens:
                    emb = emb + self.branch_gate_tokens[g].to(emb.dtype).view(1, 1, -1)
        return self._branch_transform(k, emb)

    def _regularization_loss(self):
        base = super()._regularization_loss()
        if self.rare_reg_scale <= 1.0 or not self._rare_masks or not self._embedding_regularizer:
            return base
        w = float(self._embedding_regularizer) * (self.rare_reg_scale - 1.0) / 2.0
        extra = 0.0
        layers = self.embedding_layer
        while not hasattr(layers, "embedding_layers") and hasattr(layers, "embedding_layer"):
            layers = layers.embedding_layer          # MultiHeadFeatureEmbedding -> FeatureEmbedding -> FeatureEmbeddingDict
        layers = layers.embedding_layers
        for n, mk in self._rare_masks.items():
            mod = layers[n] if n in layers else None
            if isinstance(mod, nn.Embedding):
                extra = extra + w * (mod.weight.pow(2).sum(dim=1) * mk.to(mod.weight.device, mod.weight.dtype)).sum()
        return base + extra

    def _to_head_layout(self, t):
        """B x F x D field-major tensor -> the MultiHeadFeatureEmbedding layout B x H x [F x D/(2H) | F x D/(2H)]."""
        H = self.view_num_heads
        heads = torch.stack(torch.tensor_split(t, H, dim=-1), dim=1)          # B x H x F x D/H
        a, b = torch.tensor_split(heads, 2, dim=-1)                            # two halves per head
        return torch.cat([a.flatten(start_dim=2), b.flatten(start_dim=2)], dim=-1)

    def _branch_transform(self, k: int, emb):
        """Fixed per-branch input transform: Bernoulli subspace mask and/or orthogonal rotation (block = within each field slice)."""
        if self.branch_subspace_keep > 0:
            emb = emb * self.branch_subspace_masks[k].to(emb.dtype).view(1, 1, -1)
        if self.branch_rotation == "full":
            emb = emb @ self.branch_rotations[k].to(emb.dtype)
        elif self.branch_rotation == "block":
            B, H, L = emb.shape
            emb = (emb.view(B, H, 2, self.num_fields, -1) @ self.branch_rotations[k].to(emb.dtype)).view(B, H, L)
        return emb

    def _view_for(self, k: int):
        """Tail view read by branch k: evaluator-fixed, random per step in training, else k mod V."""
        if k in self._view_fixed:
            v = self._view_fixed[k]
            return None if v == "mean" else int(v)
        if getattr(self, "view_by_route", False):                          # wave 23: route drop_j reads tail view j, other routes the mean view
            j = self._member_of_route(self.towers[k], getattr(self.towers[k], "route_override", None)) if k < len(self.towers) else None
            return None if j is None else int(j) % self.tail_views
        if self.training:
            return random.randrange(self.tail_views)
        return int(k) % self.tail_views

    def _needs_tower_embedding(self) -> bool:
        return self.branch_embeddings != "shared" or self.branch_subspace_keep > 0 or self.branch_rotation != "none"   # hashtail != shared

    def _gate_index(self, k: int) -> int:
        """Gate used by tower slot k: the evaluator's override, the rollout's sampled assignment, or gate k."""
        if self._gate_override is not None:
            return int(self._gate_override)
        if self._gate_assign is not None:
            return int(self._gate_assign[k])
        return int(k)

    def _sample_gate_assignment(self):
        """Per rollout: num_towers distinct gates from the pool while training (trajectory dropout), else 0..K-1."""
        if self.branch_embeddings == "soft" and self.gate_pool > self.num_towers and self.training:
            self._gate_assign = random.sample(range(self.gate_pool), self.num_towers)
        else:
            self._gate_assign = None

    def _feature_ownership_loss(self) -> torch.Tensor:
        """Mean over embedding dims of the entropy of the tower-ownership distribution softmax_k(sigmoid(g_k) / temp)."""
        s = torch.sigmoid(self.branch_gate_logits)                       # K x J
        p = torch.softmax(s / self.branch_gate_temp, dim=0)
        return -(p * p.clamp_min(1e-8).log()).sum(dim=0).mean()

    def branch_gate_stats(self) -> dict:
        s = torch.sigmoid(self.branch_gate_logits.detach())
        owner = (s / s.sum(dim=0, keepdim=True)).max(dim=0).values       # share of the dominant tower per dim
        return {"gate_mean": float(s.mean()), "gate_min": float(s.min()), "dims_owned_gt0.7": float((owner > 0.7).float().mean())}

    # ----------------------------------------------------------------- route dropout
    # ------------------------------------------------------- sampled branches (wave 17d)
    def _draw_latent(self, tower, rng) -> dict:
        """One branch latent from the training distribution: expert subset, rank mask, delta dim-group mask."""
        K = int(getattr(tower, "lora_bank_size", 0) or 0)
        r = int(getattr(tower, "lora_rank", 0) or (tower.lora_A_bank.shape[1] if getattr(tower, "lora_A_bank", None) is not None else 0))
        lat = {}
        if K > 1 and 0 < self.sb_experts < K:
            lat["allowed"] = sorted(rng.sample(range(K), self.sb_experts))
        if r > 0 and 0 < self.sb_rank_keep < 1:
            m = [1.0 if rng.random() < self.sb_rank_keep else 0.0 for _ in range(r)]
            if sum(m) == 0:
                m[rng.randrange(r)] = 1.0
            lat["rank_mask"] = m
        if self.sb_alpha:
            lat["z"] = [round(rng.random(), 4) for _ in range(24)]      # per-step anchor mix in [0, 1] (0 = input, 1 = state)
        if self.branch_embeddings == "capped" and self.sb_delta_groups > 1 and 0 < self.sb_delta_keep < self.sb_delta_groups:
            D = self.branch_embedding_layers[0].weight.shape[1]
            keep = set(rng.sample(range(self.sb_delta_groups), self.sb_delta_keep))
            lat["delta_mask"] = [1.0 if (d * self.sb_delta_groups) // D in keep else 0.0 for d in range(D)]
        return lat

    @staticmethod
    def _latent_hash(lat: dict) -> int:
        import hashlib, json
        return int(hashlib.blake2b(json.dumps(lat, sort_keys=True).encode(), digest_size=4).hexdigest(), 16)

    def _apply_latent_dict(self, k: int, lat: dict):
        tower = self.towers[k]
        tower.route_override = {"allowed": lat["allowed"]} if "allowed" in lat else None
        dev = tower.lora_A_bank.device if getattr(tower, "lora_A_bank", None) is not None else None
        tower.rank_mask = torch.tensor(lat["rank_mask"], device=dev) if "rank_mask" in lat else None
        tower.z_override = lat.get("z")
        if "delta_mask" in lat:
            self._delta_masks[k] = torch.tensor(lat["delta_mask"])
        else:
            self._delta_masks.pop(k, None)
        self._bag_seed[k] = self._latent_hash(lat)

    def apply_latent(self, k: int, seed: int) -> dict:
        """Evaluator: fix tower k to the latent drawn from `seed` (deterministic; unseen combinations are valid branches)."""
        lat = self._draw_latent(self.towers[k], random.Random(int(seed) * 7919 + 17))
        self._latent_fixed[k] = lat
        return lat

    def clear_latent(self, k: int):
        self._latent_fixed.pop(k, None)
        self.towers[k].rank_mask = None
        self.towers[k].z_override = None
        self._delta_masks.pop(k, None)
        self._bag_seed.pop(k, None)

    @staticmethod
    def _member_of_route(tower, route):
        """Index of the single dropped expert of a drop1 route (None for the trained router or any other route)."""
        K = int(getattr(tower, "lora_bank_size", 0) or 0)
        allowed = route.get("allowed") if route else None
        if allowed is None or K < 2:
            return None
        if len(allowed) == 1:                        # single-expert route: member = the expert index
            return int(allowed[0])
        if len(allowed) != K - 1:
            return None
        return int((set(range(K)) - set(allowed)).pop())

    def _sample_route(self, tower):
        """Training-time routing restriction for one rollout of `tower` (None = the full trained router)."""
        K = int(getattr(tower, "lora_bank_size", 0) or 0)
        if not self.training:
            return {"top_k": K} if (self.eval_route == "dense" and K > 0) else None
        if self.route_dropout_prob <= 0.0 or K < 2 or random.random() >= self.route_dropout_prob:
            return None
        family = self.route_dropout_family
        if family == "mixed":
            family = "drop1" if random.random() < 0.5 else "topk"
        if family == "topk":
            return {"top_k": random.choice([k for k in (1, 2, 4, 8) if k <= K])}
        if family == "drop1":
            drop = random.randrange(K)
            return {"allowed": [j for j in range(K) if j != drop]}
        if family == "single":                       # wave 25: route j = expert j alone (expert j is private to route j)
            return {"allowed": [random.randrange(K)]}
        h = K // 2
        if self.route_dropout_family == "halves":
            subsets = [list(range(h)), list(range(h, K)), list(range(0, K, 2)), list(range(1, K, 2))]
            return {"allowed": random.choice(subsets)}
        return {"allowed": sorted(random.sample(range(K), h))}

    # ----------------------------------------------------------------- rollout
    def _run_rollout(self, feature_emb, rarity, *, steps, requested_budget=None, return_states=False, X=None):
        budget = int(requested_budget or steps)
        auxes = []
        views = self._rollout_views()
        if self.training and self.tbptt_segment > 0:
            detach = list(range(self.tbptt_segment, int(steps), self.tbptt_segment))
        elif self.training and self.hrm_detach_segments:
            detach = self.anytime_supervision_depths
        else:
            detach = None
        if self._samples_schedules():
            if self.training:
                for tower, pattern in zip(self.towers, self._sample_schedules(len(self.towers))):
                    tower.phase_pattern = pattern
            else:
                for tower, pattern in zip(self.towers, self._eval_patterns):
                    tower.phase_pattern = pattern
        if self.training and self.stars_weight > 0:
            chosen = random.randint(0, len(self.towers) - 1)      # one tower per batch, same on all ranks
            for k, tower in enumerate(self.towers):
                tower.stars_active = (k == chosen)
        self._sample_gate_assignment()
        for k, tower in enumerate(self.towers):
            if self.sampled_branches and self.training:
                self._apply_latent_dict(k, self._draw_latent(tower, random))     # fresh latent per tower per step
            elif self.sampled_branches and k in self._latent_fixed:
                self._apply_latent_dict(k, self._latent_fixed[k])               # evaluator-chosen latent
            else:
                tower.route_override = self._sample_route(tower)      # None outside training / without route dropout
                self._route_member[k] = self._member_of_route(tower, tower.route_override)
                if self.sampled_branches:
                    tower.rank_mask = None; tower.z_override = None; self._delta_masks.pop(k, None); self._bag_seed.pop(k, None)
            base_emb = feature_emb if (not self._needs_tower_embedding() or X is None) else self.tower_embedding(k, X, feature_emb)
            for _v, mask in views:
                emb = base_emb if mask is None else base_emb * mask.to(base_emb.device, base_emb.dtype).view(1, 1, -1)
                _, aux = tower(emb, num_steps=int(steps), reference_steps=budget, requested_budget=budget,
                               detach_after_steps=detach, return_states=return_states or self.fusion_head == "bilinear")
                auxes.append(aux)
        tower_logits = [[self._mean_logit(x) for x in aux["step_logits"]] for aux in auxes]   # K x T
        for e, module in enumerate(self.extra_towers):                                        # heterogeneous branches
            ke = self.num_towers + e
            emb_e = feature_emb if (not self._needs_tower_embedding() or X is None) else self.tower_embedding(ke, X, feature_emb)
            if hasattr(module, "steps_forward"):
                tower_logits.append(module.steps_forward(emb_e, int(steps)))                  # recursive family: per-step logits
            else:
                tower_logits.append([module(emb_e)] * int(steps))                                 # static family: depth-independent logit
        fuse_idx = list(range(len(tower_logits)))
        if self.deploy_towers and not self.training and len(tower_logits) == self.num_towers:
            fuse_idx = [i for i in self.deploy_towers if i < len(tower_logits)]          # deployed towers only
        reduce = (lambda s: s.sum(dim=0)) if self.residual_branches else (lambda s: s.mean(dim=0))   # residual branches add up
        fused = [reduce(torch.stack([tower_logits[i][t] for i in fuse_idx], dim=0)) for t in range(int(steps))]
        if self.fusion_head == "bilinear" and len(auxes) > 1 and auxes[0].get("states"):
            # low-rank bilinear interaction between the towers' states at every step, added to the mean logit
            for t in range(int(steps)):
                corr = 0.0
                for i in range(len(auxes)):
                    hi = self.fusion_U(auxes[i]["states"][t].flatten(start_dim=1))
                    for j in range(i + 1, len(auxes)):
                        hj = self.fusion_V(auxes[j]["states"][t].flatten(start_dim=1))
                        corr = corr + (hi * hj).sum(dim=-1, keepdim=True) / math.sqrt(hi.size(-1))
                fused[t] = fused[t] + corr.view_as(fused[t])
        e_logits, l_logits = tower_logits[0], tower_logits[1 % len(tower_logits)]
        e_aux, l_aux = auxes[0], auxes[1 % len(auxes)]
        # early exit: the base-class halting head (features from the fused trajectory and the first two
        # towers) gives per-step stop probabilities, the halting distribution and the expected depth
        stop_probs, halt_dist, expected_depth = self._halt_outputs(fused, e_logits, l_logits, e_aux, l_aux, feature_emb, rarity)
        return {"fused": fused, "tower_logits": tower_logits, "auxes": auxes, "route_members": [self._route_member.get(k) for k in range(len(self.towers))],
                "e_logits": e_logits, "l_logits": l_logits, "e_aux": e_aux, "l_aux": l_aux,
                "stop_probs": stop_probs, "halt_dist": halt_dist, "expected_depth": expected_depth, "budget": budget}

    def analyze_batch(self, inputs, *, steps=None, return_states=False):
        X, y = self.inputs_to_device(inputs)
        feature_emb = self.embedding_layer(X)
        rarity = self._rarity_score(X)
        steps = int(steps or self._eval_steps_override or self.eval_steps)
        run = self._run_rollout(feature_emb, rarity, steps=steps, requested_budget=steps, return_states=return_states, X=X)
        return {"y_true": y, "fused_logits": run["fused"], "tower_logits": run["tower_logits"], "auxes": run["auxes"],
                "e2l_logits": run["e_logits"], "l2e_logits": run["l_logits"],
                "e2l_aux": run["e_aux"], "l2e_aux": run["l_aux"],
                "stop_probs": run["stop_probs"], "halt_dist": run["halt_dist"], "expected_depth": run["expected_depth"]}

    # --------------------------------------------------------- schedule sampling
    def _samples_schedules(self) -> bool:
        return bool(self.schedule_family) or self.schedule_sampler == "bernoulli"

    def _random_schedule(self) -> str:
        """One random schedule (python RNG: identical on every rank)."""
        if self.schedule_sampler == "bernoulli" or not self.schedule_family:
            p = random.uniform(*self.schedule_e_prob_range)
            return "".join("E" if random.random() < p else "L" for _ in range(self.schedule_sample_length))
        return random.choice(self.schedule_family)

    @staticmethod
    def mirror_schedule(s: str) -> str:
        return str(s).translate(str.maketrans("EL", "LE"))

    def _sample_schedules(self, n: int) -> list[str]:
        if self.schedule_mirror_pairs:
            n_pairs = (n + 1) // 2
            if self.schedule_sampler == "bernoulli":
                seeds = [self._random_schedule() for _ in range(n_pairs)]
            else:
                # one representative per mirror class of the family (E-first member), distinct per pair
                reps = sorted({min(s, self.mirror_schedule(s)) for s in self.schedule_family})
                draw = random.sample(reps, k=min(n_pairs, len(reps)))
                seeds = [draw[k % len(draw)] for k in range(n_pairs)]
            out = []
            for s in seeds:
                pair = [s, self.mirror_schedule(s)]
                if random.random() < 0.5:
                    pair.reverse()
                out += pair
            return out[:n]
        if self.schedule_sampler == "bernoulli":
            return [self._random_schedule() for _ in range(n)]
        draw = random.sample(self.schedule_family, k=min(n, len(self.schedule_family)))
        return [draw[k % len(draw)] for k in range(n)]

    def _loss_units(self, logits: list) -> list:
        """Units of the conditional tri-BCE: single towers, or mirrored pairs (2k, 2k+1) with pair_tri_bce."""
        if not self.pair_tri_bce or len(logits) < 2:
            return logits
        units = [(logits[k] + logits[k + 1]) / 2 for k in range(0, len(logits) - 1, 2)]
        if len(logits) % 2:
            units.append(logits[-1])
        return units

    def _rollout_steps(self):
        rollout, target = super()._rollout_steps()
        if (self.training and self.rollout_extension_depth > rollout and self.rollout_extension_prob > 0
                and random.random() < self.rollout_extension_prob):
            rollout = self.rollout_extension_depth
            target = random.choice(sorted(set(self.train_depth_choices) | {rollout}))
        return rollout, target

    def _supervised_depths(self, steps: int) -> list[int]:
        depths = [d for d in (self.anytime_supervision_depths or []) if 1 <= d <= steps]
        return depths or [steps]

    def _tower_rollout(self, tower, feature_emb, pattern: str, steps: int) -> list[torch.Tensor]:
        """Per-step mean logits of one tower along ``pattern`` (pattern restored afterwards)."""
        saved = tower.phase_pattern
        tower.phase_pattern = pattern
        try:
            _, aux = tower(feature_emb, num_steps=steps, reference_steps=steps, requested_budget=steps)
        finally:
            tower.phase_pattern = saved
        return [self._mean_logit(x) for x in aux["step_logits"]]

    def _fusion_distill(self, feature_emb, fused, y_true, steps: int) -> torch.Tensor:
        tower = self.towers[random.randrange(len(self.towers))]
        student = self._tower_rollout(tower, feature_emb, self._random_schedule(), steps)
        terms = [F.binary_cross_entropy_with_logits(student[d - 1], torch.sigmoid(fused[d - 1].detach()))
                 for d in self._supervised_depths(steps)]
        return torch.stack(terms).mean()

    def _order_consistency(self, feature_emb, y_true, steps: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Consistency between two schedules with equal symbol counts in every block."""
        a = list(self._random_schedule()[:steps].ljust(steps, "L"))
        b = list(a)
        blk = self.order_consistency_block
        for start in range(0, steps, blk):
            chunk = b[start:start + blk]
            random.shuffle(chunk)
            b[start:start + blk] = chunk
        tower = self.towers[random.randrange(len(self.towers))]
        la = self._tower_rollout(tower, feature_emb, "".join(a), steps)
        lb = self._tower_rollout(tower, feature_emb, "".join(b), steps)
        depths = self._supervised_depths(steps)
        consistency = torch.stack([F.mse_loss(la[d - 1], lb[d - 1]) for d in depths]).mean()
        supervised = torch.stack([F.binary_cross_entropy_with_logits(l[d - 1], y_true)
                                  for l in (la, lb) for d in depths]).mean()
        return consistency, supervised

    def _depth_improvement(self, fused, y_true, steps: int) -> torch.Tensor:
        depths = self._supervised_depths(steps)
        if len(depths) < 2:
            return y_true.new_zeros(())
        losses = [F.binary_cross_entropy_with_logits(fused[d - 1], y_true) for d in depths]
        return torch.stack([F.relu(self.depth_improvement_delta - (losses[i - 1] - losses[i]))
                            for i in range(1, len(losses))]).mean()

    def _eta_floor_loss(self, auxes) -> torch.Tensor:
        terms = []
        for aux in auxes:
            for t, eta in enumerate(aux["eta"]):
                if t + 1 > self.eta_floor_start:
                    terms.append(F.relu(self.eta_floor - eta).mean())
        return torch.stack(terms).mean() if terms else auxes[0]["eta"][0].new_zeros(())

    # ------------------------------------------------------------------ losses
    def _residual_loss(self, units, y_true):
        """Boosting-style branches: unit 0 fits the labels alone; unit k fits the residual of the detached sum of units < k."""
        total = F.binary_cross_entropy_with_logits(units[0], y_true)
        prefix = units[0].detach()
        for u in units[1:]:
            total = total + F.binary_cross_entropy_with_logits(prefix + u, y_true)
            prefix = prefix + u.detach()
        return total

    def _bag_weights(self, k: int, y_true):
        """Per-row Poisson(1) bootstrap weights of branch k, a deterministic function of the input row and k (mean-normalised)."""
        if not self.branch_bagging or self._bag_X is None:
            return None
        h = (self._bag_X.long() * self.bag_primes.view(1, -1)).sum(dim=1)
        seed = int(self._bag_seed.get(k, k + 1)) & 0x7FFFFFFF                       # sampled branches: seed = latent hash
        h = ((h ^ (seed * 2654435769)) * 2654435761 + 97) & 0x7FFFFFFF               # branch seed mixed before the multiply
        u = h.to(torch.float32) / 2147483647.0
        thresholds = u.new_tensor([0.3679, 0.7358, 0.9197, 0.9810, 0.9963, 0.9994])   # Poisson(1) CDF
        w = (u.view(-1, 1) > thresholds.view(1, -1)).sum(dim=1).to(torch.float32)
        w = w / w.mean().clamp_min(1e-6)
        return w.view_as(y_true)

    @staticmethod
    def _unit_loss(kind: str, u, y_true, w=None):
        """bce: (row-weighted) BCE; rank: pairwise logistic loss over in-batch positive/negative pairs (an AUC surrogate)."""
        if kind == "rank":
            lg, yy = u.reshape(-1), y_true.reshape(-1)
            pos, neg = lg[yy > 0.5], lg[yy <= 0.5]
            if pos.numel() == 0 or neg.numel() == 0:
                return F.binary_cross_entropy_with_logits(u, y_true)
            return F.softplus(neg.view(1, -1) - pos.view(-1, 1)).mean()
        if w is None:
            return F.binary_cross_entropy_with_logits(u, y_true)
        return (F.binary_cross_entropy_with_logits(u, y_true, reduction="none") * w).mean()

    def _branch_loss(self, fused, units, y_true, focus_pos=None):
        """tri-BCE over the loss units; with branch masking the fused prediction is recomputed over the masked units."""
        if self.residual_branches and len(units) > 1:
            return self._residual_loss(units, y_true)
        if self.branch_objective == "mixed" and len(units) > 1:      # joint fused BCE + decoupled per-branch BCE, weighted
            lam = self.objective_mix
            return lam * F.binary_cross_entropy_with_logits(fused, y_true) + (1.0 - lam) * torch.stack(
                [self._unit_loss("bce", u, y_true, self._bag_weights(k, y_true)) for k, u in enumerate(units)]).sum()
        if self.branch_objective == "detached" and len(units) > 1:   # symmetric detached fusion: branch k fixes the others' current errors
            K = len(units)
            terms = []
            for k, u in enumerate(units):
                others = torch.stack([units[j].detach() for j in range(K) if j != k]).sum(dim=0)
                terms.append(F.binary_cross_entropy_with_logits((u + others) / K, y_true))
            return torch.stack(terms).mean()
        if self.branch_objective == "independent":     # decoupled branches: every unit fits the labels alone, fusion only at inference
            K = len(units)
            terms = []
            for k, u in enumerate(units):
                kind = self.branch_losses[k % len(self.branch_losses)] if self.branch_losses else "bce"
                term = self._unit_loss(kind, u, y_true, self._bag_weights(k, y_true))
                if focus_pos is not None and K > 1:      # depth-budget specialists: branch k prefers budget position k/(K-1)
                    term = term * math.exp(-((focus_pos - k / (K - 1)) / self.branch_budget_focus) ** 2)
                terms.append(term)
            return torch.stack(terms).sum()
        if self.branch_bagging and self._bag_X is not None:   # tri-BCE with bagged per-branch terms
            loss = F.binary_cross_entropy_with_logits(fused, y_true)
            total = loss
            for k, tl in enumerate(units):
                loss_k = self._unit_loss("bce", tl, y_true, self._bag_weights(k, y_true))
                total = total + loss_k * torch.where(loss_k > loss, loss_k - loss, loss_k.new_zeros(()))
            return total
        if self.training and self.branch_mask_keep < 1.0 and len(units) > 1:
            loss, _ = masked_tri_bce(units, y_true, self.branch_mask_keep, self.branch_mask_mode, True)
            return loss
        return self._tri_bce(fused, units, y_true)

    def _tri_bce(self, fused, tower_logits_last, y_true):
        """K-tower conditional tri-BCE: fused BCE plus each tower's excess over the fused loss."""
        loss = F.binary_cross_entropy_with_logits(fused, y_true)
        total = loss
        for tl in tower_logits_last:
            loss_k = F.binary_cross_entropy_with_logits(tl, y_true)
            total = total + loss_k * torch.where(loss_k > loss, loss_k - loss, loss_k.new_zeros(()))
        return total

    def _budget_advantage_loss(self, budget_fused: list[torch.Tensor], y_true: torch.Tensor) -> torch.Tensor:
        """Group-relative (GRPO-style) advantage over the budget group of each example."""
        per = torch.stack([F.binary_cross_entropy_with_logits(f, y_true, reduction="none") for f in budget_fused], dim=0)
        reward = -per.detach()                                            # B_budgets x N x 1
        advantage = (reward - reward.mean(dim=0, keepdim=True)) / (reward.std(dim=0, keepdim=True) + 1e-6)
        weights = torch.softmax(advantage / self.budget_advantage_temperature, dim=0)
        return (weights * per).sum(dim=0).mean()

    @staticmethod
    def _branch_error_correlation(tower_logits_last, y_true) -> torch.Tensor:
        """Mean pairwise Pearson correlation of tower errors (sigmoid(l_k) - y) over the batch."""
        errs = torch.stack([torch.sigmoid(tl).reshape(-1) - y_true.reshape(-1) for tl in tower_logits_last], dim=0)
        errs = errs - errs.mean(dim=1, keepdim=True)
        errs = errs / errs.norm(dim=1, keepdim=True).clamp_min(1e-8)
        corr = errs @ errs.T
        k = corr.size(0)
        if k < 2:
            return corr.new_zeros(())
        mask = torch.triu(torch.ones_like(corr, dtype=torch.bool), diagonal=1)
        return corr[mask].mean()

    def _stars_terms(self, auxes) -> tuple[torch.Tensor | None, float | None]:
        pens = [aux["stars_penalty"] for aux in auxes if aux.get("stars_penalty") is not None]
        if not pens:
            return None, None
        rhos = [float(aux["stars_rho"]) for aux in auxes if aux.get("stars_rho") is not None]
        return torch.stack(pens).mean(), float(sum(rhos) / len(rhos))

    def _forward_multi_budget(self, X, y_true_original, feature_emb, rarity):
        self._bag_X = X
        budgets = self._sample_multi_budgets()
        runs = {b: self._run_rollout(feature_emb, rarity, steps=b, requested_budget=b, X=X) for b in budgets}
        primary_budget = min(budgets, key=lambda b: (abs(b - self.eval_steps), b))
        primary = runs[primary_budget]
        logit = primary["fused"][-1]
        y_pred = self.output_activation_fn(logit) if self.output_activation_fn else logit
        out = {"y_pred": y_pred}
        components = {}
        if self.loss_fn is not None:
            y_true = y_true_original.view_as(y_pred)
            losses, weights = [], []
            for b in budgets:
                run = runs[b]
                units_b = self._loss_units([tl[-1] for tl in run["tower_logits"]])
                pos = (sorted(budgets).index(b) / max(len(budgets) - 1, 1)) if self.branch_budget_focus > 0 else None
                losses.append(self._split_loss(run["fused"][-1], units_b, y_true) if self.tower_data_split else self._branch_loss(run["fused"][-1], units_b, y_true, focus_pos=pos))
                weights.append(self.multi_budget_depth_decay ** (max(budgets) - b))
            w = y_true.new_tensor(weights)
            total = (torch.stack(losses) * w).sum() / w.sum().clamp_min(1e-12)
            components["multi_budget_tri_bce"] = total.detach()
            if self.self_distill_weight > 0:      # every branch matches the detached fused prediction of its own budget (no external teacher)
                sd = torch.stack([F.binary_cross_entropy_with_logits(tl[-1].view_as(y_true), torch.sigmoid(runs[b]["fused"][-1].detach()).view_as(y_true))
                                  for b in budgets for tl in runs[b]["tower_logits"]]).mean()
                total = total + self.self_distill_weight * sd
                components["self_distill"] = sd.detach()
            if self.residual_step_weight > 0:
                terms = [F.binary_cross_entropy_with_logits((tl[t - 1].detach() + tl[t] - tl[t - 1]).view_as(y_true), y_true)
                         for b in budgets for tl in runs[b]["tower_logits"][:self.num_towers] for t in range(1, len(tl))]
                rs = torch.stack(terms).mean()
                total = total + self.residual_step_weight * rs
                components["residual_step"] = rs.detach()
            if self.stochastic_beta > 0:
                kl = self._stochastic_kl([a for b in budgets for a in runs[b]["auxes"]])
                total = total + self.stochastic_beta * kl
                components["stochastic_kl"] = kl.detach()
            if self.branch_gate_reg > 0:
                own = self._feature_ownership_loss()
                total = total + self.branch_gate_reg * own
                components["feature_ownership"] = own.detach()
            if self.bank_orth_weight > 0:
                orth = self._bank_orth_loss()
                total = total + self.bank_orth_weight * orth
                components["bank_orth"] = orth.detach()
            if self.indep_loss_weight > 0:
                ind = torch.stack([F.binary_cross_entropy_with_logits(tl[-1].view_as(y_true), y_true) for b in budgets for tl in runs[b]["tower_logits"]]).mean()
                total = total + self.indep_loss_weight * ind
                components["independent_bce"] = ind.detach()
            if self.anytime_fuse_weight > 0 and len(budgets) > 1:
                anyf = F.binary_cross_entropy_with_logits(torch.stack([runs[b]["fused"][-1].view_as(y_true) for b in budgets], dim=0).mean(dim=0), y_true)
                total = total + self.anytime_fuse_weight * anyf
                components["anytime_fused_depth"] = anyf.detach()
            if self.bsd_weight > 0 and self.deploy_towers:
                bsd = torch.stack([self._bsd_loss([tl[-1] for tl in runs[b]["tower_logits"]], y_true) for b in budgets]).mean()
                total = total + self.bsd_weight * bsd
                components["breadth_self_distill"] = bsd.detach()
                self._bsd_step += 1
            if self._distill_active():
                self._private_terms = []
                dl = torch.stack([self._distill_loss([tl[-1] for tl in runs[b]["tower_logits"]], runs[b]["fused"][-1], y_true, members=runs[b].get("route_members")) for b in budgets]).mean()
                total = total + self.distill_weight * dl
                components[f"distill_{self.distill_mode}"] = dl.detach()
                if self._private_terms:
                    out["loss_private"] = self.distill_weight * torch.stack(self._private_terms).mean()
                    components["distill_private"] = out["loss_private"].detach()
            if self.cross_budget_progressive_weight > 0:
                val = self._cross_budget_progressive([runs[b]["fused"][-1] for b in budgets], y_true)
                total = total + self.cross_budget_progressive_weight * val
                components["cross_budget_progressive"] = val.detach()
            if self.budget_advantage_weight > 0 and len(budgets) > 1:
                val = self._budget_advantage_loss([runs[b]["fused"][-1] for b in budgets], y_true)
                total = total + self.budget_advantage_weight * val
                components["budget_advantage"] = val.detach()
            if self.progressive_margin_weight > 0:
                val = torch.stack([self._progressive_margin(runs[b]["fused"], y_true) for b in budgets]).mean()
                total = total + self.progressive_margin_weight * val
                components["progressive_margin"] = val.detach()
            if self.stability_ratio_weight > 0:
                val = torch.stack([self._stability_ratio_loss(*runs[b]["auxes"]) for b in budgets]).mean()
                total = total + self.stability_ratio_weight * val
                components["stability_ratio"] = val.detach()
            corr = torch.stack([self._branch_error_correlation([tl[-1] for tl in runs[b]["tower_logits"]], y_true)
                                for b in budgets]).mean()
            components["branch_error_corr"] = corr.detach()
            if self.branch_decorrelation_weight > 0:
                total = total + self.branch_decorrelation_weight * corr
            if self.schedule_advantage_weight > 0 and len(primary["tower_logits"]) > 1:
                adv = torch.stack([self._budget_advantage_loss([tl[-1] for tl in runs[b]["tower_logits"]], y_true)
                                   for b in budgets]).mean()
                total = total + self.schedule_advantage_weight * adv
                components["schedule_advantage"] = adv.detach()
            if self.fusion_distill_weight > 0:
                # student: one tower on a random schedule at the primary budget, matched to the
                # detached fused prediction at that budget's final step
                tower = self.towers[random.randrange(len(self.towers))]
                student = self._tower_rollout(tower, feature_emb, self._random_schedule(), primary_budget)
                dist = F.binary_cross_entropy_with_logits(student[-1], torch.sigmoid(primary["fused"][-1].detach()))
                total = total + self.fusion_distill_weight * dist
                components["fusion_distill"] = dist.detach()
            if self.stars_weight > 0:
                terms = [self._stars_terms(runs[b]["auxes"]) for b in budgets]
                pens = [p for p, _ in terms if p is not None]
                if pens:
                    val = torch.stack(pens).mean()
                    total = total + self.stars_weight * val
                    components["stars_penalty"] = val.detach()
                    components["stars_rho"] = torch.tensor(sum(r for _, r in terms if r is not None) / len(pens))
            total = total + self._regularization_loss()
            out["loss"] = total
        step = getattr(self, "_total_batches", 0)
        if self.training and self.track_anytime and (step % self.aux_log_every == 0):
            payload = self._diagnostic_payload(primary["e_aux"], primary["l_aux"], None, None, primary_budget)
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
        self._bag_X = X
        if self.training and self.multi_budget_training:
            return self._forward_multi_budget(X, y_true_original, feature_emb, rarity)
        rollout_steps, target_depth = self._rollout_steps()
        has_budget = self.budget_condition_phase or self.budget_condition_router or self.budget_condition_update
        run = self._run_rollout(feature_emb, rarity, steps=rollout_steps,
                                requested_budget=target_depth if has_budget else rollout_steps, X=X)
        halt_mode = self._halt_eval_override or self.halt_eval_mode
        if (not self.training) and halt_mode == "learned" and run["stop_probs"] is not None:
            logit, _selected = self._select_learned_halt(run["fused"], run["stop_probs"])   # early exit per example
        else:
            logit = run["fused"][target_depth - 1]
        y_pred = self.output_activation_fn(logit) if self.output_activation_fn else logit
        out = {"y_pred": y_pred}
        if self.loss_fn is not None:
            y_true = y_true_original.view_as(y_pred)
            idx = target_depth - 1
            units_1 = self._loss_units([tl[idx] for tl in run["tower_logits"]])
            total = self._split_loss(run["fused"][idx], units_1, y_true) if self.tower_data_split else self._branch_loss(run["fused"][idx], units_1, y_true)
            if self.self_distill_weight > 0:
                total = total + self.self_distill_weight * torch.stack([F.binary_cross_entropy_with_logits(tl[idx].view_as(y_true), torch.sigmoid(run["fused"][idx].detach()).view_as(y_true)) for tl in run["tower_logits"]]).mean()
            if self.halt_oracle_weight > 0 and run["halt_dist"] is not None:
                val, _oracle = self._oracle_halting_loss(run["fused"], y_true, run["halt_dist"])
                total = total + self.halt_oracle_weight * val
            if self.halt_ponder_weight > 0 and run["expected_depth"] is not None:
                total = total + self.halt_ponder_weight * run["expected_depth"].mean() / float(max(rollout_steps, 1))
            if self.anytime_bce_weight > 0:
                total = total + self.anytime_bce_weight * self._anytime_bce(run["fused"], y_true)
            if self.progressive_margin_weight > 0:
                total = total + self.progressive_margin_weight * self._progressive_margin(run["fused"], y_true)
            if self.stability_ratio_weight > 0:
                total = total + self.stability_ratio_weight * self._stability_ratio_loss(*run["auxes"])
            if self.stochastic_beta > 0:
                kl = self._stochastic_kl(run["auxes"])
                total = total + self.stochastic_beta * kl
            if self.branch_gate_reg > 0:
                total = total + self.branch_gate_reg * self._feature_ownership_loss()
            if self.bank_orth_weight > 0:
                total = total + self.bank_orth_weight * self._bank_orth_loss()
            if self.indep_loss_weight > 0:
                total = total + self.indep_loss_weight * torch.stack([F.binary_cross_entropy_with_logits(tl[idx].view_as(y_true), y_true) for tl in run["tower_logits"]]).mean()
            if self.bsd_weight > 0 and self.deploy_towers:
                total = total + self.bsd_weight * self._bsd_loss([tl[idx] for tl in run["tower_logits"]], y_true)
                self._bsd_step += 1
            if self._distill_active():
                total = total + self.distill_weight * self._distill_loss([tl[idx] for tl in run["tower_logits"]], run["fused"][idx], y_true, members=run.get("route_members"))
            corr = self._branch_error_correlation([tl[idx] for tl in run["tower_logits"]], y_true)
            if self.branch_decorrelation_weight > 0:
                total = total + self.branch_decorrelation_weight * corr
            extras = {"branch_error_corr": corr.detach()}
            if self.schedule_advantage_weight > 0 and len(run["tower_logits"]) > 1:
                adv = self._budget_advantage_loss([tl[idx] for tl in run["tower_logits"]], y_true)
                total = total + self.schedule_advantage_weight * adv
                extras["schedule_advantage"] = adv.detach()
            if self.fusion_distill_weight > 0:
                dist = self._fusion_distill(feature_emb, run["fused"], y_true, rollout_steps)
                total = total + self.fusion_distill_weight * dist
                extras["fusion_distill"] = dist.detach()
            if self.depth_improvement_weight > 0:
                imp = self._depth_improvement(run["fused"], y_true, rollout_steps)
                total = total + self.depth_improvement_weight * imp
                extras["depth_improvement"] = imp.detach()
            if self.eta_floor_weight > 0:
                fl = self._eta_floor_loss(run["auxes"])
                total = total + self.eta_floor_weight * fl
                extras["eta_floor"] = fl.detach()
            if self.order_consistency_weight > 0:
                cons, sup = self._order_consistency(feature_emb, y_true, rollout_steps)
                total = total + self.order_consistency_weight * cons + max(self.anytime_bce_weight, 0.1) * sup
                extras["order_consistency"] = cons.detach()
                extras["order_consistency_bce"] = sup.detach()
            step = getattr(self, "_total_batches", 0)
            if self.track_anytime and step % self.aux_log_every == 0:
                tracking.log({f"loss/{k}": float(v) for k, v in extras.items()}, step=step)
            if self.stars_weight > 0:
                pen, rho = self._stars_terms(run["auxes"])
                if pen is not None:
                    total = total + self.stars_weight * pen
                    if self.track_anytime and step % self.aux_log_every == 0:
                        tracking.log({"loss/stars_penalty": float(pen.detach()), "anytime/stars_rho": rho}, step=step)
            out["loss"] = total + self._regularization_loss()
        return out


class MultiTowerFCNv2(dcn_layers.FCNv2):
    """Untied FCNv2 with K towers (alternating E2L / L2E schedules); K = 2 is the paper model."""

    def __init__(self, feature_map, num_towers: int = 4, exp_num_layers=5, lin_num_layers=5,
                 net_dropout=0.1, batch_norm=False, layer_norm=False, num_heads=1, embedding_dim=16,
                 tower_schedules: list | None = None, fcn_norm_cap: float | None = None,
                 tower_loss_mode: str = "tri", fcn_view_keep_prob: float = 0.0, fcn_view_learned: bool = False,
                 branch_mask_keep: float = 1.0, branch_mask_mode: str = "step",
                 teacher_logits: str | None = None, distill_mode: str = "none", distill_weight: float = 0.0,
                 distill_rank_temp: float = 1.0, distill_member_fusion_weight: float = 1.0, **kwargs):
        super().__init__(feature_map, exp_num_layers=exp_num_layers, lin_num_layers=lin_num_layers,
                         net_dropout=net_dropout, batch_norm=batch_norm, layer_norm=layer_norm,
                         num_heads=num_heads, embedding_dim=embedding_dim, **kwargs)
        # tower_schedules: optional explicit per-tower layer strings over {E, L} (one untied cross layer
        # per character: E = exponential / current-state anchor, L = linear / input anchor); overrides
        # num_towers and the alternating E{exp}L{lin} / L{lin}E{exp} layout.
        self.tower_schedules = [str(s).upper() for s in (tower_schedules or [])]
        self.fcn_norm_cap = float(fcn_norm_cap) if fcn_norm_cap else None
        # tower_loss_mode: "tri" = K-tower conditional tri-BCE (fused BCE + each tower's excess over it;
        # the fused term gives every tower a 1/K-scaled gradient); "independent" = sum of the K per-tower
        # BCEs with no fused term, so each tower is trained exactly as it would be alone (a jointly
        # embedded ensemble of K independently supervised towers).
        self.tower_loss_mode = str(tower_loss_mode)
        if self.tower_loss_mode not in ("tri", "independent"):
            raise ValueError("tower_loss_mode must be tri or independent")
        # fusion-KD control for the untied baseline: BCE(mean logit, sigmoid(teacher)) with the same precomputed teacher
        self.distill_mode = str(distill_mode)
        if self.distill_mode not in ("none", "fusion", "fusion_rank", "member"):
            raise ValueError("MultiTowerFCNv2 supports distill_mode none, fusion, fusion_rank or member")
        self.distill_rank_temp = float(distill_rank_temp); self.distill_member_fusion_weight = float(distill_member_fusion_weight)
        self.distill_weight = float(distill_weight)
        self.teacher_logits = teacher_logits
        self._teacher = None
        self.branch_mask_keep = float(branch_mask_keep)       # random branch masking in the loss (1.0 = off)
        self.branch_mask_mode = str(branch_mask_mode)         # step = branch dropout, example = bagging
        if self.branch_mask_mode not in ("step", "example"):
            raise ValueError("branch_mask_mode must be step or example")
        if any(set(s) - {"E", "L"} for s in self.tower_schedules):
            raise ValueError("tower_schedules must contain only E and L")
        self.num_towers = len(self.tower_schedules) if self.tower_schedules else int(num_towers)
        input_dim = feature_map.num_fields * embedding_dim
        e2l = ["ecn"] * int(exp_num_layers) + ["lcn"] * int(lin_num_layers)
        l2e = ["lcn"] * int(lin_num_layers) + ["ecn"] * int(exp_num_layers)
        if self.tower_schedules:
            schedules = [["ecn" if c == "E" else "lcn" for c in s] for s in self.tower_schedules]
        else:
            schedules = [(e2l if k % 2 == 0 else l2e) for k in range(self.num_towers)]
        del self.E2LCN, self.L2ECN
        self.towers = nn.ModuleList([
            dcn_layers.OfficialFCNv2CrossTower(input_dim=input_dim, schedule=sch,
                                               net_dropout=net_dropout, batch_norm=batch_norm,
                                               layer_norm=layer_norm, num_heads=num_heads,
                                               norm_cap=self.fcn_norm_cap)
            for sch in schedules
        ])
        self.hashtail = str(kwargs.get("branch_embeddings", "shared")) == "hashtail"
        self._view_fixed = {}
        if self.hashtail:
            self.tail_views = int(kwargs.get("tail_views", 4))
            self.embedding_layer = _HashTailEmbedding(feature_map, int(embedding_dim) * int(num_heads), int(num_heads), int(kwargs.get("tail_cap", 2000)),
                                                      self.tail_views, int(kwargs.get("tail_buckets", 131072)), str(kwargs.get("data_dir", "")))
        self.reset_parameters()
        # Per-tower input views: a fixed random field subspace per tower (keep prob fcn_view_keep_prob, seeded by
        # the run seed and the tower index), optionally learnable as a soft gate initialised from the mask.
        self.fcn_view_keep_prob = float(fcn_view_keep_prob)
        self.fcn_view_learned = bool(fcn_view_learned)
        if self.fcn_view_keep_prob <= 0:
            self.view_masks = None
        else:
            masks = []
            for k in range(self.num_towers):
                rng = random.Random(int(kwargs.get("seed", 2025)) * 100 + k)
                fields = [1.0 if rng.random() < self.fcn_view_keep_prob else 0.0 for _ in range(feature_map.num_fields)]
                if sum(fields) == 0:
                    fields[rng.randrange(feature_map.num_fields)] = 1.0
                half = int(embedding_dim) // (2 * int(num_heads))                     # embedding layout: two field-major halves per head
                masks.append(torch.tensor(fields).repeat_interleave(half).repeat(2))
            m = torch.stack(masks)                                                  # K x d
            if self.fcn_view_learned:
                self.view_masks = nn.Parameter(m)
            else:
                self.register_buffer("view_masks", m)
        self.compile(kwargs["optimizer"], kwargs["loss"], float(kwargs.get("learning_rate", 1e-3)))
        self.to_device()

    def _view_for(self, k: int):
        if k in self._view_fixed:
            v = self._view_fixed[k]; return None if v == "mean" else int(v)
        return random.randrange(self.tail_views) if self.training else int(k) % self.tail_views

    def tower_input(self, k: int, feature_emb: torch.Tensor) -> torch.Tensor:
        """The embedding as seen by tower k (its field view, or the full input when views are off)."""
        if self.view_masks is None:
            return feature_emb
        return feature_emb * self.view_masks[k].to(feature_emb.dtype).view(1, 1, -1)

    def inputs_to_device(self, inputs):
        teacher = None
        if isinstance(inputs, (tuple, list)) and len(inputs) == 3:
            X, y, teacher = inputs
            inputs = (X, y)
        X, y = super().inputs_to_device(inputs)
        self._teacher = None if teacher is None else teacher.float().to(self.device, non_blocking=self.device.type == "cuda")
        return X, y

    def forward(self, inputs):
        X, y_true_original = self.inputs_to_device(inputs)
        if getattr(self, "hashtail", False):                          # wave 20: every tower reads its own tail view
            tower_logits = [tower(self.tower_input(k, self.embedding_layer(X, view=self._view_for(k)))).mean(dim=1) for k, tower in enumerate(self.towers)]
        else:
            feature_emb = self.embedding_layer(X)
            tower_logits = [tower(self.tower_input(k, feature_emb)).mean(dim=1) for k, tower in enumerate(self.towers)]
        logit = torch.stack(tower_logits, dim=0).mean(dim=0)
        y_pred = self.output_activation_fn(logit) if self.output_activation_fn else logit
        out = {"y_pred": y_pred}
        if self.loss_fn is not None:
            y_true = y_true_original.view_as(y_pred)
            loss = self.loss_fn(y_pred, y_true, reduction="mean")
            if self.training and self.branch_mask_keep < 1.0 and len(tower_logits) > 1:
                total, _ = masked_tri_bce(tower_logits, y_true, self.branch_mask_keep, self.branch_mask_mode, True)
            elif self.tower_loss_mode == "independent":
                total = loss.new_zeros(())
                for tl in tower_logits:
                    pred_k = self.output_activation_fn(tl) if self.output_activation_fn else tl
                    total = total + self.loss_fn(pred_k, y_true, reduction="mean")
            else:
                total = loss
                for tl in tower_logits:
                    pred_k = self.output_activation_fn(tl) if self.output_activation_fn else tl
                    loss_k = self.loss_fn(pred_k, y_true, reduction="mean")
                    total = total + loss_k * torch.clamp(loss_k - loss, min=0.0)
            if self.distill_mode != "none" and self.distill_weight > 0 and self._teacher is not None:
                t_all = self._teacher
                t_all = t_all.view(-1, 1) if t_all.dim() == 1 else t_all
                t = (t_all.mean(dim=1, keepdim=True) if t_all.size(1) > 1 else t_all).view_as(logit)
                dl = F.binary_cross_entropy_with_logits(logit, torch.sigmoid(t))
                if self.distill_mode == "fusion_rank":                    # + in-batch pairwise ordering of the fused logit (same as the recursive class)
                    f, tt = logit.reshape(-1), t.reshape(-1)
                    h = f.numel() // 2
                    perm = torch.randperm(f.numel(), device=f.device)
                    a, b = perm[:h], perm[h:2 * h]
                    dl = dl + F.binary_cross_entropy_with_logits((f[a] - f[b]) / self.distill_rank_temp, torch.sigmoid((tt[a] - tt[b]) / self.distill_rank_temp))
                elif self.distill_mode == "member":                       # tower k -> member k mod M, fusion -> mean
                    m = t_all.size(1)
                    dl = self.distill_member_fusion_weight * dl + torch.stack([F.binary_cross_entropy_with_logits(tl.view_as(logit), torch.sigmoid(t_all[:, k % m].view_as(logit)))
                                                                             for k, tl in enumerate(tower_logits)]).mean()
                total = total + self.distill_weight * dl
            out["loss"] = total + self._regularization_loss()
        return out
