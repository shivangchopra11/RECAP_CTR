""" Weight EMA and per-step cosine LR for FuxiCTR models
    (same recipe as recap/trainer.py in CTR_ICLR/final).

    The EMA of the weights is updated after every optimizer step with a warm-up,
    d_t = min(ema_decay, (1 + t) / (10 + t)). Validation, checkpoint selection and the final
    test use the averaged weights (the best checkpoint stores them); training continues on
    the live weights. Config keys (model_config.yaml, per expid):
        ema_decay: 0.9995          # 0 / absent = off
        ema_scope: all             # all | dense (dense = all but the embedding tables)
        lr_schedule: cosine        # plateau (fuxictr default) | cosine
        cosine_epochs: 20          # per-step cosine from learning_rate to lr_min over this many epochs
        lr_min: 1.e-5
    With the cosine schedule also set reduce_lr_on_plateau: False.
"""

import math
import logging
import torch


class TrainingSchedule(object):
    def __init__(self, params):
        self.ema_decay = float(params.get("ema_decay") or 0.0)
        self.ema_scope = str(params.get("ema_scope") or "all")
        self.lr_schedule = str(params.get("lr_schedule") or "plateau")
        self.cosine_epochs = int(params.get("cosine_epochs", 20))
        self.lr_min = float(params.get("lr_min", 1e-5))
        self.ema_params, self.shadow, self.backup = [], None, None

    def watch(self, model):
        """ Hook into the FuxiCTR training loop of `model` (a fuxictr BaseModel). """
        if self.ema_decay <= 0 and self.lr_schedule != "cosine":
            return
        train_step, eval_step = model.train_step, model.eval_step
        lr0 = [group["lr"] for group in model.optimizer.param_groups]
        if self.ema_decay > 0:
            self.ema_params = [p for name, p in model.named_parameters()
                               if self.ema_scope == "all" or "embedding_layer." not in name]
            self.shadow = [p.detach().clone() for p in self.ema_params]
            logging.info("Weight EMA: decay={}, scope={} ({} parameters)".format(
                self.ema_decay, self.ema_scope, sum(p.numel() for p in self.ema_params)))
        if self.lr_schedule == "cosine":
            logging.info("Per-step cosine LR: {} -> {} over {} epochs".format(
                lr0[0], self.lr_min, self.cosine_epochs))

        def train_step_hook(batch_data):
            loss = train_step(batch_data)
            step = model._total_steps  # already counts this step (incremented by train_epoch)
            if self.lr_schedule == "cosine":
                total = max(1, self.cosine_epochs * model._steps_per_epoch)
                frac = min(step, total) / total
                for group, lr in zip(model.optimizer.param_groups, lr0):
                    group["lr"] = self.lr_min + 0.5 * (lr - self.lr_min) * (1.0 + math.cos(math.pi * frac))
            if self.shadow is not None:
                self._update(step)
            return loss

        def eval_step_hook():
            # validate + checkpoint the averaged weights, then resume training on the live ones
            self._apply()
            try:
                eval_step()
            finally:
                self._restore()

        model.train_step = train_step_hook
        model.eval_step = eval_step_hook

    @torch.no_grad()
    def _update(self, step):
        decay = min(self.ema_decay, (1.0 + step) / (10.0 + step))
        torch._foreach_mul_(self.shadow, decay)
        torch._foreach_add_(self.shadow, [p.detach() for p in self.ema_params], alpha=1.0 - decay)

    @torch.no_grad()
    def _apply(self):
        if self.shadow is None or self.backup is not None:
            return
        self.backup = [p.detach().clone() for p in self.ema_params]
        for p, s in zip(self.ema_params, self.shadow):
            p.copy_(s)

    @torch.no_grad()
    def _restore(self):
        if self.backup is None:
            return
        for p, b in zip(self.ema_params, self.backup):
            p.copy_(b)
        self.backup = None
