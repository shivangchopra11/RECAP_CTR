""" Weights & Biases logging for FuxiCTR runs.

    Per evaluation: train/{AUC,logloss,loss}, val/{AUC,logloss}, lr, epoch
    End of run:     final_val/{AUC,logloss}, test/{AUC,logloss} (also in the run summary)

    Train AUC/logloss are computed online from the predictions on the training batches since
    the last evaluation (model in train mode, i.e. with dropout), so they cost no extra pass.
    Configure in model_config.yaml (Base or per expid):
        wandb: {project: fcn-ctr, entity: null, mode: online, train_metrics: True}
    mode: online | offline | disabled  (or override with run_expid.py --wandb_mode)
"""

import os
import logging
import numpy as np

EXCLUDE_PARAMS = ["feature_cols", "label_col", "wandb", "data_loader"]


class WandbLogger(object):
    def __init__(self, params, config_dir, mode=None, run_id=None):
        cfg = params.get("wandb") or {}
        self.mode = mode or cfg.get("mode", "online")
        self.train_metrics = cfg.get("train_metrics", True)
        self.run = None
        if self.mode == "disabled":
            return
        import wandb
        self.wandb = wandb
        group = os.path.basename(os.path.normpath(config_dir))  # tuner sweep name, "config" otherwise
        init_args = dict(project=cfg.get("project", "fcn-ctr"),
                         entity=cfg.get("entity"),
                         name=params["model_id"],
                         group=None if group == "config" else group,
                         job_type="train",
                         tags=[params["model"], params["dataset_id"]],
                         config={k: v for k, v in params.items() if k not in EXCLUDE_PARAMS},
                         dir=params.get("model_root", "./checkpoints/"),
                         settings=wandb.Settings(init_timeout=120))
        init_args["config"]["slurm_job"] = os.environ.get("SLURM_JOB_ID")
        if run_id:  # resumed run (resume.py): continue the same W&B run
            init_args.update(id=run_id, resume="allow")
        try:
            self.run = wandb.init(mode=self.mode, **init_args)
        except Exception as e:  # e.g. no network on the compute node: keep a local copy
            logging.warning("wandb.init failed ({}), falling back to offline mode".format(e))
            self.run = wandb.init(mode="offline", **init_args)
        self._reset()

    def _reset(self):
        self._y_pred, self._y_true = [], []
        self._loss_sum, self._num_batches = 0., 0

    def watch(self, model):
        """ Hook into the FuxiCTR training loop of `model` (a fuxictr BaseModel). """
        if self.run is None:
            return
        compute_loss = model.compute_loss
        checkpoint_and_earlystop = model.checkpoint_and_earlystop

        def compute_loss_hook(return_dict, y_true):
            # only called by train_step, i.e. on training batches
            loss = compute_loss(return_dict, y_true)
            if self.train_metrics:
                self._y_pred.append(return_dict["y_pred"].detach().float().cpu().numpy().reshape(-1))
                self._y_true.append(y_true.detach().float().cpu().numpy().reshape(-1))
            self._loss_sum += loss.item()
            self._num_batches += 1
            return loss

        def checkpoint_and_earlystop_hook(logs, min_delta=1e-6):
            # called by eval_step with the validation metrics of this evaluation
            log = {"epoch": model._epoch_index + 1,
                   "lr": model.optimizer.param_groups[0]["lr"]}
            log.update({"val/" + k: v for k, v in logs.items()})
            if self._num_batches > 0:
                log["train/loss"] = self._loss_sum / self._num_batches
            if self._y_pred:
                train_logs = model.evaluate_metrics(np.concatenate(self._y_true).astype(np.float64),
                                                    np.concatenate(self._y_pred).astype(np.float64),
                                                    ["logloss", "AUC"])
                log.update({"train/" + k: v for k, v in train_logs.items()})
            self._reset()
            self.wandb.log(log, step=model._total_steps)
            return checkpoint_and_earlystop(logs, min_delta)

        model.compute_loss = compute_loss_hook
        model.checkpoint_and_earlystop = checkpoint_and_earlystop_hook
        self.run.summary["num_params"] = sum(p.numel() for p in model.parameters() if p.requires_grad)

    def log_final(self, valid_result, test_result):
        if self.run is None:
            return
        final = {"final_val/" + k: v for k, v in valid_result.items()}
        final.update({"test/" + k: v for k, v in test_result.items()})
        self.run.summary.update(final)
        self.wandb.log(final)
        self.run.finish()
