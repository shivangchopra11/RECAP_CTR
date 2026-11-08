"""Base model with local checkpointing and W&B logging."""
from __future__ import annotations

import logging
import os
import time
from pathlib import Path

import numpy as np
import torch
from tqdm.auto import tqdm

from . import progress, tracking, utils


class BaseModel(torch.nn.Module):
    def __init__(
        self,
        feature_map,
        model_id="BaseModel",
        gpu=0,
        monitor="AUC",
        save_best_only=True,
        monitor_mode="max",
        patience=2,
        every_x_epochs=1,
        embedding_regularizer=None,
        net_regularizer=None,
        reduce_lr_on_plateau=True,
        lr_scheduler=None,
        warmup_steps=4000,
        **kwargs,
    ):
        super().__init__()
        self.device = utils.set_device(gpu)
        self._monitor = utils.Monitor(kv=monitor)
        self._monitor_mode = monitor_mode
        self._patience = patience
        self._every_x_epochs = every_x_epochs
        self._save_best_only = save_best_only
        self._embedding_regularizer = embedding_regularizer
        self._net_regularizer = net_regularizer
        self._reduce_lr_on_plateau = reduce_lr_on_plateau
        self.lr_scheduler = lr_scheduler
        self.warmup_steps = warmup_steps
        self.lr_scale = 1.0
        self._feature_map = feature_map
        self.model_id = model_id
        self.model_dir = os.path.abspath(kwargs.get("model_dir", os.path.join("outputs", model_id)))
        Path(self.model_dir).mkdir(parents=True, exist_ok=True)
        self.checkpoint = os.path.join(self.model_dir, self.model_id + "_model.pt")
        self._validation_metrics = kwargs.get("metrics", ["logloss", "AUC"])
        self._verbose = kwargs.get("verbose", 1)
        self._heartbeat_file = kwargs.get("heartbeat_file")
        self._heartbeat_interval_seconds = float(kwargs.get("heartbeat_interval_seconds", 60.0))
        self.loss_fn = None

    def _heartbeat(self, stage: str, *, force: bool = False, **fields):
        progress.heartbeat(
            self._heartbeat_file,
            stage,
            interval_seconds=self._heartbeat_interval_seconds,
            force=force,
            model_id=self.model_id,
            total_batches=getattr(self, "_total_batches", 0),
            **fields,
        )

    def compile(self, optimizer, loss, lr=1e-3):
        self.to(self.device)
        try:
            self.optimizer = utils.set_optimizer(optimizer)(self.parameters(), lr=lr)
        except Exception as exc:
            raise NotImplementedError(f"optimizer={optimizer} is not supported.") from exc
        loss_name = utils.set_loss(loss)
        self.loss_fn = getattr(torch.nn.functional, loss_name)

    def _update_lr(self):
        if self.lr_scheduler == "transformer":
            if not hasattr(self, "d_model") or self.d_model is None:
                raise ValueError("Model must define d_model for transformer LR scheduling")
            step_num = self._total_batches + 1
            arg1 = step_num ** -0.5
            arg2 = step_num * (self.warmup_steps ** -1.5)
            lr = self.lr_scale * (self.d_model ** -0.5 * min(arg1, arg2))
            for group in self.optimizer.param_groups:
                group["lr"] = lr
            return lr
        return None

    def loss_with_reg(self, y_pred, y_true):
        total_loss = self.loss_fn(y_pred, y_true, reduction="mean")
        if self._embedding_regularizer or self._net_regularizer:
            emb_reg = utils.set_regularizer(self._embedding_regularizer)
            net_reg = utils.set_regularizer(self._net_regularizer)
            for name, param in self.named_parameters():
                if not param.requires_grad:
                    continue
                pairs = emb_reg if "embedding_layer" in name else net_reg
                for p_norm, weight in pairs:
                    total_loss += (weight / p_norm) * torch.norm(param, p_norm) ** p_norm
        return total_loss

    def init_weights(self, embedding_initializer=None):
        def _initialize(module):
            if isinstance(module, torch.nn.ModuleDict):
                for key, value in module.items():
                    if isinstance(value, torch.nn.Embedding):
                        spec = self._feature_map.feature_specs.get(key, {})
                        if "pretrained_emb" in spec:
                            continue
                        if embedding_initializer is not None:
                            try:
                                initializer = embedding_initializer.replace("(", "(value.weight,", 1)
                                eval(initializer, {"torch": torch, "value": value})
                            except Exception as exc:
                                raise NotImplementedError(
                                    f"embedding_initializer={embedding_initializer} is not supported"
                                ) from exc
                        else:
                            torch.nn.init.xavier_normal_(value.weight)
            if isinstance(module, torch.nn.Linear):
                torch.nn.init.xavier_normal_(module.weight)
                if module.bias is not None:
                    module.bias.data.zero_()
        self.apply(_initialize)

    def reset_parameters(self):
        """Match FuxiCTR==2.0.1 BaseModel.reset_parameters().

        Feature embeddings initialize themselves when they are constructed.  The
        official FCN code then resets only nn.Linear/nn.Conv1d modules with Xavier
        normal initialization; it does *not* initialize embeddings a second time.
        """
        def _reset(module):
            if type(module) in (torch.nn.Linear, torch.nn.Conv1d):
                torch.nn.init.xavier_normal_(module.weight)
                if module.bias is not None:
                    module.bias.data.fill_(0)

        self.apply(_reset)

    def inputs_to_device(self, inputs):
        X, y = inputs
        non_blocking = self.device.type == "cuda"
        X = X.to(self.device, non_blocking=non_blocking)
        y = y.float().view(-1, 1).to(self.device, non_blocking=non_blocking)
        self.batch_size = y.size(0)
        return X, y

    def reduce_learning_rate(self, factor=0.1, min_lr=1e-6):
        if self.lr_scheduler == "transformer":
            self.lr_scale *= factor
            return self._update_lr()
        last_lr = None
        for group in self.optimizer.param_groups:
            group["lr"] = max(group["lr"] * factor, min_lr)
            last_lr = group["lr"]
        return last_lr

    def checkpoint_and_earlystop(self, epoch, logs, min_delta=1e-6):
        monitor_value = self._monitor.get_value(logs)
        no_improve = (
            (self._monitor_mode == "min" and monitor_value > self._best_metric - min_delta)
            or (self._monitor_mode == "max" and monitor_value < self._best_metric + min_delta)
        )
        if no_improve:
            self._stopping_steps += 1
            logging.info("Monitor(%s) no improvement: %.6f", self._monitor_mode, monitor_value)
            if self._reduce_lr_on_plateau:
                lr = self.reduce_learning_rate()
                if lr is not None:
                    logging.info("Reduced learning rate to %.8g", lr)
        else:
            self._stopping_steps = 0
            self._best_metric = monitor_value
            self._best_epoch = epoch
            if self._save_best_only:
                logging.info("Saving best model: %.6f", monitor_value)
                self.save_weights(self.checkpoint)
        if self._stopping_steps * self._every_x_epochs >= self._patience:
            self._stop_training = True
            logging.info("Early stopping at epoch=%s", epoch)
        if not self._save_best_only:
            self.save_weights(self.checkpoint)

    def fit_generator(
        self,
        data_generator,
        epochs=1,
        validation_data=None,
        verbose=0,
        max_gradient_norm=10.0,
        log_every=50,
        **kwargs,
    ):
        """Train the model and run validation at the end of every epoch.

        The validation split is intentionally evaluated once per completed epoch.
        Test evaluation is *not* performed here; ``train.py`` runs the test split
        exactly once after training, using the best validation checkpoint.
        """
        self.valid_gen = validation_data
        self._max_gradient_norm = max_gradient_norm
        self._best_metric = np.inf if self._monitor_mode == "min" else -np.inf
        self._best_epoch = None
        self._stopping_steps = 0
        self._total_batches = 0
        self._batches_per_epoch = len(data_generator)
        self._stop_training = False
        self._verbose = verbose
        self.to(self.device)

        history = []
        self._heartbeat("training_start", force=True)
        logging.info(
            "Start training: %d batches/epoch on %s; validation will run after every epoch",
            self._batches_per_epoch,
            self.device,
        )

        for epoch in range(epochs):
            epoch_num = epoch + 1
            self._current_epoch = epoch_num
            self._heartbeat("epoch_start", force=True, epoch=epoch_num)
            epoch_loss = self.train_on_epoch(data_generator, epoch, log_every=log_every)
            logging.info("Epoch %d/%d train loss: %.6f", epoch_num, epochs, epoch_loss)

            epoch_record = {"epoch": epoch_num, "train_loss": epoch_loss}
            wandb_payload = {"epoch": epoch_num, "train/loss": epoch_loss}

            # Validation/eval is always run at the end of every epoch when a
            # validation generator is supplied.  These are the metrics used for
            # checkpoint selection, LR-on-plateau, and early stopping.
            if self.valid_gen is not None:
                logging.info("Epoch %d/%d: running validation evaluation...", epoch_num, epochs)
                self._heartbeat("validation_start", force=True, epoch=epoch_num)
                val_logs = self.evaluate_generator(self.valid_gen, heartbeat_stage="validation")
                self._heartbeat("validation_end", force=True, epoch=epoch_num)
                logging.info("Epoch %d/%d validation: %s", epoch_num, epochs, utils.print_to_list(val_logs))
                epoch_record.update({f"valid_{k}": v for k, v in val_logs.items()})
                wandb_payload.update({f"valid/{k}": v for k, v in val_logs.items()})
                self.checkpoint_and_earlystop(epoch_num, val_logs)
            elif not self._save_best_only:
                self.save_weights(self.checkpoint)

            # Log train + validation metrics together at the same epoch boundary.
            tracking.log(wandb_payload, step=self._total_batches)
            history.append(epoch_record)

            if self._stop_training:
                logging.info("Stopping training after epoch %d", epoch_num)
                break

        # Test evaluation should use the checkpoint selected by validation, not
        # simply the parameters from the last optimization step.
        if not os.path.exists(self.checkpoint):
            self.save_weights(self.checkpoint)
        self.load_weights(self.checkpoint)
        self._heartbeat("training_complete", force=True, best_epoch=self._best_epoch)
        logging.info(
            "Training finished; loaded best checkpoint %s (best epoch=%s, best %s=%.6f)",
            self.checkpoint,
            self._best_epoch if self._best_epoch is not None else "n/a",
            "+".join(self._monitor.kv_pairs.keys()),
            float(self._best_metric),
        )
        return history

    def train_on_epoch(self, data_generator, epoch, log_every=50):
        self.train()
        epoch_sum = torch.zeros((), device=self.device)
        win_sum = torch.zeros((), device=self.device)
        win_n = 0
        n_batches = 0
        iterator = data_generator if self._verbose == 0 else tqdm(data_generator, desc=f"epoch {epoch+1}")
        t0 = time.time()
        for batch_index, batch_data in enumerate(iterator):
            self._heartbeat(
                "training",
                epoch=epoch + 1,
                batch=batch_index + 1,
                batches_per_epoch=self._batches_per_epoch,
            )
            if self.lr_scheduler:
                self._update_lr()
            self.optimizer.zero_grad(set_to_none=True)
            return_dict = self.forward(batch_data)
            loss = return_dict["loss"]
            # Training losses must be scalars.  Some auxiliary formulations can
            # accidentally produce a singleton tensor with shape [1]; reshape
            # that case to a 0-D scalar so accumulation into epoch_sum/win_sum
            # remains shape-safe, while still rejecting true vector losses.
            if loss.numel() != 1:
                raise RuntimeError(
                    f"Model returned a non-scalar training loss with shape {tuple(loss.shape)}"
                )
            loss = loss.reshape(())
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(self.parameters(), self._max_gradient_norm)
            self.optimizer.step()
            self._total_batches += 1
            self._heartbeat(
                "training",
                epoch=epoch + 1,
                batch=batch_index + 1,
                batches_per_epoch=self._batches_per_epoch,
            )

            ld = loss.detach()
            epoch_sum += ld
            win_sum += ld
            win_n += 1
            n_batches += 1

            if (batch_index + 1) % log_every == 0:
                avg = (win_sum / max(win_n, 1)).item()
                dt = max(time.time() - t0, 1e-6)
                sps = log_every / dt
                lr = self.optimizer.param_groups[0]["lr"]
                payload = {
                    "train/loss_step": avg,
                    "train/lr": lr,
                    "perf/steps_per_sec": sps,
                    "perf/samples_per_sec": sps * self.batch_size,
                    "train/grad_norm": float(grad_norm),
                }
                for group in self.optimizer.param_groups:
                    group_name = group.get("group_name")
                    if group_name:
                        payload[f"train/lr_{group_name}"] = float(group["lr"])
                tracking.log(payload, step=self._total_batches)
                logging.info(
                    "epoch %d step %d/%d loss=%.6f lr=%.3g (%.1f steps/s)",
                    epoch + 1,
                    batch_index + 1,
                    self._batches_per_epoch,
                    avg,
                    lr,
                    sps,
                )
                win_sum.zero_()
                win_n = 0
                t0 = time.time()
        return (epoch_sum / max(n_batches, 1)).item()

    def evaluate_generator(self, data_generator, heartbeat_stage="evaluation"):
        self.eval()
        y_pred, y_true = [], []
        n_batches = len(data_generator) if hasattr(data_generator, "__len__") else None
        with torch.no_grad():
            for batch_index, batch_data in enumerate(data_generator):
                self._heartbeat(
                    heartbeat_stage,
                    batch=batch_index + 1,
                    batches_per_epoch=n_batches,
                )
                return_dict = self.forward(batch_data)
                self._heartbeat(
                    heartbeat_stage,
                    batch=batch_index + 1,
                    batches_per_epoch=n_batches,
                )
                y_pred.extend(return_dict["y_pred"].detach().cpu().numpy().reshape(-1))
                y_true.extend(batch_data[1].detach().cpu().numpy().reshape(-1))
        return utils.evaluate_metrics(
            np.asarray(y_true, np.float64),
            np.asarray(y_pred, np.float64),
            self._validation_metrics,
        )

    def to_device(self):
        self.to(self.device)

    def save_weights(self, checkpoint):
        Path(os.path.dirname(checkpoint)).mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), checkpoint)

    def load_weights(self, checkpoint, strict=True):
        try:
            state = torch.load(checkpoint, map_location=self.device, weights_only=True)
        except TypeError:  # PyTorch < 2.0 compatibility
            state = torch.load(checkpoint, map_location=self.device)
        # buffers that some checkpoints carry but that are no longer persistent (recomputed at init): drop them so both
        # older checkpoints (without them) and interim ones (with them) load strictly
        state = {k: v for k, v in state.items() if not k.endswith(("bag_primes", "capped_field_idx"))}
        return self.load_state_dict(state, strict=strict)

    def get_final_activation(self, task="binary_classification"):
        if task == "binary_classification":
            return torch.nn.Sigmoid()
        if task == "multi_classification":
            return torch.nn.Softmax(dim=-1)
        if task == "regression":
            return None
        raise NotImplementedError(f"task={task} is not supported")

    def count_parameters(self, count_embedding=True):
        total = 0
        for name, param in self.named_parameters():
            if not count_embedding and "embedding" in name:
                continue
            if param.requires_grad:
                total += param.numel()
        logging.info("Total trainable parameters: %d", total)
        return total
