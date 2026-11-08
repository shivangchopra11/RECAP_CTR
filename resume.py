""" Resume a run from its last completed epoch (after preemption, requeue or a node failure).

    After every epoch the full training state is written atomically to
    <model_root>/<dataset_id>/<expid>.resume.pt: live weights, optimizer, EMA shadow weights,
    epoch / step counters, early-stopping state, RNG states and the W&B run id (the best
    checkpoint <expid>.model is kept as usual). A restarted run restores it and continues with
    the next epoch; epochs already done are skipped inside FuxiCTR's own fit loop. When the run
    has finished, the state file is removed and <expid>.done is written, so a requeued job exits
    instead of training again (run_expid.py --force reruns it).
"""

import os
import random
import logging
import numpy as np
import torch
import fuxictr


class Resume(object):
    def __init__(self, params):
        self.model_dir = os.path.join(params.get("model_root", "./checkpoints/"), params["dataset_id"])
        self.expid = params["model_id"]
        self.path = os.path.join(self.model_dir, self.expid + ".resume.pt")
        self.done_path = os.path.join(self.model_dir, self.expid + ".done")
        self.state = None
        if os.path.exists(self.path):
            self.state = torch.load(self.path, map_location="cpu", weights_only=False)

    def is_done(self):
        return os.path.exists(self.done_path)

    @property
    def wandb_id(self):
        return self.state.get("wandb_id") if self.state else None

    def set_logger(self, params):
        """ fuxictr.utils.set_logger, but appending to the run's log when resuming. """
        os.makedirs(self.model_dir, exist_ok=True)
        log_file = os.path.join(self.model_dir, params.get("model_id", "") + ".log")
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
        logging.basicConfig(level=logging.INFO,
                            format='%(asctime)s P%(process)d %(levelname)s %(message)s',
                            handlers=[logging.FileHandler(log_file, mode='a' if self.state else 'w'),
                                      logging.StreamHandler()])
        logging.info("FuxiCTR version: " + fuxictr.__version__)
        if self.state:
            logging.info("Resuming from {} (epochs done: {}, steps: {})".format(
                self.path, self.state["next_epoch"], self.state["total_steps"]))

    def watch(self, model, schedule=None, wandb_logger=None):
        """ Hook into FuxiCTR's train_epoch: restore once, skip finished epochs, save after each epoch. """
        train_epoch = model.train_epoch
        restored = [self.state is None]

        def train_epoch_hook(data_generator):
            if not restored[0]:
                self._restore(model, schedule)
                restored[0] = True
            if self.state is not None and (model._epoch_index < self.state["next_epoch"] or model._stop_training):
                return  # completed before the restart
            train_epoch(data_generator)
            self._save(model, schedule, wandb_logger)

        model.train_epoch = train_epoch_hook

    def _save(self, model, schedule, wandb_logger):
        state = {"model": model.state_dict(),
                 "optimizer": model.optimizer.state_dict(),
                 "ema_shadow": [t.detach().cpu() for t in schedule.shadow] if (schedule and schedule.shadow is not None) else None,
                 "next_epoch": model._epoch_index + 1,
                 "total_steps": model._total_steps,
                 "best_metric": model._best_metric,
                 "stopping_steps": model._stopping_steps,
                 "stop_training": model._stop_training,
                 "rng": {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(),
                         "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None},
                 "wandb_id": wandb_logger.run.id if (wandb_logger and wandb_logger.run) else None}
        tmp = self.path + ".tmp"
        torch.save(state, tmp)
        os.replace(tmp, self.path)
        self.state = None  # this process is past the restore point

    def _restore(self, model, schedule):
        s = self.state
        model.load_state_dict(s["model"])
        model.optimizer.load_state_dict(s["optimizer"])
        if s["ema_shadow"] is not None and schedule is not None and schedule.shadow is not None:
            for t, saved in zip(schedule.shadow, s["ema_shadow"]):
                t.copy_(saved.to(t.device))
        model._total_steps = s["total_steps"]
        model._best_metric = s["best_metric"]
        model._stopping_steps = s["stopping_steps"]
        model._stop_training = s["stop_training"]
        random.setstate(s["rng"]["python"])
        np.random.set_state(s["rng"]["numpy"])
        torch.set_rng_state(s["rng"]["torch"])
        if s["rng"]["cuda"] is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(s["rng"]["cuda"])
        logging.info("Restored training state: {} epochs, {} steps, best monitor {:.6f}, stopping steps {}".format(
            s["next_epoch"], s["total_steps"], s["best_metric"], s["stopping_steps"]))

    def finish(self):
        if os.path.exists(self.path):
            os.remove(self.path)
        open(self.done_path, "w").close()
