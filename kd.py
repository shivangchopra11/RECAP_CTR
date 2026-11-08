""" Knowledge distillation from a precomputed teacher (same objectives as MultiTowerFCNv2 in
    CTR_ICLR/final src/ranking_ctr/recursive_models.py).

    The teacher is a row-aligned array of logits for the training split,
    <data_root>/<dataset_id>/<teacher_logits>.train.npy (scripts/make_teacher.py: mean logit of an
    ensemble). Each training batch carries its rows' teacher logits t; with f the model's fused logit:
        fusion:       KD = BCE(f, sigmoid(t))
        fusion_rank:  KD = BCE(f, sigmoid(t)) + BCE((f_a - f_b) / T, sigmoid((t_a - t_b) / T))
                      over random in-batch pairs (a, b)
        loss = model loss (tri-BCE on labels) + distill_weight * KD
    Config keys (model_config.yaml, per expid):
        teacher_logits: <name>       # file prefix of the teacher
        distill_mode: fusion         # fusion | fusion_rank
        distill_weight: 1.0
        distill_rank_temp: 1.0
    Validation and test do not use the teacher.
"""

import os
import logging
import numpy as np
import torch
import torch.nn.functional as F
from fuxictr.pytorch.dataloaders.npz_dataloader import NpzDataLoader, NpzDataset, BatchCollator

TEACHER_KEY = "__teacher_logit__"


def teacher_path(params, split="train"):
    return os.path.join(params["data_root"], params["dataset_id"],
                        "{}.{}.npy".format(params["teacher_logits"], split))


class TeacherNpzDataLoader(NpzDataLoader):
    """ NpzDataLoader whose training split also yields the teacher logit of every row. """

    def __init__(self, feature_map, data_path, batch_size=32, shuffle=False, num_workers=1,
                 split="train", teacher_file=None, **kwargs):
        if split != "train" or teacher_file is None:
            super(TeacherNpzDataLoader, self).__init__(feature_map, data_path, batch_size=batch_size,
                                                       shuffle=shuffle, num_workers=num_workers)
            return
        if not data_path.endswith(".npz"):
            data_path += ".npz"
        self.dataset = NpzDataset(feature_map, data_path)
        teacher = np.load(teacher_file).astype(np.float64).reshape(-1, 1)
        if len(teacher) != len(self.dataset):
            raise ValueError("teacher {} has {} rows, {} has {}".format(
                teacher_file, len(teacher), data_path, len(self.dataset)))
        self.dataset.darray = np.hstack([self.dataset.darray, teacher])  # teacher = last column
        logging.info("Teacher logits: {} ({} rows)".format(teacher_file, len(teacher)))
        torch.utils.data.DataLoader.__init__(self, dataset=self.dataset, batch_size=batch_size,
                                             shuffle=shuffle, num_workers=num_workers,
                                             collate_fn=TeacherBatchCollator(feature_map))
        self.num_samples = len(self.dataset)
        self.num_blocks = 1
        self.num_batches = int(np.ceil(self.num_samples * 1.0 / self.batch_size))


class TeacherBatchCollator(BatchCollator):
    def __call__(self, batch):
        batch_dict = super(TeacherBatchCollator, self).__call__(batch)
        batch_dict[TEACHER_KEY] = torch.as_tensor(np.stack(batch)[:, -1])
        return batch_dict


class Distillation(object):
    def __init__(self, params):
        self.teacher_logits = params.get("teacher_logits")
        self.mode = str(params.get("distill_mode") or "fusion")
        self.weight = float(params.get("distill_weight", 1.0))
        self.rank_temp = float(params.get("distill_rank_temp", 1.0))
        if self.teacher_logits and self.mode not in ("fusion", "fusion_rank"):
            raise ValueError("distill_mode must be fusion or fusion_rank")
        self.teacher = None

    def enabled(self):
        return bool(self.teacher_logits) and self.weight > 0

    def data_loader_kwargs(self, params):
        """ kwargs for RankDataLoader: training batches carry the teacher logits. """
        if not self.enabled():
            return {}
        return {"data_loader": TeacherNpzDataLoader, "teacher_file": teacher_path(params, "train")}

    def watch(self, model):
        """ Hook into the FuxiCTR training loop of `model` (its forward must return "logit"). """
        if not self.enabled():
            return
        # models with their own distillation losses (e.g. RecursiveFCN) only receive the teacher logits
        native = bool(getattr(model, "native_distillation", False))
        logging.info("Distillation: teacher={}, mode={}, weight={}, rank_temp={}{}".format(
            self.teacher_logits, self.mode, self.weight, self.rank_temp, " (model's native losses)" if native else ""))
        train_step, compute_loss = model.train_step, model.compute_loss

        def train_step_hook(batch_data):
            teacher = batch_data.pop(TEACHER_KEY).float().to(model.device)
            if native:
                model.teacher = teacher
            else:
                self.teacher = teacher
            try:
                return train_step(batch_data)
            finally:
                model.teacher = None
                self.teacher = None

        if native:
            model.train_step = train_step_hook
            return

        def compute_loss_hook(return_dict, y_true):
            loss = compute_loss(return_dict, y_true)
            if self.teacher is not None:
                loss = loss + self.weight * self.kd_loss(return_dict["logit"].reshape(-1), self.teacher.reshape(-1))
            return loss

        model.train_step = train_step_hook
        model.compute_loss = compute_loss_hook

    def kd_loss(self, f, t):
        loss = F.binary_cross_entropy_with_logits(f, torch.sigmoid(t))
        if self.mode == "fusion_rank":
            half = f.numel() // 2
            perm = torch.randperm(f.numel(), device=f.device)
            a, b = perm[:half], perm[half:2 * half]
            loss = loss + F.binary_cross_entropy_with_logits((f[a] - f[b]) / self.rank_temp,
                                                             torch.sigmoid((t[a] - t[b]) / self.rank_temp))
        return loss
