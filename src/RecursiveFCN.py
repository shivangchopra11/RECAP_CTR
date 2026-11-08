# =========================================================================
# FuxiCTR adapter for the anytime recursive cross network (RRT / RECAP,
# MultiTowerRecursiveFCNv2) of CTR_ICLR/final. The model itself is the
# unmodified reference code vendored in src/recap_rrt/; this wrapper only
#   - builds the feature map it expects from the FuxiCTR feature map,
#   - feeds FuxiCTR batches to it as (X, y[, teacher]); categorical features take one column of X,
#     sequence features max_len columns, pooled by the feature's FuxiCTR encoder (e.g. MaskedSumPooling),
#   - uses its own loss (multi-budget tri-BCE + native distillation + regularizer)
#     and its optimizer (per-group learning-rate multipliers).
# Reference config keys are passed through unchanged, except that the rollout
# depth used for evaluation is `rrt_eval_steps` (in FuxiCTR `eval_steps` is the
# validation interval in batches).
# =========================================================================

import collections
import torch
from torch import nn
from fuxictr.pytorch.models import BaseModel
from fuxictr.pytorch import layers as fuxictr_layers
from .recap_rrt.recursive_models import MultiTowerRecursiveFCNv2

# FuxiCTR training-loop keys that must not reach the reference model under the same name
FUXICTR_ONLY = {"eval_steps", "early_stop_patience", "monitor", "model_root", "num_workers", "wandb",
                "data_loader", "teacher_file", "feature_cols", "label_col"}


class _ReferenceFeatureMap(object):
    """ The FuxiCTR-2.0-style feature map interface used by the reference embedding layers. """

    def __init__(self, feature_map):
        self.dataset_id = feature_map.dataset_id
        self.data_dir = getattr(feature_map, "data_dir", "")
        self.feature_specs = collections.OrderedDict()
        column = 0
        for name, spec in feature_map.features.items():
            if spec["type"] == "categorical":
                index, width = column, 1
            elif spec["type"] == "sequence":
                width = int(spec["max_len"])
                index = list(range(column, column + width))
            else:
                raise NotImplementedError("RecursiveFCN supports categorical and sequence features ({}: {})".format(
                    name, spec["type"]))
            self.feature_specs[name] = {"type": spec["type"], "source": spec.get("source", ""),
                                        "vocab_size": spec["vocab_size"], "padding_idx": spec.get("padding_idx"),
                                        "index": index}
            column += width
        self.num_fields = len(self.feature_specs)
        self.num_features = sum(spec["vocab_size"] for spec in self.feature_specs.values())

    def sum_emb_out_dim(self, embedding_dim=16):
        return self.num_fields * embedding_dim


class RecursiveFCN(BaseModel):
    native_distillation = True  # kd.py passes the teacher logits to the model's own distillation losses

    def __init__(self,
                 feature_map,
                 model_id="RecursiveFCN",
                 gpu=-1,
                 learning_rate=1e-3,
                 rrt_eval_steps=8,
                 grad_accum_steps=1,
                 **kwargs):
        super(RecursiveFCN, self).__init__(feature_map, model_id=model_id, gpu=gpu, **kwargs)
        self.grad_accum_steps = int(grad_accum_steps)
        core_kwargs = {k: v for k, v in kwargs.items() if k not in FUXICTR_ONLY}
        core_kwargs.update(eval_steps=int(rrt_eval_steps), model_dir=self.model_dir,
                           patience=kwargs.get("early_stop_patience", 2))
        self.fields = list(feature_map.features.keys())
        self.core = MultiTowerRecursiveFCNv2(_ReferenceFeatureMap(feature_map), model_id=model_id, gpu=gpu,
                                             learning_rate=learning_rate, **core_kwargs)
        # sequence features: pool the item embeddings with the feature's FuxiCTR encoder (parameter-free)
        embedding_dict = self.core.embedding_layer.embedding_layer.embedding_layer
        for name, spec in feature_map.features.items():
            if spec["type"] == "sequence":
                encoder = spec.get("feature_encoder") or "MaskedSumPooling"  # e.g. "layers.MaskedSumPooling()"
                encoder = (encoder[0] if isinstance(encoder, list) else encoder).split(".")[-1].split("(")[0]
                embedding_dict.feature_encoders[name] = getattr(fuxictr_layers, encoder)()
        self.optimizer = self.core.optimizer
        self.loss_fn = self.core.loss_fn
        self.teacher = None  # set per training batch by kd.py
        self.model_to_device()

    def compile(self, optimizer, loss, lr):
        pass  # the reference model builds its own optimizer (per-group learning-rate multipliers)

    def reset_parameters(self):
        pass  # the reference model initialises itself

    def inputs_to_X(self, inputs):
        return torch.cat([inputs[f].reshape(inputs[f].shape[0], -1) for f in self.fields], dim=1).long().to(self.device)

    def forward(self, inputs):
        X = self.inputs_to_X(inputs)
        y = self.get_labels(inputs)
        self.core._total_batches = getattr(self, "_total_steps", 0)  # cadence of the model's diagnostics
        out = self.core((X, y) if self.teacher is None else (X, y, self.teacher))
        return {"y_pred": out["y_pred"], "loss": out.get("loss")}

    def add_loss(self, return_dict, y_true):
        return return_dict["loss"]

    def compute_loss(self, return_dict, y_true):
        return return_dict["loss"]  # already includes the regularizer (and distillation when a teacher is set)

    def train_step(self, batch_data):
        """ With grad_accum_steps = k, the batch is split into k chunks whose gradients are averaged before
            one optimizer step: the update of the reference's k-GPU DDP training (per-rank batch
            batch_size // k, per-rank normalisation statistics and in-batch losses) on one GPU.
        """
        k = self.grad_accum_steps
        if k <= 1:
            return super(RecursiveFCN, self).train_step(batch_data)
        self.optimizer.zero_grad()
        teacher, total_loss = self.teacher, 0.
        for idx in torch.arange(len(self.get_labels(batch_data))).tensor_split(k):
            chunk = {key: value[idx.to(value.device)] for key, value in batch_data.items()}
            self.teacher = None if teacher is None else teacher[idx.to(teacher.device)]
            loss = self.compute_loss(self.forward(chunk), self.get_labels(chunk)) / k
            loss.backward()
            total_loss = total_loss + loss.detach()
        self.teacher = teacher
        nn.utils.clip_grad_norm_(self.parameters(), self._max_gradient_norm)
        self.optimizer.step()
        return total_loss
