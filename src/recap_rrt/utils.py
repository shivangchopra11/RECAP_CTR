"""Utility functions for the portable CTR ranking project."""
from __future__ import annotations

import collections
from functools import partial
import glob
import json
import logging
import os
from pathlib import Path
import random
from typing import Any

import numpy as np
import sklearn.metrics
import torch
import yaml


def seed_everything(seed: int = 1029, deterministic: bool = True) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def set_device(gpu: int = -1) -> torch.device:
    if gpu >= 0 and torch.cuda.is_available():
        return torch.device(f"cuda:{gpu}")
    return torch.device("cpu")


def set_optimizer(optimizer):
    if isinstance(optimizer, str):
        optimizer_map = {"adam": "Adam", "adagrad": "Adagrad", "adamw": "AdamW"}
        name = optimizer_map.get(optimizer.lower(), optimizer)
        try:
            return getattr(torch.optim, name)
        except AttributeError as exc:
            raise NotImplementedError(f"optimizer={optimizer} is not supported.") from exc
    if callable(optimizer):
        return optimizer
    raise ValueError(f"Optimizer must be a string or callable, got {type(optimizer)}")


def set_loss(loss):
    if isinstance(loss, str):
        if loss.lower() in {"bce", "binary_crossentropy", "binary_cross_entropy"}:
            return "binary_cross_entropy"
        raise NotImplementedError(f"loss={loss} is not supported.")
    return loss


def set_regularizer(reg):
    if reg is None or reg == 0 or reg == 0.0:
        return []
    pairs = []
    if isinstance(reg, (int, float)):
        pairs.append((2, float(reg)))
    elif isinstance(reg, str):
        try:
            if reg.startswith("l1(") or reg.startswith("l2("):
                pairs.append((int(reg[1]), float(reg.rstrip(")").split("(")[-1])))
            elif reg.startswith("l1_l2"):
                l1_reg, l2_reg = reg.rstrip(")").split("(")[-1].split(",")
                pairs.extend([(1, float(l1_reg)), (2, float(l2_reg))])
            else:
                raise ValueError
        except Exception as exc:
            raise NotImplementedError(f"regularizer={reg} is not supported.") from exc
    return pairs


def set_activation(activation):
    if activation is None or (isinstance(activation, str) and activation.lower() in {"none", "identity"}):
        return torch.nn.Identity()
    if isinstance(activation, str):
        table = {
            "relu": torch.nn.ReLU,
            "sigmoid": torch.nn.Sigmoid,
            "tanh": torch.nn.Tanh,
            "swish": torch.nn.SiLU,
            "silu": torch.nn.SiLU,
            "gelu": torch.nn.GELU,
            "leakyrelu": torch.nn.LeakyReLU,
        }
        if activation.lower() in table:
            return table[activation.lower()]()
        try:
            return getattr(torch.nn, activation)()
        except AttributeError as exc:
            raise ValueError(f"Unsupported activation string: {activation}") from exc
    if callable(activation):
        return activation
    raise ValueError(f"Activation must be a string, callable, or None, got {type(activation)}")


def _yaml(path: str | os.PathLike[str]) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_config(config_dir: str, experiment_id: str) -> dict[str, Any]:
    """Load the original FuxiCTR-style model/dataset YAML layout from local disk."""
    config_dir = os.path.abspath(os.path.expanduser(config_dir))
    model_configs = [os.path.join(config_dir, "model_config.yaml")]
    if not os.path.exists(model_configs[0]):
        model_configs = sorted(glob.glob(os.path.join(config_dir, "model_config", "*.yaml")))
    if not model_configs:
        raise RuntimeError(f"config_dir={config_dir} has no model config YAML files")

    found: dict[str, Any] = {}
    for path in model_configs:
        cfg = _yaml(path)
        if "Base" in cfg:
            found["Base"] = cfg["Base"]
        if experiment_id in cfg:
            found[experiment_id] = cfg[experiment_id]
        if "Base" in found and experiment_id in found:
            break
    if experiment_id not in found:
        raise RuntimeError(f"experiment_id={experiment_id} is not present in model config")

    params: dict[str, Any] = {}
    params.update(found.get("Base", {}))
    params.update(found[experiment_id] or {})
    if "dataset_id" not in params:
        raise RuntimeError(f"experiment_id={experiment_id} does not define dataset_id")
    params["model_id"] = experiment_id

    dataset_configs = [os.path.join(config_dir, "dataset_config.yaml")]
    if not os.path.exists(dataset_configs[0]):
        dataset_configs = sorted(glob.glob(os.path.join(config_dir, "dataset_config", "*.yaml")))
    for path in dataset_configs:
        cfg = _yaml(path)
        if params["dataset_id"] in cfg:
            params.update(cfg[params["dataset_id"]] or {})
            break
    return params


def load_flat_config(path: str) -> dict[str, Any]:
    return _yaml(os.path.abspath(os.path.expanduser(path)))


def set_logger(model_dir: str, filename: str = "train.log") -> None:
    Path(model_dir).mkdir(parents=True, exist_ok=True)
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(model_dir, filename), mode="a"),
            logging.StreamHandler(),
        ],
    )


def print_to_json(data, sort_keys=True):
    new_data = {k: str(v) for k, v in data.items()}
    if sort_keys:
        new_data = collections.OrderedDict(sorted(new_data.items(), key=lambda x: x[0]))
    return json.dumps(new_data, indent=4)


def print_to_list(data):
    return " - ".join(f"{k}: {v:.6f}" for k, v in data.items())


class Monitor:
    def __init__(self, kv):
        self.kv_pairs = {kv: 1} if isinstance(kv, str) else kv

    def get_value(self, logs):
        return sum(logs.get(k, 0) * v for k, v in self.kv_pairs.items())


def evaluate_metrics(y_true, y_pred, metrics):
    result = {}
    for metric in metrics:
        if metric in {"logloss", "binary_crossentropy"}:
            result[metric] = sklearn.metrics.log_loss(y_true, y_pred)
        elif metric == "AUC":
            result[metric] = sklearn.metrics.roc_auc_score(y_true, y_pred)
        elif metric == "ACC":
            yp = np.argmax(y_pred, axis=1) if np.ndim(y_pred) > 1 else (np.asarray(y_pred) >= 0.5)
            result[metric] = sklearn.metrics.accuracy_score(y_true, yp)
        else:
            raise ValueError(f"Unknown metric: {metric}")
    logging.info("[Metrics] " + " - ".join(f"{k}: {v:.6f}" for k, v in result.items()))
    return result


def get_initializer(initializer):
    if isinstance(initializer, str):
        try:
            return eval(initializer, {"partial": partial, "torch": torch})
        except Exception as exc:
            raise ValueError(f"initializer={initializer} is not supported.") from exc
    return initializer


def parse_scalar(value: str) -> Any:
    """Parse CLI key=value override values using YAML scalar rules."""
    return yaml.safe_load(value)


def apply_overrides(params: dict[str, Any], overrides: list[str] | None) -> dict[str, Any]:
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"Override must be key=value, got: {item}")
        key, value = item.split("=", 1)
        params[key] = parse_scalar(value)
    return params
