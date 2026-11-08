""" Inference-time route aggregation for a trained RecursiveFCN checkpoint
    (the "Routes" row of CTR_ICLR/final: recap/evaluate_breadth.py --routes + results.py routes_row).

    Every tower is run under each routing override of its LoRA expert bank (trained router, top-1,
    dense, each expert alone, each leave-one-out subset, halves / even / odd) at a fixed depth, and
    the branch logits are fused by
        pair     the towers under the trained router (the model's own prediction at that depth)
        uniform  mean logit of all branches (logloss after validation-fitted affine calibration)
        stacked  logistic regression over all branch logits, fitted on validation (headline row)

    python scripts/route_fusion.py --expid FCNv2_... --depth 12 --gpu 0
    (the run must have been trained with --keep_model)
"""
import os
import sys
import csv
import json
import argparse
import logging
from datetime import datetime
import numpy as np
import torch
from sklearn.metrics import roc_auc_score

ROOT = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)
from fuxictr.utils import load_config
from fuxictr.features import FeatureMap
from fuxictr.pytorch.dataloaders import RankDataLoader
from fuxictr.pytorch.torch_utils import seed_everything
import src as model_zoo


def route_set(bank_size):
    """ Same routes as recap/evaluate_breadth.py route_set. """
    K = int(bank_size)
    routes = [("trained", None)]
    if K <= 0:
        return routes
    routes += [("top1", {"top_k": 1}), ("dense", {"top_k": K})]
    routes += [("expert{}".format(i), {"allowed": [i], "top_k": 1}) for i in range(K)]
    if K > 2:
        routes += [("drop{}".format(i), {"allowed": [j for j in range(K) if j != i]}) for i in range(K)]
    if K >= 4:
        h = K // 2
        routes += [("half_lo", {"allowed": list(range(h))}), ("half_hi", {"allowed": list(range(h, K))}),
                   ("even", {"allowed": list(range(0, K, 2))}), ("odd", {"allowed": list(range(1, K, 2))})]
    return routes


# ---- fusion helpers, as in CTR_ICLR/final results.py
def sig(t):
    return 1.0 / (1.0 + np.exp(-t))


def logloss(y, p):
    p = np.clip(p, 1e-7, 1 - 1e-7)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def fit_logistic(X, y, iters=20, ridge=1e-6):
    Xb = np.c_[X.astype(np.float64), np.ones(len(X))]; y = y.astype(np.float64); w = np.zeros(Xb.shape[1]); w[:-1] = 1.0 / X.shape[1]

    def obj(w):
        t = Xb @ w
        return float(np.sum(np.logaddexp(0.0, -t) * y + np.logaddexp(0.0, t) * (1.0 - y)) + 0.5 * ridge * w @ w)
    f = obj(w)
    for _ in range(iters):
        p = sig(Xb @ w); g = Xb.T @ (p - y) + ridge * w; W = p * (1 - p)
        H = (Xb * W[:, None]).T @ Xb + ridge * np.eye(len(w)); step = np.linalg.solve(H, g)
        # backtracking: with many near-collinear branches (few validation rows) full Newton steps can diverge;
        # when the full step decreases the objective (the usual case) this is plain Newton
        t = 1.0
        while t > 1e-6:
            f2 = obj(w - t * step)
            if np.isfinite(f2) and f2 <= f:
                break
            t *= 0.5
        else:
            break
        w, f = w - t * step, f2
        if np.abs(t * step).max() < 1e-8:
            break
    return w


def fit_affine(z, y, iters=60):
    z = z.astype(np.float64); y = y.astype(np.float64); a, b = 1.0, 0.0

    def obj(a, b):
        t = a * z + b
        return float(np.mean(np.logaddexp(0.0, -t) * y + np.logaddexp(0.0, t) * (1.0 - y)))
    f = obj(a, b)
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-(a * z + b))); r = p - y; w = p * (1.0 - p)
        g = np.array([np.mean(r * z), np.mean(r)])
        h = np.array([[np.mean(w * z * z), np.mean(w * z)], [np.mean(w * z), np.mean(w)]]) + 1e-9 * np.eye(2)
        step = np.linalg.solve(h, g); t = 1.0
        while t > 1e-6:
            a2, b2 = a - t * step[0], b - t * step[1]; f2 = obj(a2, b2)
            if np.isfinite(f2) and f2 <= f and a2 > 0:
                break
            t *= 0.5
        else:
            break
        moved = max(abs(a2 - a), abs(b2 - b)); a, b, f = a2, b2, f2
        if moved < 1e-9:
            break
    return float(a), float(b)


@torch.no_grad()
def branch_logits(model, data_gen, routes, depth):
    """ [rows, towers * routes] logits at `depth` (tower-major order) and the labels. """
    core = model.core
    core.eval()
    out, labels = [], []
    for batch in data_gen:
        X = model.inputs_to_X(batch)
        shared = core.embedding_layer(X)
        cols = []
        for k, tower in enumerate(core.towers):
            emb = core.tower_embedding(k, X, shared) if core._needs_tower_embedding() else shared
            tower.noise_override = None
            tower.sample_override = None
            for _, spec in routes:
                tower.route_override = spec
                _, aux = tower(emb, num_steps=depth, reference_steps=depth, requested_budget=depth)
                cols.append(core._mean_logit(aux["step_logits"][-1]).reshape(-1).float())
            tower.route_override = None
        out.append(torch.stack(cols, dim=1).cpu().numpy())
        labels.append(model.get_labels(batch).cpu().numpy().reshape(-1))
    return np.concatenate(out), np.concatenate(labels)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="./config/")
    parser.add_argument("--expid", required=True)
    parser.add_argument("--depth", type=int, default=12, help="rollout depth of every branch (reference: 12)")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--checkpoint", default=None, help="default: checkpoints/<dataset_id>/<expid>.model")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    params = load_config(args.config, args.expid)
    params["gpu"] = args.gpu
    params.update(teacher_logits=None, distill_mode="none")  # evaluation only: no teacher
    seed_everything(seed=params["seed"])
    data_dir = os.path.join(params["data_root"], params["dataset_id"])
    feature_map = FeatureMap(params["dataset_id"], data_dir)
    feature_map.load(os.path.join(data_dir, "feature_map.json"), params)
    model = getattr(model_zoo, params["model"])(feature_map, **params)
    checkpoint = args.checkpoint or model.checkpoint
    model.load_weights(checkpoint)
    model.eval()
    routes = route_set(params.get("lora_bank_size", 0))
    names = ["tower{}{}".format(k, "" if r == "trained" else "+" + r) for k in range(len(model.core.towers)) for r, _ in routes]
    logging.info("{}: {} towers x {} routes = {} branches at depth {}".format(checkpoint, len(model.core.towers), len(routes), len(names), args.depth))

    L, Y = {}, {}
    for split in ["valid", "test"]:
        gen = RankDataLoader(feature_map, stage="test", **dict(params, test_data=params[split + "_data"], data_loader=None))
        L[split], Y[split] = branch_logits(model, gen, routes, args.depth)
        logging.info("{}: logits {}".format(split, L[split].shape))

    pair = [i for i, n in enumerate(names) if n in ("tower0", "tower1")]
    result = {"expid": args.expid, "depth": args.depth, "branches": len(names), "names": names}
    for tag, cols in (("pair", pair), ("uniform", list(range(len(names))))):
        zv, zt = L["valid"][:, cols].mean(1), L["test"][:, cols].mean(1)
        a, b = fit_affine(zv, Y["valid"])
        result[tag] = {"K": len(cols), "valid_AUC": roc_auc_score(Y["valid"], zv), "test_AUC": roc_auc_score(Y["test"], zt),
                       "test_logloss_raw": logloss(Y["test"], sig(zt.astype(np.float64))),
                       "test_logloss_cal": logloss(Y["test"], sig(a * zt + b)), "scale": a, "bias": b}
    w = fit_logistic(L["valid"], Y["valid"])
    zv = np.c_[L["valid"], np.ones(len(L["valid"]))] @ w
    zt = np.c_[L["test"], np.ones(len(L["test"]))] @ w
    result["stacked"] = {"K": len(names), "valid_AUC": roc_auc_score(Y["valid"], zv), "test_AUC": roc_auc_score(Y["test"], zt),
                         "valid_logloss": logloss(Y["valid"], sig(zv)), "test_logloss": logloss(Y["test"], sig(zt)),
                         "weights": dict(zip(names, w[:-1].round(4).tolist())), "bias": float(w[-1])}
    result["single_branch_test_AUC"] = {n: roc_auc_score(Y["test"], L["test"][:, i]) for i, n in enumerate(names)}

    out_file = os.path.join(model.model_dir, args.expid + ".routes_d{}.json".format(args.depth))
    json.dump(result, open(out_file, "w"), indent=2)
    for tag in ["pair", "uniform", "stacked"]:
        r = result[tag]
        print("{:8s} K={:3d}  valid AUC {:.6f}  test AUC {:.6f}  test logloss {:.6f}{}".format(
            tag, r["K"], r["valid_AUC"], r["test_AUC"], r.get("test_logloss", r.get("test_logloss_cal")),
            "" if tag == "stacked" else " (calibrated; raw {:.6f})".format(r["test_logloss_raw"])))
    print("->", out_file)

    # results/results.csv row for the headline (stacked) fusion
    row = {"time": datetime.now().strftime("%Y%m%d-%H%M%S"), "expid": args.expid + "_Routes",
           "model": params["model"], "dataset_id": params["dataset_id"], "seed": params["seed"],
           "val_AUC": result["stacked"]["valid_AUC"], "val_logloss": result["stacked"]["valid_logloss"],
           "test_AUC": result["stacked"]["test_AUC"], "test_logloss": result["stacked"]["test_logloss"],
           "config": args.config, "slurm_job": os.environ.get("SLURM_JOB_ID", "")}
    with open("results/results.csv", "a", newline="") as fw:
        csv.DictWriter(fw, fieldnames=list(row.keys())).writerow(row)
