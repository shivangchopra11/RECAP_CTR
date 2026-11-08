""" Ensemble the saved predictions of several runs (run_expid.py writes
    checkpoints/<dataset_id>/<expid>.{valid,test}_pred.npy) and report AUC / logloss.

    python scripts/ensemble.py --dataset_id Criteo_x4_10 --expids "FCNv2_Criteo_x4_10_E4L2_00*"
"""
import os
import json
import glob
import argparse
import numpy as np
from sklearn.metrics import roc_auc_score, log_loss

EPS = 1e-7


def metrics(y_true, y_prob):
    y_prob = np.clip(y_prob, EPS, 1 - EPS)
    return roc_auc_score(y_true, y_prob), log_loss(y_true, y_prob)


def logit(p):
    p = np.clip(p.astype(np.float64), EPS, 1 - EPS)
    return np.log(p) - np.log1p(-p)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_id", default="Criteo_x4_10")
    parser.add_argument("--data_root", default="../data/")
    parser.add_argument("--model_root", default="./checkpoints/")
    parser.add_argument("--expids", required=True, help="expid or glob, e.g. 'FCNv2_Criteo_x4_10_E4L2_00*'")
    parser.add_argument("--label", default=None, help="label column (default: from feature_map.json)")
    args = parser.parse_args()
    os.chdir(os.path.join(os.path.dirname(os.path.realpath(__file__)), ".."))
    if args.label is None:
        with open(os.path.join(args.data_root, args.dataset_id, "feature_map.json")) as fm:
            args.label = json.load(fm)["labels"][0]

    pred_dir = os.path.join(args.model_root, args.dataset_id)
    expids = sorted(os.path.basename(p)[:-len(".test_pred.npy")]
                    for p in glob.glob(os.path.join(pred_dir, args.expids + ".test_pred.npy")))
    assert expids, "no predictions match {}".format(args.expids)
    for split in ["valid", "test"]:
        y_true = np.load(os.path.join(args.data_root, args.dataset_id, split + ".npz"))[args.label]
        preds = [np.load(os.path.join(pred_dir, e + ".{}_pred.npy".format(split))) for e in expids]
        print("== {} ({} rows, {} models)".format(split, len(y_true), len(preds)))
        single = [metrics(y_true, p) for p in preds]
        for e, (auc, ll) in zip(expids, single):
            print("  {:45s} AUC {:.6f}  logloss {:.6f}".format(e, auc, ll))
        aucs, lls = np.array(single).T
        print("  {:45s} AUC {:.6f} ± {:.6f}  logloss {:.6f} ± {:.6f}".format(
            "single models: mean ± std", aucs.mean(), aucs.std(ddof=1) if len(aucs) > 1 else 0,
            lls.mean(), lls.std(ddof=1) if len(lls) > 1 else 0))
        auc, ll = metrics(y_true, np.mean([p.astype(np.float64) for p in preds], axis=0))
        print("  {:45s} AUC {:.6f}  logloss {:.6f}".format("ensemble: mean probability", auc, ll))
        z = np.mean([logit(p) for p in preds], axis=0)
        auc, ll = metrics(y_true, 1.0 / (1.0 + np.exp(-z)))
        print("  {:45s} AUC {:.6f}  logloss {:.6f}".format("ensemble: mean logit", auc, ll))
