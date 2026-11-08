""" Build a distillation teacher from saved run predictions (run_expid.py with save_train_pred: True):
    mean logit of the runs, written row-aligned to <data_root>/<dataset_id>/<name>.{train,valid,test}.npy

    python scripts/make_teacher.py --name teacher_fcnsp_e4l2_5seed --expids "FCNv2_Criteo_x4_10_E4L2_00*"
"""
import os
import json
import glob
import argparse
import numpy as np
from sklearn.metrics import roc_auc_score, log_loss

EPS = 1e-7


def logit(p):
    p = np.clip(p.astype(np.float64), EPS, 1 - EPS)
    return np.log(p) - np.log1p(-p)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--expids", required=True, help="expid glob of the ensemble members")
    parser.add_argument("--dataset_id", default="Criteo_x4_10")
    parser.add_argument("--data_root", default="../data/")
    parser.add_argument("--model_root", default="./checkpoints/")
    parser.add_argument("--num_models", type=int, default=None, help="expected number of members")
    parser.add_argument("--label", default=None, help="label column (default: from feature_map.json)")
    args = parser.parse_args()
    os.chdir(os.path.join(os.path.dirname(os.path.realpath(__file__)), ".."))
    if args.label is None:
        with open(os.path.join(args.data_root, args.dataset_id, "feature_map.json")) as fm:
            args.label = json.load(fm)["labels"][0]

    pred_dir = os.path.join(args.model_root, args.dataset_id)
    expids = sorted(os.path.basename(p)[:-len(".train_pred.npy")]
                    for p in glob.glob(os.path.join(pred_dir, args.expids + ".train_pred.npy")))
    if not expids or (args.num_models and len(expids) != args.num_models):
        raise SystemExit("found {} members for {}: {}".format(len(expids), args.expids, expids))
    print("members:", expids)
    out_dir = os.path.join(args.data_root, args.dataset_id)
    for split in ["train", "valid", "test"]:
        z = np.mean([logit(np.load(os.path.join(pred_dir, "{}.{}_pred.npy".format(e, split)))) for e in expids], axis=0)
        np.save(os.path.join(out_dir, "{}.{}.npy".format(args.name, split)), z.astype(np.float32))
        y = np.load(os.path.join(out_dir, split + ".npz"))[args.label]
        assert len(y) == len(z), (split, len(y), len(z))
        p = np.clip(1.0 / (1.0 + np.exp(-z)), EPS, 1 - EPS)
        print("{:5s} teacher (mean logit of {}): AUC {:.6f}  logloss {:.6f}  -> {}".format(
            split, len(expids), roc_auc_score(y, p), log_loss(y, p), os.path.join(out_dir, "{}.{}.npy".format(args.name, split))))
