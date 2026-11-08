# =========================================================================
# Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =========================================================================
""" Train and test one experiment with FuxiCTR (>= 2.3).

    python run_expid.py --expid FCNv2_Criteo_x4_10 --gpu 0
    python run_expid.py --expid FCNv2_Criteo_x4_10 --build_data   # once per dataset: raw csv -> npz
"""

import os
os.chdir(os.path.dirname(os.path.realpath(__file__)))
import sys
import csv
import gc
import argparse
import importlib
import logging
import numpy as np
import polars as pl
from datetime import datetime
from fuxictr.utils import load_config, set_logger, print_to_json
from fuxictr.features import FeatureMap
from fuxictr.pytorch.torch_utils import seed_everything
from fuxictr.pytorch.dataloaders import RankDataLoader
from fuxictr.preprocess import FeatureProcessor, build_dataset
import src as model_zoo
from wandb_logger import WandbLogger
from ema import TrainingSchedule
from kd import Distillation
from resume import Resume

RESULTS_FILE = "results/results.csv"


def get_feature_processor(params):
    # Dataset-specific preprocess functions, e.g. convert_to_bucket for Criteo
    name = params.get("feature_processor")
    if name is None:
        return FeatureProcessor(**params)
    module = importlib.import_module("fuxictr.datasets." + name)
    return module.CustomizedFeatureProcessor(**params)


def encode_features(feature_encoder, df):
    """ Same as feature_encoder.transform(), but categorical vocab lookups run vectorized in
        polars (replace_strict) instead of a Python loop per value. Returns a dict of arrays.
    """
    exprs, other_cols = [], []
    for feature, spec in feature_encoder.feature_map.features.items():
        if feature not in df.columns:
            continue
        if spec["type"] == "categorical" and spec.get("category_processor") is None:
            vocab = feature_encoder.processor_dict[feature + "::tokenizer"].vocab
            dtype = df.schema[feature]
            key_type = int if dtype.is_integer() else (float if dtype.is_float() else str)
            keys = [k for k in vocab if type(k) is key_type]
            exprs.append(pl.col(feature).replace_strict(keys, [vocab[k] for k in keys],
                                                        default=vocab["__OOV__"],
                                                        return_dtype=pl.Int64))
        else:
            other_cols.append(feature)
    arrays = {col: series.to_numpy() for col, series in df.with_columns(exprs).to_dict().items()}
    if other_cols:
        arrays.update(feature_encoder.transform({col: arrays[col] for col in other_cols}))
    return arrays


def build_data(params):
    """ Fit the feature map on raw csv and write feature_map.json + {train,valid,test}.npz
        to data_root/dataset_id/. Same fitting as fuxictr's build_dataset, but the transform
        runs in-process (vectorized) instead of on a Ray cluster (Ray fails to start on our
        Slurm nodes), and an existing feature_processor.pkl is reused, so a build resumes.
    """
    params = dict(params, data_format="csv", rebuild_dataset=True)
    feature_encoder = get_feature_processor(params)
    pickle_file = os.path.join(feature_encoder.data_dir, "feature_processor.pkl")
    raw_data = {"train": params["raw_train_data"],
                "valid": params["raw_valid_data"],
                "test": params.get("raw_test_data")}
    if os.path.exists(pickle_file):
        feature_encoder = feature_encoder.load_pickle(pickle_file)
    else:
        train_ddf = feature_encoder.preprocess(feature_encoder.read_data(raw_data["train"], **params))
        feature_encoder.fit(train_ddf, **params)
    for split, data_path in raw_data.items():
        npz_file = os.path.join(feature_encoder.data_dir, split + ".npz")
        if data_path is None or os.path.exists(npz_file):
            continue
        logging.info("Transform {} to {}".format(data_path, npz_file))
        df = feature_encoder.preprocess(feature_encoder.read_data(data_path, **params)).collect()
        if params.get("binarize_label"):  # e.g. KDD12: the label is a click count -> Click > 0
            label = params["label_col"]["name"]
            df = df.with_columns((pl.col(label) > 0).cast(pl.Float64).alias(label))
        logging.info("{}: {} rows loaded, encoding...".format(split, len(df)))
        arrays = encode_features(feature_encoder, df)
        del df
        np.savez(npz_file + ".tmp.npz", **arrays)
        os.replace(npz_file + ".tmp.npz", npz_file)
        del arrays
        gc.collect()


def save_result(params, args, valid_result, test_result):
    os.makedirs(os.path.dirname(RESULTS_FILE), exist_ok=True)
    row = {"time": datetime.now().strftime("%Y%m%d-%H%M%S"),
           "expid": params["model_id"],
           "model": params["model"],
           "dataset_id": params["dataset_id"],
           "seed": params["seed"],
           "val_AUC": valid_result.get("AUC"),
           "val_logloss": valid_result.get("logloss"),
           "test_AUC": test_result.get("AUC"),
           "test_logloss": test_result.get("logloss"),
           "config": args["config"],
           "slurm_job": os.environ.get("SLURM_JOB_ID", "")}
    is_new = not os.path.exists(RESULTS_FILE)
    with open(RESULTS_FILE, "a", newline="") as fw:
        writer = csv.DictWriter(fw, fieldnames=list(row.keys()))
        if is_new:
            writer.writeheader()
        writer.writerow(row)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./config/', help='The config directory.')
    parser.add_argument('--expid', type=str, required=True, help='The experiment id to run.')
    parser.add_argument('--gpu', type=int, default=0, help='The gpu index, -1 for cpu')
    parser.add_argument('--build_data', action='store_true',
                        help='Only build feature_map.json + npz data from the raw csv files.')
    parser.add_argument('--keep_model', action='store_true',
                        help='Keep the best checkpoint after testing (deleted by default).')
    parser.add_argument('--wandb_mode', type=str, default=None, choices=['online', 'offline', 'disabled'],
                        help='Override the wandb mode of the config.')
    parser.add_argument('--force', action='store_true',
                        help='Rerun even if the run has already completed (<expid>.done exists).')
    args = vars(parser.parse_args())

    experiment_id = args['expid']
    params = load_config(args['config'], experiment_id)
    params['gpu'] = args['gpu']
    if args['build_data']:
        params['model_id'] = params['dataset_id'] + '_build_data'
    if args['build_data']:
        set_logger(params)
    else:
        resume = Resume(params)  # continue from the last completed epoch after a restart
        if resume.is_done() and not args['force']:
            print("{} has already completed ({}); use --force to rerun.".format(experiment_id, resume.done_path))
            sys.exit(0)
        resume.set_logger(params)
    logging.info("Params: " + print_to_json(params))
    seed_everything(seed=params['seed'])

    data_dir = os.path.join(params['data_root'], params['dataset_id'])
    feature_map_json = os.path.join(data_dir, "feature_map.json")
    if args['build_data']:
        build_data(params)
        sys.exit(0)
    if params["data_format"] == "csv":
        params["train_data"], params["valid_data"], params["test_data"] = \
            build_dataset(get_feature_processor(params), **params)
    elif not os.path.exists(feature_map_json):
        raise FileNotFoundError("{} not found. Build the dataset first: python run_expid.py "
                                "--expid {} --build_data".format(feature_map_json, experiment_id))
    feature_map = FeatureMap(params['dataset_id'], data_dir)
    feature_map.load(feature_map_json, params)
    logging.info("Feature specs: " + print_to_json(feature_map.features))

    distillation = Distillation(params)
    params.update(distillation.data_loader_kwargs(params))  # training batches carry teacher logits
    model_class = getattr(model_zoo, params['model'])
    model = model_class(feature_map, **params)
    model.count_parameters() # print number of parameters used in model
    schedule = TrainingSchedule(params)
    schedule.watch(model)  # weight EMA / cosine LR, if configured
    distillation.watch(model)  # knowledge distillation, if configured
    wandb_logger = WandbLogger(params, args["config"], mode=args["wandb_mode"], run_id=resume.wandb_id)
    wandb_logger.watch(model)
    resume.watch(model, schedule, wandb_logger)  # save / restore the training state every epoch

    train_gen, valid_gen = RankDataLoader(feature_map, stage='train', **params)
    model.fit(train_gen, validation_data=valid_gen, **params)

    # predictions of the best checkpoint, for ensembling (scripts/ensemble.py)
    pred_prefix = os.path.join(model.model_dir, params["model_id"])
    logging.info('****** Validation evaluation ******')
    valid_result = model.evaluate(valid_gen)
    np.save(pred_prefix + ".valid_pred.npy", model.predict(valid_gen).astype(np.float32))
    del train_gen, valid_gen
    gc.collect()

    test_result = {}
    if params["test_data"]:
        logging.info('******** Test evaluation ********')
        test_gen = RankDataLoader(feature_map, stage='test', **params)
        test_result = model.evaluate(test_gen)
        np.save(pred_prefix + ".test_pred.npy", model.predict(test_gen).astype(np.float32))
    if params.get("save_train_pred"):
        # row-aligned predictions on the training set (e.g. as distillation teacher)
        train_eval_gen = RankDataLoader(feature_map, stage='test', **dict(params, test_data=params["train_data"]))
        np.save(pred_prefix + ".train_pred.npy", model.predict(train_eval_gen).astype(np.float32))
        del train_eval_gen

    save_result(params, args, valid_result, test_result)
    wandb_logger.log_final(valid_result, test_result)
    if not args['keep_model'] and os.path.exists(model.checkpoint):
        os.remove(model.checkpoint)
    resume.finish()
