#%%
!pip install lightgbm

import glob
import logging
import os
import shutil
import sys
from pathlib import Path
from pprint import pprint
from typing import Dict, List, Optional, Tuple

import hydra
import matplotlib.pyplot as plt
import numpy as np
import openpack_toolkit as optk
import openpack_torch as optorch
import pandas as pd
import pytorch_lightning as pl
import seaborn as sns
import torch
import xgboost as xgb
from lightgbm import early_stopping, log_evaluation
from openpack_toolkit import OPENPACK_OPERATIONS
from openpack_toolkit.codalab.operation_segmentation import (
    construct_submission_dict,
    eval_operation_segmentation_wrapper,
    make_submission_zipfile,
)
from optuna.integration import lightgbm as lgb
from scipy.special import softmax
from sklearn import datasets
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_sample_weight
from tqdm import tqdm
from xgboost import XGBClassifier, plot_importance

from utils import splits_5_fold_new
from utils.datamodule import OpenPackAllDataModule
from utils.lightning_module import TransformerPL
from utils_depth.datamodule import OpenPackAllDataModule as OpenPackAllDataModuleDepth
from utils_depth.lightning_module import TransformerPL as TransformerPLDepth

#%%


_ = optk.utils.notebook.setup_root_logger()
logger = logging.getLogger(__name__)

logger.debug("debug")
logger.info("info")
logger.warning("warning")
optorch.configs.register_configs()


def delete_overlap_val(y, te, unixtime):  # TODO 1000以外のtime_step_width対応
    b, c, t = y.shape
    unixtime = unixtime.ravel()
    te = te.ravel()
    y = y.transpose(0, 2, 1).reshape(b * t, c)
    delta = unixtime[1:] - unixtime[:-1]
    idx_list = np.where(delta != 1000)[0]
    n_del = 0
    while len(idx_list) > 0:
        idx = idx_list[0]
        y = np.delete(y, slice(idx, idx + 2), 0)
        unixtime = np.delete(unixtime, slice(idx, idx + 2), 0)
        te = np.delete(te, slice(idx, idx + 2), 0)

        delta = unixtime[1:] - unixtime[:-1]
        idx_list = np.where(delta != 1000)[0]
        n_del += 1

    unixtime = np.concatenate([unixtime, np.repeat(unixtime[-1], n_del * 2)])
    unixtime = unixtime.reshape(b, t)
    te = np.concatenate([te, np.repeat(te[-1], n_del * 2)])
    te = te.reshape(b, t)
    y = np.concatenate([y, np.repeat([y[-1]], n_del * 2, 0)])
    y = y.reshape(b, t, c).transpose(0, 2, 1)
    return y, te, unixtime


def delete_overlap(y, unixtime):
    b, c, t = y.shape
    unixtime = unixtime.ravel()
    y = y.transpose(0, 2, 1).reshape(b * t, c)
    delta = unixtime[1:] - unixtime[:-1]
    idx_list = np.where(delta != 1000)[0]
    n_del = 0
    while len(idx_list) > 0:
        idx = idx_list[0]
        y = np.delete(y, slice(idx, idx + 2), 0)
        unixtime = np.delete(unixtime, slice(idx, idx + 2), 0)

        delta = unixtime[1:] - unixtime[:-1]
        idx_list = np.where(delta != 1000)[0]
        n_del += 1

    unixtime = np.concatenate([unixtime, np.repeat(unixtime[-1], n_del * 2)])
    unixtime = unixtime.reshape(b, t)
    y = np.concatenate([y, np.repeat([y[-1]], n_del * 2, 0)])
    y = y.reshape(b, t, c).transpose(0, 2, 1)
    return y, unixtime


def average_slide_results(slide_results):
    y_list = [r["y"] for r in slide_results]
    unixtime_list = [r["unixtime"] for r in slide_results]

    B, C, T = y_list[0].shape

    unixtime_for_use = unixtime_list[0].ravel()
    y_stack = [[] for i in range(len(unixtime_for_use))]

    unixtime_for_use = unixtime_list[0].ravel()

    for y, unixtime in zip(y_list, unixtime_list):
        y = y.transpose(0, 2, 1)
        for _y, _unixtime in zip(y, unixtime):
            for __y, __unixtime in zip(_y, _unixtime):
                if __unixtime in unixtime_for_use:
                    ind = (unixtime_for_use == __unixtime).argmax()
                    y_stack[ind].append(__y)

    y_mean = [None] * len(unixtime_for_use)
    for i in range(len(unixtime_for_use)):
        if len(y_stack[i]) == 0:  # unixtime_for_use == __unixtimeが複数ある場合
            y_mean[i] = y_mean[i - 1]
        else:
            y_mean[i] = softmax(np.stack(y_stack[i]), 1).mean(0)
    y_mean = np.array(y_mean).reshape(B, T, C).transpose(0, 2, 1)

    return y_mean


#%%

issue_list = [
    # "transformer-seed0-d6-h16-np50-timestep1000-bs32-lr0.0001-labelsm0-mixup0.8_0-shuffle0.2_0-w_kappa_loss0 imu300_keypoint400-ht30_printer30",
    # "transformer-seed1-d6-h16-np50-timestep1000-bs32-lr0.0001-labelsm0-mixup0.8_0-shuffle0.2_0-w_kappa_loss1 imu300_keypoint400-ht30_printer30",
    # "transformer-seed2-d6-h16-np50-timestep1000-bs32-lr0.0001-labelsm0-mixup0.8_0.5-shuffle0.2_0.8-w_kappa_loss0 imu300_keypoint400-ht30_printer30",
    # "transformer-seed10-d6-h16-np50-peTrue-timestep1000-bs32-lr0.0001-labelsm0-mixup0.8_0-shuffle0.2_0-w_kappa_loss0-imu300_keypoint400-ht30_printer0",
    # "transformer-seed11-d6-h16-np50-peTrue-timestep1000-bs32-lr0.0001-labelsm0-mixup0.8_0-shuffle0.2_0-w_kappa_loss0-imu300_keypoint400-ht0_printer30",
    # "transformer-seed12-d6-h16-np50-peTrue-timestep1000-bs32-lr0.0001-labelsm0-mixup0.8_0-shuffle0.2_0-w_kappa_loss0-imu300_keypoint400-ht0_printer0",
    # "simple_transformer-seed3-d6-h16-np50-timestep1000-bs32-lr0.0001-labelsm0.1-mixup0.8_0-shuffle0.2_0-w_kappa_loss0-imu300_keypoint400-ht30_printer30",
    # "simple_transformer-seed16-d6-h16-np50-peTrue-timestep1000-bs32-lr0.0001-labelsm0.1-mixup0.8_0-shuffle0.2_0-w_kappa_loss1-imu300_keypoint400-ht30_printer30",
    # "conv_transformer-seed8-d6-h16-np50-peTrue-timestep1000-bs32-lr0.0001-labelsm0-mixup0.8_0-shuffle0.2_0-w_kappa_loss0-imu300_keypoint400-ht30_printer30",
    # "conv_transformer-seed14-d6-h16-np50-peTrue-timestep1000-bs32-lr0.0001-labelsm0-mixup0.8_0-shuffle0.2_0-w_kappa_loss1-imu300_keypoint400-ht30_printer30",
    # "b2tconv_transformer-seed4-d6-h16-np50-timestep1000-bs32-lr0.0001-labelsm0-mixup0.8_0-shuffle0.2_0-w_kappa_loss0-imu300_keypoint400-ht30_printer30",
    # "b2tconv_transformer-seed13-d6-h16-np50-peTrue-timestep1000-bs32-lr0.0001-labelsm0-mixup0.8_0-shuffle0.2_0-w_kappa_loss1-imu300_keypoint400-ht30_printer30",
    # "transformer_plusLSTM-seed6-d6-h16-np50-peTrue-timestep1000-bs32-lr0.0001-labelsm0-mixup0.8_0-shuffle0.2_0-w_kappa_loss0-imu300_keypoint400-ht30_printer30",
    # "transformer_plusLSTM-seed7-d6-h16-np50-peFalse-timestep1000-bs32-lr0.0001-labelsm0-mixup0.8_0-shuffle0.2_0-w_kappa_loss0-imu300_keypoint400-ht30_printer30",
    # "transformer_plusLSTM-seed9-d6-h16-np50-peFalse-timestep1000-bs32-lr0.0001-labelsm0-mixup0.8_0-shuffle0.2_0-w_kappa_loss1-imu300_keypoint400-ht30_printer30",
    # "twotowertwostep_transformer-seed5-d6-h16-np50-peTrue-timestep1000-bs32-lr0.0001-labelsm0-mixup0.8_0-shuffle0.2_0-w_kappa_loss0-imu300_keypoint400-ht30_printer30",
    # "twotowertwostep_transformer-seed15-d6-h16-np50-peTrue-timestep1000-bs32-lr0.0001-labelsm0-mixup0.8_0-shuffle0.2_0-w_kappa_loss1-imu300_keypoint400-ht30_printer30",
    # "transformer-seed17-d6-h16-np50-peTrue-timestep1000-bs32-lr0.0001-labelsm0-mixup0.8_0.5-shuffle0.2_0-w_kappa_loss0-imu300_keypoint400-ht30_printer30",
    # "transformer-seed18-d6-h16-np50-peTrue-timestep1000-bs32-lr0.0001-labelsm0-mixup0.8_0-shuffle0.2_0.8-w_kappa_loss0-imu300_keypoint400-ht30_printer30",
    # "simple_transformer-seed19-d6-h16-np50-peTrue-timestep1000-bs32-lr0.0001-labelsm0.1-mixup0.8_0.5-shuffle0.2_0.8-w_kappa_loss0-imu300_keypoint400-ht30_printer30",
    # "conv_transformer-seed20-d6-h16-np50-peTrue-timestep1000-bs32-lr0.0001-labelsm0-mixup0.8_0.5-shuffle0.2_0.8-w_kappa_loss0-imu300_keypoint400-ht30_printer30",
    # "b2tconv_transformer-seed21-d6-h16-np50-peTrue-timestep1000-bs32-lr0.0001-labelsm0-mixup0.8_0.5-shuffle0.2_0.8-w_kappa_loss0-imu300_keypoint400-ht30_printer30",
    # "twotowertwostep_transformer-seed22-d6-h16-np50-peTrue-timestep1000-bs32-lr0.0001-labelsm0-mixup0.8_0.5-shuffle0.2_0.8-w_kappa_loss0-imu300_keypoint400-ht30_printer30",
    # "transformer_plusLSTM-seed23-d6-h16-np50-peFalse-timestep1000-bs32-lr0.0001-labelsm0-mixup0.8_0.5-shuffle0.2_0.8-w_kappa_loss0-imu300_keypoint400-ht30_printer30",
    # "transformer-seed24-d6-h16-np50-peTrue-timestep1000-bs32-lr0.0001-labelsm0-mixup0.8_0.5-shuffle0.2_0-w_kappa_loss0-imu300_keypoint400-ht30_printer30",
    # "simple_transformer-seed25-d6-h16-np50-peTrue-timestep1000-bs32-lr0.0001-labelsm0.1-mixup0.8_0.5-shuffle0.2_0-w_kappa_loss0-imu300_keypoint400-ht30_printer30",
    # "b2tconv_transformer-seed26-d6-h16-np50-peTrue-timestep1000-bs32-lr0.0001-labelsm0-mixup0.8_0.5-shuffle0.2_0-w_kappa_loss0-imu300_keypoint400-ht30_printer30",
    # "conv_transformer-seed27-d6-h16-np50-peTrue-timestep1000-bs32-lr0.0001-labelsm0-mixup0.8_0.5-shuffle0.2_0-w_kappa_loss0-imu300_keypoint400-ht30_printer30",
    # "twotowertwostep_transformer-seed28-d6-h16-np50-peTrue-timestep1000-bs32-lr0.0001-labelsm0-mixup0.8_0.5-shuffle0.2_0-w_kappa_loss0-imu300_keypoint400-ht30_printer30",
    # "transformer_plusLSTM-seed29-d6-h16-np50-peFalse-timestep1000-bs32-lr0.0001-labelsm0-mixup0.8_0.5-shuffle0.2_0-w_kappa_loss0-imu300_keypoint400-ht30_printer30",
    "transformer_avec-seed30-d6-h20-np50-peTrue-timestep1000-bs32-lr0.0001-labelsm0-mixup0.8_0-shuffle0.2_0-w_kappa_loss0-imu300_keypoint300-ht30_printer30",
    "transformer_avec-seed32-d6-h20-np50-peTrue-timestep1000-bs32-lr0.0001-labelsm0-mixup0.8_0-shuffle0.2_0-w_kappa_loss1-imu300_keypoint300-ht30_printer30",
    "transformer_avec-seed34-d6-h20-np50-peTrue-timestep1000-bs32-lr0.0001-labelsm0-mixup0.8_0.5-shuffle0.2_0.8-w_kappa_loss0-imu300_keypoint300-ht30_printer30",
    "transformer_avec-seed37-d6-h20-np50-peTrue-timestep1000-bs32-lr0.0001-labelsm0-mixup0.8_0.5-shuffle0.2_0-w_kappa_loss0-imu300_keypoint300-ht30_printer30",
    "conformer_avec-seed31-d6-h20-np50-peTrue-timestep1000-bs32-lr0.0001-labelsm0-mixup0.8_0-shuffle0.2_0-w_kappa_loss0-imu300_keypoint300-ht30_printer30",
    "conformer_avec-seed33-d6-h20-np50-peTrue-timestep1000-bs32-lr0.0001-labelsm0-mixup0.8_0-shuffle0.2_0-w_kappa_loss1-imu300_keypoint300-ht30_printer30",
    "conformer_avec-seed38-d6-h20-np50-peTrue-timestep1000-bs32-lr0.0001-labelsm0-mixup0.8_0.5-shuffle0.2_0-w_kappa_loss0-imu300_keypoint300-ht30_printer30",
    "transformer_avec_plusLSTM-seed41-d6-h20-np50-peFalse-timestep1000-bs32-lr0.0001-labelsm0-mixup0.8_0-shuffle0.2_0-w_kappa_loss0-imu300_keypoint300-ht30_printer30",
    "transformer_avec_plusLSTM-seed47-d6-h20-np50-peFalse-timestep1000-bs32-lr0.0001-labelsm0-mixup0.8_0.5-shuffle0.2_0.8-w_kappa_loss1-imu300_keypoint300-ht30_printer30",
    "conformer_avec_plusLSTM-seed40-d6-h20-np50-peFalse-timestep1000-bs32-lr0.0001-labelsm0-mixup0.8_0-shuffle0.2_0-w_kappa_loss0-imu300_keypoint300-ht30_printer30",
    "conformer_avec_plusLSTM-seed46-d6-h20-np50-peFalse-timestep1000-bs32-lr0.0001-labelsm0-mixup0.8_0.5-shuffle0.2_0.8-w_kappa_loss1-imu300_keypoint300-ht30_printer30",
]
config_dir_list = [
    # "/workspace/logs/all/transformer/",
    # "/workspace/logs/all/transformer/",
    # "/workspace/logs/all/transformer/",
    # "/workspace/logs/all/transformer/",
    # "/workspace/logs/all/transformer/",
    # "/workspace/logs/all/transformer/",
    # "/workspace/logs/all/simple_transformer/",
    # "/workspace/logs/all/simple_transformer/",
    # "/workspace/logs/all/conv_transformer/",
    # "/workspace/logs/all/conv_transformer/",
    # "/workspace/logs/all/b2tconv_transformer/",
    # "/workspace/logs/all/b2tconv_transformer/",
    # "/workspace/logs/all/transformer_plusLSTM/",
    # "/workspace/logs/all/transformer_plusLSTM/",
    # "/workspace/logs/all/transformer_plusLSTM/",
    # "/workspace/logs/all/twotowertwostep_transformer/",
    # "/workspace/logs/all/twotowertwostep_transformer/",
    # "/workspace/logs/all/transformer/",
    # "/workspace/logs/all/transformer/",
    # "/workspace/logs/all/simple_transformer/",
    # "/workspace/logs/all/conv_transformer/",
    # "/workspace/logs/all/b2tconv_transformer/",
    # "/workspace/logs/all/twotowertwostep_transformer/",
    # "/workspace/logs/all/transformer_plusLSTM/",
    # "/workspace/logs/all/transformer/",
    # "/workspace/logs/all/simple_transformer/",
    # "/workspace/logs/all/b2tconv_transformer/",
    # "/workspace/logs/all/conv_transformer/",
    # "/workspace/logs/all/twotowertwostep_transformer/",
    # "/workspace/logs/all/transformer_plusLSTM/",
    "/workspace/logs/all/transformer_avec/",
    "/workspace/logs/all/transformer_avec/",
    "/workspace/logs/all/transformer_avec/",
    "/workspace/logs/all/transformer_avec/",
    "/workspace/logs/all/conformer_avec/",
    "/workspace/logs/all/conformer_avec/",
    "/workspace/logs/all/conformer_avec/",
    "/workspace/logs/all/transformer_avec_plusLSTM/",
    "/workspace/logs/all/transformer_avec_plusLSTM/",
    "/workspace/logs/all/conformer_avec_plusLSTM/",
    "/workspace/logs/all/conformer_avec_plusLSTM/",
]
datamodule_class_list = [OpenPackAllDataModule] * len(issue_list)
plmodel_class_list = [
    TransformerPL,
] * len(issue_list)

# issue_list = [
#     # 単体submission0.88 "transformer_plusLSTM-d6-h16-np150-timestep1000-lr_scheori_warmup-bs32-lr0.0001-gradclip1.0-labelsm0-mixup0.8_0-shuffle0.2_0-imu400_keypoint500-ht12_printer12-",
#     "transformer-d6-h16-np50-timestep1000-lr_scheori_warmup-bs32-lr0.0001-gradclip1.0-labelsm0-mixup0.8_0-shuffle0.2_0.5-imu400_keypoint500-ht12_printer12-",
#     "twotowertwostep_transformer-d3-h16-np50-timestep1000-lr_scheori_warmup-bs32-lr0.0001-gradclip1.0-labelsm0-mixup0.8_0-shuffle0.2_0-imu300_keypoint300-ht12_printer12-",
#     "transformer_plusLSTM-d6-h16-np50-timestep1000-lr_scheori_warmup-bs32-lr0.0001-gradclip1.0-labelsm0-mixup0.8_0-shuffle0.2_0-imu400_keypoint500-ht12_printer12-",
#     # "useDepth-transformer-d6-h16-np50-timestep1000-lr_scheori_warmup-bs32-lr0.0001-gradclip1.0-labelsm0-mixup0.8_0-shuffle0.2_0-imu300_keypoint300-ht12_printer12-ki_depth0-rs_depth200",
#     "useDepth-transformer-d6-h16-np50-timestep1000-lr_scheori_warmup-bs32-lr0.0001-gradclip1.0-labelsm0-mixup0.8_0-shuffle0.2_0-imu300_keypoint300-ht12_printer12-ki_depth200-rs_depth0",
# ]
# config_dir_list = [
#     # "/workspace/logs/all/transformer_plusLSTM/",
#     "/workspace/logs/all/transformer/",
#     "/workspace/logs/all/twotowertwostep_transformer/",
#     "/workspace/logs/all/transformer_plusLSTM/",
#     # "/workspace/logs/all/transformer/",
#     "/workspace/logs/all/transformer/",
# ]
# datamodule_class_list = [
#     OpenPackAllDataModule,
#     OpenPackAllDataModule,
#     OpenPackAllDataModule,
#     # OpenPackAllDataModuleDepth,
#     OpenPackAllDataModuleDepth,
# ]
# plmodel_class_list = [
#     TransformerPL,
#     TransformerPL,
#     TransformerPL,
#     # TransformerPLDepth,
#     TransformerPLDepth,
# ]

average_slide = True

stack_results_val = []
stack_results_test = []

nums_folds = 5

for issue, config_base, datamodule_class, plmodel_class in zip(
    issue_list, config_dir_list, datamodule_class_list, plmodel_class_list
):
    config_dir = os.path.join(config_base, issue, ".hydra")
    with hydra.initialize_config_dir(version_base=None, config_dir=config_dir):
        cfg = hydra.compose(
            config_name="config.yaml",
            # config_name="unet-tutorial2.yaml",
        )

    cfg.dataset.annotation.activity_sets = dict()  # Remove this attribute just for the simpler visualization.
    cfg.dataset.split = optk.configs.datasets.splits.OPENPACK_CHALLENGE_2022_SPLIT  # DEBUG_SPLIT
    # cfg.dataset.split = optk.configs.datasets.splits.DEBUG_SPLIT
    optorch.utils.reset_seed(seed=0)

    save_results_name = f"stacking_results_average_slide{average_slide}.npy"
    save_results_path = os.path.join(cfg.path.logdir.rootdir, save_results_name)
    if os.path.exists(save_results_path):
        results_val, results_test = np.load(save_results_path, allow_pickle=True)
        stack_results_val.append(results_val)
        stack_results_test.append(results_test)
        continue

    device = torch.device("cuda")
    logdir = Path(cfg.path.logdir.rootdir)
    logger.debug(f"logdir = {logdir}")
    num_epoch = cfg.train.debug.epochs if cfg.debug else cfg.train.epochs
    # num_epoch = 20 # NOTE: Set epochs manually for debugging

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=[0],
        max_epochs=num_epoch,
        logger=False,  # disable logging module
        default_root_dir=logdir,
        enable_progress_bar=False,  # disable progress bar
        enable_checkpointing=True,
    )
    logger.debug(f"logdir = {logdir}")

    # パッチずらしあり

    results_val = []
    results_test = []

    # for k in range(nums_folds):
    for k in range(nums_folds):
        cfg.dataset.split = getattr(splits_5_fold_new, f"OPENPACK_CHALLENGE_{k+1}_FOLD_SPLIT")
        datamodule = datamodule_class(cfg)
        datamodule.set_fold(k)
        cfg.mode = "test"
        datamodule.setup("test")
        dataloaders = datamodule.test_dataloader()
        split = cfg.dataset.split.test

        outputs = dict()
        chk_dir = os.path.join(cfg.path.logdir.rootdir, f"checkpoints_k{k}", "*")
        chk_path = glob.glob(chk_dir)[0]
        plmodel = plmodel_class.load_from_checkpoint(chk_path, cfg=cfg)
        plmodel.to(dtype=torch.float, device=device)
        plmodel.eval()
        plmodel.set_fold(k)
        for i, dataloader in enumerate(dataloaders):
            user, session = split[i]
            if len(dataloader) == 0:
                outputs[f"{user}-{session}"] = {
                    "t": None,
                    "y": None,
                    "unixtime": None,
                }
                continue
            logger.info(f"test on {user}-{session}")

            if average_slide:
                slide_results = []
                for n in tqdm(range(cfg.model.num_patches)):
                    dataloader.dataset.set_test_start_time(n)

                    with torch.inference_mode():
                        trainer.test(plmodel, dataloader)

                    # save model outputs
                    # pred_dir = Path(cfg.path.logdir.predict.format(user=user, session=session))
                    # pred_dir.mkdir(parents=True, exist_ok=True)

                    # for key, arr in plmodel.test_results.items():
                    #     path = Path(pred_dir, f"{key}.npy")
                    #     np.save(path, arr)
                    #     logger.info(f"save {key}[shape={arr.shape}] to {path}")

                    t = plmodel.test_results.get("t")
                    y = plmodel.test_results.get("y")
                    unixtime = plmodel.test_results.get("unixtime")

                    y, t, unixtime = delete_overlap_val(y, t, unixtime)
                    slide_results.append({"y": y, "t": t, "unixtime": unixtime})

                y = average_slide_results(slide_results)
                t = slide_results[0]["t"]
                unixtime = slide_results[0]["unixtime"]
            else:
                dataloader.dataset.set_test_start_time(0)

                with torch.inference_mode():
                    trainer.test(plmodel, dataloader)

                # save model outputs
                # pred_dir = Path(cfg.path.logdir.predict.format(user=user, session=session))
                # pred_dir.mkdir(parents=True, exist_ok=True)

                # for key, arr in plmodel.test_results.items():
                #     path = Path(pred_dir, f"{key}.npy")
                #     np.save(path, arr)
                #     logger.info(f"save {key}[shape={arr.shape}] to {path}")

                t = plmodel.test_results.get("t")
                y = plmodel.test_results.get("y")
                unixtime = plmodel.test_results.get("unixtime")

                y, t, unixtime = delete_overlap_val(y, t, unixtime)
                y = softmax(y, 1)

            outputs[f"{user}-{session}"] = {
                "t": t,
                "y": y,
                "unixtime": unixtime,
            }
        results_val.append(outputs)

        outputs = dict()
        cfg.mode = "submission"
        datamodule.setup("submission")
        dataloaders = datamodule.submission_dataloader()
        split = cfg.dataset.split.submission
        for i, dataloader in enumerate(dataloaders):
            user, session = split[i]
            if len(dataloader) == 0:
                outputs[f"{user}-{session}"] = {
                    "y": None,
                    "unixtime": None,
                }
                continue
            logger.info(f"test on {user}-{session}")

            if average_slide:
                slide_results = []
                for n in tqdm(range(cfg.model.num_patches)):
                    dataloader.dataset.set_test_start_time(n)

                    with torch.inference_mode():
                        trainer.test(plmodel, dataloader)

                    # save model outputs
                    # pred_dir = Path(cfg.path.logdir.predict.format(user=user, session=session))
                    # pred_dir.mkdir(parents=True, exist_ok=True)

                    # for key, arr in plmodel.test_results.items():
                    #     path = Path(pred_dir, f"{key}.npy")
                    #     np.save(path, arr)
                    #     logger.info(f"save {key}[shape={arr.shape}] to {path}")

                    y = plmodel.test_results.get("y")
                    unixtime = plmodel.test_results.get("unixtime")

                    y, unixtime = delete_overlap(y, unixtime)
                    slide_results.append({"y": y, "unixtime": unixtime})

                y = average_slide_results(slide_results)
                unixtime = slide_results[0]["unixtime"]

            else:
                dataloader.dataset.set_test_start_time(0)

                with torch.inference_mode():
                    trainer.test(plmodel, dataloader)

                # save model outputs
                # pred_dir = Path(cfg.path.logdir.predict.format(user=user, session=session))
                # pred_dir.mkdir(parents=True, exist_ok=True)

                # for key, arr in plmodel.test_results.items():
                #     path = Path(pred_dir, f"{key}.npy")
                #     np.save(path, arr)
                #     logger.info(f"save {key}[shape={arr.shape}] to {path}")

                y = plmodel.test_results.get("y")
                unixtime = plmodel.test_results.get("unixtime")

                y, unixtime = delete_overlap(y, unixtime)
                y = softmax(y, 1)

            outputs[f"{user}-{session}"] = {
                "y": y,
                "unixtime": unixtime,
            }
        results_test.append(outputs)

    stack_results_val.append(results_val)
    stack_results_test.append(results_test)
    np.save(save_results_path, [results_val, results_test])

#%%
base_issue_results_val_ind = 0
base_issue_results_test_ind = 0

val_keys = [x.keys() for x in stack_results_val[base_issue_results_val_ind]]

X_train_base = []
y_train = []
for i, keys in enumerate(val_keys):
    for j, key in enumerate(keys):
        t = []
        y = []
        unixtime = []
        for results_val in stack_results_val:
            t.append(results_val[i][key]["t"])
            y.append(results_val[i][key]["y"])
            unixtime.append(results_val[i][key]["unixtime"])
        stack_t = []
        stack_y = []
        stack_unixtime = []
        for u in unixtime[base_issue_results_val_ind]:
            for u2 in u:
                _stack_t = []
                _stack_y = []
                _stack_unixtime = []
                names = []
                names_single = []
                for n, _u in enumerate(unixtime):
                    if not (t[n] is None) and (_u == u2).sum() >= 1:
                        _stack_t.append(t[n][_u == u2].squeeze())
                        _stack_y.append(y[n].transpose(0, 2, 1)[_u == u2].squeeze())
                        _stack_unixtime.append(unixtime[n][_u == u2].squeeze())
                        names.append(np.arange(11) + n * 100)
                        names_single.append(n)
                stack_y.append(pd.DataFrame([np.concatenate(_stack_y)], columns=np.concatenate(names)))
                stack_t.append(pd.DataFrame([np.stack(_stack_t)], columns=names_single))
                stack_unixtime.append(pd.DataFrame([np.stack(_stack_unixtime)], columns=names_single))
        X_train_base.append(pd.concat(stack_y, axis=0))
        y_train.append(pd.concat(stack_t, axis=0))
X_train_base = pd.concat(X_train_base, axis=0)
y_train = pd.concat(y_train, axis=0)
y_train = y_train[base_issue_results_val_ind]

test_keys = [x.keys() for x in stack_results_test[base_issue_results_test_ind]]

userss_test = []
X_test_base = []
unixtime_test_base = []
for key in test_keys[0]:
    y = []
    unixtime = []
    for results_test in stack_results_test:
        _y = []
        _unixtime = []
        for j in range(nums_folds):
            if results_test[j][key]["y"] is not None:
                _y.append(results_test[j][key]["y"])
                _unixtime.append(results_test[j][key]["unixtime"])
        if len(_y) == 0:
            _y = None
            _unixtime = None
        else:
            _y = np.stack(_y).mean(0)
            _unixtime = _unixtime[0]
        y.append(_y)
        unixtime.append(_unixtime)
    stack_y = []
    stack_unixtime = []
    for u in unixtime[base_issue_results_test_ind]:
        for u2 in u:
            _stack_y = []
            _stack_unixtime = []
            names = []
            names_single = []
            for n, _u in enumerate(unixtime):
                if not (y[n] is None) and (_u == u2).sum() >= 1:
                    _stack_y.append(y[n].transpose(0, 2, 1)[_u == u2][[0]].squeeze())
                    _stack_unixtime.append(unixtime[n][_u == u2][[0]].squeeze())
                    names.append(np.arange(11) + n * 100)
                    names_single.append(n)
            stack_y.append(pd.DataFrame([np.concatenate(_stack_y)], columns=np.concatenate(names)))
            stack_unixtime.append(
                pd.DataFrame([list(np.stack(_stack_unixtime)) + [key]], columns=names_single + ["key"])
            )
    X_test_base.append(pd.concat(stack_y, axis=0))
    unixtime_test_base.append(pd.concat(stack_unixtime, axis=0))
    userss_test.append(key)
X_test_base_input = pd.concat(X_test_base, axis=0)
unixtime_test_base = pd.concat(unixtime_test_base, axis=0)

X_train_base = X_train_base[X_train_base.columns.drop([10 + 100 * n for n in range(len(issue_list))])]



#%% mean
prediction = np.array(X_test_base_input[X_train_base.columns]).reshape(-1,len(issue_list),10).mean(1)
prediction_mean=prediction
outputs = dict()
key_list = unixtime_test_base["key"].values
unixtime_list = unixtime_test_base[base_issue_results_test_ind].values
for key in test_keys[0]:
    unixtime = unixtime_list[key_list == key]
    y = prediction[key_list == key]
    outputs[key] = {
        "y": y.reshape(-1, 50, 10).transpose(0, 2, 1),
        "unixtime": np.array(unixtime).reshape(-1, 50),
    }

output_dir = f"/workspace/logs/tf_mean_stacking{len(issue_list)}_0111_average_slide{average_slide}_del10col"
os.makedirs(output_dir, exist_ok=True)

submission_dict = construct_submission_dict(outputs, OPENPACK_OPERATIONS)
make_submission_zipfile(submission_dict, output_dir)


# # %% L1
# from sklearn.linear_model import LogisticRegression
# from sklearn.model_selection import GridSearchCV, StratifiedKFold
# param_grid = [
#     {'C': [10, 20, 30]},
#     ]
# model=LogisticRegression(penalty="l1", solver="saga")#,n_jobs=-1)
# cv_shuffle = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
# clf = GridSearchCV(model, param_grid,cv=cv_shuffle, scoring="f1_macro")
# clf.fit(X_train_base[y_train != 10], y_train[y_train != 10])
# df = pd.DataFrame(clf.cv_results_)
# df

# #%%
# prediction = clf.best_estimator_.predict_proba(X_test_base_input[X_train_base.columns])
# prediction_l1 = prediction

# outputs = dict()
# key_list = unixtime_test_base["key"].values
# unixtime_list = unixtime_test_base[base_issue_results_test_ind].values
# for key in test_keys[0]:
#     unixtime = unixtime_list[key_list == key]
#     y = prediction[key_list == key]
#     outputs[key] = {
#         "y": y.reshape(-1, 50, 10).transpose(0, 2, 1),
#         "unixtime": np.array(unixtime).reshape(-1, 50),
#     }

# output_dir = f"/workspace/logs/l1_stacking{len(issue_list)}_0110_average_slide{average_slide}_del10col"
# os.makedirs(output_dir, exist_ok=True)

# submission_dict = construct_submission_dict(outputs, OPENPACK_OPERATIONS)
# make_submission_zipfile(submission_dict, output_dir)


# #%% L2
# param_grid = [
#     {'C': [1, 5, 10, 20, 30]},
#     ]
# model=LogisticRegression(penalty="l2", solver="saga",n_jobs=-1)
# cv_shuffle = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
# clf = GridSearchCV(model, param_grid,cv=cv_shuffle, scoring="f1_macro")
# clf.fit(X_train_base[y_train != 10], y_train[y_train != 10])
# df = pd.DataFrame(clf.cv_results_)
# df

# #%%
# prediction = clf.best_estimator_.predict_proba(X_test_base_input[X_train_base.columns])
# prediction_l2 = prediction

# outputs = dict()
# key_list = unixtime_test_base["key"].values
# unixtime_list = unixtime_test_base[base_issue_results_test_ind].values
# for key in test_keys[0]:
#     unixtime = unixtime_list[key_list == key]
#     y = prediction[key_list == key]
#     outputs[key] = {
#         "y": y.reshape(-1, 50, 10).transpose(0, 2, 1),
#         "unixtime": np.array(unixtime).reshape(-1, 50),
#     }

# output_dir = f"/workspace/logs/l2_stacking{len(issue_list)}_0110_average_slide{average_slide}_del10col"
# os.makedirs(output_dir, exist_ok=True)

# submission_dict = construct_submission_dict(outputs, OPENPACK_OPERATIONS)
# make_submission_zipfile(submission_dict, output_dir)


# # %%
# xgbc2_params = {
#     "n_estimators": 100,
#     "max_depth": 5,
#     "random_state": 42,
# }

# xgbc2 = XGBClassifier(**xgbc2_params)
# xgbc2.fit(X_train_base[y_train != 10], y_train[y_train != 10])  # , sample_weight=weight_train)
# prediction = xgbc2.predict_proba(X_test_base_input[X_train_base.columns])
# prediction_xgb = prediction

# #%%
# outputs = dict()
# key_list = unixtime_test_base["key"].values
# unixtime_list = unixtime_test_base[base_issue_results_test_ind].values
# for key in test_keys[0]:
#     unixtime = unixtime_list[key_list == key]
#     y = prediction[key_list == key]
#     outputs[key] = {
#         "y": y.reshape(-1, 50, 10).transpose(0, 2, 1),
#         "unixtime": np.array(unixtime).reshape(-1, 50),
#     }

# output_dir = f"/workspace/logs/stacking{len(issue_list)}_1211_average_slide{average_slide}_del10col"
# os.makedirs(output_dir, exist_ok=True)

# submission_dict = construct_submission_dict(outputs, OPENPACK_OPERATIONS)
# make_submission_zipfile(submission_dict, output_dir)


# #%%
# # plot_importance(xgbc2)
# # plt.show()

# # %%
# lgb_train = lgb.Dataset(X_train_base[y_train != 10], y_train[y_train != 10])
# lgb_test = lgb.Dataset(X_test_base_input[X_train_base.columns])
# folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# callbacks = [early_stopping(50), log_evaluation(10)]

# lgbm_params = {
#     "objective": "multiclass",
#     "metric": "multi_logloss",
#     "num_class": 10,
#     "verbosity": -1,
#     "n_jobs": -1,
# }

# tuner_cv = lgb.LightGBMTunerCV(
#     lgbm_params,
#     lgb_train,
#     num_boost_round=1000,
#     folds=folds,
#     callbacks=callbacks,
#     return_cvbooster=True,
# )
# tuner_cv.run()

# print(f"Best score: {tuner_cv.best_score}")
# print("Best params:")
# pprint(tuner_cv.best_params)

# # %%
# # 最も良かったパラメータをキーにして学習済みモデルを取り出す
# cv_booster = tuner_cv.get_best_booster()
# # Averaging でホールドアウト検証データを予測する
# y_pred_proba_list = cv_booster.predict(
#     X_test_base_input[X_train_base.columns], num_iteration=cv_booster.best_iteration
# )
# y_pred_proba_avg = np.array(y_pred_proba_list).mean(axis=0)
# prediction_tunedlgb = y_pred_proba_avg

# #%%
# outputs = dict()
# key_list = unixtime_test_base["key"].values
# unixtime_list = unixtime_test_base[base_issue_results_test_ind].values
# for key in test_keys[0]:
#     unixtime = unixtime_list[key_list == key]
#     y = y_pred_proba_avg[key_list == key]
#     outputs[key] = {
#         "y": y.reshape(-1, 50, 10).transpose(0, 2, 1),
#         "unixtime": np.array(unixtime).reshape(-1, 50),
#     }

# output_dir = f"/workspace/logs/tuned_stacking{len(issue_list)}_1211_average_slide{average_slide}_del10col"
# os.makedirs(output_dir, exist_ok=True)

# submission_dict = construct_submission_dict(outputs, OPENPACK_OPERATIONS)
# make_submission_zipfile(submission_dict, output_dir)

# # %%
# raw_importances = cv_booster.feature_importance(importance_type="gain")
# feature_name = cv_booster.boosters[0].feature_name()
# importance_df = pd.DataFrame(data=raw_importances, columns=feature_name)
# # 平均値でソートする
# sorted_indices = importance_df.mean(axis=0).sort_values(ascending=False).index
# sorted_importance_df = importance_df.loc[:, sorted_indices]
# # 上位をプロットする
# PLOT_TOP_N = 20
# plot_cols = sorted_importance_df.columns  # [:PLOT_TOP_N]
# _, ax = plt.subplots(figsize=(8, 8))
# ax.grid()
# ax.set_xscale("log")
# ax.set_ylabel("Feature")
# ax.set_xlabel("Importance")
# sns.boxplot(data=sorted_importance_df[plot_cols], orient="h", ax=ax)
# plt.show()


#%% 各々の平均
# prediction = np.stack([prediction_l1,prediction_l2,prediction_xgb,prediction_tunedlgb,prediction_mean]).mean(0)
# outputs = dict()
# key_list = unixtime_test_base["key"].values
# unixtime_list = unixtime_test_base[base_issue_results_test_ind].values
# for key in test_keys[0]:
#     unixtime = unixtime_list[key_list == key]
#     y = prediction[key_list == key]
#     outputs[key] = {
#         "y": y.reshape(-1, 50, 10).transpose(0, 2, 1),
#         "unixtime": np.array(unixtime).reshape(-1, 50),
#     }

# output_dir = f"/workspace/logs/allmean_stacking{len(issue_list)}_0110_average_slide{average_slide}_del10col"
# os.makedirs(output_dir, exist_ok=True)

# submission_dict = construct_submission_dict(outputs, OPENPACK_OPERATIONS)
# make_submission_zipfile(submission_dict, output_dir)
