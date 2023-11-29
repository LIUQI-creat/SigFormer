#%%
import glob
import logging
import os
import shutil
from pathlib import Path
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
from openpack_toolkit import OPENPACK_OPERATIONS
from openpack_toolkit.codalab.operation_segmentation import (
    construct_submission_dict,
    eval_operation_segmentation_wrapper,
    make_submission_zipfile,
)
from scipy.special import softmax
from tqdm import tqdm

from utils.datamodule import OpenPackAllDataModule
from utils.lightning_module import TransformerPL

#%%


_ = optk.utils.notebook.setup_root_logger()
logger = logging.getLogger(__name__)

logger.debug("debug")
logger.info("info")
logger.warning("warning")
optorch.configs.register_configs()

issue = "normal-transformer-d6-h16-np50-timestep1000-lr_scheori_warmup-bs32-lr0.0001-gradclip1.0-labelsm0-mixup0.8_0-shuffle0.2_0-imu400_keypoint500-ht12_printer12-ki_depth0-rs_depth0"
config_dir = os.path.join("/workspace/logs/all/transformer/", issue, ".hydra")
with hydra.initialize_config_dir(version_base=None, config_dir=config_dir):
    cfg = hydra.compose(
        config_name="config.yaml",
        # config_name="unet-tutorial2.yaml",
    )
cfg.dataset.annotation.activity_sets = dict()  # Remove this attribute just for the simpler visualization.
cfg.dataset.split = optk.configs.datasets.splits.OPENPACK_CHALLENGE_2022_SPLIT  # DEBUG_SPLIT
# cfg.dataset.split = optk.configs.datasets.splits.DEBUG_SPLIT
optorch.utils.reset_seed(seed=0)

#%%
# class OpenPackImuDataModule(optorch.data.OpenPackBaseDataModule):
#     dataset_class = optorch.data.datasets.OpenPackImu

#     def get_kwargs_for_datasets(self, stage: Optional[str] = None) -> Dict:
#         kwargs = {
#             "window": self.cfg.train.window,
#             "debug": self.cfg.debug,
#         }
#         return kwargs


# datamodule = OpenPackImuDataModule(cfg)
# datamodule.setup("test")
# dataloaders = datamodule.test_dataloader()

# batch = dataloaders[0].dataset.__getitem__(0)

#%%
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

#%%
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
        y_mean[i] = softmax(np.stack(y_stack[i]), 1).mean(0)
    y_mean = np.array(y_mean).reshape(B, T, C).transpose(0, 2, 1)

    return y_mean


# %%
datamodule = OpenPackAllDataModule(cfg)
datamodule.set_fold(0)
cfg.mode = "submission"
datamodule.setup("submission")
dataloaders = datamodule.submission_dataloader()
split = cfg.dataset.split.submission

results = []
nums_folds = 5

for k in range(nums_folds):
    outputs = dict()
    chk_dir = os.path.join(cfg.path.logdir.rootdir, f"checkpoints_k{k}", "*")
    chk_path = glob.glob(chk_dir)[0]
    plmodel = TransformerPL.load_from_checkpoint(chk_path, cfg=cfg)
    plmodel.to(dtype=torch.float, device=device)
    plmodel.eval()
    plmodel.set_fold(k)
    for i, dataloader in enumerate(dataloaders):
        user, session = split[i]
        logger.info(f"test on {user}-{session}")

        slide_results = []
        for n in tqdm(range(cfg.model.num_patches)):
            dataloader.dataset.set_test_start_time(n)
            with torch.inference_mode():
                trainer.test(plmodel, dataloader)

            # save model outputs
            pred_dir = Path(cfg.path.logdir.predict.format(user=user, session=session))
            pred_dir.mkdir(parents=True, exist_ok=True)

            for key, arr in plmodel.test_results.items():
                path = Path(pred_dir, f"{key}.npy")
                np.save(path, arr)
                logger.info(f"save {key}[shape={arr.shape}] to {path}")

            y = plmodel.test_results.get("y")
            unixtime = plmodel.test_results.get("unixtime")

            y, unixtime = delete_overlap(y, unixtime)
            slide_results.append({"y": y, "unixtime": unixtime})

        y = average_slide_results(slide_results)
        unixtime = slide_results[0]["unixtime"]
        outputs[f"{user}-{session}"] = {
            "y": y,
            "unixtime": unixtime,
        }
        results.append(outputs)

outputs = dict()
for i in range(len(results[0].keys())):
    userss = list(results[0].keys())[i]
    all_y = []
    for j in range(nums_folds):
        all_y.append(results[j][userss]["y"])
    all_y = np.array(all_y)
    new_y = all_y.mean(0)  # TODO average_slide_resultsでsoftmaxしてるならここではしない
    unixtime = results[0][userss]["unixtime"]
    outputs[userss] = {
        "y": new_y,
        "unixtime": unixtime,
    }

# %%
output_dir = str(cfg.path.logdir.rootdir)

submission_dict = construct_submission_dict(outputs, OPENPACK_OPERATIONS)
make_submission_zipfile(submission_dict, output_dir)

# %%
