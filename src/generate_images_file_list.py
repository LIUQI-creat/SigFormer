#%%
import datetime
import glob
import logging
import math
import os
import re
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import hydra
import numpy as np
import openpack_toolkit as optk
import openpack_torch as optorch
import pandas
import pandas as pd
import torch
from omegaconf import DictConfig, OmegaConf, open_dict
from openpack_toolkit import OPENPACK_OPERATIONS
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.datamodule import OpenPackAllDataModule
from utils.splits_all import OPENPACK_CHALLENGE_ALL_SPLIT, OPENPACK_CHALLENGE_ALL_SPLIT_DEBUG

#%%
optorch.configs.register_configs()
optorch.utils.reset_seed(seed=0)
with hydra.initialize_config_dir(version_base=None, config_dir="/workspace/configs"):
    cfg = hydra.compose(
        config_name="transformer_debug.yaml",
        # config_name="unet-tutorial2.yaml",
    )

cfg.dataset.annotation.activity_sets = dict()  # Remove this attribute just for the simpler visualization.
cfg.dataset.split = OPENPACK_CHALLENGE_ALL_SPLIT
# cfg.dataset.split = OPENPACK_CHALLENGE_ALL_SPLIT_DEBUG


cfg.dataset.split.train


def numericalSort(value):
    numbers = re.compile(r"(\d+)")
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts


# %%
for user, session in cfg.dataset.split.train:
    with open_dict(cfg):
        cfg.user = {"name": user}
    rootpath = Path(cfg.dataset.stream.path_kinect_depth.dir, session)

    col_names = ["unixtime", "path"]
    data = []

    png_path_list = sorted(glob.glob(os.path.join(rootpath, "*/*/*/*/*/*.png")), key=numericalSort)
    print(rootpath, len(png_path_list))
    for png_path in png_path_list:
        time = os.path.basename(png_path).split(".")[0]
        assert len(time) == 19 or time[-4:] == "1000"
        time = datetime.datetime.strptime(time, "%Y%m%d_%H%M%S_%f")
        time = (time - datetime.timedelta(hours=9)).timestamp()
        time = int(time * 1000)

        relative_png_path = png_path.replace(str(rootpath) + "/", "")

        df = pd.DataFrame([[time, relative_png_path]], columns=col_names)
        data.append(df)

    if len(data) != 0:
        df_conc = pd.concat(data, axis=0)
        df_conc.to_csv(Path(cfg.dataset.stream.path_kinect_depth.dir, f"{session}.csv"), index=False)


# %%
for user, session in cfg.dataset.split.train:
    with open_dict(cfg):
        cfg.user = {"name": user}
    rootpath = Path(cfg.dataset.stream.path_rs02_depth.dir, session)

    col_names = ["unixtime", "path"]
    data = []

    jpeg_path_list = sorted(glob.glob(os.path.join(rootpath, "*/*/*/*/*/*.jpeg")), key=numericalSort)
    print(rootpath, len(jpeg_path_list))
    for png_path in jpeg_path_list:
        time = os.path.basename(png_path).split(".")[0]
        assert len(time) == 22
        time = datetime.datetime.strptime(time, "%Y%m%d_%H%M%S_%f")
        time = (time - datetime.timedelta(hours=9)).timestamp()
        # time = time.timestamp()
        time = int(time * 1000)

        relative_png_path = png_path.replace(str(rootpath) + "/", "")

        df = pd.DataFrame([[time, relative_png_path]], columns=col_names)
        data.append(df)

    if len(data) != 0:
        df_conc = pd.concat(data, axis=0)
        df_conc.to_csv(Path(cfg.dataset.stream.path_rs02_depth.dir, f"{session}.csv"), index=False)
# %%
