#%%
import os
from pathlib import Path

import hydra
import numpy as np
from omegaconf import DictConfig, open_dict
from PIL import Image
from tqdm import tqdm

from utils import splits_5_fold
from utils.dataloader import load_depth

#%%
with hydra.initialize_config_dir(version_base=None, config_dir="/workspace/configs"):
    cfg = hydra.compose(
        config_name="transformer_single_run.yaml",
        # config_name="unet-tutorial2.yaml",
    )
nums_folds = 5

for k in range(nums_folds):

    split = getattr(splits_5_fold, f"OPENPACK_CHALLENGE_{k+1}_FOLD_SPLIT")
    im_sum = []
    for info in tqdm(split.train):
        user, session = info
        with open_dict(cfg):
            cfg.user = {"name": user}
            cfg.session = session
        path_kinect_depth = Path(
            cfg.dataset.stream.path_kinect_depth.dir,
            cfg.dataset.stream.path_kinect_depth.fname,
        )
        _, path_list = load_depth(path_kinect_depth)

        if path_list is None:
            continue

        total = 0
        _im_sum = []
        for path in path_list:
            im = Image.open(path)
            im = np.array(im)
            total += 1
            if len(_im_sum) == 0:
                _im_sum = np.zeros_like(im)
                _im_sum += im
            else:
                _im_sum += im
        im_sum.append(_im_sum / total)
    im_sum = np.array(im_sum)
    pil_image = Image.fromarray(im_sum.mean(0)).convert("I")
    os.makedirs("/workspace/data_local/v0.3.1/mean_image/kinect_depth/", exist_ok=True)
    pil_image.save(
        f"/workspace/data_local/v0.3.1/mean_image/kinect_depth/split{k}.png",
    )


#%%
for k in range(nums_folds):

    split = getattr(splits_5_fold, f"OPENPACK_CHALLENGE_{k+1}_FOLD_SPLIT")
    im_sum = []
    for info in tqdm(split.train):
        user, session = info
        with open_dict(cfg):
            cfg.user = {"name": user}
            cfg.session = session
        path_rs02_depth = Path(
            cfg.dataset.stream.path_rs02_depth.dir,
            cfg.dataset.stream.path_rs02_depth.fname,
        )
        _, path_list = load_depth(path_rs02_depth)

        if path_list is None:
            continue

        total = 0
        _im_sum = []
        for path in path_list:
            im = Image.open(path)
            im = np.array(im)
            total += 1
            if len(_im_sum) == 0:
                _im_sum = np.zeros_like(im, dtype=np.uint64)
                _im_sum += im
            else:
                _im_sum += im
        im_sum.append(_im_sum / total)
    im_sum = np.array(im_sum)
    pil_image = Image.fromarray(im_sum.mean(0).astype(np.uint8)).convert("RGB")
    os.makedirs("/workspace/data_local/v0.3.1/mean_image/rs02_depth/", exist_ok=True)
    pil_image.save(
        f"/workspace/data_local/v0.3.1/mean_image/rs02_depth/split{k}.png",
    )


# %% test
path = "/workspace/data_local/v0.3.1/U0101/rs02/depth/frames_resize224/S0400/2021/10/14/15/20/20211014_152004_452110.jpeg"
a = Image.open(path).convert("L")
# %%
path = "/workspace/data_local/v0.3.1/mean_image/kinect_depth/split0.png"
a = Image.open(path)
a = np.array(a)
