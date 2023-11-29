#%%
import glob
import logging
import os
from pathlib import Path

import hydra
#import matplotlib.pyplot as plt
import numpy as np
import openpack_toolkit as optk
import openpack_torch as optorch
import pandas as pd
import pytorch_lightning as pl
import torch
from openpack_toolkit import OPENPACK_OPERATIONS
from openpack_toolkit.codalab.operation_segmentation import construct_submission_dict, make_submission_zipfile
from scipy.special import softmax
from tqdm import tqdm

from utils import splits_5_fold_new
from utils.datamodule import OpenPackAllDataModule
from utils.lightning_module import TransformerPL

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
    "SigFormer-seed65-d6-h20-np50-peTrue-timestep1000-bs32-lr0.0001-labelsm0-mixup0.8_0-shuffle0.2_0-w_kappa_loss0-boundary_loss0.2-boundary_threshold0.85",
]
config_dir_list = [
    "/workspace/logs/all/SigFormer/",
]
datamodule_class_list = [OpenPackAllDataModule] * len(issue_list)
plmodel_class_list = [
    TransformerPL,
] * len(issue_list)

average_slide = False #True

stack_results_val = []
stack_results_test = []

nums_folds = 1

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
    cfg.dataset.split = optk.configs.datasets.splits.OPENPACK_CHALLENGE_2022_SPLIT
    optorch.utils.reset_seed(seed=0)

    save_results_name = f"stacking_mean_results_average_slide{average_slide}.npy"
    save_results_path = os.path.join(cfg.path.logdir.rootdir, save_results_name)
    if os.path.exists(save_results_path):
        results_test = np.load(save_results_path, allow_pickle=True)
        stack_results_test.append(results_test)
        continue

    device = torch.device("cuda")
    logdir = Path(cfg.path.logdir.rootdir)
    logger.debug(f"logdir = {logdir}")
    num_epoch = cfg.train.debug.epochs if cfg.debug else cfg.train.epochs

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

    results_test = []

    for k in range(nums_folds):
        cfg.dataset.split = getattr(splits_5_fold_new, f"OPENPACK_CHALLENGE_0_FOLD_SPLIT")  # {k+1}
        datamodule = datamodule_class(cfg)
        datamodule.set_fold(k)

        outputs = dict()
        chk_dir = os.path.join(cfg.path.logdir.rootdir, f"checkpoints_k{k}", "*")
        chk_path = glob.glob(chk_dir)[0]
        plmodel = plmodel_class.load_from_checkpoint(chk_path, cfg=cfg)
        plmodel.to(dtype=torch.float, device=device)
        plmodel.eval()
        plmodel.set_fold(k)
        cfg.mode = "test"
        datamodule.setup("test")
        dataloaders = datamodule.test_dataloader()
        split = cfg.dataset.split.test
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

                y = plmodel.test_results.get("y")
                unixtime = plmodel.test_results.get("unixtime")

                y, unixtime = delete_overlap(y, unixtime)
                y = softmax(y, 1)

            outputs[f"{user}-{session}"] = {
                "y": y,
                "unixtime": unixtime,
            }
        results_test.append(outputs)
    stack_results_test.append(results_test)
    np.save(save_results_path, results_test)

#%%
base_issue_results_test_ind = 0

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


#%% mean
X_test_base_input = X_test_base_input[
    X_test_base_input.columns.drop([10 + 100 * n for n in range(len(issue_list))])
]

prediction = np.array(X_test_base_input).reshape(-1, len(issue_list), 10).mean(1)

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

output_dir = f"/workspace/logs/avec/avec_mean_stacking{len(issue_list)}_0116_average_slide{average_slide}_del10col"
os.makedirs(output_dir, exist_ok=True)

submission_dict = construct_submission_dict(outputs, OPENPACK_OPERATIONS)
make_submission_zipfile(submission_dict, output_dir)
