import logging
import multiprocessing
import os
from pathlib import Path

import hydra
import numpy as np
import openpack_toolkit as optk
import openpack_torch as optorch
import pytorch_lightning as pl
import torch
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig, OmegaConf
from openpack_toolkit import OPENPACK_OPERATIONS
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger

from utils import splits_5_fold_new
from utils.datamodule import OpenPackAllDataModule
from utils.lightning_module import TransformerPL
from utils.splits_all import OPENPACK_CHALLENGE_ALL_SPLIT_DEBUG

logger = logging.getLogger(__name__)
optorch.configs.register_configs()


@hydra.main(version_base=None, config_name="config1", config_path="../configs")
def main(cfg: DictConfig) -> None:
    _ = optk.utils.notebook.setup_root_logger()

    cfg.dataset.annotation.activity_sets = dict()  # Remove this attribute just for the simpler visualization.
    cfg.train.num_workers = (
        multiprocessing.cpu_count() if cfg.train.num_workers == -1 else cfg.train.num_workers
    )

    pl.seed_everything(cfg.seed)

    wandb_logger = WandbLogger(name=cfg.issue, project=cfg.wandb.project)

    if cfg.train.grad_clip_norm > 0:
        kwargs_grad_clip = {"gradient_clip_algorithm": "norm", "gradient_clip_val": cfg.train.grad_clip_norm}
    else:
        kwargs_grad_clip = {}

    device = torch.device("cuda")
    logdir = Path(cfg.path.logdir.rootdir)
    logger.debug(f"logdir = {logdir}")

    num_epoch = cfg.train.debug.epochs if cfg.debug else cfg.train.epochs

    logger.debug(f"logdir = {logdir}")

    results = []
    results_f1 = []
    nums_folds = 5

    for k in range(nums_folds):
        split = getattr(splits_5_fold_new, f"OPENPACK_CHALLENGE_{k+1}_FOLD_SPLIT")
        if cfg.debug:
            split = OPENPACK_CHALLENGE_ALL_SPLIT_DEBUG
        cfg.dataset.split = split
        datamodule = OpenPackAllDataModule(cfg)
        datamodule.set_fold(k)
        # datamodule.setup("fit")

        checkpoint_dir = os.path.join(cfg.path.logdir.rootdir, f"checkpoints_k{k}")
        checkpoint_callback = pl.callbacks.ModelCheckpoint(
            save_top_k=1,
            mode="max",
            dirpath=checkpoint_dir,
            filename="op-{epoch:03d}-{val_f1macro:.4f}",
            # save_last=True,
            monitor="val_f1macro",
        )
        lr_monitor = LearningRateMonitor(logging_interval="epoch")

        trainer = pl.Trainer(
            accelerator="gpu",
            devices=[0],
            max_epochs=num_epoch,
            logger=wandb_logger,
            # default_root_dir=logdir,
            enable_progress_bar=True,
            enable_checkpointing=True,
            callbacks=[checkpoint_callback, lr_monitor],
            **kwargs_grad_clip
            # auto_lr_find gradient_clip_val try
            # https://rest-term.com/archives/3697/?utm_source=rss&utm_medium=rss&utm_campaign=pytorch-lightning%25e3%2581%25aelearning-rate-finder%25e3%2581%25ae%25e4%25bd%25bf%25e3%2581%2584%25e6%2596%25b9
        )
        plmodel = TransformerPL(cfg).to(dtype=torch.float, device=device)
        plmodel.set_fold(k)

        logger.info(f"Start training for {num_epoch} epochs (k={k}).")
        trainer.fit(plmodel, datamodule)

        score = max([dic["val/acc"] for dic in plmodel.log_dict["val"]])
        wandb_logger.log_metrics({f"hold-{k}_acc": score})
        score_f1 = max([dic["val/f1macro"] for dic in plmodel.log_dict["val"]])
        wandb_logger.log_metrics({f"fold-{k}_f1macro": score_f1})
        results.append(score)
        results_f1.append(score_f1)

        del plmodel
        del datamodule

    logger.info("Finish training!")
    final_score = sum(results) / nums_folds
    logger.info("Final result: {}".format(final_score))
    wandb_logger.log_metrics({"final_acc": final_score})
    final_score_f1 = sum(results_f1) / nums_folds
    logger.info("Final result: {}".format(final_score_f1))
    wandb_logger.log_metrics({"final_f1macro": final_score_f1})

    with open(os.path.join(cfg.path.logdir.rootdir, "k-fold_val_acc.txt"), "w") as f:
        f.write(str(final_score))


if __name__ == "__main__":
    main()
