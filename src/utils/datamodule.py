import copy
import math
import multiprocessing
from logging import getLogger
from typing import Dict, List, Optional, Tuple

import numpy as np
import openpack_toolkit as optk
import openpack_torch as optorch
import torch
from joblib import Parallel, delayed
from omegaconf import DictConfig, open_dict
from torch.utils.data import DataLoader

from utils.datasets import OpenPackAll

logger = getLogger(__name__)


class OpenPackAllDataModule(optorch.data.OpenPackBaseDataModule):
    dataset_class = OpenPackAll

    def set_fold(self, fold):
        with open_dict(self.cfg):
            self.cfg.dataset.k_fold = fold

    def get_kwargs_for_datasets(self, stage: Optional[str] = None) -> Dict:
        if stage == "train":
            kwargs = {
                "debug": self.cfg.debug,
                "mode": "train",
                "clip_time": self.cfg.model.num_patches,
                "time_step_width": self.cfg.model.time_step_width,
            }
        elif stage == "validate":
            kwargs = {
                "debug": self.cfg.debug,
                "mode": "validate",
                "clip_time": self.cfg.model.num_patches,
                "time_step_width": self.cfg.model.time_step_width,
            }
        elif stage == "test" or "submission":
            kwargs = {
                "debug": self.cfg.debug,
                "mode": "test",
                "clip_time": self.cfg.model.num_patches,
                "time_step_width": self.cfg.model.time_step_width,
            }
        return kwargs

    def train_dataloader(self) -> DataLoader:
        num_workers = (
            multiprocessing.cpu_count() if self.cfg.train.num_workers == -1 else self.cfg.train.num_workers
        )
        return DataLoader(
            self.op_train,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=num_workers,
            worker_init_fn=worker_init_fn,
            pin_memory=True,
            # prefetch_factor=3,
            # persistent_workers=True,
        )

    def val_dataloader(self) -> List[DataLoader]:
        # num_workers = (
        #     multiprocessing.cpu_count() if self.cfg.train.num_workers == -1 else self.cfg.train.num_workers
        # )
        return DataLoader(
            self.op_val,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=10,
            worker_init_fn=worker_init_fn,
            # persistent_workers=True,
            pin_memory=True,
            # prefetch_factor=3,
        )
        # dataloaders = []
        # for key, dataset in self.op_val.items():
        #     dataloaders.append(
        #         DataLoader(
        #             dataset,
        #             batch_size=self.batch_size,
        #             shuffle=False,
        #             pin_memory=True,
        #             num_workers=1,
        #             worker_init_fn=worker_init_fn,
        #         )
        #     )
        return dataloaders

    def test_dataloader(self) -> List[DataLoader]:
        # num_workers = (
        #     multiprocessing.cpu_count() if self.cfg.train.num_workers == -1 else self.cfg.train.num_workers
        # )
        # return DataLoader(
        #     self.op_test,
        #     batch_size=self.batch_size,
        #     shuffle=False,
        #     # num_workers=num_workers,
        #     # worker_init_fn=worker_init_fn,
        #     # persistent_workers=True,
        #     pin_memory=True,
        #     # prefetch_factor=3,
        # )
        dataloaders = []
        for key, dataset in self.op_test.items():
            dataloaders.append(
                DataLoader(
                    dataset,
                    batch_size=self.batch_size,
                    shuffle=False,
                    pin_memory=True,
                    num_workers=1,
                    worker_init_fn=worker_init_fn,
                )
            )
        return dataloaders

    def submission_dataloader(self) -> List[DataLoader]:
        num_workers = (
            multiprocessing.cpu_count() if self.cfg.train.num_workers == -1 else self.cfg.train.num_workers
        )
        dataloaders = []
        for key, dataset in self.op_submission.items():
            dataloaders.append(
                DataLoader(
                    dataset,
                    batch_size=self.batch_size,
                    shuffle=False,
                    num_workers=1,
                    worker_init_fn=worker_init_fn,
                )
            )
        return dataloaders

    # 並列化するとメモリ足らなくなる
    # def _init_datasets(
    #     self,
    #     user_session: Tuple[int, int],
    #     kwargs: Dict,
    # ) -> Dict[str, torch.utils.data.Dataset]:
    #     """Returns list of initialized dataset object.

    #     Args:
    #         rootdir (Path): _description_
    #         user_session (Tuple[int, int]): _description_
    #         kwargs (Dict): _description_

    #     Returns:
    #         Dict[str, torch.utils.data.Dataset]: dataset objects
    #     """

    #     def __init_dataset(user, session, kwargs):
    #         key = f"{user}-{session}"
    #         dataset = self.dataset_class(copy.deepcopy(self.cfg), [(user, session)], **kwargs)
    #         return key, dataset

    #     n_jobs = 1 if self.cfg.debug == True else -1
    #     data = Parallel(n_jobs=n_jobs)(
    #         delayed(__init_dataset)(user, session, kwargs) for user, session in user_session
    #     )
    #     datasets = dict()
    #     for key, dataset in data:
    #         datasets[key] = dataset
    #     return datasets

    def setup(self, stage: Optional[str] = None) -> None:
        split = self.cfg.dataset.split

        if stage in (None, "fit"):
            kwargs = self.get_kwargs_for_datasets(stage="train")
            self.op_train = self.dataset_class(self.cfg, split.train, **kwargs)
        else:
            self.op_train = None

        if stage in (None, "fit", "validate"):
            kwargs = self.get_kwargs_for_datasets(stage="validate")
            self.op_val = self.dataset_class(self.cfg, split.val, **kwargs)
            # self.op_val = self._init_datasets(split.val, kwargs)
        else:
            self.op_val = None

        if stage in (None, "test"):
            kwargs = self.get_kwargs_for_datasets(stage="test")
            # self.op_test = self.dataset_class(self.cfg, split.test, **kwargs)
            self.op_test = self._init_datasets(split.test, kwargs)
        else:
            self.op_test = None

        if stage in (None, "submission"):
            kwargs = self.get_kwargs_for_datasets(stage="submission")
            kwargs.update({"submission": True})
            self.op_submission = self._init_datasets(split.submission, kwargs)
        elif stage == "test-on-submission":
            kwargs = self.get_kwargs_for_datasets(stage="submission")
            self.op_submission = self._init_datasets(split.submission, kwargs)
        else:
            self.op_submission = None

        logger.info(f"dataset[train]: {self.op_train}")
        logger.info(f"dataset[val]: {self.op_val}")
        logger.info(f"dataset[test]: {self.op_test}")
        logger.info(f"dataset[submission]: {self.op_submission}")


def worker_init_fn(worker_id):
    worker_info = torch.utils.data.get_worker_info()
    dataset = worker_info.dataset  # the dataset copy in this worker process
    overall_n = len(dataset.data)
    # configure the dataset to only process the split workload
    worker_id = worker_info.id
    l = np.linspace(0, overall_n, worker_info.num_workers + 1)
    start = int(math.ceil(l[worker_id]))
    end = int(math.ceil(l[worker_id + 1]))
    dataset.data = dataset.data[start:end]
