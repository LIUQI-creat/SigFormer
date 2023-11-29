"""Dataset Class for OpenPack dataset.
"""
from logging import getLogger
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import openpack_toolkit as optk
import PIL
import torch
from joblib import Parallel, delayed
from omegaconf import DictConfig, open_dict
from openpack_toolkit import OPENPACK_OPERATIONS
from PIL import Image, ImageFile

from utils.dataloader import (
    load_and_resample_scan_log,
    load_bbox,
    load_depth,
    load_e4acc,
    load_feature,
    load_feature_all,
    load_mean_feature,
    load_mean_image,
    pre_load_image,
)

logger = getLogger(__name__)
ImageFile.LOAD_TRUNCATED_IMAGES = True


class OpenPackAll(torch.utils.data.IterableDataset):  # FIXME torch.utils.data.ChainDatasetの検討
    data: List[Dict] = None
    index: Tuple[Dict] = None

    def __init__(
        self,
        cfg: DictConfig,
        user_session_list: Tuple[Tuple[int, int], ...],
        classes: optk.ActSet = OPENPACK_OPERATIONS,
        submission: bool = False,
        debug: bool = False,
        epoch_size=None,
        frame_transform=None,
        video_transform=None,
        clip_time=30,
        time_step_width=1000,
        mode="train",
    ) -> None:
        """Initialize OpenPackImu dataset class.

        Args:
            cfg (DictConfig): instance of ``optk.configs.OpenPackConfig``. path, dataset, and
                annotation attributes must be initialized.
            user_session (Tuple[Tuple[int, int], ...]): the list of pairs of user ID and session ID
                to be included.
            classes (optk.ActSet, optional): activity set definition.
                Defaults to OPENPACK_OPERATION_CLASSES.
            window (int, optional): window size [steps]. Defaults to 30*60 [s].
            submission (bool, optional): Set True when you want to load test data for submission.
                If True, the annotation data will no be replaced by dummy data. Defaults to False.
            debug (bool, optional): enable debug mode. Defaults to False.
        """
        super().__init__()
        self.classes = classes
        self.submission = submission
        self.debug = debug
        self.epoch_size = epoch_size
        self.clip_time = clip_time
        self.frame_transform = frame_transform
        self.video_transform = video_transform
        self.time_step_width = time_step_width
        self.mode = mode

        self.load_dataset(cfg, user_session_list, submission=submission)

        # Allow for temporal jittering
        if epoch_size is None:
            epoch_size = len(self.data)
        self.epoch_size = epoch_size

        self.preprocessing(cfg)

    def load_dataset(
        self,
        cfg: DictConfig,
        user_session_list: Tuple[Tuple[int, int], ...],
        submission: bool = False,
    ) -> None:
        """Called in ``__init__()`` and load required data.

        Args:
            user_session (Tuple[Tuple[str, str], ...]): _description_
            window (int, optional): _description_. Defaults to None.
            submission (bool, optional): _description_. Defaults to False.
        """

        def _load_data(user, session):
            with open_dict(cfg):
                cfg.user = {"name": user}
                cfg.session = session

            ts_list = []
            x_list = []
            data_name = []
            # imu
            paths_imu = []
            for device in cfg.dataset.stream.devices:
                with open_dict(cfg):
                    cfg.device = device

                path = Path(cfg.dataset.stream.path_imu.dir, cfg.dataset.stream.path_imu.fname)
                paths_imu.append(path)

            ts_sess_imu, x_sess_imu = optk.data.load_imu(
                paths_imu,
                use_acc=cfg.dataset.stream.acc,
                use_gyro=cfg.dataset.stream.gyro,
                use_quat=cfg.dataset.stream.quat,
            )
            x_sess_imu = x_sess_imu.T
            ts_list.append(ts_sess_imu)
            x_list.append(x_sess_imu)
            data_name.append("imu")

            # keypoint
            path_keypoint = Path(
                cfg.dataset.stream.path_keypoint.dir,
                cfg.dataset.stream.path_keypoint.fname,
            )
            ts_sess_keypoint, x_sess_keypoint = optk.data.load_keypoints(path_keypoint)
            x_sess_keypoint = x_sess_keypoint[: (x_sess_keypoint.shape[0] - 1)]  # Remove prediction score.
            x_sess_keypoint = x_sess_keypoint.transpose(1, 0, 2)
            ts_list.append(ts_sess_keypoint)
            x_list.append(x_sess_keypoint)
            data_name.append("keypoint")

            # if cfg.model.kinect_depth_dim != 0:
            #     if cfg.model.use_cnn_feature:
            #         # path_kinect_depth = Path(
            #         #     cfg.dataset.stream.path_kinect_feature.dir,
            #         #     cfg.dataset.stream.path_kinect_feature.fname,
            #         # )
            #         path_kinect_depth_all_unixtime = Path(
            #             cfg.dataset.stream.path_kinect_feature_all.dir,
            #             cfg.dataset.stream.path_kinect_feature_all.unixtime_fname,
            #         )
            #         path_kinect_depth_all_feat = Path(
            #             cfg.dataset.stream.path_kinect_feature_all.dir,
            #             cfg.dataset.stream.path_kinect_feature_all.feat_fname,
            #         )
            #         ts_sess_kinect_depth, x_sess_kinect_depth = load_feature_all(
            #             path_kinect_depth_all_unixtime, path_kinect_depth_all_feat
            #         )
            #         ts_list.append(ts_sess_kinect_depth)
            #         x_list.append(x_sess_kinect_depth)
            #         data_name.append("kinect_feature")
            #     else:
            #         # kinect_depth
            #         path_kinect_depth = Path(
            #             cfg.dataset.stream.path_kinect_depth.dir,
            #             cfg.dataset.stream.path_kinect_depth.fname,
            #         )
            #         ts_sess_kinect_depth, x_sess_kinect_depth = load_depth(path_kinect_depth)
            #         if cfg.dataload.pre_image:
            #             all_image_path = Path(
            #                 cfg.dataload.all_image_path.dir,
            #                 cfg.dataload.all_image_path.fname,
            #             )
            #             x_sess_kinect_depth = pre_load_image(
            #                 x_sess_kinect_depth,
            #                 all_image_path,
            #                 cfg.model.image_size,
            #                 min_value=cfg.dataset.stream.min_value_kinect_depth,
            #                 max_value=cfg.dataset.stream.max_value_kinect_depth,
            #             )
            #         ts_list.append(ts_sess_kinect_depth)
            #         x_list.append(x_sess_kinect_depth)
            #         data_name.append("kinect_depth")

            # else:
            #     ts_list.append(None)
            #     x_list.append(None)
            #     data_name.append("kinect_depth")

            # # rs02_depth
            # if cfg.model.rs02_depth_dim != 0:
            #     path_rs02_depth = Path(
            #         cfg.dataset.stream.path_rs02_depth.dir,
            #         cfg.dataset.stream.path_rs02_depth.fname,
            #     )
            #     ts_sess_rs02_depth, x_sess_rs02_depth = load_depth(path_rs02_depth)
            #     ts_list.append(ts_sess_rs02_depth)
            #     x_list.append(x_sess_rs02_depth)
            #     data_name.append("rs02_depth")
            # else:
            #     ts_list.append(None)
            #     x_list.append(None)
            #     data_name.append("rs02_depth")

            # e4acc
            if cfg.model.e4acc_dim != 0:
                paths_e4acc = []
                for device in cfg.dataset.stream.devices_e4acc:
                    with open_dict(cfg):
                        cfg.device = device

                    path = Path(cfg.dataset.stream.path_e4acc.dir, cfg.dataset.stream.path_e4acc.fname)
                    paths_e4acc.append(path)

                ts_sess_e4acc, x_sess_e4acc = load_e4acc(
                    paths_e4acc,
                )
                if x_sess_e4acc is not None:
                    x_sess_e4acc = x_sess_e4acc.T
                ts_list.append(ts_sess_e4acc)
                x_list.append(x_sess_e4acc)
                data_name.append("e4acc")
            else:
                ts_list.append(None)
                x_list.append(None)
                data_name.append("e4acc")

            # bbox
            path_keypoint = Path(
                cfg.dataset.stream.path_keypoint.dir,
                cfg.dataset.stream.path_keypoint.fname,
            )
            ts_sess_bbox, x_sess_bbox = load_bbox(path_keypoint)
            x_sess_bbox = x_sess_bbox.transpose(1, 0)
            ts_list.append(ts_sess_bbox)
            x_list.append(x_sess_bbox)
            data_name.append("bbox")

            ts_sess, clip_ts_list, clip_x_list = self._clip_data(
                cfg,
                ts_list,
                x_list,
                data_name=data_name,  # kinect_depth
            )

            # ht
            path_ht = Path(
                cfg.dataset.stream.path_ht.dir,
                cfg.dataset.stream.path_ht.fname,
            )
            x_sess_ht = load_and_resample_scan_log(path_ht, ts_sess)

            # printer
            path_printer = Path(
                cfg.dataset.stream.path_printer.dir,
                cfg.dataset.stream.path_printer.fname,
            )
            x_sess_printer = load_and_resample_scan_log(path_printer, ts_sess)

            if submission:
                # For set dummy data.
                label = np.zeros((len(ts_sess),), dtype=np.int64)
            else:
                path = Path(cfg.dataset.annotation.path.dir, cfg.dataset.annotation.path.fname)
                df_label = optk.data.load_and_resample_operation_labels(path, ts_sess, classes=self.classes)
                label = df_label["act_idx"].values

            return {
                "user": user,
                "session": session,
                "imu": clip_x_list[0],
                "keypoint": clip_x_list[1],
                "imu_unixtime": clip_ts_list[0],
                "keypoint_unixtime": clip_ts_list[1],
                # "kinect_depth": clip_x_list[2],
                # "rs02_depth": clip_x_list[3],
                # "kinect_depth_unixtime": clip_ts_list[2],
                # "rs02_depth_unixtime": clip_ts_list[3],
                "label": label,
                "label_unixtime": ts_sess,  # =ht_unixtime, printer_unixtime
                "ht": x_sess_ht,
                "printer": x_sess_printer,
                "e4acc": clip_x_list[2],
                "e4acc_unixtime": clip_ts_list[2],
                "bbox": clip_x_list[3],
                "bbox_unixtime": clip_ts_list[3],
            }

        n_jobs = 1 if cfg.debug == True else -1
        data = Parallel(n_jobs=n_jobs)(
            delayed(_load_data)(user, session) for seq_idx, (user, session) in enumerate(user_session_list)
        )

        # if cfg.dataload.pre_image:
        #     mean_path = Path(
        #         cfg.dataset.stream.mean_image_kinect_depth.dir,
        #         cfg.dataset.stream.mean_image_kinect_depth.fname,
        #     )
        #     self.mean_kinect_depth = load_mean_image(
        #         mean_path,
        #         cfg.model.image_size,
        #         min_value=cfg.dataset.stream.min_value_kinect_depth,
        #         max_value=cfg.dataset.stream.max_value_kinect_depth,
        #     )

        self.data = data
        self.cfg = cfg

    def _get_start_end_ts(self, ts_list, th):
        def ceil(src, range=self.time_step_width):
            if src % range == 0:
                return src
            else:
                return ((int)(src / range) + 1) * range

        def floor(src, range=self.time_step_width):
            return (int)(src / range) * range

        def leftShiftIndex(arr, n):
            result = arr[n:] + arr[:n]
            return result

        min_list = []
        max_list = []
        not_None_list = []
        for ts in ts_list:
            if type(ts) == np.ndarray:
                min_list.append(ts[0])
                max_list.append(ts[-1])
                not_None_list.append(True)
            else:
                assert ts == None
                min_list.append(10000000000000)
                max_list.append(0)
                not_None_list.append(False)

        # # check threshold
        min_within_th = (min_list - min(min_list)) < th
        max_within_th = (max(max_list) - max_list) < th
        not_None_list = np.array(not_None_list)
        return (
            ceil(max(np.array(min_list)[np.logical_and(not_None_list, min_within_th)])),
            floor(min(np.array(max_list)[np.logical_and(not_None_list, max_within_th)])),
            min_within_th,
            max_within_th,
            not_None_list,
        )

    def _clip_data(self, cfg, ts_list, x_list, data_name, th=5000):
        start_ts, end_ts, min_within_th, max_within_th, not_None_list = self._get_start_end_ts(ts_list, th)
        base_ts = self._get_clip_base_ts(ts_list, cfg.dataset.stream.clip_base_ts, data_name)
        clip_ts_list = []
        clip_x_list = []
        for i, (ts, x) in enumerate(zip(ts_list, x_list)):
            if not_None_list[i] == False:
                if data_name[i] in ["kinect_depth", "rs02_depth"]:
                    fps = self._get_fps(cfg, data_name, i)
                    ts = np.floor(np.arange(start_ts, end_ts, 1000 / fps)).astype(int)
                    x = [None] * len(ts)
                elif data_name[i] in ["kinect_feature"]:
                    fps = self._get_fps(cfg, data_name, i)
                    ts = np.floor(np.arange(start_ts, end_ts, 1000 / fps)).astype(int)
                    x = np.zeros((len(ts), cfg.dataset.stream.kinect_feature_dim))
                elif data_name[i] in ["e4acc"]:
                    fps = self._get_fps(cfg, data_name, i)
                    ts = np.floor(np.arange(start_ts, end_ts, 1000 / fps)).astype(int)
                    x = np.zeros((len(ts), cfg.dataset.stream.e4acc_dim))

                else:
                    assert NotImplementedError

            else:
                use_index = np.logical_and(ts >= start_ts, ts <= end_ts)
                ts = ts[use_index]
                x = x[use_index]

                if min_within_th[i] == False and data_name[i] in [
                    "kinect_depth",
                    "rs02_depth",
                    "kinect_feature",
                ]:
                    fps = self._get_fps(cfg, data_name, i)
                    add_ts = np.floor(np.arange(start_ts, ts[0], 1000 / fps)).astype(int)
                    ts = np.concatenate([add_ts[:-1], ts])
                    if x.ndim == 2:
                        x = np.concatenate(
                            [np.repeat(np.zeros_like(x[0])[np.newaxis, :], len(add_ts[:-1]), 0), x]
                        )
                    else:
                        x = np.concatenate([[None] * len(add_ts[:-1]), x])
                if max_within_th[i] == False and data_name[i] in [
                    "kinect_depth",
                    "rs02_depth",
                    "kinect_feature",
                ]:
                    fps = self._get_fps(cfg, data_name, i)
                    add_ts = np.floor(np.arange(ts[-1], end_ts, 1000 / fps)).astype(int)
                    ts = np.concatenate([ts, add_ts[1:]])
                    if x.ndim == 2:
                        x = np.concatenate(
                            [x, np.repeat(np.zeros_like(x[0])[np.newaxis, :], len(add_ts[1:]), 0)]
                        )
                    else:
                        x = np.concatenate([x, [None] * len(add_ts[1:])])

            clip_ts_list.append(ts)
            clip_x_list.append(x)

        label_ts = np.arange(start_ts, end_ts + self.time_step_width, self.time_step_width)

        return label_ts, clip_ts_list, clip_x_list

    def _get_clip_base_ts(self, ts_list, base_name, data_name):
        ts = ts_list[data_name.index(base_name)]
        return ts

    def _get_fps(self, cfg, data_name, i):
        name = data_name[i]
        return cfg.dataset.stream["frame_rate_" + name]

    def preprocessing(self, cfg) -> None:
        """This method is called after ``load_dataset()`` and apply preprocessing to loaded data."""
        assert cfg.dataset.stream.quat == False and len(cfg.dataset.stream.devices) == 4, NotImplementedError
        for seq_dict in self.data:
            imu = seq_dict["imu"]

            tmp = []
            if cfg.dataset.stream.acc == True:
                min_value = cfg.dataset.stream.min_value_imu_acc
                tmp += [min_value, min_value, min_value]  # xyzの3次元
            if cfg.dataset.stream.gyro == True:
                min_value = cfg.dataset.stream.min_value_imu_gyro
                tmp += [min_value, min_value, min_value]  # xyzの3次元
            tmp = np.array(tmp)
            min_arr = np.concatenate([tmp, tmp, tmp, tmp])  # デバイス数
            tmp = []
            if cfg.dataset.stream.acc == True:
                max_value = cfg.dataset.stream.max_value_imu_acc
                tmp += [max_value, max_value, max_value]  # xyzの3次元
            if cfg.dataset.stream.gyro == True:
                max_value = cfg.dataset.stream.max_value_imu_gyro
                tmp += [max_value, max_value, max_value]  # xyzの3次元
            tmp = np.array(tmp)
            max_arr = np.concatenate([tmp, tmp, tmp, tmp])  # デバイス数

            imu = (imu - min_arr) / (max_arr - min_arr)
            imu = (imu - 0.5) / 0.5
            seq_dict["imu"] = imu

            keypoint = seq_dict["keypoint"]
            min_keypoint = cfg.dataset.stream.min_value_keypoint
            max_keypoint = cfg.dataset.stream.max_value_keypoint
            # keypoint = np.clip(keypoint, min_keypoint, max_keypoint)
            keypoint = (keypoint - min_keypoint) / (max_keypoint - min_keypoint)
            keypoint = (keypoint - 0.5) / 0.5
            seq_dict["keypoint"] = keypoint

            e4acc = seq_dict["e4acc"]
            min_e4acc = cfg.dataset.stream.min_value_e4acc
            max_e4acc = cfg.dataset.stream.max_value_e4acc
            # keypoint = np.clip(keypoint, min_keypoint, max_keypoint)
            e4acc = (e4acc - min_e4acc) / (max_e4acc - min_e4acc)
            e4acc = (e4acc - 0.5) / 0.5
            seq_dict["e4acc"] = e4acc

            bbox = seq_dict["bbox"]
            min_bbox = cfg.dataset.stream.min_value_bbox
            max_bbox = cfg.dataset.stream.max_value_bbox
            # keypoint = np.clip(keypoint, min_keypoint, max_keypoint)
            bbox = (bbox - min_bbox) / (max_bbox - min_bbox)
            bbox = (bbox - 0.5) / 0.5
            seq_dict["bbox"] = bbox

    @property
    def num_classes(self) -> int:
        """Returns the number of classes

        Returns:
            int
        """
        return len(self.classes)

    def __str__(self) -> str:
        s = "OpenPackImu(" f"num_sequence={len(self.data)}, " f"submission={self.submission}" ")"
        return s

    def __len__(self) -> int:
        num_max_clip = []
        for d in self.data:
            num_max_clip.append(len(d["label"]) // self.clip_time + 1)
        return sum(num_max_clip)

    def set_test_start_time(self, n):
        assert 0 <= n and n < self.clip_time
        self.test_start_time = n

    def __iter__(self):
        num_max_clip = []
        for d in self.data:
            num_max_clip.append(len(d["label"]) // self.clip_time + 1)

        cumsum = np.cumsum(num_max_clip)
        n_clip = cumsum[-1]
        if self.mode == "train":
            epoch_start_time = torch.randint(0, self.clip_time - 1, (len(self.data),)).numpy()
            index_order = torch.randperm(n_clip).numpy()
        elif self.mode == "test":
            epoch_start_time = np.ones(len(self.data), dtype=int) * self.test_start_time
            index_order = np.arange(n_clip)
        else:
            epoch_start_time = np.zeros(len(self.data), dtype=int)
            index_order = np.arange(n_clip)

        for ind in index_order:
            data_ind = np.where(cumsum > ind)[0][0]  # 怪しい　train_3dcnn参照
            data = self.data[data_ind]
            clip_position = ind - ([0] + list(cumsum))[data_ind]
            start_ts = (
                epoch_start_time[data_ind] * self.time_step_width
                + clip_position * self.clip_time * self.time_step_width
            )
            end_ts = (
                epoch_start_time[data_ind] * self.time_step_width
                + (clip_position + 1) * self.clip_time * self.time_step_width
            )

            data_start_ts = data["label_unixtime"][0]
            start_ts += data_start_ts
            end_ts += data_start_ts

            # IMU
            imu, _ = self.segment_and_padding(
                data["imu"], data["imu_unixtime"], start_ts, end_ts, self.cfg.dataset.stream.frame_rate_imu
            )

            # keypoint
            keypoint, _ = self.segment_and_padding(
                data["keypoint"],
                data["keypoint_unixtime"],
                start_ts,
                end_ts,
                self.cfg.dataset.stream.frame_rate_keypoint,
            )

            # e4acc
            e4acc, _ = self.segment_and_padding(
                data["e4acc"],
                data["e4acc_unixtime"],
                start_ts,
                end_ts,
                self.cfg.dataset.stream.frame_rate_e4acc,
            )

            # bbox
            bbox, _ = self.segment_and_padding(
                data["bbox"],
                data["bbox_unixtime"],
                start_ts,
                end_ts,
                self.cfg.dataset.stream.frame_rate_keypoint,
            )

            # ht
            ht, _ = self.segment_and_padding(data["ht"], data["label_unixtime"], start_ts, end_ts, 1)
            ht = np.array(ht)[:, 0]

            # printer
            printer, _ = self.segment_and_padding(
                data["printer"], data["label_unixtime"], start_ts, end_ts, 1
            )
            printer = np.array(printer)[:, 0]

            # label
            label, ts = self.segment_and_padding(data["label"], data["label_unixtime"], start_ts, end_ts, 1)
            label = np.array(label)[:, 0]

            # # kinect_depth
            # if self.cfg.model.kinect_depth_dim != 0:
            #     if self.cfg.model.use_cnn_feature:
            #         kinect_depth, exist_data_kinect_depth = self.segment_and_padding_for_feature(
            #             data["kinect_depth"],
            #             data["kinect_depth_unixtime"],
            #             start_ts,
            #             end_ts,
            #             self.cfg.dataset.stream.frame_rate_kinect_feature,
            #             label_ts=ts,
            #         )
            #     else:
            #         kinect_depth, _, exist_data_kinect_depth = self.segment_and_padding(
            #             data["kinect_depth"],
            #             data["kinect_depth_unixtime"],
            #             start_ts,
            #             end_ts,
            #             self.cfg.dataset.stream.frame_rate_kinect_depth,
            #             c=1,
            #             resize=self.cfg.model.image_size,
            #             label_ts=ts,
            #         )
            # else:  # dummy
            #     kinect_depth = np.zeros((self.clip_time, 1))
            #     exist_data_kinect_depth = np.array([False] * self.clip_time)

            # # rs02_depth
            # if self.cfg.model.rs02_depth_dim != 0:
            #     rs02_depth, _, exist_data_rs02_depth = self.segment_and_padding(
            #         data["rs02_depth"],
            #         data["rs02_depth_unixtime"],
            #         start_ts,
            #         end_ts,
            #         self.cfg.dataset.stream.frame_rate_rs02_depth,
            #         c=3,
            #         resize=self.cfg.model.image_size,
            #         label_ts=ts,
            #     )
            # else:  # dummy
            #     rs02_depth = np.zeros((self.clip_time, 1))
            #     exist_data_rs02_depth = np.array([False] * self.clip_time)

            # if self.cfg.pre_data_aug.use and self.mode == "train":
            #     imu, keypoint, e4acc, kinect_depth, rs02_depth = self.pre_data_aug(
            #         [imu, keypoint, e4acc, kinect_depth, rs02_depth],
            #         data_names=["imu", "keypoint", "e4acc", "kinect_depth", "rs02_depth"],
            #     )

            # padding ts
            ts = np.concatenate(
                [
                    ts,
                    np.arange(
                        ts[-1] + self.time_step_width,
                        ts[-1] + self.time_step_width * (self.clip_time - len(ts) + 1),
                        self.time_step_width,
                    ),
                ]
            )

            output = {
                "imu": imu,
                "keypoint": keypoint,
                "e4acc": e4acc,
                "bbox": bbox,
                "label": label,
                "ts": ts,
                "ht": ht,
                "printer": printer,
                # "kinect_depth": kinect_depth,
                # "rs02_depth": rs02_depth,
                # "exist_data_kinect_depth": exist_data_kinect_depth,
                # "exist_data_rs02_depth": exist_data_rs02_depth,
            }
            yield output

    def segment_and_padding(self, data, unixtime, start_ts, end_ts, fs, c=None, resize=None, label_ts=None):
        _start_ind = np.where(unixtime >= start_ts)[0]
        if len(_start_ind) == 0:
            start_ind = len(unixtime) - 2
            end_ind = len(unixtime) - 1
        else:
            start_ind = _start_ind[0]
            _end_ind = np.where(unixtime <= end_ts)[0]
            if len(_end_ind) == 0:
                end_ind = None
            else:
                end_ind = _end_ind[-1]
                if start_ind == end_ind:
                    end_ind = None

        data = data[start_ind:end_ind]
        unixtime = unixtime[start_ind:end_ind]

        if c != None:
            if self.cfg.model.use_only_one_image:
                use_ind = []
                for i in range(len(label_ts)):
                    _ind = np.where(unixtime > label_ts[i])[0]
                    if len(_ind) != 0:
                        use_ind.append(_ind[0])
                    else:
                        use_ind.append(-1)
                data = np.array(data)[use_ind]
                unixtime = np.array(unixtime)[use_ind]
                fs = 1
            if self.cfg.dataload.pre_image and c == 1:  # only kinect
                data, exist_data = self._prepro_image(data)
            else:
                data, exist_data = self._load_image(data, c, resize)
            data = self._segment_and_padding(data, unixtime, fs)
            exist_data = self._segment_and_padding(exist_data, unixtime, fs)
            return data, unixtime, exist_data

        data = self._segment_and_padding(data, unixtime, fs)
        return data, unixtime

    def segment_and_padding_for_feature(self, data, unixtime, start_ts, end_ts, fs, label_ts):
        assert data is not None

        seg_data = []
        seg_ts = []
        exist_data = []
        for i in range(len(label_ts)):
            _ind = np.where(unixtime >= label_ts[i])[0]
            if len(_ind) == 0:
                if self.cfg.model.use_mean_image:
                    seg_data.append(self.mean_feature)
                else:
                    seg_data.append(np.zeros_like(data[0]))  # ゼロベクトルを使用
                exist_data.append(False)
            else:
                if np.all(data[_ind[0]] == 0):  # 存在しないデータなら
                    if self.cfg.model.use_mean_image:
                        seg_data.append(self.mean_feature)
                    else:
                        seg_data.append(data[_ind[0]])  # そのままゼロベクトルを使用
                    exist_data.append(False)
                else:
                    seg_data.append(data[_ind[0]])
                    exist_data.append(True)
                seg_ts.append(unixtime[_ind[0]])
        seg_data = np.array(seg_data)
        exist_data = np.array(exist_data)

        # TODO seg_dataの中にallゼロの2048ベクトルがあったら、exist_data　＆　self.mean_feature代入

        if len(seg_data) < self.clip_time:
            seg_data = np.concatenate(
                [seg_data, np.repeat(seg_data[[-1]], self.clip_time - len(seg_data), 0)]
            )
            exist_data = np.concatenate(
                [exist_data, np.repeat(exist_data[[-1]], self.clip_time - len(exist_data), 0)]
            )

        return seg_data, exist_data

    def _load_image(self, data, c, resize):
        image_data = []
        exist_data = []

        for path in data:
            if path == None:
                if self.cfg.model.use_mean_image:
                    if c == 1:
                        mean_path = Path(
                            self.cfg.dataset.stream.mean_image_kinect_depth.dir,
                            self.cfg.dataset.stream.mean_image_kinect_depth.fname,
                        )
                    elif c == 3:
                        mean_path = Path(
                            self.cfg.dataset.stream.mean_image_rs02_depth.dir,
                            self.cfg.dataset.stream.mean_image_rs02_depth.fname,
                        )
                    im = Image.open(mean_path)
                    # im = crop_max_square(im)  crop済み
                    if resize != 224:
                        im = im.resize((resize, resize))
                    im = np.array(im)
                    if c == 3:  # RGB rs02
                        im = im / 255
                    elif c == 1:  # kinect
                        im = (im - self.cfg.dataset.stream.min_value_kinect_depth) / (
                            self.cfg.dataset.stream.max_value_kinect_depth
                            - self.cfg.dataset.stream.min_value_kinect_depth
                        )
                    im = (im - 0.5) / 0.5
                    if im.ndim == 2:
                        im = im[:, :, np.newaxis]
                    im = im.transpose(2, 0, 1)
                    image_data.append(im)
                else:
                    image_data.append(np.zeros((c, resize, resize)))
                exist_data.append(False)
            else:
                # resize時に読めないファイルは排除済み
                # try:
                #     im = Image.open(path)
                # except PIL.UnidentifiedImageError:
                #     with open("cannot_read.txt", "a") as f:
                #         f.write(str(path) + "\n")
                #     image_data.append(np.zeros((c, resize, resize)))
                #     exist_data.append(False)
                #     continue
                im = Image.open(path)
                # im = crop_max_square(im)  crop済み
                if resize != 224:
                    im = im.resize((resize, resize))
                im = np.array(im)
                if c == 3:  # RGB rs02
                    im = im / 255
                    im = (im - 0.5) / 0.5
                elif c == 1:  # kinect
                    im = (im - self.cfg.dataset.stream.min_value_kinect_depth) / (
                        self.cfg.dataset.stream.max_value_kinect_depth
                        - self.cfg.dataset.stream.min_value_kinect_depth
                    )
                im = (im - 0.5) / 0.5
                if im.ndim == 2:
                    im = im[:, :, np.newaxis]
                im = im.transpose(2, 0, 1)
                image_data.append(im)
                exist_data.append(True)
        image_data = np.array(image_data)
        exist_data = np.array(exist_data)
        assert image_data.dtype != np.object

        return image_data, exist_data

    def _prepro_image(self, data):
        image_data = []
        exist_data = []

        for _data in data:
            if _data is None:
                if self.cfg.model.use_mean_image:
                    im = self.mean_kinect_depth
                    im = (im - self.cfg.dataset.stream.min_value_kinect_depth) / (
                        self.cfg.dataset.stream.max_value_kinect_depth
                        - self.cfg.dataset.stream.min_value_kinect_depth
                    )  # only kinect
                    im = (im - 0.5) / 0.5
                    if im.ndim == 2:
                        im = im[:, :, np.newaxis]
                    im = im.transpose(2, 0, 1)
                    image_data.append(im)
                else:
                    im = self.mean_kinect_depth
                    im = (im - 0.5) / 0.5
                    if im.ndim == 2:
                        im = im[:, :, np.newaxis]
                    im = im.transpose(2, 0, 1)
                    image_data.append(np.zeros_like(im))
                exist_data.append(False)
            else:
                im = _data
                im = (im - self.cfg.dataset.stream.min_value_kinect_depth) / (
                    self.cfg.dataset.stream.max_value_kinect_depth
                    - self.cfg.dataset.stream.min_value_kinect_depth
                )  # only kinect
                im = (im - 0.5) / 0.5
                if im.ndim == 2:
                    im = im[:, :, np.newaxis]
                im = im.transpose(2, 0, 1)
                image_data.append(im)
                exist_data.append(True)
        image_data = np.array(image_data)
        exist_data = np.array(exist_data)
        assert image_data.dtype != np.object

        return image_data, exist_data

    def _segment_and_padding(self, data, ts, fs):
        data_segment = []
        num_data = int(self.time_step_width / 1000 * fs)
        if num_data == 0:  # for label's unixtime
            num_data = 1
        for i in range(self.clip_time):
            _segment_start = np.where(ts >= ts[0] + (i * self.time_step_width))
            if len(_segment_start[0]) == 0:
                segment_start = ts.shape[0] - 1
            else:
                segment_start = _segment_start[0][0]
            segment_end = segment_start + num_data
            if segment_end >= len(data):
                # padding with last data
                last_data = data[-1]
                len_repeat = num_data - len(data[segment_start:])
                # if len_repeat < 0:
                #     segment_end = segment_start + num_data
                #     data_segment.append(data[segment_start:segment_end])
                # else:
                repeat_data = np.repeat([last_data], len_repeat, axis=0)
                data_segment.append(np.concatenate([data[segment_start:], repeat_data]))
                for j in range(i + 1, self.clip_time):
                    len_repeat = num_data
                    repeat_data = np.repeat([last_data], len_repeat, axis=0)
                    data_segment.append(repeat_data)
                break
            else:
                assert segment_end - segment_end <= fs
            data_segment.append(data[segment_start:segment_end])

        data_segment = np.array(data_segment)
        assert len(data_segment) == self.clip_time
        assert data_segment.shape[1] == num_data

        return np.array(data_segment)

    def pre_data_aug(self, x_list, data_names):
        if self.cfg.pre_data_aug.rotate != 0:
            max_ang = self.cfg.pre_data_aug.rotate
            ang = 2 * max_ang * torch.rand(1)[0] - max_ang
            ang = np.rint(ang.numpy()).astype(np.int)
            t = np.deg2rad(ang)
            R = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])

            for i, name in enumerate(data_names):
                if name == "keypoint":
                    x_list[i] = np.matmul(x_list[i].transpose(0, 1, 3, 2), R).transpose(0, 1, 3, 2)

                elif name in ["kinect_depth", "rs02_depth"]:
                    if x_list[i].ndim == 2:  # rs02 dummy
                        continue
                    p, f, c, h, w = x_list[i].shape
                    M = cv2.getRotationMatrix2D((w / 2, h / 2), ang, 1)
                    for _p in range(p):
                        for _f in range(f):
                            tmp = cv2.warpAffine(x_list[i][_p, _f].transpose(1, 2, 0), M, (h, w))
                            if tmp.ndim == 2:
                                tmp = tmp[:, :, np.newaxis]
                            tmp = tmp.transpose(2, 0, 1)
                            x_list[i][_p, _f] = tmp

        if self.cfg.pre_data_aug.shift != 0:
            max_shift = self.cfg.pre_data_aug.shift
            shift_x = 2 * max_shift * torch.rand(1)[0] - max_shift
            shift_x = np.rint(shift_x.numpy()).astype(np.int)
            shift_y = 2 * max_shift * torch.rand(1)[0] - max_shift
            shift_y = np.rint(shift_y.numpy()).astype(np.int)
            for i, name in enumerate(data_names):
                if name == "keypoint":
                    im_size = self.cfg.model.image_size
                    shift_x_per = shift_x / im_size
                    shift_y_per = shift_y / im_size
                    x_list[i][:, :, 0, :] += shift_x_per * 2  # xyの順番合ってるか怪しい
                    x_list[i][:, :, 1, :] += shift_y_per * 2
                elif name in ["kinect_depth", "rs02_depth"]:
                    if x_list[i].ndim == 2:  # rs02 dummy
                        continue
                    p, f, c, h, w = x_list[i].shape
                    M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
                    for _p in range(p):
                        for _f in range(f):
                            tmp = cv2.warpAffine(x_list[i][_p, _f].transpose(1, 2, 0), M, (h, w))
                            if tmp.ndim == 2:
                                tmp = tmp[:, :, np.newaxis]
                            tmp = tmp.transpose(2, 0, 1)
                            x_list[i][_p, _f] = tmp

        if self.cfg.pre_data_aug.flip_p != 0:
            if torch.rand(1)[0] < self.cfg.pre_data_aug.flip_p:
                for i, name in enumerate(data_names):
                    if name == "imu":
                        p, f, d = x_list[i].shape
                        x_list[i] = (
                            x_list[i]
                            .reshape(p, f, 4, self.cfg.dataset.stream.imu_dim // 4)[:, :, [1, 0, 3, 2]]
                            .reshape(p, f, d)
                        )
                    if name == "keypoint":
                        x_list[i][:, :, 0, :] *= -1
                    elif name in ["kinect_depth", "rs02_depth"]:
                        if x_list[i].ndim == 2:  # rs02 dummy
                            continue
                        p, f, c, h, w = x_list[i].shape
                        for _p in range(p):
                            for _f in range(f):
                                tmp = cv2.flip(x_list[i][_p, _f].transpose(1, 2, 0), 1)
                                if tmp.ndim == 2:
                                    tmp = tmp[:, :, np.newaxis]
                                tmp = tmp.transpose(2, 0, 1)
                                x_list[i][_p, _f] = tmp

        return x_list
