"""``dataloader`` provide utility function to load files saved in OpenPack dataset format.
"""
import json
import os
from logging import getLogger
from pathlib import Path
from typing import List, Tuple, Union

import numpy as np
import pandas as pd
from openpack_toolkit.activity import ActSet
from PIL import Image


def load_and_resample_scan_log(
    path: Path,
    unixtimes_ms: np.ndarray,
) -> np.ndarray:
    """Load scan log data such as HT, and make binary vector for given timestamps.
    Elements that have the same timestamp in second precision are marked as 1.
    Other values are set to 0.

    Args:
        path (Path): path to a scan log CSV file.
        unixtimes_ms (np.ndarray):  unixtime seqeuence (milli-scond precision).
            shape=(T,).

    Returns:
        np.ndarray: binary 1d vector.
    """
    assert unixtimes_ms.ndim == 1
    df = pd.read_csv(path)

    unixtimes_sec = unixtimes_ms // 1000

    X_log = np.zeros(len(unixtimes_ms)).astype(np.int32)
    for utime_ms in df["unixtime"].values:
        utime_sec = utime_ms // 1000
        ind = np.where(unixtimes_sec == utime_sec)[0]
        X_log[ind] = 1

    return X_log


def load_e4acc(
    paths: Union[Tuple[Path, ...], List[Path]],
    th: int = 30,
) -> Tuple[np.ndarray, np.ndarray]:
    """Load e4acc data from CSVs.

    Args:
        paths (Union[Tuple[Path, ...], List[Path]]): list of paths to target CSV.
            (e.g., [**/atr01/S0100.csv])
        th (int, optional): threshold of timestamp difference [ms].
            Default. 30 [ms] (<= 1 sample)
    Returns:
        Tuple[np.ndarray, np.ndarray]: unixtime and loaded sensor data.
    """
    assert isinstance(
        paths, (tuple, list)
    ), f"the first argument `paths` expects tuple of Path, not {type(paths)}."

    channels = ["acc_x", "acc_y", "acc_z"]

    ts_ret, x_ret, ts_list = None, [], []
    for path in paths:
        df = pd.read_csv(path)
        if df.empty:
            return None, None
        assert set(channels) < set(df.columns)

        ts = df["time"].values
        x = df[channels].values.T

        ts_list.append(ts)
        x_ret.append(x)

    min_len = min([len(ts) for ts in ts_list])
    ts_ret = None
    for i in range(len(paths)):
        x_ret[i] = x_ret[i][:, :min_len]
        ts_list[i] = ts_list[i][:min_len]

        if ts_ret is None:
            ts_ret = ts_list[i]
        else:
            # Check whether the timestamps are equal or not.
            delta = np.abs(ts_list[i] - ts_ret)
            assert delta.max() < th, (
                f"max difference is {delta.max()} [ms], " f"but difference smaller than th={th} is allowed."
            )

    x_ret = np.concatenate(x_ret, axis=0)
    return ts_ret, x_ret


def load_depth(
    csv_path: Path,
) -> Tuple[np.ndarray, np.ndarray]:
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        unixtime = df["unixtime"].values
        data_path = df["path"].values
        absolute_data_path = []
        for p in data_path:
            absolute_data_path.append(csv_path.parent / csv_path.stem / Path(p))
        absolute_data_path = np.array(absolute_data_path)

        return unixtime, absolute_data_path
    else:
        return None, None


def load_feature(
    csv_path: Path,
    load_all,
) -> Tuple[np.ndarray, np.ndarray]:
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        unixtime = df["unixtime"].values
        data_path = df["path"].values
        absolute_data_path = []
        for p in data_path:
            absolute_data_path.append(csv_path.parent / csv_path.stem / Path(p))
        absolute_data_path = np.array(absolute_data_path)

        data = []
        for path in absolute_data_path:
            data.append(np.load(path).squeeze())
        data = np.array(data)

        return unixtime, data
    else:
        return None, None


def load_feature_all(
    path_all_unixtime: Path,
    path_all_feat: Path,
) -> Tuple[np.ndarray, np.ndarray]:
    if os.path.exists(path_all_unixtime):
        unixtime = np.load(path_all_unixtime)
        data = np.load(path_all_feat)
        data = data.squeeze()

        return unixtime[:, 0], data
    else:
        return None, None


def load_mean_feature(
    path,
):
    data = np.load(path).squeeze()

    return data


def pre_load_image(x_sess_kinect_depth, all_image_path, resize, min_value=0, max_value=13000):
    if os.path.exists(all_image_path):
        data = np.load(all_image_path)
        return data
    else:
        if x_sess_kinect_depth is None:
            return None
        else:
            all = []
            for path in x_sess_kinect_depth:
                im = Image.open(path)
                if resize != 224:
                    im = im.resize((resize, resize))
                im = np.array(im)
                im = im.clip(min_value, max_value)

                all.append(im.astype(np.int16))
            all = np.array(all)
            np.save(all_image_path, all)

            return all


def load_mean_image(path, resize, min_value=0, max_value=1300):
    im = Image.open(path)
    if resize != 224:
        im = im.resize((resize, resize))
    im = np.array(im)
    im = im.clip(min_value, max_value)
    return im.astype(np.int16)


def load_bbox(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Load keypoints from JSON.

    Args:
        path (Path): path to a target JSON file.
    Returns:
        Tuple[np.ndarray, np.ndarray]:
            * T (np.ndarray): unixtime for each frame.
            * X (np.ndarray): xy-cordinates of keypoints. and the score of corresponding
                prediction. shape=(3, FRAMES, NODE). The first dim is corresponding to
                [x-cordinate, y-cordinate, score].
    Todo:
        * Handle the JSON file that contains keypoints from multiple people.
    """
    with open(path, "r") as f:
        data = json.load(f)

    T, X = [], []
    for i, d in enumerate(data["annotations"][:]):
        ut = d.get("image_id", -1)
        kp = np.array(d.get("bbox", []))

        X.append(kp.T)
        T.append(ut)

    T = np.array(T)
    X = np.stack(X, axis=1)

    return T, X
