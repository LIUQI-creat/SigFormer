from dataclasses import dataclass
from typing import Dict, List, Optional

from omegaconf import MISSING
from openpack_toolkit.activity import ActSet


@dataclass
class AllDataStreamConfig:
    @dataclass
    class Paths:
        # path to the root directory of this stream.
        dir: Optional[str] = None
        fname: Optional[str] = None

    schema: str = MISSING
    name: str = MISSING
    description: Optional[str] = None
    super_stream: Optional[str] = None
    path_imu: Paths = MISSING
    path_keypoint: Paths = MISSING
    file_format: Optional[Dict] = None
    frame_rate_imu: int = MISSING  # [Hz, fps]
    frame_rate_keypoint: int = MISSING  # [Hz, fps]
    path_e4acc: Paths = MISSING
    frame_rate_e4acc: int = MISSING
    # devices_e4acc
    path_e4bvp: Paths = MISSING
    frame_rate_e4bvp: int = MISSING
    # device_e4bvps
    path_e4eda: Paths = MISSING
    frame_rate_e4eda: int = MISSING
    # devices_e4eda
    path_e4temp: Paths = MISSING
    frame_rate_e4temp: int = MISSING
    # devices_e4temp
    path_ht: Paths = MISSING
    frame_rate_ht: int = MISSING
    path_order: Paths = MISSING
    frame_rate_order: int = MISSING


ATR_ACC_STREAM = AllDataStreamConfig(
    schema="AllConfig",
    name="all_imu_keypoint",
    super_stream="None",
    # imu
    path_imu=AllDataStreamConfig.Paths(
        dir="${path.openpack.rootdir}/${user.name}/atr/${device}",
        fname="${session}.csv",
    ),
    frame_rate_imu=30,
    devices_imu=["atr01", "atr02", "atr03", "atr04"],
    acc=True,
    gyro=False,
    quat=False,
    # keypoint
    path_keypoint=AllDataStreamConfig.Paths(
        dir="${path.openpack.rootdir}/${user.name}/kinect/${..category}/${..model}/single",
        fname="${session}.json",
    ),
    frame_rate_keypoint=15,
    category="2d-kpt",
    model="mmpose-hrnet-w48-posetrack18-384x288-posewarper-stage2",
    nodes={
        0: "nose",
        1: "left_eye",
        2: "right_eye",
        3: "left_ear",
        4: "right_ear",
        5: "left_shoulder",
        6: "right_shoulder",
        7: "left_elbow",
        8: "right_elbow",
        9: "left_wrist",
        10: "right_wrist",
        11: "left_hip",
        12: "right_hip",
        13: "left_knee",
        14: "right_knee",
        15: "left_ankle",
        16: "right_ankle",
    },
    # e4acc
    path_e4acc=AllDataStreamConfig.Paths(
        dir="${path.openpack.rootdir}/${user.name}/e4/${device}/acc",
        fname="${session}.csv",
    ),
    frame_rate_e4acc=32,
    devices_e4acc=["e401", "e402"],
    # e4bvp
    path_e4bvp=AllDataStreamConfig.Paths(
        dir="${path.openpack.rootdir}/${user.name}/e4/${device}/bvp",
        fname="${session}.csv",
    ),
    frame_rate_e4bvp=64,
    devices_e4bvp=["e401", "e402"],
    # e4eda
    path_e4eda=AllDataStreamConfig.Paths(
        dir="${path.openpack.rootdir}/${user.name}/e4/${device}/eda",
        fname="${session}.csv",
    ),
    frame_rate_e4eda=4,
    devices_e4eda=["e401", "e402"],
    # e4temp
    path_e4temp=AllDataStreamConfig.Paths(
        dir="${path.openpack.rootdir}/${user.name}/e4/${device}/temp",
        fname="${session}.csv",
    ),
    frame_rate_e4temp=4,
    devices_e4temp=["e401", "e402"],
    # ht
    path_ht=AllDataStreamConfig.Paths(
        dir="${path.openpack.rootdir}/${user.name}/system/ht",
        fname="${session}.csv",
    ),
    frame_rate_ht=-1,
    # order
    path_order=AllDataStreamConfig.Paths(
        dir="${path.openpack.rootdir}/${user.name}/system/order-sheet/",
        fname="${session}.csv",
    ),
    frame_rate_order=-1,
)
