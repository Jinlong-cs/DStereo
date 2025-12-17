import json
import os
from typing import List

import numpy as np
from aidisdk.model import DeviceMeta
from hatbc.message import LidarFrame, LidarParam, PointCloud
from pyquaternion import Quaternion

from projects.cloudmodel.tools.instance_engine import CloudModelInference

init_cfg = dict(
    aidi_config="projects/cloudmodel/model_service/aidi_configs/lidar_detection/aidi_config_lidar_detection_at128.py",
    run_device=DeviceMeta("gpu", 0),
)

infer_engine = CloudModelInference(
    infer_mode=CloudModelInference.InferenceMode.InferModelLocal,
    init_config=init_cfg,
)


def bin_to_frames(bin_root, attribute_param_file) -> List[LidarFrame]:
    frames = []

    # load attributes
    with open(attribute_param_file, "r") as f:
        calib_dict = json.load(f)

    if "atf_2_chassis" in calib_dict["calibration"]:
        matrix = np.array(
            calib_dict["calibration"]["atf_2_chassis"], dtype=np.float32
        )
        roatation = Quaternion._from_matrix(matrix[:3, :3])
        translation = matrix[:3][:, -1]

    bin_list = os.listdir(bin_root)
    bin_list.sort()
    bin_list = bin_list[123:400:5]

    for bin_file in bin_list:
        if not bin_file.endswith(".bin"):
            continue

        frame = LidarFrame(
            pcl=PointCloud(url=os.path.join(bin_root, bin_file)),
            lidar_param=LidarParam(
                pitch=roatation.yaw_pitch_roll[1],
                yaw=roatation.yaw_pitch_roll[0],
                roll=roatation.yaw_pitch_roll[2],
                x_offset=translation[0],
                y_offset=translation[1],
                z_offset=translation[2],
            ),
        )

        # load pcl data
        if os.path.exists(frame.pcl.url):
            with open(frame.pcl.url, "rb") as f:
                pcl_data = np.fromfile(f, dtype=np.float32)
                pcl_data = np.reshape(pcl_data, (-1, 4))
                frame.pcl.data = pcl_data

        frames.append(frame)

    return frames


if __name__ == "__main__":
    data_root = "/horizon-bucket/SD_Algorithm/08_perception_lidar/03_datasets/HDE_UT263_20230321/UT263_20230321_091211/lidar_front"
    attribute_param_file = "/horizon-bucket/SD_Algorithm/08_perception_lidar/03_datasets/HDE_UT263_20230321/UT263_20230321_091211/attribute.json"  # noqa
    lidar_frames = bin_to_frames(data_root, attribute_param_file)
    res = infer_engine(lidar_frames)
