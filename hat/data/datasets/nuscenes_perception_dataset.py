import os
import random

import cv2
import numpy as np

try:
    from nuscenes import NuScenes
    from pyquaternion import Quaternion
except ImportError:
    NuScenes = None
    Quaternion = None
from torch.utils.data import Dataset

from hat.registry import OBJECT_REGISTRY
from hat.utils.package_helper import require_packages


@OBJECT_REGISTRY.register
class NuScenesPerceptionDataset(Dataset):
    CAMERAS = [
        "CAM_FRONT",
        "CAM_FRONT_RIGHT",
        "CAM_BACK_RIGHT",
        "CAM_BACK",
        "CAM_BACK_LEFT",
        "CAM_FRONT_LEFT",
    ]
    NameMapping = {
        "movable_object.barrier": "barrier",
        "vehicle.bicycle": "bicycle",
        "vehicle.bus.bendy": "bus",
        "vehicle.bus.rigid": "bus",
        "vehicle.car": "car",
        "vehicle.construction": "construction_vehicle",
        "vehicle.motorcycle": "motorcycle",
        "human.pedestrian.adult": "pedestrian",
        "human.pedestrian.child": "pedestrian",
        "human.pedestrian.construction_worker": "pedestrian",
        "human.pedestrian.police_officer": "pedestrian",
        "movable_object.trafficcone": "traffic_cone",
        "vehicle.trailer": "trailer",
        "vehicle.truck": "truck",
    }
    CLASSES = (
        "car",
        "truck",
        "trailer",
        "bus",
        "construction_vehicle",
        "bicycle",
        "motorcycle",
        "pedestrian",
        "traffic_cone",
        "barrier",
    )

    @require_packages(
        "nuscenes",
        "pyquaternion",
        raise_msg="Please `pip3 install nuscenes-devkit pyquaternion`",
    )
    def __init__(
        self,
        version,
        dataroot,
        video_frame=1,
        max_interval=1,
        min_interval=1,
        fix_interval=True,
        tracking=False,
        max_tracking_interval=1,
        classes=None,
        transforms=None,
    ):
        self.nusc = NuScenes(version=version, dataroot=dataroot, verbose=True)
        self.version = version
        self.dataroot = dataroot
        self.video_frame = video_frame
        self.max_interval = max_interval
        self.min_interval = min_interval
        self.fix_interval = fix_interval
        self.tracking = tracking
        self.max_tracking_interval = max_tracking_interval
        if classes is not None:
            self.CLASSES = classes
        self.transforms = transforms
        if self.tracking:
            self.fix_interval = True

    def __len__(self):
        return len(self.nusc.sample)

    def __getitem__(self, index):
        det_index_list = self.get_seq_index(index)
        key_frame_flags = [False] * len(det_index_list)
        key_frame_flags[0] = True
        if self.tracking and len(det_index_list) > 1:
            interval = det_index_list[0] - det_index_list[1]
            track_index = index - interval * (
                int(random.random() * self.max_tracking_interval) + 1
            )
            track_index = max(track_index, 0)
            track_index_list = self.get_seq_index(track_index, interval)
            for idx in track_index_list:
                if idx not in det_index_list:
                    det_index_list.append(idx)
                    key_frame_flags.append(False)
            track_key_frame = key_frame_flags.index(track_index)
            key_frame_flags[track_key_frame] = True

        video_data = []
        for index, flag in zip(det_index_list, key_frame_flags):
            video_data.append(self.get_frame_data(index, flag))

        if self.transforms is not None:
            for i in range(len(video_data)):
                video_data[i] = self.transforms(video_data[i])

        output = video_data[0]
        output["seq_data"] = video_data[1:]
        if self.tracking and len(det_index_list) > 1:
            output["track_key_frame"] = track_key_frame - 1
        return output

    def get_seq_index(self, end_index, interval=None):
        scene_token = self.nusc.sample[end_index]["scene_token"]
        index_list = [end_index]
        interval = int(random.random() * self.max_interval) + self.min_interval
        for _ in range(self.video_frame - 1):
            if index_list[-1] == 0:
                break
            next_index = index_list[-1] - interval
            next_index = max(next_index, 0)
            if scene_token != self.nusc.sample[next_index]["scene_token"]:
                break
            index_list.append(next_index)
            if not self.fix_interval:
                interval = (
                    int(random.random() * self.max_interval)
                    + self.min_interval
                )
        return index_list

    def get_frame_data(self, index, return_anno=True):
        sample = self.nusc.sample[index]
        frame_data = {
            "timestamp": sample["timestamp"] / 1e6,
            "scene_token": sample["scene_token"],
        }
        frame_data.update(self.get_images(index))
        frame_data.update(self.get_all_sensor_pose(index))
        if return_anno and "test" not in self.version:
            frame_data.update(
                self.get_annos(index, frame_data["lidar2global"])
            )
        return frame_data

    def get_images(self, index):
        sample = self.nusc.sample[index]
        images = []
        for cam in self.CAMERAS:
            sample_data = self.nusc.get("sample_data", sample["data"][cam])
            filename = os.path.join(self.dataroot, sample_data["filename"])
            images.append(cv2.imread(filename))
        images = np.stack(images, axis=0)
        return {"img": images}

    def get_annos(self, index, lidar2global):
        sample = self.nusc.sample[index]
        boxes = self.nusc.get_sample_data(sample["data"]["LIDAR_TOP"])[1]
        names = np.array(
            [
                self.NameMapping[b.name]
                if b.name in self.NameMapping
                else b.name
                for b in boxes
            ]
        )
        cat_ids = np.array(
            [
                self.CLASSES.index(name) if name in self.CLASSES else -1
                for name in names
            ]
        )
        locs = np.array([b.center for b in boxes]).reshape(-1, 3)
        dims = np.array([b.wlh for b in boxes]).reshape(-1, 3)
        rots = np.array(
            [b.orientation.yaw_pitch_roll[0] for b in boxes]
        ).reshape(-1, 1)
        velocity = np.array(
            [self.nusc.box_velocity(token) for token in sample["anns"]]
        )
        if len(velocity) > 0:
            velocity = (
                np.linalg.inv(lidar2global[:3, :3]) @ velocity[..., None]
            )
            velocity = velocity.reshape(-1, 3)
            boxes = np.concatenate(
                [locs, dims[:, [1, 0, 2]], rots, velocity], axis=-1
            )
        else:
            boxes = np.zeros((0, 10))

        instance_inds = []
        num_lidar_pts = []
        for ann_token in sample["anns"]:
            ann = self.nusc.get("sample_annotation", ann_token)
            inds = self.nusc.getind("instance", ann["instance_token"])
            instance_inds.append(inds)
            num_lidar_pts.append(ann["num_lidar_pts"])
        instance_inds = np.array(instance_inds)

        mask = np.logical_and(np.array(num_lidar_pts) > 0, cat_ids != -1)
        boxes = boxes[mask]
        names = names[mask]
        instance_inds = instance_inds[mask]
        cat_ids = cat_ids[mask]
        return {
            "boxes": boxes,
            "names": names,
            "instance_inds": instance_inds,
            "category_ids": cat_ids,
        }

    def get_all_sensor_pose(self, index):
        sample = self.nusc.sample[index]
        sample_data = sample["data"]
        lidar2global, lidar2ego = self.get_sensor_pose(
            sample_data["LIDAR_TOP"]
        )

        cam2global_list = []
        cam2ego_list = []
        cam_intrinsic_list = []
        for cam in self.CAMERAS:
            (cam2global, cam2ego, cam_intrinsic) = self.get_sensor_pose(
                sample_data[cam]
            )
            cam2global_list.append(cam2global)
            cam2ego_list.append(cam2ego)
            cam_intrinsic_list.append(cam_intrinsic)
        cam2global = np.stack(cam2global_list)
        cam2ego = np.stack(cam2ego_list)
        cam_intrinsic = np.stack(cam_intrinsic_list)

        lidar2cam = np.linalg.inv(cam2ego) @ lidar2ego

        lidar2img = cam_intrinsic @ lidar2cam[:, :3]
        return {
            "lidar2global": lidar2global,
            "lidar2ego": lidar2ego,
            "cam2global": cam2global,
            "cam2ego": cam2ego,
            "cam_intrinsic": cam_intrinsic,
            "lidar2cam": lidar2cam,
            "lidar2img": lidar2img,
        }

    def get_sensor_pose(self, token):
        sensor_data = self.nusc.get("sample_data", token)
        ego_pose = self.nusc.get("ego_pose", sensor_data["ego_pose_token"])
        ego2global = np.eye(4)
        ego2global[:3, :3] = Quaternion(ego_pose["rotation"]).rotation_matrix
        ego2global[:3, 3] = ego_pose["translation"]

        calib = self.nusc.get(
            "calibrated_sensor", sensor_data["calibrated_sensor_token"]
        )
        sensor2ego = self.get_trans_matrix(
            calib["rotation"], calib["translation"]
        )
        senor2global = ego2global @ sensor2ego
        if sensor_data["sensor_modality"] == "camera":
            cam_intrinsic = np.array(calib["camera_intrinsic"])
            return senor2global, sensor2ego, cam_intrinsic
        return senor2global, sensor2ego

    def get_trans_matrix(self, rotation, translation):
        trans_mat = np.eye(4)
        trans_mat[:3, :3] = Quaternion(rotation).rotation_matrix
        trans_mat[:3, 3] = translation
        return trans_mat
