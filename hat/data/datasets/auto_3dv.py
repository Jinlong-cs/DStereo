# Copyright (c) Horizon Robotics. All rights reserved.

import json
import logging
import math
import os
import pickle
from typing import Mapping, Optional, Sequence, Union

import numpy as np
import timeout_decorator
import torch.utils.data as data
from PIL import Image

try:
    from pycocotools import mask as coco_mask
except ImportError:
    coco_mask = None

from hat.core.bev_elevation_utils import HOMO_PAD_VALUE, reformat_homo_info
from hat.data.datasets.auto_3dv_packer import FileList2LMDB
from hat.data.datasets.bev import HomoGenerator
from hat.data.datasets.e2e_dynamic_dataset import DatumParserE2E
from hat.data.datasets.img_dataset import ImgLmdbDataset, NamedIndexDatasetV2
from hat.data.datasets.lidar_3d import PointCloudLmdbDataset
from hat.data.datasets.split_dataset import find_index
from hat.data.datasets.utils import Real3DRecReader
from hat.data.utils import pil_loader
from hat.registry import OBJECT_REGISTRY
from hat.utils.apply_func import _as_list
from hat.utils.filesystem import join_path
from hat.utils.pack_type.lmdb import Lmdb, LmdbReadList

try:
    from horizon_driving_dataset import DatasetReader, PoseTransformer
except ImportError:
    PoseTransformer = None

__all__ = ["ANCAuto3DV", "ANCBev3DDatasetRec"]

logger = logging.getLogger(__name__)

BUCKET_ROOT = "/horizon-bucket"


class SyncInfo(object):
    """
    A SyncInfo object desribes all sample`s sync info of a dataset.

    It can extract sync infomation from a text file or a lmdb file.
    The format of one sample`s sync infomation is explain in __getitem__ func.

    num_max_frames: 时序数据打包过程中，每个sync_info会打包一定数目的冗余帧数，在实际
        读取数据时，希望只使用特定数目的帧数。默认为None，不对数据进行额外处理。
    num_frames_per_iter: 时序数据长度往往较大，希望将每个sample的时序数据进行拆分，
        每个iter只输出特定数目的帧数，需要和num_max_frames配合使用。默认为None，
        不对数据进行额外处理。
        num_max_frames与num_frames_per_iter适用于端到端等时序任务训练：
        e.g.
            dataset 一个sample中SyncInfo的长度是12，dataset共有50个sample，
            sample_interval = 2，
            num_max_frames = 9，
            num_frames_per_iter = 3，
            此时SyncInfo的长度为 (9 / 3) * (50 / 2) = 75.
            SyncInfo[0]返回第一个iter，它包含了第一个sample中1-3帧的数据，
            SyncInfo[1]返回第二个iter，它包含了第一个sample中4-6帧的数据，
            SyncInfo[2]返回第三个iter，它包含了第一个sample中7-9帧的数据，
            SyncInfo[3]返回第四个iter，它包含了第三个sample中1-3帧的数据。
        默认情况下，num_max_frames和num_frames_per_iter都为None，适用于单帧数据格式的
        情况，此时不对数据进行额外处理。
        e.g.:
            sample_interval = 2，
            num_max_frames = None，
            num_frames_per_iter = None，
            SyncInfo[0]返回第一个iter，它包含了第一个sample的数据，
            SyncInfo[1]返回第二个iter，它包含了第三个sample的数据，
            SyncInfo[2]返回第三个iter，它包含了第五个sample的数据。
    reverse：将读取出来的sync_info信息逆序排列。期望sync_file中的顺序是:(t,t-1,t-2,...)，
        如果是相反顺序，可以设置reverse=False进行反转.
    fill_fake_temporal_data：sync_info的时序长度小于设置的num_max_frames时，
        需要设置fill_fake_temporal_data=True，用历史帧的sync_info进行填充
    reverse_select: 从clip的尾部选择帧.
    """

    def __init__(
        self,
        root: str,
        sync_file: Optional[Sequence[str]] = None,
        sync_file_lmdb: Optional[Sequence[str]] = None,
        sample_interval: Union[int, Sequence[int]] = 1,
        num_max_frames: Optional[int] = None,
        num_frames_per_iter: Optional[int] = None,
        reverse: bool = False,
        fill_fake_temporal_data: bool = False,
        reverse_select: bool = False,
    ):

        self.sample_interval = _as_list(sample_interval)
        self.reverse = reverse
        self.fill_fake_temporal_data = fill_fake_temporal_data

        assert (
            sync_file is not None or sync_file_lmdb is not None
        ), "please provede sync_file or sync_file_lmdb!"
        self.use_lmdb = sync_file_lmdb is not None

        self.sync_file_lmdb = _as_list(sync_file_lmdb)
        self.sync_file = _as_list(sync_file)

        # 如果num_max_frames和num_frames_per_iter是None，不进行额外处理。
        # 只有给定了num_max_frames后，num_frames_per_iter才有效，并对sample进行划分。
        assert (num_max_frames is None and num_frames_per_iter is None) or (
            num_max_frames is not None and num_frames_per_iter is not None
        )
        self.num_max_frames = num_max_frames
        self.num_frames_per_iter = num_frames_per_iter

        if self.num_frames_per_iter is not None:
            assert self.num_max_frames is not None
            self.split_num_per_sample = (
                self.num_max_frames // self.num_frames_per_iter
            )
        else:
            self.split_num_per_sample = None

        self.num_samples = []
        self.resample_num_samples = []

        if self.use_lmdb:
            assert len(self.sample_interval) == len(self.sync_file_lmdb)
            self.lmdb = []
            for file in self.sync_file_lmdb:
                cur_lmdb = Lmdb(
                    file,
                    False,
                    True,
                    readonly=True,
                    map_size=1024 * 10,
                )
                self.lmdb.append(cur_lmdb)
                self.num_samples.append(len(cur_lmdb))
        else:
            assert len(self.sample_interval) == len(self.sync_file)
            self.items = []
            for file in self.sync_file:
                cur_items = self.readlines(file)
                self.items.append(cur_items)
                self.num_samples.append(len(cur_items))

        for num, interval in zip(self.num_samples, self.sample_interval):
            self.resample_num_samples.append(math.ceil(num / interval))
        # 如果进行数据划分，则将样本数目扩充self.split_num_per_sample倍。遵循，先
        # 进行interval sampling，再进行split的顺序。
        if self.split_num_per_sample is not None:
            for i in range(len(self.resample_num_samples)):
                self.resample_num_samples[i] *= self.split_num_per_sample

        self.root = root
        self.reverse_select = reverse_select

    def readlines(self, filename):
        """Read all the lines in a text file and return as a list."""
        with open(filename, "r") as f:
            lines = f.read().splitlines()
        return lines

    def __len__(self):
        return sum(self.resample_num_samples)

    def __getitem__(self, idx: int) -> dict:  # noqa: D205,D400

        # calculate true index before repeating.
        part_idx, offset = find_index(self.resample_num_samples, idx)

        # 如果self.split_num_per_sample 不为None，根据该值计算当时offset所在的
        # sample index 以及 split sample index。

        if self.split_num_per_sample is None:
            sample_split_index = None
            sample_info = self._getitem(
                part_idx, offset * self.sample_interval[part_idx]
            )
            if self.reverse:
                sample_info = sample_info[::-1]
        else:
            sample_split_index = offset % self.split_num_per_sample
            # sample_split_index表示sub-clip在大clip的编号
            real_sample_split_index = (
                -sample_split_index + self.split_num_per_sample - 1
            )
            # 由于sync_info的时序帧是倒序排列[t,t-1,t-2,...]，需要sub_clip的编号转化为真正用于索引的编号  # noqa

            offset = offset // self.split_num_per_sample
            sample_info = self._getitem(
                part_idx, offset * self.sample_interval[part_idx]
            )
            if len(sample_info) < self.num_max_frames:
                # 如果sync_info中的时序长度小于指定的最大长度，那么必须设置fill_fake_temporal_data
                assert self.fill_fake_temporal_data, f"{self.sync_file_lmdb}"
                sample_info = sample_info + [sample_info[-1]] * (
                    self.num_max_frames - len(sample_info)
                )

            if not self.reverse_select:
                sample_info = sample_info[: self.num_max_frames]
            else:
                sample_info = sample_info[-self.num_max_frames :]
            if self.reverse:
                sample_info = sample_info[::-1]

            if self.split_num_per_sample is not None:
                begin_idx = real_sample_split_index * self.num_frames_per_iter
                end_idx = (
                    real_sample_split_index + 1
                ) * self.num_frames_per_iter
                # 在时序训练的场景中，往往需要多返回一帧信息(主要是为了计算姿态)
                if sample_split_index == 0:  # 表示当前sub_clip是大clip的第一个sub_clip
                    sample_info = sample_info[begin_idx:end_idx]
                    sample_info.append(sample_info[-1])  # 将最后一帧额外加入返回的list
                else:
                    sample_info = sample_info[
                        begin_idx : end_idx + 1
                    ]  # 额外多返回一帧
        return sample_info, sample_split_index

    def _getitem(
        self, part_index: int, offset: int
    ) -> dict:  # noqa: D205,D400
        """
        Return one sample contains multiple frame`s sync info as a list of
        dict organized like this:
        [
            # first frame
            {
                pack_dir: 'pack_dir_relative_to_dataset_root',
                lidar_top: 'xxx',
                camera_front: 'xxx',
                camera_front_left: 'xxx',
                camera_front_right: 'xxx',
                camera_rear: 'xxx',
                camera_rear_left: 'xxx',
                camera_rear_right: 'xxx',
                bev_seg: 'bev_seg/xxx',
                bev_3d: 'bev_3d/xxx',
                bev_motion_flow: 'bev_motion_flow/xxx',
                ......
            },

            # second frame
            {
                pack_dir: 'pack_dir_relative_to_dataset_root',
                lidar_top: 'xxx',
                camera_front: 'xxx',
                camera_front_left: 'xxx',
                camera_front_right: 'xxx',
                camera_rear: 'xxx',
                camera_rear_left: 'xxx',
                camera_rear_right: 'xxx',
                bev_seg: 'bev_seg/xxx',
                bev_3d: 'bev_3d/xxx',
                bev_motion_flow: 'bev_motion_flow/xxx',
                ......
            },
            ......
        ]
        """
        if self.use_lmdb:
            one_item = self.lmdb[part_index].read(offset).decode()
        else:
            one_item = self.items[part_index][offset]

        cur_sample = json.loads(one_item)
        """ cur_sample`s format is like this:
        {
            pack_dir: 'xxx',
            camera_front: [xxx,xxx,...]
            camera_front_left: [xxx,xxx,...]
            ...
        }
        """

        sync_infos = []
        pack_dir = cur_sample.pop("pack_dir")
        root = self.root
        if "root" in cur_sample:
            root = cur_sample.pop("root")
            root = os.path.join(BUCKET_ROOT, _as_list(root)[0])
        key = list(cur_sample.keys())[0]
        for i in range(len(cur_sample[key])):
            sync_info = {"pack_dir": pack_dir, "root": root}
            for key in cur_sample:
                if isinstance(cur_sample[key], (list, tuple)):
                    sync_info[key] = cur_sample[key][i]
                elif isinstance(cur_sample[key], (str, bool)):
                    sync_info[key] = cur_sample[key]
                else:
                    raise NotImplementedError(
                        f"SyncInfo cannot support type: \
                        {type(cur_sample[key])}..."
                    )
            sync_infos.append(sync_info)
        return sync_infos


class HDELoader(object):
    """
    A loader to load image/pose from lmdb.

    打包数据的说明在：https://horizonrobotics.feishu.cn/wiki/wikcnlzXtDwElferLlkbx9mnBNe

    """

    def __init__(self, HDE_data_path: str) -> None:
        self.HDE_data_path = HDE_data_path

        basename = os.path.basename(os.path.normpath(HDE_data_path))
        assert basename in ["site_data", "park_data", "lmdb_pack"]

        self.is_site_data = basename == "site_data"

        self.img_lmdb_loader_dict = {}
        self.liosam_lmdb_loader_dict = {}
        self.wheel_lmdb_loader_dict = {}

        self.img_key_dict = {}
        self.liosam_key_dict = {}
        self.wheel_key_dict = {}

    def _get_lmdb_key_set(self, key_dict, lmdb_path, lmdb_name):
        split_path = os.path.split(lmdb_path)

        for sub_dir in split_path:
            if sub_dir not in key_dict:
                key_dict[sub_dir] = {}
            key_dict = key_dict[sub_dir]

        if lmdb_name not in key_dict:
            full_path = os.path.join(self.HDE_data_path, lmdb_path, lmdb_name)
            txt_path = full_path + ".txt"

            if not os.path.exists(full_path) or not os.path.exists(txt_path):
                key_dict[lmdb_name] = None
            else:
                with open(txt_path, "r") as f:
                    lines = f.read().splitlines()
                key_dict[lmdb_name] = set(lines)
        key_set = key_dict[lmdb_name]
        return key_set

    def _get_lmdb_loader(
        self,
        lmdb_loader_dict: Mapping,
        lmdb_path: str,
        lmdb_name: str,
        lmdb_class: Union[Lmdb, ImgLmdbDataset] = Lmdb,
        **lmdb_kwargs,
    ):
        split_path = os.path.split(lmdb_path)

        for sub_dir in split_path:
            if sub_dir not in lmdb_loader_dict:
                lmdb_loader_dict[sub_dir] = {}
            lmdb_loader_dict = lmdb_loader_dict[sub_dir]

        if lmdb_name not in lmdb_loader_dict:
            full_path = os.path.join(self.HDE_data_path, lmdb_path, lmdb_name)
            if not os.path.exists(full_path):
                raise FileExistsError(full_path)

            txt_path = full_path + ".txt"
            if not os.path.exists(txt_path):
                raise FileExistsError(txt_path)

            lmdb_loader_dict[lmdb_name] = lmdb_class(
                full_path, map_size=1024 * 10, **lmdb_kwargs
            )

        lmdb_loader = lmdb_loader_dict[lmdb_name]
        return lmdb_loader

    def load_img(self, *img_paths):
        """
        Return a ndarray image with given paths.

        NOTE: 请确保提供的img_paths和HDE_data_path能够拼凑出待加载图像的完整路径
        对于site 格式数据，你提供的img_paths至少要有5级
        e.g. FSD_Site_NC109_20221212/Site_121_34973_31_45059_1/NC109_20221212_182822/camera_front/1670840902300.jpg  # noqa
        对于park 格式数据，你提供的img_paths至少要有4级
        e.g. UT0Q9_20230215_D/20230215-122546_955/camera_front_right/1676435147306.jpg  # noqa

        """
        relative_dir, img_key = self._parse_path(*img_paths)
        view_name = img_key.split("/")[-2]
        lmdb_loader = self._get_lmdb_loader(
            self.img_lmdb_loader_dict, relative_dir, view_name, ImgLmdbDataset
        )
        return lmdb_loader[img_key]

    def check_img(self, *img_paths):
        relative_dir, img_key = self._parse_path(*img_paths)
        view_name = img_key.split("/")[-2]
        key_set = self._get_lmdb_key_set(
            self.img_key_dict, relative_dir, view_name
        )
        return key_set is not None and img_key in key_set

    def load_liosam_pose(self, *img_paths):
        """

        返回4x4的vcs姿态矩阵,这个姿态的含义是: 从当前时间戳的vcs坐标系到世界坐标系的姿态.

        NOTE:
        1.这个vcs姿态是从liosam姿态转换得到
        2.在实际打包的姿态中，key都是以“前视”的时间戳生成的，因此必须保证img_paths必须包含"camera_front"字段  # noqa
        3.对于site 格式数据，你提供的img_paths至少要有5级
          e.g. FSD_Site_NC109_20221212/Site_121_34973_31_45059_1/NC109_20221212_182822/camera_front/1670840902300.jpg  # noqa
          对于park 格式数据，你提供的img_paths至少要有4级
          e.g. UT0Q9_20230215_D/20230215-122546_955/camera_front/1676435147306.jpg  # noqa

        """
        relative_dir, img_key = self._parse_path(*img_paths)
        lmdb_name = "liosam_pose"
        lmdb_loader = self._get_lmdb_loader(
            self.liosam_lmdb_loader_dict,
            relative_dir,
            lmdb_name,
            Lmdb,
            writable=False,
        )
        pose_byte = lmdb_loader.read(img_key)
        pose = np.frombuffer(pose_byte, "float32").reshape((4, 4))
        return pose

    def check_liosam_pose(self, *img_paths):
        relative_dir, img_key = self._parse_path(*img_paths)
        lmdb_name = "liosam_pose"
        key_set = self._get_lmdb_key_set(
            self.liosam_key_dict, relative_dir, lmdb_name
        )
        return key_set is not None and img_key in key_set

    def load_wheel_pose(self, *img_paths):
        """

        返回4x4的vcs姿态矩阵,这个姿态的含义是: 从当前时间戳的vcs坐标系到世界坐标系的姿态.

        NOTE:
        1.这个函数返回的姿态的含义和load_liosam_pose函数相同
        2.不同点是这个vcs姿态是从wheel姿态转换得到
        """
        relative_dir, img_key = self._parse_path(*img_paths)
        lmdb_name = "wheel_pose"
        lmdb_loader = self._get_lmdb_loader(
            self.wheel_lmdb_loader_dict,
            relative_dir,
            lmdb_name,
            Lmdb,
            writable=False,
        )
        pose_byte = lmdb_loader.read(img_key)
        if pose_byte is None:
            raise IndexError(f"{img_key} not exist!")
        pose = np.frombuffer(pose_byte, "float32").reshape((4, 4))
        return pose

    def check_wheel_pose(self, *img_paths):
        relative_dir, img_key = self._parse_path(*img_paths)
        lmdb_name = "wheel_pose"
        key_set = self._get_lmdb_key_set(
            self.wheel_key_dict, relative_dir, lmdb_name
        )
        return key_set is not None and img_key in key_set

    def _parse_path(self, *img_paths):
        """

        将提供的img_paths转换成相对路径和key.

        NOTE:
        对于site 格式数据，你提供的img_paths至少要有5级
        e.g. FSD_Site_NC109_20221212/Site_121_34973_31_45059_1/NC109_20221212_182822/camera_front/1670840902300.jpg  # noqa
        对于park 格式数据，你提供的img_paths至少要有4级
        e.g. UT0Q9_20230215_D/20230215-122546_955/camera_front/1676435147306.jpg  # noqa

        """

        full_path = os.path.join(*img_paths)
        split_path = full_path.split("/")
        key = os.path.join(*split_path[-4:])

        if self.is_site_data:
            relative_dir = os.path.join(split_path[-5], split_path[-4])
            # e.g. FSD_Site_NC109_20221212/Site_121_34973_31_45059_1
        else:
            relative_dir = os.path.join(split_path[-4], split_path[-3])
            # e.g. UT0Q9_20230215_D/20230215-122546_955

        return relative_dir, key


class Reader3DV(object):
    """Read img/depth/seg/bev_seg... data from origin file or rec."""

    def __init__(
        self,
        img_data_path: Optional[Union[str, Sequence[str]]] = None,
        HDE_data_path: Optional[Union[str, Sequence[str]]] = None,
        seg_data_path: Optional[Union[str, Sequence[str]]] = None,
        depth_data_path: Optional[Union[str, Sequence[str]]] = None,
        bev_seg_data_path: Optional[Union[str, Sequence[str]]] = None,
        bev_elevation_data_path: Optional[Union[str, Sequence[str]]] = None,
        bev_elevation_vismask_data_path: Optional[
            Union[str, Sequence[str]]
        ] = None,
        bev_freespace_data_path: Optional[Union[str, Sequence[str]]] = None,
        bev_freespace_mask_path: Optional[str] = None,
        bev_occlusion_data_path: Optional[Union[str, Sequence[str]]] = None,
        bev3d_lmdb_path: Optional[Union[str, Sequence[str]]] = None,
        multi_view_lmdb_path: Optional[Union[str, Sequence[str]]] = None,
        bev_static_lmdb_path: Optional[Union[str, Sequence[str]]] = None,
        bev_motion_flow_lmdb_path: Optional[Union[str, Sequence[str]]] = None,
        bev_parking_obj_lmdb_path: Optional[Union[str, Sequence[str]]] = None,
        bev_discrete_obj_lmdb_path: Optional[Union[str, Sequence[str]]] = None,
        e2e_dynamic_lmdb_path: Optional[Union[str, Sequence[str]]] = None,
        tag_info_lmdb_path: Optional[Union[str, Sequence[str]]] = None,
        odometry_file_path: Optional[str] = None,
        object_tag_infos_lmdb_path: Optional[Union[str, Sequence[str]]] = None,
        point_cloud_data_path: Optional[Union[str, Sequence[str]]] = None,
        lidar_det_gt_path: Optional[Union[str, Sequence[str]]] = None,
    ) -> None:

        self.HDE_data_path = HDE_data_path

        if self.HDE_data_path is not None:
            self.HDE_loader = HDELoader(HDE_data_path)
        else:
            self.img_reader = self._load_img_reader(img_data_path)

        self.seg_reader = self._load_img_reader(
            seg_data_path, cv_format=-1, rgb=False
        )
        self.bev_seg_reader = self._load_img_reader(
            bev_seg_data_path, cv_format=-1, rgb=False
        )
        self.bev_elevation_reader = self._load_img_reader(
            bev_elevation_data_path, cv_format=-1, rgb=False
        )
        self.bev_elevation_vismask_reader = self._load_img_reader(
            bev_elevation_vismask_data_path,
            cv_format=-1,
            rgb=False,
        )
        self.bev_freespace_reader = self._load_img_reader(
            bev_freespace_data_path, cv_format=-1, rgb=False
        )
        self.bev_freespace_mask_reader = self._load_img_reader(
            bev_freespace_mask_path, cv_format=-1, rgb=False
        )
        self.bev_occlusion_reader = self._load_img_reader(
            bev_occlusion_data_path, cv_format=-1, rgb=False
        )
        self.depth_reader = self._load_img_reader(
            depth_data_path, cv_format=-1, rgb=False
        )
        self.pack_front_pose = {}

        self.bev3d_lmdb_path = bev3d_lmdb_path
        if bev3d_lmdb_path is not None:
            self.bev_3d_lmdb = LmdbReadList(
                bev3d_lmdb_path,
                readonly=True,
                map_size=1024 * 10,
            )
        self.multi_view_lmdb_path = multi_view_lmdb_path
        if multi_view_lmdb_path is not None:
            self.multi_view_lmdb = LmdbReadList(
                multi_view_lmdb_path,
                readonly=True,
                map_size=1024 * 10,
            )
        self.bev_static_lmdb_path = bev_static_lmdb_path
        if bev_static_lmdb_path is not None:
            self.bev_static_lmdb = LmdbReadList(
                bev_static_lmdb_path,
                readonly=True,
                map_size=1024 * 10,
            )
        self.bev_motion_flow_lmdb_path = bev_motion_flow_lmdb_path
        if bev_motion_flow_lmdb_path is not None:
            self.bev_motion_flow_lmdb = LmdbReadList(
                bev_motion_flow_lmdb_path, readonly=True, map_size=1024 * 10
            )

        self.bev_parking_obj_lmdb_path = bev_parking_obj_lmdb_path
        if bev_parking_obj_lmdb_path is not None:
            self.bev_parking_obj_lmdb = LmdbReadList(
                bev_parking_obj_lmdb_path,
                readonly=True,
                map_size=1024 * 10,
            )

        self.bev_discrete_obj_lmdb_path = bev_discrete_obj_lmdb_path
        if bev_discrete_obj_lmdb_path is not None:
            self.bev_discrete_obj_lmdb = LmdbReadList(
                bev_discrete_obj_lmdb_path,
                readonly=True,
                map_size=1024 * 10,
            )

        self.om_lmdb = None
        self.om_lmdb_path = bev_static_lmdb_path
        if self.om_lmdb_path is not None:
            self.om_lmdb = LmdbReadList(
                self.om_lmdb_path,
                readonly=True,
                map_size=1024 * 10,
            )

        self.e2e_dynamic_lmdb_path = e2e_dynamic_lmdb_path
        if e2e_dynamic_lmdb_path is not None:
            self.e2e_dynamic_lmdb = LmdbReadList(
                e2e_dynamic_lmdb_path,
                readonly=True,
                map_size=1024 * 10,
            )
        self.tag_info_lmdb_path = tag_info_lmdb_path
        if tag_info_lmdb_path is not None:
            self.tag_info_lmdb = LmdbReadList(
                tag_info_lmdb_path,
                readonly=True,
                map_size=1024 * 10,
            )
        self.object_tag_infos_lmdb_path = object_tag_infos_lmdb_path
        if object_tag_infos_lmdb_path is not None:
            self.object_tag_infos_lmdb = LmdbReadList(
                object_tag_infos_lmdb_path,
                readonly=True,
                map_size=1024 * 10,
            )
        self.odometry_file_path = odometry_file_path

        self.pack_pose = {}

        self.point_cloud_data_path = point_cloud_data_path
        if point_cloud_data_path is not None:
            self.point_cloud_reader = self._load_point_cloud_reader(
                point_cloud_data_path,
            )

        if lidar_det_gt_path is not None:
            with open(lidar_det_gt_path, "rb") as f:
                gt_infos = pickle.load(f)
            gt_names = [info["lidar_path"] for info in gt_infos]
            assert len(gt_names) == len(
                gt_infos
            ), f"pkl bad: please check your gt {lidar_det_gt_path}"
            self.lidar_det_gt_infos = dict(zip(gt_names, gt_infos))

    def _load_img_reader(self, data_path, **kwargs):
        if data_path is None:
            return None
        data_path = _as_list(data_path)
        if os.path.isdir(data_path[0]):
            # use lmdb to read
            return ImgLmdbDataset(data_path, map_size=1024 * 10, **kwargs)
        else:
            # use rec to read
            return NamedIndexDatasetV2(data_path, read_by_name=True, **kwargs)

    def _load_point_cloud_reader(self, data_path):
        if data_path is None:
            return None
        data_path = _as_list(data_path)
        return PointCloudLmdbDataset(
            data_path,
            load_dim=4,  # does not matter
            map_size=1024 * 10,
            parse_raw_buffer=False,
        )

    def __getstate__(self):
        state = self.__dict__
        state["pack_pose"] = {}
        return state

    def __setstate__(self, state):
        self.__dict__ = state
        self.pack_pose = {}

    def get_img_data(
        self, root, data_path, mode="RGB", size=None, rec_dataset=None
    ):
        # root is the root path of each sample recorded in syncfile.
        # Sometimes samples in the same syncfile are stored in different
        # paths, so we supplement the root for each sample in syncfile
        # additionally, which has higher priority than self.root, which
        # is obtained from dataset.yaml.
        if data_path is None:
            return None
        if rec_dataset:
            img_np = rec_dataset[data_path]
            img = Image.fromarray(img_np).convert(mode)
        else:
            img = pil_loader(os.path.join(root, data_path), mode, size=size)
        return img

    def get_img_data_from_HDELoader(self, root, data_path):
        # root is the root path of each sample recorded in syncfile.
        # Sometimes samples in the same syncfile are stored in different
        # paths, so we supplement the root for each sample in syncfile
        # additionally, which has higher priority than self.root, which
        # is obtained from dataset.yaml.
        if data_path is None:
            return None
        img_np = self.HDE_loader.load_img(root, data_path)
        img = Image.fromarray(img_np).convert("RGB")
        return img

    def read_rgb(self, root, data_path, img_load_size):
        if self.HDE_data_path is not None:
            return self.get_img_data_from_HDELoader(root, data_path)
        else:
            return self.get_img_data(
                root,
                data_path,
                "RGB",
                size=img_load_size,
                rec_dataset=self.img_reader,
            )

    def read_depth(self, root, data_path):
        return self.get_img_data(
            root, data_path, "I", rec_dataset=self.depth_reader
        )

    def read_seg(self, root, data_path):
        return self.get_img_data(
            root, data_path, "I", rec_dataset=self.seg_reader
        )

    def read_bev_seg(self, root, data_path):
        return self.get_img_data(
            root, data_path, "I", rec_dataset=self.bev_seg_reader
        )

    def read_bev_freespace(self, root, data_path):
        return self.get_img_data(
            root,
            data_path,
            "RGB",
            rec_dataset=self.bev_freespace_reader,
        )

    def read_bev_freespace_mask(self, root, data_path):
        return self.get_img_data(
            root,
            data_path,
            "L",
            rec_dataset=self.bev_freespace_mask_reader,
        )

    def read_bev_elevation(self, root, data_path):
        return self.get_img_data(
            root,
            data_path,
            "RGB",
            rec_dataset=self.bev_elevation_reader,
        )

    def read_bev_elevation_vismask(self, root, data_path):
        return self.get_img_data(
            root,
            data_path,
            "RGB",
            rec_dataset=self.bev_elevation_vismask_reader,
        )

    def read_bev_occlusion(self, root, data_path):
        return self.get_img_data(
            root, data_path, "I", rec_dataset=self.bev_occlusion_reader
        )

    def read_occlusion_mask(self, data_path):
        img = None
        if self.bev_occlusion_reader:
            try:
                img = self.bev_occlusion_reader[data_path]
                img = Image.fromarray(img)
            except (UnboundLocalError, IndexError):
                img = None

            if img is None:
                data_path = data_path.replace(
                    "/dense_ele_bev_vis_mask_camera_front", ""
                )
                try:
                    img = self.bev_occlusion_reader[data_path]
                    img = Image.fromarray(img)
                except (UnboundLocalError, IndexError):
                    img = None
        return img

    def read_bev_motion_flow(self, root, data_key, data_path):
        if self.bev_motion_flow_lmdb is not None:
            data = self.bev_motion_flow_lmdb.read(data_key).decode()
            return json.loads(data)
        else:
            annos_json = os.path.join(root, data_path)
            with open(annos_json, "rb") as fin:
                annos = json.load(fin)
            return annos[data_key]

    def read_tag_info(self, root, data_key):
        if self.tag_info_lmdb_path is not None:
            data = self.tag_info_lmdb.read(data_key).decode()
            return json.loads(data)
        else:
            return {}

    def read_object_tag_info(self, root, data_key):
        if self.object_tag_infos_lmdb_path is not None:
            data = self.object_tag_infos_lmdb.read(data_key).decode()
            return json.loads(data)
        else:
            return {}

    def read_bev_3d(self, root, data_key, data_path):
        if self.bev3d_lmdb_path is not None:
            data = self.bev_3d_lmdb.read(data_key).decode()
            return json.loads(data)
        else:
            annos_json = os.path.join(root, data_path)
            with open(annos_json, "rb") as fin:
                annos = json.load(fin)
            return annos[data_key]

    def read_static_anno(self, root, data_key, data_path):
        if self.bev_static_lmdb_path is not None:
            data = self.bev_static_lmdb.read(data_key).decode()
            return json.loads(data)
        else:
            annos_json = os.path.join(root, data_path)
            with open(annos_json, "rb") as fin:
                annos = json.load(fin)
            return annos[data_key]

    def read_multi_view_info(self, root, data_key):
        if self.multi_view_lmdb_path is not None:
            data = self.multi_view_lmdb.read(data_key).decode()
            return json.loads(data)
        else:
            return []

    def read_bev_parking_obj(self, data_key):
        data = self.bev_parking_obj_lmdb.read(data_key).decode()
        return json.loads(data)

    def read_bev_discrete_obj(self, data_key):
        data = self.bev_discrete_obj_lmdb.read(data_key).decode()
        return json.loads(data)

    # TODO(kefan.chen): move transform code below to OnlineMappingTargetGenerator  # noqa: E501
    def read_om(self, om_path):
        pt_pair_3d_dim = 6
        timestamp = os.path.split(om_path)[-1]
        target = {}

        try:
            """
            label_dict = {
                "solid_lane": [ins1, ...],
                "roadedges": [ins1, ...],
                "crosspoints": [ins1, ...],
                "changepoints": [ins1, ...],
                "ignores": [ins1, ...],
                "data_version": "xxx",
            }
            ins egs: {"pts": [[[x1,y1],[x2,y2]] ,...], "type": "xxx", ...}
            """
            label = self.om_lmdb.read(timestamp).decode()
            label_dict = json.loads(label)
            for k, v in label_dict.items():
                if k == "data_version":
                    target[k] = v
                    continue
                if len(v[0]) == 0:
                    continue
                res = []
                for ins in label_dict[k]:
                    if not ins:
                        continue
                    if "pts" not in ins.keys():
                        continue
                    if k not in ["crosspoints", "changepoints"]:
                        ins_pts = np.array(ins["pts"], np.float)
                    else:
                        ins_pts = np.array([ins["pts"]], np.float)
                    if ins_pts.shape[1] == pt_pair_3d_dim:
                        ins["pts"] = ins_pts
                    else:
                        # trans_ins format: [x1,y1,0,x2,y2,0]
                        # ins_pts format: [x1,y1,x2,y2]
                        trans_ins = np.zeros(
                            (ins_pts.shape[0], pt_pair_3d_dim), np.float
                        )
                        trans_ins[:, :2] = ins_pts[:, :2]
                        if k not in ["crosspoints", "changepoints"]:
                            trans_ins[:, 3:5] = ins_pts[:, 2:]
                        ins["pts"] = trans_ins
                    res.append(ins)
                target[k] = res
        except Exception:
            raise Exception(f"Error timestamp: {timestamp}")
        return target

    def read_bev_crosspoint(self, data_path):
        cls_2_idx = {
            "merge_start": 0,
            "merge_stop": 1,
            "split_start": 2,
            "split_stop": 3,
            "u_turn": 4,
            "other": 5,
        }
        timestamp = os.path.split(data_path)[-1]
        target = {
            "crosspoints": [],
            "changepoints": [],
            "ignores": [],
            "data_version": 0,
        }
        try:
            label = self.om_lmdb.read(timestamp)
            label_dict = json.loads(label)
            for k, v in label_dict.items():
                if v == [{}]:
                    continue
                elif k == "crosspoints":
                    trans_ins = np.zeros((len(v), 3), np.float)
                    for idx, ins in enumerate(label_dict[k]):
                        if ins == {}:
                            continue
                        if ins["crosspoint_category"] == "no":
                            trans_ins[idx, 0] = cls_2_idx["other"]
                        else:
                            trans_ins[idx, 0] = cls_2_idx[
                                ins["crosspoint_category"]
                            ]
                        trans_ins[idx, 1:3] = ins["pts"]
                    target[k] = trans_ins.tolist()
                elif k == "ignores":
                    ignore_regions = []
                    for ins in label_dict[k]:
                        if ins == {}:
                            continue
                        ignore_regions.append(ins["pts"])
                    target[k] = ignore_regions
                elif k == "changepoints":
                    trans_ins = np.zeros((len(v), 3), np.float)
                    for idx, ins in enumerate(label_dict[k]):
                        trans_ins[idx, 0] = 0
                        trans_ins[idx, 1:3] = ins["pts"]
                    target[k] = trans_ins.tolist()
                elif k == "data_version":
                    target[k] = v
        except Exception as e:
            logger.info(e)
            logger.info(f"Error timestamp {timestamp}")
        return target

    def read_e2e_dynamic_anno(self, data_key):
        """Load e2e dynamic clip's annos."""
        assert self.e2e_dynamic_lmdb is not None, ValueError(
            "e2e_dynamic_lmdb can't None..."
        )
        data_bin = self.e2e_dynamic_lmdb.read(data_key)
        data = DatumParserE2E.parse_from_string(data_bin)
        return data

    def read_lidar_detection(self, gt_name):
        return self.lidar_det_gt_infos[gt_name]

    def load_pose(self, dataset_root, pack_dir, timestamp):
        """Load vcs pose(represented as a 4x4 matrix) with specify timestamp.

        NOTE: Load from liosam_offline_10HZ.txt which represented in
        lidar coordinate system first, then convert to vcs coordinate system.
        """  # noqa

        if self.HDE_data_path is not None:
            vcs_pose = self.HDE_loader.load_liosam_pose(
                dataset_root, pack_dir, "camera_front", timestamp
            )
        else:
            if timestamp not in self.pack_pose:
                pt = PoseTransformer()
                pack_path = os.path.join(dataset_root, pack_dir)

                dr = DatasetReader()
                dr.read_pack(pack_path)
                transform_matrix = dr.get_extrinsic(
                    from_sensor="chassis", to_sensor="lidar_top"
                )
                if os.path.exists(
                    join_path(pack_path, "odometry/liosam_offline_10HZ.txt")
                ):
                    loam_array = dr.get_odometry("liosam_offline_10HZ.txt")
                else:
                    raise FileExistsError(
                        f"liosam_offline_10HZ.txt not in {pack_path}"
                    )

                pt.loadarray(loam_array)
                pt.rotate(transform_matrix)
                pt.normalize2origin()

                try:
                    vcs_pose = pt.seek_by_timestamp(
                        timestamp, t_max_diff=0.11
                    ).astype("float32")
                    self.pack_pose[timestamp] = vcs_pose
                except Exception:
                    raise IndexError(
                        f"pose of {timestamp} in {pack_dir} load error!"
                    )
            else:
                vcs_pose = self.pack_pose[timestamp]

        return vcs_pose

    def load_wheel_pose_base(
        self, dataset_root, pack_dir, timestamp, task_name=None
    ):
        """Load vcs pose(represented as a 4x4 matrix) with specify timestamp.

        NOTE: Load from wheel.txt which represented in
        vcs coordinate system.
        """  # noqa
        if self.HDE_data_path is not None:
            vcs_pose = self.HDE_loader.load_wheel_pose(
                dataset_root, pack_dir, "camera_front", timestamp
            )
        else:
            timestamp = float(timestamp) / 1000
            # Each pack dir shares a wheel.txt. When the pack dir is read
            # firstly, load wheel.txt and perform data transformation, then
            # save the results in PoseTransformer(). If a constant pack dir
            # is loaded, get the result in memory directly.
            if pack_dir not in self.pack_pose:

                pt = PoseTransformer()
                pack_path = os.path.join(dataset_root, pack_dir)
                dr = DatasetReader()
                dr.read_pack(pack_path)

                if self.odometry_file_path:
                    wheel_path = self.odometry_file_path
                    assert os.path.exists(
                        wheel_path
                    ), f"{wheel_path} not exists"
                    loam_array = np.loadtxt(wheel_path)
                elif task_name:
                    wheel_path = join_path(
                        pack_path, f"odometry/{task_name}/wheel.txt"
                    )
                    assert os.path.exists(
                        wheel_path
                    ), f"wheel.txt not in {pack_path}/odometry/{task_name}"
                    loam_array = np.loadtxt(wheel_path)
                elif os.path.exists(
                    join_path(pack_path, "odometry/wheel.txt")
                ):
                    loam_array = dr.get_odometry("wheel.txt")
                else:
                    raise FileExistsError(f"wheel.txt not in {pack_path}")
                pt.loadarray(loam_array)
                self.pack_pose[pack_dir] = pt
            else:
                pt = self.pack_pose[pack_dir]

            try:
                vcs_pose = pt.seek_by_timestamp(
                    timestamp, t_max_diff=0.121
                ).astype("float32")
                self.pack_pose[timestamp] = vcs_pose
            except Exception:
                # 如果左右两帧的diff大于t_max_diff，odo选择最接近的帧。
                # 这种情况一般出现在第一帧和最后一帧。
                # 此时最接近的帧的diff必须不超过50ms。
                odo_timestamps = pt.get_timestamps()
                nearest_odo_idx = (
                    np.abs(odo_timestamps[:, 0] - timestamp)
                ).argmin()
                vcs_pose = pt.absolute_transform[nearest_odo_idx]
                self.pack_pose[timestamp] = vcs_pose
                if np.abs(odo_timestamps[nearest_odo_idx] - timestamp) > 0.05:
                    raise IndexError(
                        f"pose of {timestamp} in {pack_dir} load error!"
                    )

        return vcs_pose

    def load_point_cloud(self, root, data_path):
        if self.point_cloud_data_path is not None:
            points = self.point_cloud_reader[data_path]
        else:  # for debug
            load_type = np.float32
            load_dim = 4
            keep_dim = 4
            points = np.fromfile(
                os.path.join(root, data_path), dtype=load_type
            ).reshape(-1, load_dim)[:, :keep_dim]

        return points

    def load_wheel_pose(self, dataset_root, pack_dir, timestamp):
        return self.load_wheel_pose_base(dataset_root, pack_dir, timestamp)

    def load_bev3d_pose(self, dataset_root, pack_dir, timestamp, task_name):
        """Load vcs pose(represented as a 4x4 matrix) with specify timestamp for bev3d.

        NOTE: The data production of bev3d includes hde links and 4dgt links.
        The hde use wheel pose and the 4dgt use liosam pose. And the 4dgt
        liosom pose has been converted to the VCS coordinate system by the
        packaging link. So far, the 4dgt pose file name is wheel.txt which
        will be changed in feature version.
        """
        return self.load_wheel_pose_base(
            dataset_root, pack_dir, timestamp, task_name
        )


class Frame(object):  # noqa: D205,D400
    """
    A Frame object could load many kinds type of img data using
    giving sync information, include img, depth, segmentation,
    bev_seg and so on.

    """

    def __init__(
        self,
        reader,
        frame_sync_info: Mapping,
        camera_view_names: Sequence,
        img_load_size: Sequence,
        lidar_view_name: str,
    ) -> None:

        self.reader = reader
        self.frame_sync_info = frame_sync_info
        self.camera_view_names = camera_view_names
        self.img_load_size = img_load_size
        self.lidar_view_name = lidar_view_name

    def img(self):
        """Return multi-view img as a list."""
        # NOTE: if camera_view_name not in sync info,
        # return zero-paddings PIL Image.
        imgs = []
        for (
            view_name,
            img_load_size,
        ) in zip(self.camera_view_names, self.img_load_size):
            if view_name in self.frame_sync_info:
                img_path = (
                    os.path.join(
                        self.frame_sync_info["pack_dir"],
                        view_name,
                        self.frame_sync_info[view_name],
                    )
                    + ".jpg"
                )
                img = self.reader.read_rgb(
                    self.frame_sync_info["root"], img_path, img_load_size
                )
            else:
                assert (
                    img_load_size is not None
                ), "padding img should set \
                img_load_size in Auto3dv"
                img = Image.fromarray(
                    np.zeros(img_load_size[::-1], dtype="uint8")
                ).convert(mode="RGB")
            imgs.append(img)
        return imgs

    def depth(self):
        """Return multi-view depth as a list."""
        img_paths = join_path(
            self.frame_sync_info["pack_dir"],
            [
                os.path.join(
                    "depth_" + view_name,
                    self.frame_sync_info[view_name] + ".png",
                )
                for view_name in self.camera_view_names
            ],
        )
        return [
            self.reader.read_depth(self.frame_sync_info["root"], img_path)
            for img_path in img_paths
        ]

    def seg(self):
        """Return multi-view segmentation as a list."""
        segs = []
        for (
            view_name,
            img_load_size,
        ) in zip(self.camera_view_names, self.img_load_size):
            if view_name in self.frame_sync_info:
                seg_path = (
                    os.path.join(
                        self.frame_sync_info["pack_dir"],
                        "seg_" + view_name,
                        self.frame_sync_info[view_name],
                    )
                    + ".png"
                )
                seg = self.reader.read_seg(
                    self.frame_sync_info["root"], seg_path
                )
            else:
                assert (
                    img_load_size is not None
                ), "padding seg should set \
                img_load_size in Auto3dv"
                seg = Image.fromarray(
                    np.ones(img_load_size[::-1] * 255, dtype="uint8")
                ).convert(mode="I")
            segs.append(seg)
        return segs

    def bev_seg(self):
        """Return bev segmentation."""
        bev_seg_path = os.path.join(
            self.frame_sync_info["pack_dir"],
            self.frame_sync_info["bev_seg"] + ".png",
        )
        return self.reader.read_bev_seg(
            self.frame_sync_info["root"], bev_seg_path
        )

    def bev_freespace(self):
        """Return bev freespace."""
        if "bev_freespace" in self.frame_sync_info:
            bev_freespace_path = os.path.join(
                self.frame_sync_info["pack_dir"],
                self.frame_sync_info["bev_freespace"] + ".png",
            )
        else:
            bev_freespace_path = os.path.join(
                self.frame_sync_info["pack_dir"],
                self.frame_sync_info["ele_bev_camera_front"] + ".png",
            ).replace("dense_ele_bev_camera_front/", "")
        return self.reader.read_bev_freespace(
            self.frame_sync_info["root"], bev_freespace_path
        )

    def bev_freespace_mask(self):
        """Return bev freespace mask."""
        if "bev_freespace_mask" in self.frame_sync_info:
            bev_freespace_path = os.path.join(
                self.frame_sync_info["pack_dir"],
                self.frame_sync_info["bev_freespace_mask"],
            )
        else:
            bev_freespace_path = os.path.join(
                self.frame_sync_info["pack_dir"],
                self.frame_sync_info["camera_front"],
            )
        return self.reader.read_bev_freespace_mask(
            self.frame_sync_info["root"], bev_freespace_path
        )

    def bev_elevation(self):
        """Return bev elevation."""
        if "bev_elevation" in self.frame_sync_info:
            bev_elevation_path = os.path.join(
                self.frame_sync_info["pack_dir"],
                self.frame_sync_info["bev_elevation"] + ".png",
            )
        else:
            bev_elevation_path = os.path.join(
                self.frame_sync_info["pack_dir"],
                self.frame_sync_info["ele_bev_camera_front"] + ".png",
            )
        return self.reader.read_bev_elevation(
            self.frame_sync_info["root"], bev_elevation_path
        )

    def bev_elevation_vismask(self):
        """Return bev elevation vis mask."""
        if "bev_vismask" in self.frame_sync_info:
            bev_elevation_vismask_path = os.path.join(
                self.frame_sync_info["pack_dir"],
                self.frame_sync_info["bev_vismask"] + ".png",
            )
        else:
            bev_elevation_vismask_path = os.path.join(
                self.frame_sync_info["pack_dir"],
                self.frame_sync_info["ele_bev_camera_front"] + ".png",
            ).replace(
                "dense_ele_bev_camera_front",
                "dense_ele_bev_vis_mask_camera_front",
            )
        return self.reader.read_bev_elevation_vismask(
            self.frame_sync_info["root"], bev_elevation_vismask_path
        )

    def bev_occlusion(self):
        """Return bev occlusion."""
        bev_occlusion_path = os.path.join(
            self.frame_sync_info["pack_dir"],
            self.frame_sync_info["occlusion"] + ".png",
        )
        return self.reader.read_bev_occlusion(
            self.frame_sync_info["root"], bev_occlusion_path
        )

    def read_static_anno(self):
        """Return bev_static key to get anno from annos file and bev_static_json_path."""  # noqa: E501
        bev_seg_key = os.path.join(
            self.frame_sync_info["camera_front"],
        )
        bev_seg_json_path = os.path.join(
            self.frame_sync_info["pack_dir"], "bevseg/annotations.json"
        )
        return self.reader.read_static_anno(
            self.frame_sync_info["root"], bev_seg_key, bev_seg_json_path
        )

    def e2e_dynamic_anno(self):
        """Return e2e_dynamic key to get anno from annos file."""
        return self.reader.read_e2e_dynamic_anno(
            self.frame_sync_info["e2e_dynamic"]
        )

    def front_seg(self):
        """Return segmentation for front view."""
        front_seg_path = os.path.join(
            self.frame_sync_info["pack_dir"],
            os.path.join(
                "seg_camera_front",
                self.frame_sync_info["camera_front"] + ".png",
            ),
        )
        return self.reader.read_seg(
            self.frame_sync_info["root"], front_seg_path
        )

    def timestamp(self):
        """Return timestamp.

        We use the timestamp of the first view in datasets as the criterion.
        The first item of `camera_view_names` must be:
            a) "camera_front" in 6v, 10v and 11v datasets.
            b) "fisheye_front" in 4v datasoets.
            c) "camera_front_left" in 5v datasets.  # hard code in 5v datasets
            d) any view name in 1v datasets.
        """

        if "camera_front" in self.frame_sync_info:
            view_name = "camera_front"
        elif "fisheye_front" in self.frame_sync_info:
            view_name = "fisheye_front"
        elif "camera_front_left" in self.frame_sync_info:
            view_name = "camera_front_left"
        else:
            view_name = None
            for k in self.frame_sync_info.keys():
                if "camera" in k or "fisheye" in k:
                    view_name = k
                    break
            if view_name is None:
                raise ValueError("no valid view_name in frame_sync_info")
        timestamp = (
            np.array([self.frame_sync_info[view_name]], dtype="float64") / 1000
        )

        return timestamp

    def pose(self):
        """Load pose from liosam.

        Return vcs pose from current timestamp to first timestamp
        in corresponding pack and represented as transformation matrix.

        """
        return self.reader.load_pose(
            self.frame_sync_info["root"],
            self.frame_sync_info["pack_dir"],
            self.frame_sync_info["camera_front"],
        )

    def wheel_pose(self):
        """Load pose from wheel.

        Return vcs pose from current timestamp to first timestamp
        in corresponding pack and represented as transformation matrix.

        """
        return self.reader.load_wheel_pose(
            self.frame_sync_info["root"],
            self.frame_sync_info["pack_dir"],
            self.frame_sync_info["camera_front"],
        )

    def bev3d_vehicle_pose(self):
        return self.reader.load_bev3d_pose(
            self.frame_sync_info["root"],
            self.frame_sync_info["pack_dir"],
            self.frame_sync_info["camera_front"],
            "vehicle",
        )

    def bev3d_vrumerge_pose(self):
        return self.reader.load_bev3d_pose(
            self.frame_sync_info["root"],
            self.frame_sync_info["pack_dir"],
            self.frame_sync_info["camera_front"],
            "vrumerge",
        )

    @property
    def eye_pose(self):
        return np.eye(4)

    def bev_3d(self):
        """Return bev_3d key to get anno from annos file and bev_3d_json_path."""  # noqa: E501
        bev_3d_key = os.path.join(
            self.frame_sync_info["pack_dir"], self.frame_sync_info["bev_3d"]
        )
        bev_3d_json_path = os.path.join(
            self.frame_sync_info["pack_dir"], "bev3d/annotations.json"
        )
        return self.reader.read_bev_3d(
            self.frame_sync_info["root"], bev_3d_key, bev_3d_json_path
        )

    def multi_view(self):
        """Return multi view key to get multi view info from multi view lmdb."""  # noqa: E501
        multi_view_info_key = os.path.join(
            self.frame_sync_info["pack_dir"], self.frame_sync_info["bev_3d"]
        )
        multi_view_infos_dict = self.reader.read_multi_view_info(
            self.frame_sync_info["root"], multi_view_info_key
        )
        if not multi_view_infos_dict:
            return multi_view_infos_dict
        else:
            multi_view_infos = []
            for view_name in self.camera_view_names:
                if view_name in self.frame_sync_info:
                    multi_view_info = multi_view_infos_dict[view_name]
                    ignore_mask_2d = multi_view_info["meta"]["ignore_mask"][
                        "ignore_mask_2d"
                    ]
                    ignore_mask_2d = coco_mask.decode(ignore_mask_2d).astype(
                        np.uint8
                    )
                    ignore_mask_2d = Image.fromarray(ignore_mask_2d)
                    multi_view_info["meta"]["ignore_mask"][
                        "ignore_mask_2d"
                    ] = ignore_mask_2d
                    multi_view_infos.append(multi_view_info)
                else:
                    multi_view_infos.append({})
            return multi_view_infos

    def bev_parking_obj(self):
        """Return bev_parking_obj key to get anno from annos file."""
        if "bev_parking" in self.frame_sync_info.keys():
            bev_parking_obj_key = self.frame_sync_info["bev_parking"]
        else:
            bev_parking_obj_key = self.frame_sync_info["camera_front"]
        return self.reader.read_bev_parking_obj(bev_parking_obj_key)

    def bev_discrete_obj(self):
        """Return bev_discrete_obj key to get anno from annos file."""
        if "bev_parking" in self.frame_sync_info.keys():
            bev_discrete_obj_key = self.frame_sync_info["bev_parking"]
        else:
            bev_discrete_obj_key = self.frame_sync_info["camera_front"]
        return self.reader.read_bev_discrete_obj(bev_discrete_obj_key)

    def bev_occlusion_mask(self):
        """Return occlusion mask."""
        bev_occlusion_mask_path = os.path.join(
            self.frame_sync_info["pack_dir"],
            "dense_ele_bev_vis_mask_camera_front",
            self.frame_sync_info["camera_front"] + ".png",
        )
        return self.reader.read_occlusion_mask(bev_occlusion_mask_path)

    def om(self):
        """Return om gt."""
        om_path = os.path.join(
            self.frame_sync_info["root"],
            self.frame_sync_info["pack_dir"],
            self.frame_sync_info["camera_front"],
        )
        return self.reader.read_om(om_path)

    def bev_crosspoint(self):
        """Return bev crosspoint gt."""
        bev_crosspoint_path = os.path.join(
            self.frame_sync_info["root"],
            self.frame_sync_info["pack_dir"],
            self.frame_sync_info["camera_front"],
        )
        return self.reader.read_bev_crosspoint(bev_crosspoint_path)

    def bev_motion_flow(self):
        """Return bev motion flow."""
        bev_motion_flow_key = os.path.join(
            self.frame_sync_info["pack_dir"],
            self.frame_sync_info["bev_motion_flow"],
        )
        bev_motion_flow_json_path = os.path.join(
            self.frame_sync_info["pack_dir"],
            "bev_motion_flow/annotations.json",
        )
        return self.reader.read_bev_3d(
            self.frame_sync_info["root"],
            bev_motion_flow_key,
            bev_motion_flow_json_path,
        )

    def lidar_detection(self):
        return self.reader.read_lidar_detection(
            self.frame_sync_info["gt_name"]
        )

    def point_cloud(self):
        """Load a from bin file and return point cloud."""
        point_cloud_path = (
            os.path.join(
                self.frame_sync_info["pack_dir"],
                self.lidar_view_name,
                self.frame_sync_info["lidar"],
            )
            + ".bin"
        )

        points = self.reader.load_point_cloud(
            self.frame_sync_info["root"],
            point_cloud_path,
        )
        return points

    @property
    def pack_dir(self):
        return join_path(
            self.frame_sync_info["root"], self.frame_sync_info["pack_dir"]
        )

    @property
    def clip_key(self):
        return self.frame_sync_info["e2e_dynamic"]

    @property
    def img_paths(self):
        img_paths = []
        for view_name in self.camera_view_names:
            if view_name in self.frame_sync_info:
                img_path = (
                    os.path.join(view_name, self.frame_sync_info[view_name])
                    + ".jpg"
                )
                img_paths.append(img_path)
            else:
                dake_data_path = ""  # fake data path is ""
                img_paths.append(dake_data_path)
        return img_paths

    @property
    def tag_info(self):
        """Return tag_info key to get tag from tags file."""  # noqa: E501
        bev_3d_key = os.path.join(
            self.frame_sync_info["pack_dir"], self.frame_sync_info["bev_3d"]
        )
        return self.reader.read_tag_info(
            self.frame_sync_info["root"], bev_3d_key
        )

    @property
    def object_tag_info(self):
        """Return tag_info key to get tag from tags file."""  # noqa: E501
        bev_3d_key = os.path.join(
            self.frame_sync_info["pack_dir"], self.frame_sync_info["bev_3d"]
        )
        return self.reader.read_object_tag_info(
            self.frame_sync_info["root"], bev_3d_key
        )


@OBJECT_REGISTRY.register
class ANCAuto3DV(data.Dataset):  # noqa: D205,D400
    """A dataset for generating specific data to support many
    kinds of 3dv task. e.g. 2.5d task(depth,pose,resflow), bev_seg task,
    bev_3d task, bev_motion_flow task and so on.

    Args:
        root str: Path of dataset root, which is obtained from dataset.yaml.
        camera_view_names: sub directory name of each view.
        per_view_shape: img shape of each view.
        img_load_size: img load size of each view
        sync_file: path of sync_file.
        sync_file_lmdb: The path of LMDB sync file
        sample_interval: resample interval of each sync file.
        img_data_path:
            image rec path, load image data from rec file if not None,
            will load from original img otherwise.
        seg_data_path:
            seg rec path, load seg/obj_maks/ data from rec file
            if not None, will load from original seg file otherwise.
        depth_data_path:
            depth rec path, load depth from rec file if not None,
            will load from original depth file otherwise.
        bev_seg_data_path:
            bev seg rec path, load bev_seg data from rec file if not None,
            will load from original bev_seg file otherwise.
        bev_elevation_data_path:
            bev elevation rec path, load bev_elevation data from rec file
            if not None, will load from original bev_elevation file otherwise.
        bev_elevation_vismask_data_path:
            bev elevation vis mask rec path, load bev elevation vis_mask data
            from rec file. if not None, will load from original bev elevation
            vis_mask file otherwise.
        bev_freespace_data_path:
            bev freespace rec path, load bev_freespace data from rec file
            if not None, will load from original bev_freespace file otherwise.
        bev_occlusion_data_path:
            bev occlusion rec path, load bev_occlusion data from rec file
            if not None, will load from original bev_occlusion file otherwise.
        bev_3d_lmdb_path:
            bev3d annotation lmdb path.
        multi_view_lmdb_path:
            multi view annotation lmdb path for each view.
        bev_static_lmdb_path:
            bev static annotation lmdb path.
        bev_motion_flow_lmdb_path:
            bev motion flow annotation lmdb path.
        e2e_dynamic_lmdb_path:
            end2end dynamic task's annotation lmdb path.
        front_mask_path: path of front mask file.
        load_intrinsics: whether load intrinsics.
        load_distortcoef: whether load distortion coefficient.
        homo_gen: HomoGenerator dict for compute homography.
        homo_gen_small: HomoGenerator dict for compute
             homography of small range.
        homo_gen_high_sp: HomoGenerator dict for compute
             homography of high spatial resolution.
        flag_for_group: flag to distinguish data in different shape of
             the camera module. Data of the same shape has the same flag
             value, even if it comes from different camera modules, so as
             to be concatenated by the DistributedGroupSampler correctly.
        num_max_frames: max applied length for every samples, remove extra
            length in packaged data.
        num_frames_per_iter: split every sample into slices of a specific
            length for every interation.
        tag_info_lmdb_path: Image tag info lmdb path.
        odometry_file_path: 4dgt liosam wheel.txt file path.
        convert_mf_to_sf: Convert temporal data to single frame data.
            Default is False.
        reverse_select: Select frame from tail of clips.
            Default is False.
        object_tag_infos_lmdb_path: Object tag info lmdb path.
    """

    def __init__(
        self,
        root: str,
        camera_view_names: Sequence[str],
        per_view_shape: Mapping,
        sync_file: Optional[Sequence[str]] = None,
        transforms: Optional[Sequence] = None,
        img_load_size: Optional[Sequence] = None,
        sync_file_lmdb: Optional[Sequence[str]] = None,
        sample_interval: Union[int, Sequence[int]] = 1,
        img_data_path: Optional[Union[str, Sequence]] = None,
        HDE_data_path: Optional[Union[str, Sequence]] = None,
        seg_data_path: Optional[Union[str, Sequence]] = None,
        depth_data_path: Optional[Union[str, Sequence]] = None,
        bev_seg_data_path: Optional[Union[str, Sequence]] = None,
        bev_elevation_data_path: Optional[Union[str, Sequence]] = None,
        bev_elevation_vismask_data_path: Optional[Union[str, Sequence]] = None,
        bev_freespace_data_path: Optional[Union[str, Sequence]] = None,
        bev_freespace_mask_path: Optional[str] = None,
        bev_occlusion_data_path: Optional[Union[str, Sequence]] = None,
        bev_3d_lmdb_path: Optional[Union[str, Sequence]] = None,
        multi_view_lmdb_path: Optional[Union[str, Sequence]] = None,
        bev_static_lmdb_path: Optional[Union[str, Sequence]] = None,
        bev_motion_flow_lmdb_path: Optional[Union[str, Sequence]] = None,
        bev_parking_obj_lmdb_path: Optional[Union[str, Sequence]] = None,
        bev_discrete_obj_lmdb_path: Optional[Union[str, Sequence]] = None,
        e2e_dynamic_lmdb_path: Optional[Union[str, Sequence]] = None,
        front_mask_path: Optional[str] = None,
        load_intrinsics: bool = False,
        load_distortcoef: bool = False,
        homo_gen: Optional[dict] = None,
        homo_gen_small: Optional[dict] = None,
        homo_gen_high_sp: Optional[dict] = None,
        flag_for_group: Optional[int] = 1,
        num_max_frames: Optional[int] = None,
        num_frames_per_iter: Optional[int] = None,
        fill_fake_temporal_data: bool = False,
        reverse: bool = False,
        tag_info_lmdb_path: Optional[Sequence[str]] = None,
        odometry_file_path: Optional[str] = None,
        convert_mf_to_sf: Optional[bool] = False,
        reverse_select: Optional[bool] = False,
        object_tag_infos_lmdb_path: Optional[Sequence[str]] = None,
        point_cloud_data_path: Optional[Union[str, Sequence]] = None,
        lidar_det_gt_path: Optional[Union[str, Sequence]] = None,
        lidar_view_name: Optional[str] = None,
    ):
        self.root = root
        self.camera_view_names = camera_view_names
        self.per_view_shape = per_view_shape
        if img_load_size is not None:
            assert len(img_load_size) == len(camera_view_names)
        else:
            img_load_size = [None] * len(camera_view_names)
        self.img_load_size = img_load_size

        self.transforms = transforms

        self.load_intrinsics = load_intrinsics
        self.load_distortcoef = load_distortcoef
        self.intrinsics = None
        self.distcoef = None

        # get meta_info from homogen
        self.meta_info = None
        self.homo_gen = homo_gen
        if homo_gen is not None:
            self.homo_genor = HomoGenerator(**homo_gen)

        # TODO(ning.xu): Cancel homo_gen_small, support
        # multi vcs_range in homo_generator.
        self.homo_gen_small = homo_gen_small
        if homo_gen_small is not None:
            self.homo_genor_small = HomoGenerator(**homo_gen_small)

        self.homo_gen_high_sp = homo_gen_high_sp
        if homo_gen_high_sp is not None:
            self.homo_genor_high_sp = HomoGenerator(**homo_gen_high_sp)

        self.front_mask_img = None
        if front_mask_path is not None:
            assert os.path.exists(
                front_mask_path
            ), f"{front_mask_path} not exist!"
            self.front_mask_img = pil_loader(front_mask_path, "F")

        self.reader = None

        self.img_data_path = img_data_path
        self.HDE_data_path = HDE_data_path
        self.seg_data_path = seg_data_path
        self.depth_data_path = depth_data_path
        self.bev_seg_data_path = bev_seg_data_path
        self.bev_elevation_data_path = bev_elevation_data_path
        self.bev_elevation_vismask_data_path = bev_elevation_vismask_data_path
        self.bev_freespace_data_path = bev_freespace_data_path
        self.bev_freespace_mask_path = bev_freespace_mask_path
        self.bev_occlusion_data_path = bev_occlusion_data_path
        self.bev3d_lmdb_path = bev_3d_lmdb_path
        self.multi_view_lmdb_path = multi_view_lmdb_path
        self.bev_static_lmdb_path = bev_static_lmdb_path
        self.bev_motion_flow_lmdb_path = bev_motion_flow_lmdb_path
        self.bev_parking_obj_lmdb_path = bev_parking_obj_lmdb_path
        self.bev_discrete_obj_lmdb_path = bev_discrete_obj_lmdb_path
        self.e2e_dynamic_lmdb_path = e2e_dynamic_lmdb_path
        self.num_max_frames = num_max_frames
        self.num_frames_per_iter = num_frames_per_iter
        self.tag_info_lmdb_path = tag_info_lmdb_path
        self.object_tag_infos_lmdb_path = object_tag_infos_lmdb_path
        self.odometry_file_path = odometry_file_path
        self.convert_mf_to_sf = convert_mf_to_sf
        self.sync_info = SyncInfo(
            root,
            sync_file,
            sync_file_lmdb,
            sample_interval,
            num_max_frames,
            num_frames_per_iter,
            reverse=reverse,
            fill_fake_temporal_data=fill_fake_temporal_data,
            reverse_select=reverse_select,
        )
        self.flag = flag_for_group * np.ones(len(self), dtype=np.uint8)

        self.lidar_view_name = lidar_view_name
        self.lidar_det_gt_path = lidar_det_gt_path
        self.point_cloud_data_path = point_cloud_data_path

    def _init_reader(self):
        self.reader = Reader3DV(
            img_data_path=self.img_data_path,
            HDE_data_path=self.HDE_data_path,
            seg_data_path=self.seg_data_path,
            depth_data_path=self.depth_data_path,
            bev_seg_data_path=self.bev_seg_data_path,
            bev_elevation_data_path=self.bev_elevation_data_path,
            bev_elevation_vismask_data_path=self.bev_elevation_vismask_data_path,  # noqa
            bev_freespace_data_path=self.bev_freespace_data_path,
            bev_freespace_mask_path=self.bev_freespace_mask_path,
            bev_occlusion_data_path=self.bev_occlusion_data_path,
            bev3d_lmdb_path=self.bev3d_lmdb_path,
            multi_view_lmdb_path=self.multi_view_lmdb_path,
            bev_static_lmdb_path=self.bev_static_lmdb_path,
            bev_motion_flow_lmdb_path=self.bev_motion_flow_lmdb_path,
            bev_parking_obj_lmdb_path=self.bev_parking_obj_lmdb_path,
            bev_discrete_obj_lmdb_path=self.bev_discrete_obj_lmdb_path,
            e2e_dynamic_lmdb_path=self.e2e_dynamic_lmdb_path,
            tag_info_lmdb_path=self.tag_info_lmdb_path,
            odometry_file_path=self.odometry_file_path,
            object_tag_infos_lmdb_path=self.object_tag_infos_lmdb_path,
            point_cloud_data_path=self.point_cloud_data_path,
            lidar_det_gt_path=self.lidar_det_gt_path,
        )

    def __getstate__(self):
        state = self.__dict__
        state["reader"] = None
        return state

    def __setstate__(self, state):
        self.__dict__ = state

    def _load_homo_offset_path(self, homo_offset_path):
        """Load _load_homo_offset_path for each view."""

        H = []
        for sub_dir in self.camera_view_names:
            hom_file = os.path.join(homo_offset_path, sub_dir + ".npy")
            assert os.path.exists(hom_file), hom_file
            ori_homo = np.load(hom_file).astype("float32")
            H.append(ori_homo)
        return np.stack(H, axis=0)

    def _load_attr(self, root, pack_dir):
        if root is not None:
            attribute_file = os.path.join(root, pack_dir, "attribute.json")
        else:
            attribute_file = os.path.join(
                self.root, pack_dir, "attribute.json"
            )

        @timeout_decorator.timeout(60)
        def load():
            with open(attribute_file, "r") as fin:
                att_file = json.load(fin)
            return att_file

        try:
            return load()
        except (timeout_decorator.TimeoutError, FileNotFoundError) as e:
            if isinstance(e, timeout_decorator.TimeoutError):
                raise FileNotFoundError(
                    f"read {attribute_file} timeout > 60 sec"
                )
            elif isinstance(e, FileNotFoundError):
                raise FileNotFoundError(f"{attribute_file} FileNotFoundError")

    def _set_intrinsics(self, root, pack_dir):
        """Load intrinsics from attribute file.

        Note: only return intrinsics matrix for front camera.

        """
        att_file = self._load_attr(root, pack_dir)
        cam_front_intri = np.array(
            att_file["calibration"]["camera_front"]["K"], dtype=np.float32
        )
        cam_front_intri[0] /= self.per_view_shape["camera_front"][1]
        cam_front_intri[1] /= self.per_view_shape["camera_front"][0]
        self.intrinsics = cam_front_intri

    def _set_distcoefs(self, root, pack_dir):
        """Load distort coefficients from attribute file.

        Note: only return distcoefs matrix for front camera.

        """
        att_file = self._load_attr(root, pack_dir)
        cam_front_discoef = np.array(
            att_file["calibration"]["camera_front"]["d"], dtype=np.float32
        )
        self.distcoef = cam_front_discoef

    def _get_frames(self, sync_infos: Sequence):
        return [
            Frame(
                self.reader,
                one_frame,
                self.camera_view_names,
                self.img_load_size,
                self.lidar_view_name,
            )
            for one_frame in sync_infos
        ]

    def __getitem__(self, idx):

        # lazy open reader to avoid unecessary initialization.
        if self.reader is None:
            self._init_reader()

        multi_frame_sync_info, sample_split_index = self.sync_info[idx]
        if self.convert_mf_to_sf:
            multi_frame_sync_info = multi_frame_sync_info[:1]
            sample_split_index = None

        frames = self._get_frames(multi_frame_sync_info)

        data_dict = {}
        data_dict["frames"] = frames

        if sample_split_index is not None:
            data_dict["sample_split_index"] = sample_split_index

        dataset_root = self.sync_info[0][0][0]["root"]

        if self.homo_gen is not None:
            data_dict["meta_info"] = self.homo_genor.load_homo_offset()

        if self.homo_gen_small is not None:
            data_dict[
                "meta_info_small"
            ] = self.homo_genor_small.load_homo_offset()

        if self.homo_gen_high_sp is not None:
            data_dict[
                "meta_info_high_sp"
            ] = self.homo_genor_high_sp.load_homo_offset()

        data_dict["temporal_info"] = {
            "num_frames_per_iter": 1
            if self.convert_mf_to_sf or self.num_frames_per_iter is None
            else self.num_frames_per_iter,
        }

        if self.front_mask_img is not None:
            data_dict["front_mask"] = self.front_mask_img.copy()

        if self.load_intrinsics:
            if self.intrinsics is None:
                self._set_intrinsics(
                    dataset_root, self.sync_info[0][0][0]["pack_dir"]
                )
            data_dict["intrinsics"] = self.intrinsics.copy()

        if self.load_distortcoef:
            if self.distcoef is None:
                self._set_distcoefs(
                    dataset_root, self.sync_info[0][0][0]["pack_dir"]
                )
            data_dict["distortcoef"] = self.distcoef.copy()

        if self.transforms:
            data_dict = self.transforms(data_dict)
        return data_dict

    def __len__(self):
        return len(self.sync_info)


def generate_sync_info(
    pack_list_path,
    dataset_root,
    sync_info_path,
    frames_offset=(0,),
    frame_interval_time=30,
    to_lmdb=False,
    lmdb_path=None,
    drop_frame_num=10,
    fisheye=False,
):
    """
    Generate sync information file used for Auto3DV class.

    As a Example code.

    Args:
        pack_list_path str: Path of pack list.
        dataset_root str: dataset root path.
        sync_info_path str: result path of sync_info.
        frames_offset list(int):
            interval between the reference frame and current frame.
            NOTE: first element in frames_offset must be 0.
            suppose an img sequence is like [0,1,2....T]
            Example 1:
                (0) means we only load one frame in a sample.
                the sample that dataset will generate is [t] at index t.
            Example 2:
                (0,-1) means we load two frames in a sample.
                the sample that dataset will generate is [t,t-1] at index t.
            Example 3:
                (0,-1,-3) means we load three frames in a sample.
                the sample that dataset will generate is [t,t-1,t-3] at index t.  # noqa
            Example 3:
                (0,-1,-4,-5) means we load four frames in a sample.
                the sample that dataset will generate is [t,t-1,t-4,t-5] at index t,  # noqa
        frame_interval_time int: The interval between adjecent
        to_lmdb bool: whehter convert sync_info file to lmdb format.
        lmdb_path str: sync_info lmdb path.
        drop_frame_num int: frame index in [0,drop_frame_num] and [-drop_frame_num,-1]  # noqa
            are invalid and will be droped.
        fisheye bool: whether generate the sync_info for fisheye dataset, default: False.
    """

    def _check_valid_frame_interval(
        sync_infos, frames_offset, frame_interval_time, fisheye
    ):

        for i in range(1, len(sync_infos)):
            interval_time = -frames_offset[i] * frame_interval_time
            min_time, max_time = interval_time - 10, interval_time + 10
            cam_front_name = "camera_front" if not fisheye else "fisheye_front"
            cur_interval_time = abs(
                int(sync_infos[i][cam_front_name])
                - int(sync_infos[0][cam_front_name])
            )
            if cur_interval_time >= max_time or cur_interval_time <= min_time:
                return False
        return True

    frames_offset = (
        frames_offset if frames_offset[0] == 0 else [0] + frames_offset
    )
    # make sure all value in frames_offset is <=0
    assert all(
        list(map(lambda x: x <= 0, frames_offset))
    ), "frame interval must be less than or equal to 0"

    with open(pack_list_path, "r") as f:
        pack_list = f.read().splitlines()

    sub_dirs = (
        [
            "camera_front",
            "camera_front_left",
            "camera_front_right",
            "camera_rear_left",
            "camera_rear_right",
            "camera_rear",
        ]
        if not fisheye
        else [
            "fisheye_front",
            "fisheye_rear",
            "fisheye_left",
            "fisheye_right",
        ]
    )
    all_samples = []
    for _pack_idx, pack_dir in enumerate(pack_list):
        pack_path = os.path.join(dataset_root, pack_dir)
        pack_json = os.path.join(pack_path, "attribute.json")
        with open(pack_json, "r", encoding="utf8") as fp:
            json_data = json.load(fp)
            sync = json_data["sync"]
            total_frame = len(sync["lidar_top"])

            start_idx = drop_frame_num - frames_offset[-1]
            end_idx = total_frame - drop_frame_num

            for i in range(start_idx, end_idx):
                cur_sample = {}
                cur_sample["pack_dir"] = pack_dir
                sync_infos = []
                for offset in frames_offset:
                    sync_info = {}
                    for sub_dir in sub_dirs:
                        sync_info[sub_dir] = os.path.join(
                            str(sync[sub_dir][i + offset])
                        )
                    sync_info["bev_seg"] = os.path.join(
                        "bevGT_3", sync_info["camera_front"]
                    )
                    sync_info["bev_3d"] = str(sync["lidar_top"][i + offset])

                    sync_info["bev_motion_flow"] = str(
                        sync["camera_front"][i + offset]
                    )
                    # fill any other content to sync_info,
                    # set None if not exist.
                    # e.g.
                    # sync_info['lidar_top']=xxx
                    # sync_info['bev_motion_flow']=xxx

                    sync_infos.append(sync_info)
                if _check_valid_frame_interval(
                    sync_infos, frames_offset, frame_interval_time, fisheye
                ):
                    for key in sync_infos[0]:
                        cur_sample[key] = [
                            sync_info[key] for sync_info in sync_infos
                        ]

                    # do other check to decide whether append current sample
                    # only check bev_seg of frame idx 0 here
                    if os.path.exists(
                        os.path.join(
                            pack_path, sync_infos[0]["bev_seg"] + ".png"
                        )
                    ):
                        all_samples.append(json.dumps(cur_sample) + "\n")

    all_sample_num = len(all_samples)
    if all_sample_num > 0:
        logger.info(f"all sample num: {all_sample_num}")
        with open(sync_info_path, "w") as w:
            for one in all_samples:
                w.write(one)
        if to_lmdb:
            assert lmdb_path is not None, "please provide valid lmdb path"
            f = FileList2LMDB(sync_info_path, lmdb_path)
            f()


@OBJECT_REGISTRY.register
class ANCAuto3DVFromImage(data.Dataset):
    """Dataset which gets img data from the data_path.

    This dataset can used for inference on unlabeled data.

    Args:
        data_path: The path where the image is stored.
        att_json_path: path of attribute.json.
        transforms: List of transform.
        per_view_shape : img shape of each view.
        camera_view_names : sub directory name of each view.
        only_front_view :
            only load front view img. setting True for
            depth_pose_resflow training.
        get_pack_dir: whether to get the pack_dir,
            now only for inference of bev task when need
            6v images.
        homo_gen: HomoGenerator dict for compute homography.


    """

    def __init__(
        self,
        data_path: str,
        intrinsics: np.ndarray,
        distortcoef: np.ndarray,
        per_view_shape: Mapping,
        camera_view_names: Optional[Sequence] = None,
        img_load_size: Optional[Sequence] = None,
        att_json_path: Optional[Sequence] = None,
        transforms: Optional[Sequence] = None,
        only_front_view: Optional[bool] = True,
        get_pack_dir: Optional[bool] = True,
        homo_gen: Optional[dict] = None,
    ):
        self.data_path = os.path.split(data_path)[
            0
        ]  # strip the `camera_view` level of the path hierarchy  # noqa
        self.intrinsics = intrinsics
        self.distortcoef = distortcoef
        self.transforms = transforms
        self.att_json_path = att_json_path
        self.img_load_size = img_load_size
        self.per_view_shape = per_view_shape
        self.camera_view_names = camera_view_names
        self.only_front_view = only_front_view
        self.get_pack_dir = get_pack_dir
        self.front_img_name = camera_view_names[
            0
        ]  # fisheye_front or camera_front
        assert self.front_img_name in ["fisheye_front", "camera_front"]
        self.collect_samples()

        # get homography and homo offset matrix
        self.H_origin = None
        self.homo_offset = None
        self.homo_gen = homo_gen
        if homo_gen is not None:
            homo_genor = HomoGenerator(**homo_gen)
            self.H_origin = homo_genor.get_homography()
            self.homo_offset = homo_genor.get_homo_offset()
            self.H_origin = reformat_homo_info(
                self.H_origin, self.camera_view_names, HOMO_PAD_VALUE
            )
            self.homo_offset = reformat_homo_info(
                self.homo_offset, self.camera_view_names, HOMO_PAD_VALUE
            )
        self.pers_view_intr_norm_mat = [
            np.array(
                [
                    [1 / self.per_view_shape[camera_view_name][1], 0, 0],
                    [0, 1 / self.per_view_shape[camera_view_name][0], 0],
                    [0, 0, 1],
                ],
                dtype="float32",
            ).reshape(3, 3)
            for camera_view_name in self.camera_view_names
        ]

    def __getstate__(self):
        state = self.__dict__
        state["H_origin"] = None
        state["homo_offset"] = None
        return state

    def __setstate__(self, state):
        self.__dict__ = state
        if self.homo_gen is not None:
            homo_genor = HomoGenerator(**self.homo_gen)
            self.H_origin = homo_genor.get_homography()
            self.homo_offset = homo_genor.get_homo_offset()
            self.H_origin = reformat_homo_info(
                self.H_origin, self.camera_view_names, HOMO_PAD_VALUE
            )
            self.homo_offset = reformat_homo_info(
                self.homo_offset, self.camera_view_names, HOMO_PAD_VALUE
            )

    def load_homography(self):
        """Load homography for each view and retuan as list."""
        H = []
        for camera_view_name in self.camera_view_names:
            hom_file = os.path.join(self.homo_path, camera_view_name + ".npy")
            assert os.path.exists(hom_file), hom_file
            H.append(np.load(hom_file).astype("float32"))
        return H

    def _load_homo_offset_path(self, homo_offset_path):
        """Load homo_offset_path for each view."""
        H = []
        for sub_dir in self.camera_view_names:
            hom_file = os.path.join(homo_offset_path, sub_dir + ".npy")
            assert os.path.exists(hom_file), hom_file
            ori_homo = np.load(hom_file).astype("float32")
            H.append(ori_homo)
        return np.stack(H, axis=0)

    def collect_samples(self):
        """Create data samples from data_path."""

        postfix = ".jpg"
        if self.front_img_name == "camera_front":
            att_json_path = os.path.join(self.data_path, "attribute.json")
        else:
            att_json_path = os.path.join(self.data_path, "attribute_v1.json")
        assert os.path.exists(att_json_path)
        with open(att_json_path, "r") as fin:
            att_file = json.load(fin)

        # whether to use 30fps
        if (
            "unsync" in att_file.keys()
            and self.front_img_name in att_file["unsync"].keys()
        ):
            imgs_list = att_file["unsync"]
        else:
            imgs_list = att_file["sync"]

        self.sample_lines = []
        sync_list = att_file["sync"]
        for i in range(len(sync_list[self.front_img_name])):
            file_list_before = []
            file_list_current = []
            for camera_view_name in self.camera_view_names:
                if (
                    sync_list[camera_view_name][i]
                    >= imgs_list[camera_view_name][1]
                ):
                    # get index in imgs_list according sync_list
                    _time_stamp = sync_list[camera_view_name][i]
                    unsync_view_index = imgs_list[camera_view_name].index(
                        _time_stamp
                    )
                    file_list_before.append(
                        str(imgs_list[camera_view_name][unsync_view_index - 1])
                        + postfix
                    )
                    file_list_current.append(
                        str(imgs_list[camera_view_name][unsync_view_index])
                        + postfix
                    )
            if len(file_list_current) == len(self.camera_view_names):
                if self.only_front_view:
                    # for 2.5d
                    self.front_idx = self.camera_view_names.index(
                        self.front_img_name
                    )
                    self.sample_lines.append(
                        [
                            file_list_current[self.front_idx],
                            file_list_before[self.front_idx],
                        ]
                    )
                else:
                    # for bev
                    self.sample_lines.append(
                        [file_list_current, file_list_before]
                    )

    def get_image_path(self, img_dir, name, postfix=".jpg"):
        image_path = os.path.join(self.data_path, img_dir, name + postfix)
        assert os.path.exists(image_path), image_path
        return image_path

    def get_image(self, img_dir, name, postfix, mode="RGB", size=None):
        image = pil_loader(
            self.get_image_path(img_dir, name, postfix), mode, size=size
        )
        return image

    def __len__(self):
        return len(self.sample_lines)

    @property
    def sample_list(self):
        return self.sample_lines

    def __getitem__(self, index):
        data_dict = {}
        img_names = self.sample_lines[index]

        data_dict["pil_imgs"] = []
        for img_names_i in img_names:
            imgs = []
            if self.only_front_view:  # only load front view img
                imgs.append(
                    self.get_image(
                        self.front_img_name,
                        img_names_i,
                        "",
                        "RGB",
                        size=self.img_load_size,
                    )
                )
            else:
                for camera_view_name, img_name in zip(
                    self.camera_view_names, img_names_i
                ):
                    imgs.append(
                        self.get_image(
                            camera_view_name,
                            img_name,
                            "",
                            "RGB",
                            size=self.img_load_size,
                        )
                    )
            data_dict["pil_imgs"].append(imgs)

        # homograpy
        H = []
        if self.H_origin is not None:
            for camera_view_name, _img_name in zip(
                self.camera_view_names, img_names[-1]
            ):
                i = self.camera_view_names.index(camera_view_name)
                currennt_H = self.H_origin[i]
                currennt_H = self.pers_view_intr_norm_mat[i] @ currennt_H
                H.append(currennt_H)
            data_dict["homography"] = np.stack(H, axis=0)
        if self.homo_offset is not None:
            data_dict["homo_offset"] = self.homo_offset.copy()
        if self.only_front_view:
            data_dict["obj_mask"] = self.get_image(
                f"seg_{self.front_img_name}",
                img_names[1].replace("jpg", "png"),
                "",
                "F",
            )

        # To compatible with 2d task, set t-1 as img_name frame.
        # data_dict['img_name'] = _as_list(img_names[-1])[0]
        data_dict["img_name"] = _as_list(img_names[0])[0]
        cam_front_intri = self.intrinsics.copy()[:3, :3]
        cam_front_intri[0] /= self.per_view_shape[self.front_img_name][1]
        cam_front_intri[1] /= self.per_view_shape[self.front_img_name][0]
        data_dict["intrinsics"] = cam_front_intri
        data_dict["distortcoef"] = self.distortcoef.copy()
        if self.get_pack_dir:
            data_dict["pack_dir"] = self.data_path

        if self.transforms is not None:
            data_dict = self.transforms(data_dict)
        return data_dict

    def __repr__(self):
        repr_str = self.__class__.__name__ + ": "
        repr_str += f"data_path={self.data_path}, "
        return repr_str


@OBJECT_REGISTRY.register
class ANCBev3DDatasetRec(data.Dataset):
    """The Bev3D dataset only for parsing real3d recs and \
        for bev3d training(validation).

    Args:
        img_data_path: Paths for recs.
        homo_path: Path of homograph matrix npy file.
        num_classes: Number of classes.
        transforms: Transforms that applies to the data.
        views: View number of dataset.
        per_view_shape: Img shape of each view.
        sub_dirs: Sub directory name of each view.
        homo_gen: HomoGenerator object for compute homography.
        flag_for_group int: flag to distinguish data in different shape of
             the camera module. Data of the same shape has the same flag
             value, even if it comes from different camera modules, so as
             to be concatenated by the DistributedGroupSampler correctly.
    """

    def __init__(
        self,
        img_data_path: Union[str, Sequence],
        num_classes: int,
        per_view_shape: Mapping,
        sub_dirs: Sequence[str],
        views: int = 6,
        transforms: Optional[Sequence] = None,
        homo_gen: Optional[object] = None,
        flag_for_group: Optional[int] = 1,
        homo_gen_small: Optional[object] = None,
    ):
        super(ANCBev3DDatasetRec, self).__init__()
        self.rec_paths = img_data_path
        self.transforms = transforms
        self.num_classes = num_classes
        self.views = views
        self.sub_dirs = sub_dirs
        self.per_view_shape = per_view_shape

        assert self.num_classes in [
            1,
            3,
        ], "currently the number of classes must be 3 or 1"

        # get homography matrix.
        self.meta_info = None
        self.homo_gen = homo_gen
        if homo_gen is not None:
            self.camera_view_names = [
                f"camera_{_sub_dir}" for _sub_dir in self.sub_dirs
            ]
            self.homo_genor = HomoGenerator(**homo_gen)

        self.homo_gen_small = homo_gen_small
        if homo_gen_small is not None:
            self.camera_view_names = [
                f"camera_{_sub_dir}" for _sub_dir in self.sub_dirs
            ]
            self.homo_genor_small = HomoGenerator(**homo_gen_small)

        # load rec img
        self.load_rec(self.rec_paths)
        self.num_samples = self.acc_lengths[-1]
        self.flag = flag_for_group * np.ones(len(self), dtype=np.uint8)

    def __getstate__(self):
        state = self.__dict__
        return state

    def __setstate__(self, state):
        self.__dict__ = state

    def load_rec(self, rec_paths):
        assert isinstance(rec_paths, list) and len(rec_paths) % 2 == 0
        self.recs = []
        for rec_paths in [
            rec_paths[i : i + 2] for i in range(0, len(rec_paths), 2)
        ]:
            recs_tmp = []
            for rec_path in rec_paths:
                logger.info(f"loading {rec_path}")
                rec_reader = Real3DRecReader(rec_path)
                recs_tmp.append(rec_reader)
            self.recs.append(recs_tmp)
        lengths = [len(rec[-1]) for rec in self.recs]
        self.acc_lengths = np.cumsum(lengths)

    def __len__(self):
        return self.num_samples

    def _get_rec_from_idx(self, idx):
        """Get recio from index."""
        assert idx < len(self)
        for length_idx in range(len(self.acc_lengths)):
            length_i = self.acc_lengths[length_idx]
            if idx > (length_i - 1):
                continue
            rec = self.recs[length_idx]
            if length_idx == 0:
                previous_idx = 0
            else:
                previous_idx = self.acc_lengths[length_idx - 1]
            idx_in_rec = idx - previous_idx
            assert idx_in_rec >= 0
            if idx_in_rec >= len(rec[-1]):
                idx_in_rec = len(rec[-1]) - 1
            return rec, idx_in_rec

    def __getitem__(self, index):
        anns, img, img_info = [], [], []
        rec, idx = self._get_rec_from_idx(index)
        imgs, labels = [rec[-1][idx][0]], [rec[-1][idx][-1]]
        for i in range(
            idx * (self.views - 1), idx * (self.views - 1) + (self.views - 1)
        ):
            imgs.append(rec[0][i][0])
            labels.append(rec[0][i][-1])

        # Adapt to different real3d data formats
        image_key = (
            "file_name"
            if "file_name" in labels[0]["meta"].keys()
            else "image_key"
        )

        # reorder the image by the sub_dir'
        for sub_dir in self.sub_dirs:
            for _label in labels:
                _label_view_name = _label["meta"][image_key].split("__")[1]
                if _label_view_name.split("_0820")[0] == sub_dir:
                    index = labels.index(_label)
                    img_info.append(_label["meta"])
                    img.append(Image.fromarray(imgs[index]))
                    anns.append(_label["objects"])
                    break

        img_id = [str(_iminfo["id"]) for _iminfo in img_info]
        image_name = [_iminfo[image_key] for _iminfo in img_info]

        data_dict = {
            "image_name": image_name,
            "image_id": img_id,
            "img": img,
            "timestamp": np.array([float(labels[0]["meta"]["timestamp"])]),
            "num_classes": self.num_classes,
            "annotations": anns,
        }
        if self.homo_gen is not None:
            data_dict["meta_info"] = self.homo_genor.load_homo_offset()
        if self.homo_gen_small is not None:
            data_dict[
                "meta_info_small"
            ] = self.homo_genor_small.load_homo_offset()
        if self.transforms:
            data_dict = self.transforms(data_dict)
        return data_dict
