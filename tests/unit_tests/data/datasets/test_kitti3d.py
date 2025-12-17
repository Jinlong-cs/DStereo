import os
import uuid

import numpy as np
import pytest

from hat.data.datasets.kitti3d import (
    Kitti3D,
    Kitti3DDetection,
    Kitti3DDetectionPacker,
    create_groundtruth_database,
    create_kitti_info_file,
    create_reduced_point_cloud,
)
from tests.utils import execute_cmd


class TestKitti3D:
    @classmethod
    def setup_class(cls):
        uuid1 = str(uuid.uuid1())
        cls.tmp_dir = os.path.join("kitti3d_" + uuid1)
        if not os.path.exists(cls.tmp_dir):
            os.makedirs(cls.tmp_dir)
        cmd = f"""
        cp /horizon-bucket/HDLTAlgorithm/pipeline_test_data/kitti3d_data/origin_data/kitti3d_mini.tar.gz . # noqa: E501
        tar xf kitti3d_mini.tar.gz -C {cls.tmp_dir} --strip-components 2
        rm kitti3d_mini.tar.gz
        """
        execute_cmd(cmd)

    def test_reader(self):
        create_kitti_info_file(self.tmp_dir)
        create_reduced_point_cloud(self.tmp_dir)
        create_groundtruth_database(self.tmp_dir)

        expected_pkl_names = [
            "kitti3d_infos_val.pkl",
            "kitti3d_infos_train.pkl",
            "kitti3d_dbinfos_train.pkl",
        ]

        for pkl_name in expected_pkl_names:
            assert os.path.exists(os.path.join(self.tmp_dir, pkl_name))

    @pytest.mark.parametrize(
        [
            "split",
        ],
        [
            pytest.param("train"),
            pytest.param("val"),
        ],
    )
    def test_origin_kitti3d(self, split):
        dataset = Kitti3DDetection(
            source_path=self.tmp_dir,
            split_name=split,
        )

        assert len(dataset) == 3
        for _, data in enumerate(dataset):
            lidar_info = data["lidar"]
            points = lidar_info["points"]
            assert isinstance(points, np.ndarray)
            assert points.shape[-1] == 4

            gt_bbox = lidar_info["annotations"]["boxes"]

            assert isinstance(gt_bbox, np.ndarray)
            assert gt_bbox.shape[-1] == 7

    @pytest.mark.parametrize(
        [
            "split",
        ],
        [
            pytest.param("train"),
            pytest.param("val"),
        ],
    )
    def test_pack_kitti3d(self, split, pack_type="lmdb"):
        target_dir = os.path.join(self.tmp_dir, f"{split}_{pack_type}")
        packer = Kitti3DDetectionPacker(
            src_data_dir=self.tmp_dir,
            target_data_dir=target_dir,
            split_name="train",
            num_workers=2,
            pack_type=pack_type,
        )
        packer()

        assert os.path.exists(os.path.join(target_dir, "data.mdb"))
        assert os.path.exists(os.path.join(target_dir, "lock.mdb"))

    @pytest.mark.parametrize(
        [
            "split",
        ],
        [
            pytest.param("train"),
            pytest.param("val"),
        ],
    )
    def test_lmdb_kitti3d(self, split, pack_type="lmdb"):

        kitti3d_data = Kitti3D(
            data_path=os.path.join(self.tmp_dir, f"{split}_{pack_type}"),
            transforms=None,
        )

        assert len(kitti3d_data) == 3
        for _, data in enumerate(kitti3d_data):
            lidar_info = data["lidar"]
            points = lidar_info["points"]
            assert isinstance(points, np.ndarray)
            assert points.shape[-1] == 4

            gt_bbox = lidar_info["annotations"]["boxes"]

            assert isinstance(gt_bbox, np.ndarray)
            assert gt_bbox.shape[-1] == 7

    # @classmethod
    # def teardown_class(cls):
    #     shutil.rmtree(cls.tmp_dir)
