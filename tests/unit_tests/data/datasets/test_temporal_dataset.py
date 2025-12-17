import os

import numpy as np
import pytest

from hat.data.datasets.temporal_dataset import TemporalLMDBDataset
from hat.registry import build_from_registry
from tests import HAT_BUCKET_PATH, MATRIX_BUCKET_EXISTS, MATRIX_BUCKET_PATH
from tests.utils import check, check_shape, check_type


@pytest.mark.skipif(not MATRIX_BUCKET_EXISTS, reason="requiring MATRIX bucket")
def test_temporal_lmdb_dataset():
    history_frames = 2
    future_frames = 2
    lmdb_path = os.path.join(
        MATRIX_BUCKET_PATH, "users/xuewu.lin/data/nuScenes/LMDB/v1.0-trainval"
    )
    dataset = TemporalLMDBDataset(
        lmdb_path=lmdb_path,
        num_seq_split=2,
        history_frames=history_frames,
        future_frames=future_frames,
    )
    assert len(dataset) == 34149
    for i in range(0, 34149, 1000):
        data = dataset[i]
        assert len(data["data_queue"]) <= history_frames
        assert len(data["future_data_queue"]) <= future_frames


def test_temporal_json_dataset():

    json_file = os.path.join(
        HAT_BUCKET_PATH,
        "users/zixiang.pei/unit_test/test_temporal_json_dataset/mv_json_file.json",  # noqa
    )
    datset_config = dict(
        type="TemporalJsonDataset",
        json_file=json_file,
        max_interval=600,
    )

    dataset = build_from_registry(datset_config)

    assert len(dataset) == 2

    data_1 = dataset[0]
    data_2 = dataset[1]
    assert "meta" in data_1
    assert "timestamp" in data_1["meta"]
    assert int(data_1["meta"]["timestamp"]) == 1676686645533
    assert (
        int(data_2["meta"]["timestamp"]) - int(data_1["meta"]["timestamp"])
        == 500
    )


def test_temporal_lmdb_dataset_v2():

    lmdb_path = os.path.join(
        HAT_BUCKET_PATH,
        "users/zixiang.pei/unit_test/test_temporal_lmdb_dataset",  # noqa
    )
    datset_config = dict(
        type="TemporalLmdbDataset",
        idx_path=os.path.join(lmdb_path, "idx"),
        anno_path=os.path.join(lmdb_path, "anno"),
        max_interval=600,
        max_len_in_clip=20,
        event_key="event_id",
    )

    dataset = build_from_registry(datset_config)

    assert len(dataset) == 116

    data_1 = dataset[0]
    data_2 = dataset[1]
    assert "meta" in data_1
    assert "timestamp" in data_1["meta"]
    assert int(data_1["meta"]["timestamp"]) == 1673799701966
    assert (
        int(data_2["meta"]["timestamp"]) - int(data_1["meta"]["timestamp"])
        == 100
    )


def get_transforms(model_setting="x3c"):

    view_names = [
        "front_left",
        "front_right",
        "rear_left",
        "rear_right",
        "rear",
        "front",
    ]

    category_id_dict = {
        1: 0,  # Pedestrian
        2: 1,  # Car        -> Car
        3: 2,  # Cyclist
        4: 1,  # Bus        -> Car
        5: 1,  # Truck      -> Car
        6: 1,  # SpecialCar -> Car
        7: 1,  # Blur       -> Car
        8: -99,  # Other    -> ignore
    }  # 'Dontcare' -> Ignore

    def get_view_shape(model_setting="x3c"):
        if "as33" in model_setting:
            raw_image_hw = (1280, 2048)
        else:
            raw_image_hw = (1280, 1920)
        per_view_shape = {
            "camera_front_left": raw_image_hw,
            "camera_front_right": raw_image_hw,
            "camera_rear_left": raw_image_hw,
            "camera_rear_right": raw_image_hw,
            "camera_rear": raw_image_hw,
            "camera_front": (2160, 3840),
        }
        return per_view_shape

    transforms_ = [
        dict(
            type="MultiViewRecPadView",
            camera_view_names=view_names,
            view_shapes=get_view_shape(model_setting),
        ),
        dict(
            type="GetCalibParams",
            camera_view_names=view_names,
            view_shapes=get_view_shape(model_setting),
        ),
        dict(
            type="MultiViewRecTransform",
            category_id_dict=category_id_dict,
            camera_view_names=view_names,
        ),
    ]

    return transforms_


@pytest.mark.skipif(True, reason="need update")
def test_temporal_json_dataset_for_mvjson():

    json_file = os.path.join(
        HAT_BUCKET_PATH,
        "users/zixiang.pei/unit_test/test_temporal_json_dataset/mv_json_file.json",  # noqa
    )
    img_dir = os.path.join(
        HAT_BUCKET_PATH,
        "users/zixiang.pei/unit_test/test_temporal_json_dataset/imgs",  # noqa
    )

    datset_config = dict(
        type="TemporalJsonDataset",
        json_file=json_file,
        transforms=[
            dict(
                type="MVImgJsonReader",
                img_dir=img_dir,
                to_rgb=True,
            )
        ]
        + get_transforms("x3c"),
        max_interval=600,
    )

    dataset = build_from_registry(datset_config)
    assert len(dataset) == 2
    data = dataset[1]

    assert type(data["imgs"]) == list
    assert len(data["imgs"]) == 6
    check(data["imgs"], check_type, instance=np.ndarray)
    for i in range(5):
        check(data["imgs"][i], check_shape, shape=(1280, 1920, 3))
    check(data["imgs"][5], check_shape, shape=(2160, 3840, 3))
    assert data["imgs"][5].dtype == np.uint8
    check(data["ori_camera_matrix"], check_type, instance=np.ndarray)
    check(data["ori_camera_matrix"], check_shape, shape=(6, 3, 3))
    check(data["ori_distcoeffs"], check_shape, shape=(6, 8))
    check(data["ori_vcs2cam"], check_shape, shape=(6, 4, 4))
    check(data["ori_image_size"], check_shape, shape=(6, 2))
    assert tuple(data["ori_image_size"][0]) == (1920, 1280)
    assert tuple(data["ori_image_size"][5]) == (3840, 2160)
    check(data["camera_matrix"], check_shape, shape=(6, 3, 3))
    check(data["T_vcs2cam"], check_shape, shape=(6, 4, 4))
    check(data["T_vcs2global"], check_shape, shape=(4, 4))
    check(data["T_global2vcs"], check_shape, shape=(4, 4))
    assert data["gt_bboxes_3d"].dtype == np.float32
    check(data["gt_bboxes_3d"], check_shape, shape=(32, 7))
    assert data["gt_labels_3d"].dtype == np.int64
    check(data["gt_labels_3d"], check_shape, shape=(32,))
    assert data["timestamp"] * 1e3 == 1676686646033


@pytest.mark.skipif(True, reason="need update")
def test_temporal_lmdb_dataset_for_mvjson():

    lmdb_path = os.path.join(
        HAT_BUCKET_PATH,
        "users/zixiang.pei/unit_test/test_temporal_lmdb_dataset",  # noqa
    )
    datset_config = dict(
        type="TemporalLmdbDataset",
        idx_path=os.path.join(lmdb_path, "idx"),
        anno_path=os.path.join(lmdb_path, "anno"),
        transforms=[
            dict(
                type="MVImgLmdbReader",
                img_path=os.path.join(lmdb_path, "img"),
                to_rgb=True,
            )
        ]
        + get_transforms("x3c"),
        max_interval=600,
        max_len_in_clip=20,
        event_key="event_id",
    )

    dataset = build_from_registry(datset_config)
    assert len(dataset) == 116
    data = dataset[1]

    assert data["views_pad"] == [5]

    assert type(data["imgs"]) == list
    assert len(data["imgs"]) == 6
    check(data["imgs"], check_type, instance=np.ndarray)
    for i in range(5):
        check(data["imgs"][i], check_shape, shape=(1280, 1920, 3))
    check(data["imgs"][5], check_shape, shape=(2160, 3840, 3))
    assert data["imgs"][5].dtype == np.uint8
    check(data["ori_camera_matrix"], check_type, instance=np.ndarray)
    check(data["ori_camera_matrix"], check_shape, shape=(6, 3, 3))
    check(data["ori_distcoeffs"], check_shape, shape=(6, 8))
    check(data["ori_vcs2cam"], check_shape, shape=(6, 4, 4))
    check(data["ori_image_size"], check_shape, shape=(6, 2))
    assert tuple(data["ori_image_size"][0]) == (1920, 1280)
    assert tuple(data["ori_image_size"][5]) == (3840, 2160)
    check(data["camera_matrix"], check_shape, shape=(6, 3, 3))
    check(data["T_vcs2cam"], check_shape, shape=(6, 4, 4))
    check(data["T_vcs2global"], check_shape, shape=(4, 4))
    check(data["T_global2vcs"], check_shape, shape=(4, 4))
    assert data["gt_bboxes_3d"].dtype == np.float32
    check(data["gt_bboxes_3d"], check_shape, shape=(2, 7))
    assert data["gt_labels_3d"].dtype == np.int64
    check(data["gt_labels_3d"], check_shape, shape=(2,))
    assert data["timestamp"] * 1e3 == 1673799702066


if __name__ == "__main__":
    pytest.main(["-s", __file__])
