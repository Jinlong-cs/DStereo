from hat.registry import build_from_registry
from projects.halo.cv.configs.gesture.plugins.get_sta_demo_config import (
    get_lmdb_dataset,
)


def test_landmark_rec_dataset():
    # "/horizon-bucket/HDLTAlgorithm/data/orig_data/action/gesture/test_lmdb_data/T18_hand_gesture_IMS-IR_front_v5_static_v2.0_ldmk2.5dv3.0.3"  # noqa
    test_data_path = "./tmp_orig_data/action/gesture/test_lmdb_data/T18_hand_gesture_IMS-IR_front_v5_static_v2.0_ldmk2.5dv3.0.3"  # noqa
    train_dataset_dict, val_dataset_dict = get_lmdb_dataset()
    val_dataset_dict["datasets"] = [val_dataset_dict["datasets"][0]]
    val_dataset_dict["datasets"][0]["dataset"]["lmdb_path"] = test_data_path
    train_dataset = build_from_registry(val_dataset_dict)

    item = train_dataset[0]
    assert "frames" in item
    assert "clip_keypoints" in item
    assert "act_label" in item
    assert "label_weight" in item
    assert len(item["frames"]) == 8
