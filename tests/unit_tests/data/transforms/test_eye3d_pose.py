import pytest

from hat.data.datasets.eye3d_pose_dataset import Eye3dPoseDataset
from hat.data.transforms.eye3d_pose import Eye3dPoseTransformList


def test_eye3d_pose_dataset():
    """Test eye3d_pose dataset using lmdb image, mask and anno."""
    argvs = [
        {"txtpath": "tmp_orig_data/pnpnet/test_40pts_mini.txt"},
    ]
    data_list = [
        {
            "input_shape": [29, 1, 1],
            "ldmk_idx": [36, 39, 45, 42, 30, 48, 54, 19, 21, 22, 24, 51, 57],
            "transform_args": {
                "t_mean": 521.34283,
                "t_std": 69.54176,
                "use_pnp_eye3d": True,
                "use_pnp_eye3d_test": True,
                "use_face_center": False,
            },
        },
        {
            "input_shape": [29, 1, 1],
            "ldmk_idx": [36, 39, 45, 42, 30, 19, 21, 22, 24],
            "transform_args": {
                "t_mean": 521.34283,
                "t_std": 69.54176,
                "use_pnp_eye3d": False,
                "use_pnp_eye3d_test": False,
                "use_face_center": True,
            },
        },
    ]
    augm = {"distribution": "normal", "params": {"clip": 0.04, "std": 0.02}}
    transform = Eye3dPoseTransformList(data_list, augm)
    dataset = Eye3dPoseDataset(argvs, transform)
    item = dataset[0]
    assert isinstance(item, dict)
    assert "img" in item
    assert item["img"].numpy().shape == (64, 1, 1)
    assert "gt_eye3d_pose" in item
    assert "gt_pose" in item["gt_eye3d_pose"]
    assert "gt_eye3d" in item["gt_eye3d_pose"]
    assert item["gt_eye3d_pose"]["gt_pose"].numpy().shape == (6, 1, 1)
    assert item["gt_eye3d_pose"]["gt_eye3d"].numpy().shape == (6, 1, 1)


if __name__ == "__main__":
    pytest.main(["-s", __file__])
