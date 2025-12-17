import pytest

from hat.data.datasets.eye3d_pose_dataset import Eye3dPoseDataset


def test_eye3d_pose_dataset():
    """Test eye3d_pose dataset using lmdb image, mask and anno."""

    argvs = [
        {"txtpath": "tmp_orig_data/pnpnet/test_40pts_mini.txt"},
        {
            "txtpath": "",
            "synthetic_argv": {
                "length": 5,
                "face_model": "tmp_orig_data/pnpnet/facemodel.txt",
                "range_list": [
                    {
                        "mean": [0, 0, 500],
                        "scale": [150, 150, 300],
                        "pose_scale": [30, 40, 50],
                        "pose_mean": [0, 0, 0],
                        "ratio": 2,
                    },
                    {
                        "mean": [0, 0, 400],
                        "scale": [500, 200, 200],
                        "pose_scale": [30, 40, 50],
                        "pose_mean": [0, 0, 0],
                        "ratio": 1,
                    },
                ],
            },
        },
    ]
    dataset = Eye3dPoseDataset(argvs)
    for idx in [0, len(dataset) - 1]:
        item = dataset[idx]
        assert isinstance(item, dict)
        for key in ["undistort_points_70", "mtx"]:
            assert key in item


if __name__ == "__main__":
    pytest.main(["-s", __file__])
