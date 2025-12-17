import pytest
import torch
from torchvision.transforms import Compose

from hat.data.datasets.gaze.gaze_dataset import GazeRecDataset
from hat.data.transforms.detection import Normalize, Resize, ToTensor
from hat.data.transforms.gaze import GazeRandomCropWoResize, GazeYUVTransform
from hat.registry import build_from_registry

transforms = Compose(
    [
        Resize(img_scale=(192, 320), keep_ratio=False),
        ToTensor(),
        Normalize(mean=128.0, std=128.0),
    ]
)

filename = "./tmp_orig_data/gaze/hat_test/mega.rec"


@pytest.mark.parametrize("", [()])
def test_landmark_rec_dataset():
    dataset = GazeRecDataset(
        filename=filename,
        input_size=(320, 192),
        transforms=transforms,
    )
    item = dataset[0]
    assert "img" in item
    assert isinstance(item["img"], torch.Tensor)
    assert "gaze_label" in item
    assert "gt_gaze" in item["gaze_label"]
    assert "gt_normed_eye_ldmk" in item["gaze_label"]


@pytest.mark.parametrize("", [()])
def test_gaze_dataset():
    config = dict(
        type="GazeDataset",
        rec_list=[
            "./tmp_orig_data/face/gaze_selfsup/"
            + "gaze_mtl_eyeldmk_202/training/mega.rec",
            "./tmp_orig_data/face/gaze_selfsup/"
            + "gaze_mtl_eyeldmk_da05_new/training/mega.rec",
        ],
        input_size=(320, 192),
        transforms=transforms,
    )
    gaze_dataset = build_from_registry(config)
    item = gaze_dataset[0]
    assert "img" in item
    assert isinstance(item["img"], torch.Tensor)
    assert "gaze_label" in item
    assert "gt_gaze" in item["gaze_label"]
    assert "gt_normed_eye_ldmk" in item["gaze_label"]


@pytest.mark.parametrize("", [()])
def test_gaze_dataset_wt_gazemap():
    gazemap_settings = {
        "active": True,
        "size": (80, 48),
        "feature_dim": 64,
        "eyeball_weight": 1.0,
        "iris_weight": 1.0,
        "ignore_sample_with_eyeball_only": True,
        "visualize": True,
    }

    for method in [
        "calc",
        # "fix",
        # "fix_circle",
        "fix_cuberoot",
        "fix_circle_squeeze",
        "fix_cuberoot_squeeze",
    ]:
        _transforms = Compose(
            [
                GazeRandomCropWoResize(
                    size=(320, 192),
                    prob=1.0,
                    area=(0.85, 1.0),
                    ratio=(1.25, 2),
                    is_train=False,
                ),
                GazeYUVTransform(
                    rgb_data=False,
                    nc=3,
                    equalize_hist=True,
                    equalize_hist_method=method,
                ),
                Resize(img_scale=(192, 320), keep_ratio=False),
                ToTensor(),
                Normalize(mean=128.0, std=128.0),
            ]
        )

        config = dict(
            type="GazeDataset",
            rec_list=[
                "./tmp_orig_data/face/gazemap/cd569-01-trainset-Cam0_norm_noalign_sim_wt_gazemap/cd569-01-trainset-Cam0_norm_noalign_sim_0_wt_gazemap.rec",  # noqa
                "./tmp_orig_data/face/gazemap/cd569-02-trainset-Cam0_norm_noalign_sim_wt_gazemap/cd569-02-trainset-Cam0_norm_noalign_sim_0_wt_gazemap.rec",  # noqa
            ],
            input_size=(320, 192),
            transforms=_transforms,
            angle_form="degree",
            gazemap_settings=gazemap_settings,
        )
        gaze_dataset = build_from_registry(config)
        item = gaze_dataset[0]
        assert "img" in item
        assert isinstance(item["img"], torch.Tensor)
        assert "gaze_label" in item
        assert "gt_gaze" in item["gaze_label"]
        assert "gt_normed_eye_ldmk" in item["gaze_label"]
        assert "gt_gazemap" in item["gaze_label"]
        assert "gt_gazemap_weight" in item["gaze_label"]


if __name__ == "__main__":
    pytest.main(["-s", __file__])
