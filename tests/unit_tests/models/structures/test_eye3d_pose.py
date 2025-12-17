import pytest
import torch
from torch.utils.data import DataLoader

from hat.data.datasets.eye3d_pose_dataset import Eye3dPoseDataset
from hat.data.transforms.eye3d_pose import Eye3dPoseTransformList
from hat.models.structures.eye3d_pose import Eye3dPoseModel
from hat.registry import build_from_registry
from tests.unit_tests.models.base import qat_test


def test_eye3d_pose_model():
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
    input_channels = dataset[0]["img"].shape[0]
    dataloader = DataLoader(dataset, 2)

    model_train = build_from_registry(
        dict(
            type="Eye3dPoseModel",
            num_mod=len(data_list),
            input_channels=input_channels,
            backbone_scale_list=[128] * 4,
            neck_scale_list=[64] * 2,
            losses=torch.nn.L1Loss(),
            deploy=False,
            bn_kwargs={
                "eps": 2e-05,
                "momentum": 0.1,
            },
        )
    )
    model_val = Eye3dPoseModel(
        num_mod=len(data_list),
        input_channels=input_channels,
        backbone_scale_list=[128] * 4,
        neck_scale_list=[64] * 2,
        losses=torch.nn.L1Loss(),
        deploy=True,
        bn_kwargs={
            "eps": 2e-05,
            "momentum": 0.1,
        },
    )
    model_val.eval()

    for batch in dataloader:
        pred, loss = model_train(batch)
        assert isinstance(pred, dict)
        assert isinstance(loss, torch.Tensor)
        for key in ["pred_pose", "pred_eye3d", "loss_dict"]:
            assert key in pred
        assert pred["pred_pose"].detach().numpy().shape == (2, 6, 1, 1)
        assert pred["pred_eye3d"].detach().numpy().shape == (2, 6, 1, 1)

        rot, eye3d = model_val(batch)
        assert rot is not None
        assert eye3d is not None
        qat_test(model_val, batch)
        break


if __name__ == "__main__":
    pytest.main(["-s", __file__])
