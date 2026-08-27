from __future__ import annotations
import copy

import numpy as np
import pytest
import torch

from hat.data.collates.collates import collate_disp_cat
from hat.data.datasets.multi_disp_dataset.augment_dataset import AugDataset
from hat.data.datasets.multi_disp_dataset.resize_aware import (
    FIXED_RESIZE_SCALE,
    ResizeAwareStereo,
    build_resize_aware_spec,
    resize_aware_config,
)


class _SyntheticStereoDataset:
    def __init__(self, disparity=None):
        self.left = np.full((352, 640, 3), 127, dtype=np.uint8)
        self.right = np.full((352, 640, 3), 63, dtype=np.uint8)
        self.disparity = (
            np.full((352, 640), 10.0, dtype=np.float32)
            if disparity is None
            else disparity
        )
        self.file_list = [["left.jpg", "right.jpg", "disparity.npz"]]
        self.root_dir = "/synthetic"
        self.name = "SyntheticStereo"

    def __getitem__(self, index):
        assert index == 0
        return self.left.copy(), self.right.copy(), self.disparity.copy()

    def __len__(self):
        return 1

    @staticmethod
    def pad_image_to_multiple_of_32(array, value=0):
        return array


def _build_augmented_dataset(disparity=None, resize=True):
    return AugDataset(
        _SyntheticStereoDataset(disparity),
        False,
        96,
        None,
        None,
        None,
        ["center", 352, 640],
        resize_aware_args=resize_aware_config() if resize else None,
    )


def test_s080_geometry_and_disparity_units():
    spec = build_resize_aware_spec()
    assert spec.scale == FIXED_RESIZE_SCALE
    assert spec.content_shape == (282, 512)
    assert spec.tensor_shape == (288, 512)
    assert spec.padding == (3, 3, 0, 0)
    assert spec.horizontal_scale == pytest.approx(0.8)
    assert spec.vertical_scale == pytest.approx(282 / 352)

    image = np.full((352, 640, 3), 127, dtype=np.uint8)
    disparity = np.full((352, 640), 10.0, dtype=np.float32)
    left, right, disparity_out, metadata = ResizeAwareStereo()(
        image, image, disparity
    )

    assert left.shape == right.shape == (288, 512, 3)
    assert disparity_out.shape == (288, 512)
    np.testing.assert_array_equal(left[:3], 0)
    np.testing.assert_array_equal(right[-3:], 0)
    np.testing.assert_array_equal(disparity_out[:3], 0.0)
    np.testing.assert_array_equal(disparity_out[-3:], 0.0)
    np.testing.assert_allclose(disparity_out[3:-3], 8.0)
    assert metadata["resize_content_shape"] == (282, 512)
    assert metadata["resize_tensor_shape"] == (288, 512)


def test_invalid_disparity_cannot_become_valid_after_scaling():
    disparity = np.full((352, 640), 10.0, dtype=np.float32)
    disparity[:, :128] = 96.0
    disparity[:, 128:256] = 120.0
    disparity[:, 256:320] = np.nan
    image = np.zeros((352, 640, 3), dtype=np.uint8)

    _, _, disparity_out, _ = ResizeAwareStereo()(image, image, disparity)

    assert np.isfinite(disparity_out).all()
    assert set(np.unique(disparity_out)).issubset({0.0, 8.0})
    assert np.count_nonzero(disparity_out == 0.0) > 0


def test_profile_rejects_other_scales_and_noncanonical_inputs():
    with pytest.raises(ValueError, match="only supports fixed scale"):
        build_resize_aware_spec(scale=0.75)

    transform = ResizeAwareStereo()
    image = np.zeros((352, 639, 3), dtype=np.uint8)
    disparity = np.zeros((352, 639), dtype=np.float32)
    with pytest.raises(ValueError, match="canonical 640x352"):
        transform(image, image, disparity)


def test_augdataset_and_collate_emit_static_s080_tensors():
    dataset = _build_augmented_dataset()
    sample = dataset[0]

    assert sample["img"].shape == (2, 3, 288, 512)
    assert sample["gt_disp"].shape == (1, 288, 512)
    assert sample["resize_scale"] == pytest.approx(0.8)
    assert sample["origin_shape"] == (282, 512)
    assert sample["mask_flag"] is True

    batch = collate_disp_cat([sample, dataset[0]])
    assert batch["img"].shape == (4, 3, 288, 512)
    assert batch["gt_disp"].shape == (2, 1, 288, 512)
    expected_scale = torch.tensor(
        [0.8, 0.8], dtype=batch["resize_scale"].dtype
    )
    torch.testing.assert_close(batch["resize_scale"], expected_scale)


def test_resize_validity_ratio_is_frozen_before_padding():
    disparity = np.full((352, 640), 10.0, dtype=np.float32)
    disparity[:, :220] = 96.0
    sample = _build_augmented_dataset(disparity)[0]
    assert sample["mask_flag"] is False


def test_legacy_augdataset_contract_is_unchanged():
    sample = _build_augmented_dataset(resize=False)[0]
    assert sample["img"].shape == (2, 3, 352, 640)
    assert sample["gt_disp"].shape == (352, 640)
    assert "resize_scale" not in sample


def test_s080_config_is_isolated_and_keeps_the_base_recipe():
    import DStereo.DStereoPlus as base

    base_data_loader = copy.deepcopy(base.data_loader)
    base_val_data_loader = copy.deepcopy(base.val_data_loader)
    import DStereo.DStereoPlus_resize_aware_s080 as config

    assert base.data_loader == base_data_loader
    assert base.val_data_loader == base_val_data_loader
    assert config.resize_aware_scale == 0.8
    assert config.resize_content_shape == (282, 512)
    assert config.resize_tensor_shape == (288, 512)
    assert config.resize_padding == (3, 3, 0, 0)

    dataset = config.data_loader["dataset"]
    assert dataset["res_args"] == [352, 640, False]
    assert dataset["resize_aware_args"]["scale"] == 0.8
    assert config.data_loader["sampler"] == base.data_loader["sampler"]
    assert config.data_loader["batch_size"] == 8

    val_dataset = config.val_data_loader["dataset"]
    assert "resize_aware_args" not in val_dataset
    assert val_dataset["res_args"] == [352, 640, False]
    assert val_dataset["crop_args"] == ["center", 352, 640]

    assert config.model == base.model
    assert config.num_steps == 200000
    assert config.maxdisp == 96
    assert config.base_lr == 0.0001
    assert config.val_interval == 5000
    assert config.cudnn_benchmark == base.cudnn_benchmark
    assert config.float_trainer["data_loader"] is config.data_loader
    initialization = config.float_trainer["model_convert_pipeline"][
        "converters"
    ][0]
    assert initialization["checkpoint_path"] == base.checkpoint_path
    assert initialization["allow_miss"] is False
    assert initialization["ignore_extra"] is False
    assert initialization["ignore_tensor_shape"] is False


def test_channel_aligned_s080_labels_satisfy_the_base_float_loss():
    from hat.models.structures.stereo_match.dstereoplus import DStereoPlus

    class _LossContext:
        maxdisp = 96

        @staticmethod
        def _maybe_dequantize(value):
            return value

    prediction = torch.full(
        (2, 1, 16, 32), 9.0, dtype=torch.float32, requires_grad=True
    )
    labels = torch.full((2, 1, 16, 32), 10.0, dtype=torch.float32)
    losses = DStereoPlus.sequence_loss(
        _LossContext(), prediction, [prediction, prediction], labels
    )
    total_loss = torch.stack(losses).sum()

    assert torch.isfinite(total_loss)
    total_loss.backward()
    assert prediction.grad is not None
    assert torch.isfinite(prediction.grad).all()


class _ScalarMetric:
    def __init__(self):
        self.value = None

    def update(self, value):
        self.value = value


class _EpeMetric:
    def __init__(self):
        self.values = None

    def update(self, labels, predictions, masks):
        self.values = labels, predictions, masks


def test_s080_metric_normalizes_train_and_validation_shapes():
    import DStereo.DStereoPlus_resize_aware_s080 as config

    loss_metric = _ScalarMetric()
    epe_metric = _EpeMetric()
    labels = torch.full((2, 16, 32), 10.0)
    predictions = torch.full((2, 1, 16, 32), 9.0)
    losses = [torch.tensor(1.0), torch.tensor(2.0)]
    config.update_s080_loss_metric(
        [loss_metric, epe_metric],
        {"gt_disp": labels},
        {"losses": losses, "pred_disps": predictions},
    )

    assert loss_metric.value.item() == pytest.approx(3.0)
    metric_labels, metric_predictions, masks = epe_metric.values
    assert metric_labels.shape == metric_predictions.shape == (2, 16, 32)
    assert masks.all()


def test_s080_metric_rejects_unexpected_geometry():
    import DStereo.DStereoPlus_resize_aware_s080 as config

    with pytest.raises(ValueError, match="geometry mismatch"):
        config.update_s080_loss_metric(
            [_ScalarMetric(), _EpeMetric()],
            {"gt_disp": torch.ones(1, 16, 32)},
            (torch.ones(1, 1, 8, 16),),
        )
