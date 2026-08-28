from __future__ import annotations
import copy

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from hat.data.collates.collates import collate_disp_cat
from hat.data.datasets.multi_disp_dataset.augment_dataset import AugDataset
from hat.data.datasets.multi_disp_dataset.cat_random_datast import (
    CatRandomDataset,
)
from hat.data.datasets.multi_disp_dataset.resize_aware import (
    DEFAULT_RESIZE_SCALES,
    ResizeAwareStereo,
    build_resize_aware_specs,
    resize_aware_config,
)
from hat.data.samplers.stereo_scale_sampler import StereoScaleSampler

SCALES = (1.0, 0.8)


class _SyntheticStereoDataset:
    def __init__(self, disparity=None, length=1):
        self.left = np.full((352, 640, 3), 127, dtype=np.uint8)
        self.right = np.full((352, 640, 3), 63, dtype=np.uint8)
        self.disparity = (
            np.full((352, 640), 10.0, dtype=np.float32)
            if disparity is None
            else disparity
        )
        self.length = length
        self.file_list = [
            [f"left-{index}.jpg", f"right-{index}.jpg", "disparity.npz"]
            for index in range(length)
        ]
        self.root_dir = "/synthetic"
        self.name = "SyntheticStereo"

    def __getitem__(self, index):
        assert 0 <= index < self.length
        return self.left.copy(), self.right.copy(), self.disparity.copy()

    def __len__(self):
        return self.length

    @staticmethod
    def pad_image_to_multiple_of_32(array, value=0):
        return array


class _ScaleShapeDataset(Dataset):
    def __init__(self, length):
        self.length = length
        self.specs = {
            spec.scale: spec for spec in build_resize_aware_specs(SCALES)
        }

    def __len__(self):
        return self.length

    def __getitem__(self, item):
        index, scale = item
        spec = self.specs[scale]
        return {
            "index": index,
            "payload": torch.zeros(spec.tensor_shape),
            "resize_scale": scale,
            "mask_flag": True,
        }


class _RetryDataset(Dataset):
    def __init__(self):
        self.calls = []

    def __len__(self):
        return 4

    def __getitem__(self, item):
        self.calls.append(item)
        return {"mask_flag": len(self.calls) > 1, "item": item}


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


class _NamedMetric:
    def __init__(self, name, value):
        self.name = name
        self.value = value

    def get(self):
        return self.name, self.value


def _build_augmented_dataset(
    disparity=None,
    *,
    resize=True,
    scales=SCALES,
    test_mode=False,
    length=1,
):
    return AugDataset(
        _SyntheticStereoDataset(disparity, length=length),
        test_mode,
        96,
        None,
        None,
        None,
        ["center", 352, 640],
        resize_aware_args=(
            resize_aware_config(scales, max_disp=96) if resize else None
        ),
    )


def test_dual_scale_geometry_and_disparity_units():
    s100, s080 = build_resize_aware_specs(SCALES)
    assert DEFAULT_RESIZE_SCALES == SCALES
    assert s100.content_shape == (352, 640)
    assert s100.tensor_shape == (352, 640)
    assert s100.padding == (0, 0, 0, 0)
    assert s100.horizontal_scale == pytest.approx(1.0)
    assert s080.content_shape == (282, 512)
    assert s080.tensor_shape == (288, 512)
    assert s080.padding == (3, 3, 0, 0)
    assert s080.horizontal_scale == pytest.approx(0.8)
    assert s080.vertical_scale == pytest.approx(282 / 352)


def test_s100_is_pixel_exact_for_valid_inputs():
    row = np.arange(640, dtype=np.uint8)
    left = np.repeat(row[None, :, None], 352, axis=0)
    left = np.repeat(left, 3, axis=2)
    right = np.flip(left, axis=1).copy()
    disparity = np.full((352, 640), 10.0, dtype=np.float32)

    left_out, right_out, disparity_out, metadata = ResizeAwareStereo(SCALES)(
        left, right, disparity, 1.0
    )

    np.testing.assert_array_equal(left_out, left)
    np.testing.assert_array_equal(right_out, right)
    np.testing.assert_array_equal(disparity_out, disparity)
    assert metadata["resize_tensor_shape"] == (352, 640)


def test_s080_resizes_full_fov_and_pads_black():
    image = np.full((352, 640, 3), 127, dtype=np.uint8)
    disparity = np.full((352, 640), 10.0, dtype=np.float32)

    left, right, disparity_out, metadata = ResizeAwareStereo(SCALES)(
        image, image, disparity, 0.8
    )

    assert left.shape == right.shape == (288, 512, 3)
    assert disparity_out.shape == (288, 512)
    np.testing.assert_array_equal(left[:3], 0)
    np.testing.assert_array_equal(right[-3:], 0)
    np.testing.assert_array_equal(disparity_out[:3], 0.0)
    np.testing.assert_array_equal(disparity_out[-3:], 0.0)
    np.testing.assert_allclose(disparity_out[3:-3], 8.0)
    assert metadata["resize_content_shape"] == (282, 512)


def test_invalid_disparity_cannot_become_valid_after_scaling():
    disparity = np.full((352, 640), 10.0, dtype=np.float32)
    disparity[:, :128] = 96.0
    disparity[:, 128:256] = 120.0
    disparity[:, 256:320] = np.nan
    image = np.zeros((352, 640, 3), dtype=np.uint8)

    _, _, disparity_out, _ = ResizeAwareStereo(SCALES)(
        image, image, disparity, 0.8
    )

    assert np.isfinite(disparity_out).all()
    assert set(np.unique(disparity_out)).issubset({0.0, 8.0})
    assert np.count_nonzero(disparity_out == 0.0) > 0


def test_transform_rejects_unconfigured_scale_and_noncanonical_input():
    transform = ResizeAwareStereo(SCALES)
    image = np.zeros((352, 639, 3), dtype=np.uint8)
    disparity = np.zeros((352, 639), dtype=np.float32)
    with pytest.raises(ValueError, match="not in configured scales"):
        transform(image, image, disparity, 0.75)
    with pytest.raises(ValueError, match="canonical 640x352"):
        transform(image, image, disparity, 0.8)


@pytest.mark.parametrize(
    ("scale", "image_shape", "disparity_shape"),
    [
        (1.0, (2, 3, 352, 640), (1, 352, 640)),
        (0.8, (2, 3, 288, 512), (1, 288, 512)),
    ],
)
def test_augdataset_emits_both_channel_aligned_shapes(
    scale, image_shape, disparity_shape
):
    dataset = _build_augmented_dataset(test_mode=True)
    sample = dataset[(0, scale)]

    assert sample["img"].shape == image_shape
    assert sample["gt_disp"].shape == disparity_shape
    assert sample["metric_gt_disp"].shape == (1, 352, 640)
    assert sample["resize_scale"] == pytest.approx(scale)
    assert sample["mask_flag"] is True

    batch = collate_disp_cat([sample, dataset[(0, scale)]])
    assert batch["img"].shape == (4, *image_shape[1:])
    assert batch["gt_disp"].shape == (2, *disparity_shape)
    assert batch["metric_gt_disp"].shape == (2, 1, 352, 640)
    torch.testing.assert_close(
        torch.as_tensor(batch["resize_scale"], dtype=torch.float64),
        torch.tensor([scale, scale], dtype=torch.float64),
    )

    train_sample = _build_augmented_dataset(test_mode=False)[(0, scale)]
    assert "metric_gt_disp" not in train_sample


def test_multiscale_requires_tag_while_single_scale_val_accepts_scalar():
    with pytest.raises(ValueError, match="requires"):
        _build_augmented_dataset()[0]

    sample = _build_augmented_dataset(scales=(0.8,), test_mode=True)[0]
    assert sample["resize_scale"] == pytest.approx(0.8)
    assert sample["img"].shape == (2, 3, 288, 512)


def test_resize_validity_ratio_is_frozen_before_padding():
    disparity = np.full((352, 640), 10.0, dtype=np.float32)
    disparity[:, :220] = 96.0
    sample = _build_augmented_dataset(disparity)[(0, 0.8)]
    assert sample["mask_flag"] is False


def test_legacy_augdataset_contract_is_unchanged():
    dataset = _build_augmented_dataset(resize=False)
    sample = dataset[0]
    assert sample["img"].shape == (2, 3, 352, 640)
    assert sample["gt_disp"].shape == (352, 640)
    assert "resize_scale" not in sample
    assert "metric_gt_disp" not in sample
    with pytest.raises(ValueError, match="disabled"):
        dataset[(0, 0.8)]


def test_sampler_keeps_batches_uniform_and_covers_both_scales_per_cycle():
    dataset = CatRandomDataset([_ScaleShapeDataset(16)])
    sampler = StereoScaleSampler(
        dataset,
        batch_size=2,
        scales=SCALES,
        shuffle=True,
        seed=17,
        num_replicas=1,
        rank=0,
    )
    loader = DataLoader(dataset, batch_size=2, sampler=sampler)

    observed = []
    for batch in loader:
        scales = batch["resize_scale"].tolist()
        assert scales[0] == scales[1]
        observed.append(scales[0])

    assert len(observed) == 8
    for start in range(0, len(observed), 2):
        assert set(observed[start : start + 2]) == set(SCALES)
    assert observed.count(1.0) == observed.count(0.8) == 4


def test_sampler_drops_incomplete_cycle_to_keep_exact_scale_balance():
    dataset = CatRandomDataset([_ScaleShapeDataset(10)])
    sampler = StereoScaleSampler(
        dataset,
        batch_size=2,
        scales=SCALES,
        shuffle=False,
        num_replicas=1,
        rank=0,
    )

    tokens = list(sampler)
    observed = [tokens[index][1] for index in range(0, len(tokens), 2)]
    assert sampler.num_batches == 4
    assert observed == [1.0, 0.8, 1.0, 0.8]

    with pytest.raises(ValueError, match="complete scale cycle"):
        StereoScaleSampler(
            CatRandomDataset([_ScaleShapeDataset(2)]),
            batch_size=2,
            scales=SCALES,
            shuffle=False,
            num_replicas=1,
            rank=0,
        )


def test_sampler_is_reproducible_and_ddp_ranks_share_step_scale():
    dataset = CatRandomDataset([_ScaleShapeDataset(32)])
    first = StereoScaleSampler(
        dataset, 2, SCALES, seed=23, num_replicas=1, rank=0
    )
    clone = StereoScaleSampler(
        dataset, 2, SCALES, seed=23, num_replicas=1, rank=0
    )
    assert list(first) == list(clone)
    first.set_epoch(1)
    clone.set_epoch(1)
    assert list(first) == list(clone)

    rank_zero = StereoScaleSampler(
        dataset, 2, SCALES, seed=31, num_replicas=2, rank=0
    )
    rank_one = StereoScaleSampler(
        dataset, 2, SCALES, seed=31, num_replicas=2, rank=1
    )
    zero_tokens = list(rank_zero)
    one_tokens = list(rank_one)
    for start in range(0, len(zero_tokens), 2):
        zero_batch = zero_tokens[start : start + 2]
        one_batch = one_tokens[start : start + 2]
        assert {token[1] for token in zero_batch} == {
            token[1] for token in one_batch
        }
        assert {token[0] for token in zero_batch}.isdisjoint(
            {token[0] for token in one_batch}
        )


def test_tagged_retry_preserves_scale():
    child = _RetryDataset()
    dataset = CatRandomDataset([child])
    result = dataset[(0, 0.8)]

    assert result["mask_flag"]
    assert len(child.calls) == 2
    assert [item[1] for item in child.calls] == [0.8, 0.8]
    with pytest.raises(ValueError, match="pairs"):
        dataset[(0, 0.8, "extra")]
    with pytest.raises(IndexError, match="out of range"):
        dataset[(len(dataset), 0.8)]


def test_dual_scale_config_is_isolated_and_keeps_training_recipe():
    import DStereo.DStereoPlus as base

    base_data_loader = copy.deepcopy(base.data_loader)
    base_val_data_loader = copy.deepcopy(base.val_data_loader)
    import DStereo.DStereoPlus_resize_aware_s080 as config

    assert base.data_loader == base_data_loader
    assert base.val_data_loader == base_val_data_loader
    assert config.resize_aware_scales == SCALES
    assert config.s100_spec.tensor_shape == (352, 640)
    assert config.s080_spec.tensor_shape == (288, 512)
    assert config.cudnn_benchmark is False

    dataset = config.data_loader["dataset"]
    assert dataset["res_args"] == [352, 640, False]
    assert dataset["resize_aware_args"]["scales"] == [1.0, 0.8]
    assert config.data_loader["sampler"]["type"] is StereoScaleSampler
    assert config.data_loader["sampler"]["scales"] == SCALES
    assert config.data_loader["batch_size"] == 8
    assert config.data_loader["drop_last"] is True

    assert len(config.val_data_loader) == 2
    assert config.val_data_loader_s100["dataset"]["resize_aware_args"][
        "scales"
    ] == [1.0]
    assert config.val_data_loader_s080["dataset"]["resize_aware_args"][
        "scales"
    ] == [0.8]
    assert config.val_callback["share_callbacks"] is False
    assert config.val_callback["callbacks"] == [
        [config.val_metric_updater_s100],
        [config.val_metric_updater_s080],
    ]
    assert (
        config.val_metric_updater_s100["metrics"][0] is config.val_metrics[0]
    )
    assert (
        config.val_metric_updater_s080["metrics"][0] is config.val_metrics[1]
    )
    assert [metric["name"] for metric in config.val_metrics] == [
        "EPE_s100",
        "EPE_s080",
    ]
    assert config.ckpt_callback["monitor_metric_key"] == "EPE_s100"
    assert config.float_trainer["val_metrics"] is config.val_metrics
    assert config.train_callbacks[3] is config.val_callback
    assert config.train_callbacks[4] is config.ckpt_callback

    assert config.model == base.model
    assert config.num_steps == 200000
    assert config.maxdisp == 96
    assert config.base_lr == 0.0001
    assert config.val_interval == 5000
    initialization = config.float_trainer["model_convert_pipeline"][
        "converters"
    ][0]
    assert initialization["checkpoint_path"] == base.checkpoint_path
    assert initialization["allow_miss"] is False
    assert initialization["ignore_extra"] is False
    assert initialization["ignore_tensor_shape"] is False


def test_registry_reuses_scale_metrics_between_updaters_and_trainer():
    import DStereo.DStereoPlus_resize_aware_s080 as config

    from hat.registry import build_from_registry

    probe = {
        "callbacks": config.val_callback["callbacks"],
        "val_metrics": config.val_metrics,
    }
    built = build_from_registry(copy.deepcopy(probe))

    assert built["callbacks"][0][0].metrics[0] is built["val_metrics"][0]
    assert built["callbacks"][1][0].metrics[0] is built["val_metrics"][1]
    assert (
        built["callbacks"][0][0].metrics[0]
        is not built["callbacks"][1][0].metrics[0]
    )


def test_channel_aligned_labels_satisfy_base_float_loss():
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


def test_train_metric_normalizes_disparity_channel():
    import DStereo.DStereoPlus_resize_aware_s080 as config

    loss_metric = _ScalarMetric()
    epe_metric = _EpeMetric()
    labels = torch.full((2, 1, 16, 32), 10.0)
    predictions = torch.full((2, 1, 16, 32), 9.0)
    losses = [torch.tensor(1.0), torch.tensor(2.0)]
    config.update_resize_aware_train_metric(
        [loss_metric, epe_metric],
        {"gt_disp": labels},
        {"losses": losses, "pred_disps": predictions},
    )

    assert loss_metric.value.item() == pytest.approx(3.0)
    metric_labels, metric_predictions, masks = epe_metric.values
    assert metric_labels.shape == metric_predictions.shape == (2, 16, 32)
    assert masks.all()


def test_s080_validation_restores_canonical_units_and_ignores_padding():
    import DStereo.DStereoPlus_resize_aware_s080 as config

    metric = _EpeMetric()
    predictions = torch.full((1, 1, 288, 512), 1000.0)
    predictions[:, :, 3:-3] = 7.2
    canonical = torch.full((1, 1, 352, 640), 10.0)
    config.update_s080_val_metric(
        [metric],
        {"metric_gt_disp": canonical},
        (predictions,),
    )

    labels, restored, masks = metric.values
    assert labels.shape == restored.shape == (1, 352, 640)
    torch.testing.assert_close(restored, torch.full_like(restored, 9.0))
    assert masks.all()


def test_s080_validation_restores_nonconstant_disparity():
    import DStereo.DStereoPlus_resize_aware_s080 as config

    metric = _EpeMetric()
    content = torch.arange(282 * 512, dtype=torch.float32).reshape(
        1, 1, 282, 512
    )
    predictions = torch.full((1, 1, 288, 512), -1000.0)
    predictions[:, :, 3:-3] = content
    canonical = torch.ones((1, 1, 352, 640), dtype=torch.float32)

    config.update_s080_val_metric(
        [metric],
        {"metric_gt_disp": canonical},
        (predictions,),
    )

    labels, restored, masks = metric.values
    expected = (
        torch.nn.functional.interpolate(
            content,
            size=(352, 640),
            mode="bilinear",
            align_corners=False,
        )[:, 0]
        / 0.8
    )
    torch.testing.assert_close(restored, expected)
    torch.testing.assert_close(labels, canonical[:, 0])
    assert masks.all()


def test_s100_validation_keeps_canonical_units():
    import DStereo.DStereoPlus_resize_aware_s080 as config

    metric = _EpeMetric()
    canonical = torch.full((1, 1, 352, 640), 10.0)
    predictions = torch.full((1, 1, 352, 640), 9.0)
    config.update_s100_val_metric(
        [metric],
        {"metric_gt_disp": canonical},
        {"pred_disps": predictions},
    )

    labels, restored, masks = metric.values
    torch.testing.assert_close(restored, predictions[:, 0])
    torch.testing.assert_close(labels, canonical[:, 0])
    assert masks.all()


def test_validation_rejects_unexpected_scale_geometry():
    import DStereo.DStereoPlus_resize_aware_s080 as config

    with pytest.raises(ValueError, match="prediction geometry mismatch"):
        config.update_s080_val_metric(
            [_EpeMetric()],
            {"metric_gt_disp": torch.ones(1, 1, 352, 640)},
            (torch.ones(1, 1, 282, 512),),
        )


def test_wandb_logs_only_two_independent_validation_epe_keys():
    from hat.callbacks.wandb_logger import WandbLogger

    logger = WandbLogger()
    captured = []
    logger.log_metrics = lambda data, step: captured.append((data, step))
    logger.log_val_metrics(
        [
            _NamedMetric("EPE_s100", 1.25),
            _NamedMetric("EPE_s080", 1.5),
        ],
        step=4999,
    )

    assert captured == [
        (
            {
                "val/epe_s100": 1.25,
                "val/epe_s080": 1.5,
            },
            4999,
        )
    ]
    assert "val/epe" not in captured[0][0]
    assert not any("mean" in key for key in captured[0][0])
