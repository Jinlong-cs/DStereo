from __future__ import annotations
import inspect

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from hat.data.datasets.multi_disp_dataset.cat_random_datast import (
    CatRandomDataset,
)
from hat.data.datasets.multi_disp_dataset.resize_aware import (
    ResizeAwareStereo,
    build_resize_aware_specs,
    resize_aware_config,
)
from hat.data.samplers.stereo_scale_sampler import StereoScaleSampler

SCALES = tuple(round(1.00 - 0.05 * index, 2) for index in range(15))
EXPECTED_GEOMETRY = (
    (1.00, (352, 640), (352, 640), (0, 0, 0, 0)),
    (0.95, (334, 608), (352, 608), (9, 9, 0, 0)),
    (0.90, (317, 576), (320, 576), (1, 2, 0, 0)),
    (0.85, (299, 544), (320, 544), (10, 11, 0, 0)),
    (0.80, (282, 512), (288, 512), (3, 3, 0, 0)),
    (0.75, (264, 480), (288, 480), (12, 12, 0, 0)),
    (0.70, (246, 448), (256, 448), (5, 5, 0, 0)),
    (0.65, (229, 416), (256, 416), (13, 14, 0, 0)),
    (0.60, (211, 384), (224, 384), (6, 7, 0, 0)),
    (0.55, (194, 352), (224, 352), (15, 15, 0, 0)),
    (0.50, (176, 320), (192, 320), (8, 8, 0, 0)),
    (0.45, (158, 288), (160, 288), (1, 1, 0, 0)),
    (0.40, (141, 256), (160, 256), (9, 10, 0, 0)),
    (0.35, (123, 224), (128, 224), (2, 3, 0, 0)),
    (0.30, (106, 192), (128, 192), (11, 11, 0, 0)),
)


class _RangeDataset(Dataset):
    def __init__(self, length):
        self.length = length

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        return index


class _ScaleShapeDataset(Dataset):
    def __init__(self, length):
        self.length = length
        self.specs = {
            spec.scale: spec for spec in build_resize_aware_specs(SCALES)
        }

    def __len__(self):
        return self.length

    def __getitem__(self, item):
        _, scale = item
        spec = self.specs[scale]
        return {
            "payload": torch.zeros(spec.tensor_shape),
            "scale": scale,
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


class _FakeStereoDataset(Dataset):
    name = "fake-stereo"
    root_dir = "/fake"
    file_list = [["left.png", "right.png"]]

    def __init__(self, disparity):
        self.disparity = disparity.astype(np.float32)

    def __len__(self):
        return 1

    def __getitem__(self, index):
        left = np.full((352, 640, 3), (10, 20, 30), dtype=np.uint8)
        right = np.full((352, 640, 3), (30, 20, 10), dtype=np.uint8)
        return left, right, self.disparity.copy()


def _content(array, spec):
    height_end = spec.tensor_height - spec.pad_bottom
    width_end = spec.tensor_width - spec.pad_right
    return array[spec.pad_top : height_end, spec.pad_left : width_end]


def _nested_strings(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from _nested_strings(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _nested_strings(item)


def test_all_requested_scales_have_expected_32_aligned_geometry():
    specs = build_resize_aware_specs(SCALES)
    observed = tuple(
        (spec.scale, spec.content_shape, spec.tensor_shape, spec.padding)
        for spec in specs
    )
    assert observed == EXPECTED_GEOMETRY
    assert all(spec.tensor_height % 32 == 0 for spec in specs)
    assert all(spec.tensor_width % 32 == 0 for spec in specs)


def test_effective_disparity_scale_uses_configured_base_width():
    spec = build_resize_aware_specs([0.5], base_height=200, base_width=320)[0]
    assert spec.content_shape == (100, 160)
    assert spec.horizontal_scale == pytest.approx(0.5)


def test_scale_one_preserves_valid_images_and_disparity():
    row = np.arange(640, dtype=np.uint8)
    left = np.repeat(row[None, :, None], 352, axis=0)
    left = np.repeat(left, 3, axis=2)
    right = np.flip(left, axis=1).copy()
    disparity = np.full((352, 640), 12.5, dtype=np.float32)

    transformed = ResizeAwareStereo([1.0])
    left_out, right_out, disparity_out, metadata = transformed(
        left, right, disparity, 1.0
    )

    np.testing.assert_array_equal(left_out, left)
    np.testing.assert_array_equal(right_out, right)
    np.testing.assert_array_equal(disparity_out, disparity)
    assert metadata["resize_content_shape"] == (352, 640)
    assert metadata["resize_padding"] == (0, 0, 0, 0)


@pytest.mark.parametrize(
    "invalid_value", [0.0, -1.0, 96.0, 100.0, np.inf, np.nan]
)
def test_invalid_disparity_stays_invalid_after_downscale(invalid_value):
    image = np.zeros((352, 640, 3), dtype=np.uint8)
    disparity = np.full((352, 640), invalid_value, dtype=np.float32)
    transformed = ResizeAwareStereo([0.9], max_disp=96.0)
    spec = transformed.spec_for(0.9)

    _, _, disparity_out, _ = transformed(image, image, disparity, 0.9)

    np.testing.assert_array_equal(disparity_out, 0.0)
    assert _content(disparity_out, spec).shape == (317, 576)


def test_valid_disparity_uses_actual_horizontal_scale():
    image = np.zeros((352, 640, 3), dtype=np.uint8)
    disparity = np.full((352, 640), 95.0, dtype=np.float32)
    transformed = ResizeAwareStereo([0.9], max_disp=96.0)
    spec = transformed.spec_for(0.9)

    _, _, disparity_out, metadata = transformed(image, image, disparity, 0.9)

    np.testing.assert_allclose(_content(disparity_out, spec), 85.5)
    assert metadata["resize_horizontal_scale"] == pytest.approx(0.9)


def test_padding_is_black_for_bgr_and_zero_for_disparity():
    left = np.full((352, 640, 3), 127, dtype=np.uint8)
    right = np.full((352, 640, 3), 63, dtype=np.uint8)
    disparity = np.full((352, 640), 10.0, dtype=np.float32)
    transformed = ResizeAwareStereo([0.3])

    left_out, right_out, disparity_out, metadata = transformed(
        left, right, disparity, 0.3
    )

    assert left_out.shape == (128, 192, 3)
    assert right_out.shape == (128, 192, 3)
    assert disparity_out.shape == (128, 192)
    np.testing.assert_array_equal(left_out[:11], 0)
    np.testing.assert_array_equal(right_out[-11:], 0)
    np.testing.assert_array_equal(disparity_out[:11], 0.0)
    np.testing.assert_array_equal(disparity_out[-11:], 0.0)
    assert metadata["resize_padding"] == (11, 11, 0, 0)


def test_sampler_keeps_batches_uniform_and_covers_each_scale_per_cycle():
    dataset = CatRandomDataset([_ScaleShapeDataset(60)])
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
    assert loader.batch_size == sampler.batch_size
    assert sampler.num_batches == 30
    for batch in loader:
        scales = batch["scale"].tolist()
        assert scales[0] == scales[1]
        assert batch["payload"].shape[0] == 2
        observed.append(scales[0])

    assert set(observed[:15]) == set(SCALES)
    assert set(observed[15:30]) == set(SCALES)
    assert observed[:15] != observed[15:30]


def test_sampler_is_reproducible_and_changes_with_epoch():
    dataset = CatRandomDataset([_RangeDataset(120)])
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
    fresh_epoch_zero = StereoScaleSampler(
        dataset, 2, SCALES, seed=23, num_replicas=1, rank=0
    )
    assert list(first) != list(fresh_epoch_zero)


def test_two_ranks_use_the_same_scale_at_each_step():
    dataset = CatRandomDataset([_RangeDataset(240)])
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
            token[0] for token in one_batch
        )


def test_sampler_drops_only_incomplete_global_tail_and_can_skip_empty_child():
    dataset = CatRandomDataset([_RangeDataset(5), _RangeDataset(5)])
    rank_zero = StereoScaleSampler(
        dataset, 2, [1.0], shuffle=False, num_replicas=2, rank=0
    )
    rank_one = StereoScaleSampler(
        dataset, 2, [1.0], shuffle=False, num_replicas=2, rank=1
    )
    combined = {token[0] for token in list(rank_zero) + list(rank_one)}
    assert len(rank_zero) == len(rank_one) == 4
    assert combined == set(range(10)) - {4, 9}

    with_empty = CatRandomDataset([_RangeDataset(0), _RangeDataset(8)])
    sampler = StereoScaleSampler(
        with_empty,
        2,
        [1.0],
        shuffle=False,
        drop_empty=True,
        num_replicas=1,
        rank=0,
    )
    assert {token[0] for token in sampler} == set(range(8))


def test_cat_random_retry_preserves_scale_and_malformed_tags_fail_fast():
    child = _RetryDataset()
    dataset = CatRandomDataset([child])
    result = dataset[(0, 0.3)]
    assert result["mask_flag"]
    assert len(child.calls) == 2
    assert [item[1] for item in child.calls] == [0.3, 0.3]

    with pytest.raises(ValueError, match="pairs"):
        dataset[(0, 0.3, "extra")]
    with pytest.raises(IndexError, match="out of range"):
        dataset[(len(dataset), 0.3)]


def test_aug_dataset_applies_resize_before_nv12_and_collates_metadata():
    from hat.data.collates.collates import collate_disp_cat
    from hat.data.datasets.multi_disp_dataset.augment_dataset import AugDataset

    disparity = np.full((352, 640), 10.0, dtype=np.float32)
    disparity[:112] = 0.0
    dataset = AugDataset(
        base_dataset=_FakeStereoDataset(disparity),
        test_mode=False,
        max_disp=96,
        norm_args=["MixVarGENet"],
        crop_args=["center", 352, 640],
        resize_aware_args=resize_aware_config([0.3], max_disp=96),
    )

    first = dataset[(0, 0.3)]
    second = dataset[(0, 0.3)]
    batch = collate_disp_cat([first, second])

    assert first["img"].shape == (2, 3, 128, 192)
    assert first["gt_disp"].shape == (128, 192)
    assert first["origin_shape"] == (106, 192)
    assert first["resize_tensor_shape"] == (128, 192)
    assert first["resize_padding"] == (11, 11, 0, 0)
    assert first["sample_idx"] == 0
    assert first["mask_flag"]
    np.testing.assert_array_equal(first["left_img"][:11], 0)
    torch.testing.assert_close(
        first["img"][0, :, :11],
        torch.tensor([-0.875, 0.0, 0.0])[:, None, None].expand(-1, 11, 192),
    )

    assert batch["img"].shape == (4, 3, 128, 192)
    assert batch["gt_disp"].shape == (2, 128, 192)
    assert batch["resize_scale"] == [0.3, 0.3]
    assert batch["resize_content_shape"] == [(106, 192), (106, 192)]


def test_aug_dataset_rejects_mismatched_geometry_and_index_modes():
    from hat.data.datasets.multi_disp_dataset.augment_dataset import AugDataset

    base_dataset = _FakeStereoDataset(np.ones((352, 640), dtype=np.float32))
    config = resize_aware_config([0.3], max_disp=96)

    with pytest.raises(ValueError, match="max_disp must match"):
        AugDataset(
            base_dataset=base_dataset,
            test_mode=False,
            max_disp=95,
            crop_args=["center", 352, 640],
            resize_aware_args=config,
        )
    with pytest.raises(ValueError, match="crop size"):
        AugDataset(
            base_dataset=base_dataset,
            test_mode=False,
            max_disp=96,
            crop_args=["center", 320, 640],
            resize_aware_args=config,
        )

    aware = AugDataset(
        base_dataset=base_dataset,
        test_mode=False,
        max_disp=96,
        crop_args=["center", 352, 640],
        resize_aware_args=config,
    )
    with pytest.raises(ValueError, match="requires"):
        aware[0]

    legacy = AugDataset(
        base_dataset=base_dataset,
        test_mode=False,
        max_disp=96,
        crop_args=["center", 352, 640],
    )
    legacy_sample = legacy[0]
    assert legacy_sample["img"].shape == (2, 3, 352, 640)
    assert "resize_scale" not in legacy_sample
    with pytest.raises(ValueError, match="disabled"):
        legacy[(0, 0.3)]


def test_resize_aware_options_preserve_legacy_positional_parameters():
    from hat.data.datasets.multi_disp_dataset.augment_dataset import AugDataset
    from hat.data.datasets.multi_disp_dataset.stereo_multi_data import (
        StereoMultiData,
    )

    assert tuple(inspect.signature(AugDataset).parameters) == (
        "base_dataset",
        "test_mode",
        "max_disp",
        "aug_args",
        "res_args",
        "norm_args",
        "crop_args",
        "debug",
        "img_open_mode",
        "skip",
        "resize_aware_args",
    )
    assert tuple(inspect.signature(StereoMultiData).parameters) == (
        "dataset_list",
        "max_disp",
        "test_mode",
        "aug_args",
        "res_args",
        "norm_args",
        "crop_args",
        "debug",
        "img_open_mode",
        "resize_aware_args",
    )
    aug_signature = inspect.signature(AugDataset)
    stereo_signature = inspect.signature(StereoMultiData)
    assert (
        aug_signature.parameters["resize_aware_args"].kind
        is inspect.Parameter.KEYWORD_ONLY
    )
    assert aug_signature.parameters["resize_aware_args"].default is None
    assert (
        stereo_signature.parameters["resize_aware_args"].kind
        is inspect.Parameter.KEYWORD_ONLY
    )
    assert stereo_signature.parameters["resize_aware_args"].default is None
    assert "resize_aware_args" not in aug_signature.bind(*range(10)).arguments
    assert (
        "resize_aware_args" not in stereo_signature.bind(*range(9)).arguments
    )


def test_resize_aware_config_scopes_predict_and_export_artifacts():
    import os

    import DStereo.DStereoPlus as base_config
    import DStereo.DStereoPlus_resize_aware as config

    assert (
        config.float_predictor["model_convert_pipeline"]["converters"][0][
            "checkpoint_path"
        ]
        == config.float_checkpoint_path
    )
    assert (
        config.onnx_cfg["model_convert_pipeline"]["converters"][0][
            "checkpoint_path"
        ]
        == config.float_checkpoint_path
    )
    assert config.onnx_cfg["out_dir"] == config.export_dir
    assert config.predict_callbacks[0]["output_dir"] == os.path.join(
        config.export_dir, "calib_data"
    )
    assert config.floatonnx_predictor["model"]["onnx_path"] == os.path.join(
        config.export_dir, "float.onnx"
    )
    assert config.quantonnx_predictor["model"]["onnx_path"] == os.path.join(
        config.export_dir,
        "Bin_model",
        "DStereo_quantized_model.onnx",
    )
    assert config.floatonnx_predictor["callbacks"][2][
        "output_dir"
    ] == os.path.join(config.export_dir, "vis", "float")
    assert config.quantonnx_predictor["callbacks"][2][
        "output_dir"
    ] == os.path.join(config.export_dir, "vis", "quant")

    for predictor in (
        config.float_predictor,
        config.floatonnx_predictor,
        config.quantonnx_predictor,
    ):
        assert predictor["data_loader"][0] is config.calib_data_loader

    scoped_objects = (
        config.predict_callbacks,
        config.float_predictor,
        config.onnx_cfg,
        config.floatonnx_predictor,
        config.quantonnx_predictor,
    )
    for path in _nested_strings(scoped_objects):
        assert path != base_config.ckpt_dir
        assert not path.startswith(base_config.ckpt_dir + os.sep)
        assert path != "ptq_V21"
        assert not path.startswith("ptq_V21" + os.sep)
