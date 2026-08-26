from __future__ import annotations
import os

import numpy as np
import pytest

from hat.data.datasets.multi_disp_dataset.resize_aware import ResizeAwareStereo
from hat.data.samplers.stereo_scale_sampler import StereoScaleSampler


def _nested_strings(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from _nested_strings(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _nested_strings(item)


def test_s080_profile_geometry_and_transform():
    import DStereo.DStereoPlus_resize_aware_s080 as config

    assert config.resize_aware_scales == (0.8,)
    assert config.resize_content_shape == (282, 512)
    assert config.resize_tensor_shape == (288, 512)
    assert config.resize_padding == (3, 3, 0, 0)
    assert config.resize_aware_spec.horizontal_scale == pytest.approx(0.8)

    image = np.full((352, 640, 3), 127, dtype=np.uint8)
    disparity = np.full((352, 640), 10.0, dtype=np.float32)
    transform = ResizeAwareStereo(**config.resize_aware_args)

    left, right, disparity_out, metadata = transform(
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
    assert metadata["resize_tensor_shape"] == (288, 512)


def test_s080_training_and_calibration_use_the_fixed_profile():
    import DStereo.DStereoPlus_resize_aware_s080 as config

    for loader in (config.data_loader, config.calib_data_loader):
        assert loader["dataset"]["res_args"] == [352, 640, False]
        assert loader["dataset"]["resize_aware_args"] == (
            config.resize_aware_args
        )
        assert loader["sampler"]["type"] is StereoScaleSampler
        assert loader["sampler"]["scales"] == (0.8,)
        assert loader["sampler"]["drop_last"] is True
        assert loader["drop_last"] is True

    assert config.data_loader["sampler"]["batch_size"] == (
        config.train_batch_size_per_gpu
    )
    assert config.calib_data_loader["sampler"]["batch_size"] == (
        config.test_batch_size_per_gpu
    )
    assert config.calib_data_loader["sampler"]["sampler_len"] == 50
    assert config.float_trainer["data_loader"] is config.data_loader

    for predictor in (
        config.float_predictor,
        config.floatonnx_predictor,
        config.quantonnx_predictor,
    ):
        assert predictor["data_loader"][0] is config.calib_data_loader


def test_s080_export_is_static_and_validation_remains_canonical():
    import DStereo.DStereoPlus_resize_aware_s080 as config

    assert os.path.basename(config.export_dir) == "export_s080_512x288"
    assert config.onnx_cfg["inputs"] is config.deploy_inputs
    assert config.onnx_cfg["out_dir"] == config.export_dir
    assert (
        config.onnx_cfg["model_convert_pipeline"]["converters"][0][
            "checkpoint_path"
        ]
        == config.float_checkpoint_path
    )
    for tensor in config.deploy_inputs["data"].values():
        assert tuple(tensor.shape) == (1, 3, 288, 512)

    val_dataset = config.val_data_loader["dataset"]
    assert "resize_aware_args" not in val_dataset
    assert val_dataset["res_args"] == [352, 640, False]
    assert val_dataset["crop_args"] == ["center", 352, 640]
    assert config.val_callback["data_loader"] is config.val_data_loader

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


def test_s080_profile_does_not_mutate_multiscale_config():
    import DStereo.DStereoPlus_resize_aware as multiscale
    import DStereo.DStereoPlus_resize_aware_s080 as config

    expected_scales = tuple(
        round(1.00 - 0.05 * index, 2) for index in range(15)
    )
    assert multiscale.resize_aware_scales == expected_scales
    assert multiscale.data_loader["sampler"]["scales"] == expected_scales
    assert os.path.basename(multiscale.export_dir) == "export_640x352"
    assert config.data_loader is not multiscale.data_loader
    assert config.calib_data_loader is not multiscale.calib_data_loader

    scoped_objects = (
        config.predict_callbacks,
        config.float_predictor,
        config.onnx_cfg,
        config.floatonnx_predictor,
        config.quantonnx_predictor,
    )
    for path in _nested_strings(scoped_objects):
        assert path != multiscale.ckpt_dir
        assert not path.startswith(multiscale.ckpt_dir + os.sep)
        assert path != multiscale.export_dir
        assert not path.startswith(multiscale.export_dir + os.sep)
