import horizon_plugin_pytorch as horizon
import numpy as np
import pytest
import torch

from hat.core.differentiable_camera.warping_module import WarpingModule


@pytest.mark.parametrize(
    "image_size, uv_map_size",
    [
        ([1280, 1920], [640, 960]),
        ([100, 200], [50, 40]),
        ([30, 30], [224, 224]),
    ],
)
def test_warping_module(image_size, uv_map_size):
    img = torch.tensor(
        np.random.random(size=[4, 3, *image_size]) * 2 - 1,
        dtype=torch.float,
    )
    uv_map = torch.tensor(
        np.random.random(size=[4, *uv_map_size, 2]),
        dtype=torch.float,
    )
    max_side_length = max(image_size + uv_map_size)
    warping_module = WarpingModule(max_side_length=max_side_length)
    output = warping_module(img, uv_map)
    assert output.shape[-2] == uv_map_size[0]
    assert output.shape[-1] == uv_map_size[1]

    warping_module.set_qconfig()
    horizon.quantization.prepare_qat(warping_module)
    output = warping_module(img, uv_map)
    assert output.shape[-2] == uv_map_size[0]
    assert output.shape[-1] == uv_map_size[1]

    quantized_model = horizon.quantization.convert(warping_module.eval())
    output = quantized_model(img, uv_map)
    assert output.shape[-2] == uv_map_size[0]
    assert output.shape[-1] == uv_map_size[1]
