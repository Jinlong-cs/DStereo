import copy

import pytest

from hat.models.task_modules.depth import DepthDecoder, DepthMultiDecoder
from tests.utils import gen_fake_seg_data


@pytest.mark.parametrize(
    ["out_stride"],
    [
        pytest.param([2]),
        pytest.param([4]),
    ],
)
def test_depth_decoder(out_stride):
    w, h = 896, 896
    preds_with_input_size, _ = gen_fake_seg_data(
        w, h, strides=[1], num_classes=2, n=8, label_name="gt_depth"
    )
    preds, labels = gen_fake_seg_data(
        w, h, strides=out_stride, num_classes=2, n=8, label_name="gt_depth"
    )
    depth_decoder = DepthDecoder(
        in_strides=out_stride[0],
        out_stride=out_stride[0],
        gt_scale=1.0,
        output_name="depth_preds",
    )
    preds = tuple(preds)
    ret = depth_decoder(preds, labels)
    assert ret["depth_preds"].shape == preds_with_input_size[0].shape


@pytest.mark.parametrize(
    ["out_stride"],
    [
        pytest.param([2, 4, 8]),
    ],
)
def test_multi_input_decoder(out_stride):
    pred_names = []
    decoder_modules = []
    w, h = 896, 896
    preds, labels = gen_fake_seg_data(
        w, h, strides=out_stride, num_classes=2, n=8, label_name="gt_depth"
    )
    preds_new = {}
    for i in range(len(out_stride)):
        pred_names.append(f"depth_{i}")
        preds_new[pred_names[i]] = copy.deepcopy([preds[i]])
        decoder_module = DepthDecoder(
            in_strides=out_stride[i],
            out_stride=out_stride[i],
            gt_scale=1.0,
            output_name=pred_names[i],
        )
        decoder_modules.append(decoder_module)
    depth_multi_decoder = DepthMultiDecoder(
        pred_names=pred_names,
        decoder_modules=decoder_modules,
    )
    resize_preds = depth_multi_decoder(preds_new, labels)
    for _, preds in resize_preds.items():
        assert preds.shape[2:] == (h, w), f"{preds.shape}"


@pytest.mark.parametrize(
    ["out_stride"],
    [
        pytest.param([2, 4, 8]),
    ],
)
def test_multi_scale_decoder(out_stride):

    w, h = 896, 896
    for i in range(len(out_stride)):
        preds, labels = gen_fake_seg_data(
            w, h, strides=out_stride, num_classes=2, n=8, label_name="gt_depth"
        )
        pred_names = "depth"
        decoded_pred_name = "decoded_depth"
        decoder_module = DepthDecoder(
            in_strides=out_stride,
            out_stride=out_stride[i],
            gt_scale=1.0,
            output_name=decoded_pred_name,
        )
        decoder_modules = [decoder_module]
        depth_multi_decoder = DepthMultiDecoder(
            pred_names=pred_names,
            decoder_modules=decoder_modules,
        )
        preds = {pred_names: preds}
        resize_preds = depth_multi_decoder(preds, labels)
        for _, preds in resize_preds.items():
            assert preds.shape[2:] == (h, w), f"{preds.shape}"
