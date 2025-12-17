from typing import List

import numpy as np
import pytest

from hat.core.data_struct.base_struct import Mask
from hat.models.task_modules.semanticfpn import BMSegDecoder
from tests.utils import gen_fake_feats


@pytest.mark.parametrize(
    [
        "out_stride",
        "input_shape",
        "do_inverse_transform",
        "class_mapping",
        "to_cpu",
    ],
    [
        pytest.param(2, [512, 1024], False, None, False),
        pytest.param(4, [512, 1024], True, None, False),
        pytest.param(2, [256, 512], True, [0, 1, 2, 2, 3], True),
    ],
)
def test_semanticfpn_decoder(
    out_stride, input_shape, do_inverse_transform, class_mapping, to_cpu
):
    cls_num = 5
    batch_size = 2

    semanticfpn_decoder = BMSegDecoder(
        out_strides=[out_stride],
        do_inverse_transform=do_inverse_transform,
        to_cpu=to_cpu,
        class_mapping=class_mapping,
    )

    fake_res_data, _, _ = gen_fake_feats(
        input_shape[1], input_shape[0], [out_stride], batch_size, cls_num
    )
    if do_inverse_transform:
        ori_shape = (540, 1280)
        fake_label = dict(
            img_shape=[np.array(input_shape)] * batch_size,
            img_height=[ori_shape[0]] * batch_size,
            img_width=[ori_shape[1]] * batch_size,
        )

    else:
        ori_shape = input_shape
        fake_label = None

    pred_rets = semanticfpn_decoder(fake_res_data, fake_label)

    assert isinstance(pred_rets, List) and len(pred_rets) == batch_size
    for ret in pred_rets:
        assert isinstance(ret, Mask)
        if class_mapping:
            assert ret.mask.max() <= max(class_mapping)
        if to_cpu:
            assert ret.mask.device.type == "cpu"
        ret_shape = ret.mask.shape
        assert ret_shape[0] == ori_shape[0] and ret_shape[1] == ori_shape[1]
