import pytest
import torch

from hat.models.backbones.mmdetbackbone_adaptor import MMDetBackboneAdaptor

try:
    import mmcv
    import mmdet

    _MMDET_IMPORTED = True
except ImportError:
    _MMDET_IMPORTED = False


@pytest.mark.skipif(not _MMDET_IMPORTED, reason="mmdet and mmcv is required")
def test_mmdetbackbone_adaptor():

    assert mmcv.__version__
    assert mmdet.__version__

    model = MMDetBackboneAdaptor(
        # backbone can be copied from mmdet config, here we copied from
        # mmdet/configs/swin/mask_rcnn_swin-t-p4-w7_fpn_ms-crop-3x_coco.py
        backbone=dict(
            type="SwinTransformer",
            embed_dims=96,
            depths=[2, 2, 6, 2],
            num_heads=[3, 6, 12, 24],
            window_size=7,
            mlp_ratio=4,
            qkv_bias=True,
            qk_scale=None,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            drop_path_rate=0.2,
            patch_norm=True,
            out_indices=(0, 1, 2, 3),
            with_cp=False,
            convert_weights=True,
            # init_cfg=dict(type='Pretrained', checkpoint=pretrained)
        ),
        # skip pretrained model for tests
    )
    model.eval()
    data = torch.randn(1, 3, 224, 224)
    feature_strides = [4, 8, 16, 32]
    # The backbone takes a 4D image tensor and returns a list of tensors.
    outputs = model(data)
    for (i, output) in enumerate(outputs):
        assert isinstance(output, torch.Tensor)
        assert len(output.shape) == 4
        assert output.shape[-1] == 224 // feature_strides[i]
