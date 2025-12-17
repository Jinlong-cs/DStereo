import torch

from hat.registry import build_from_registry


def test_resize_parser_inplace():
    size = (512, 960)

    config = dict(
        type="ResizeParser",
        data_name="pred_segs",
        resize_kwargs=dict(size=size, mode="bilinear"),
    )

    resize_parser = build_from_registry(config)
    pred_segs = [
        torch.randn((1, 9, 512 // 2, 960 // 2)),
        torch.randn((1, 9, 512 // 4, 960 // 4)),
        torch.randn((1, 9, 512 // 8, 960 // 8)),
    ]

    pred_dict = dict()
    pred_dict["pred_segs"] = pred_segs
    pred_dict = resize_parser(pred_dict)

    for one_stride in pred_dict["pred_segs"]:
        assert one_stride.shape[-2:] == size


def test_resize_parser():
    size = (512, 960)

    config = dict(
        type="ResizeParser",
        data_name="pred_segs",
        resized_data_name="pred_segs_resized",
        resize_kwargs=dict(size=size, mode="bilinear"),
    )

    resize_parser = build_from_registry(config)
    pred_segs = [
        torch.randn((1, 9, 512 // 2, 960 // 2)),
        torch.randn((1, 9, 512 // 4, 960 // 4)),
        torch.randn((1, 9, 512 // 8, 960 // 8)),
    ]

    pred_dict = dict()
    pred_dict["pred_segs"] = pred_segs
    pred_dict = resize_parser(pred_dict)

    for one_stride in pred_dict["pred_segs_resized"]:
        assert one_stride.shape[-2:] == size
