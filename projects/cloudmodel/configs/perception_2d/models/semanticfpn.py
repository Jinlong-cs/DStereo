from typing import Dict, Optional


def build_segmentor_semanticfpn(
    backbone: Dict,
    neck: Dict,
    task_name: str,
    mode: str = "train",
    head_cfg: Optional[Dict] = None,
):
    """Build the solov2 segmentor.

    Args:
        backbone: the backbone of this fcos detector, e.g., EfficientNet.
        neck: the neck of this solov2 segmentor, e.g. BiFPN.
        task_name: the name of a specific task, e.g., lane_instanceseg, used
            to distinguish nodes of different tasks in MultitaskGraphModel.
        mode: the mode of this fcos detector, e.g. "train", "val", "test".
            Different mode corresponds to different segmentor composition.

    Returns:
        A complete solov2 segmentor can be used as a training model,
        a validation model or a test model.
    """

    head_cfg = {} if head_cfg is None else head_cfg

    num_classes = head_cfg.get("num_classes")
    parsing_desc = head_cfg.get("parsing_desc")
    upscale = head_cfg.get("upscale", True)
    class_weight = head_cfg.get("class_weight", None)
    task_loss_weight = head_cfg.get("task_loss_weight", 1.0)
    ignore_index = head_cfg.get("ignore_index", 255)
    in_strides = head_cfg.get("in_strides", [4, 8, 16])
    in_channels = head_cfg.get("in_channels", [256, 256, 256])

    head_out_stride = min(in_strides)
    if upscale:
        head_out_stride //= 2

    semanticfpn_segmentor = dict(
        type="BMSegmentor",
        backbone=backbone,
        neck=neck,
        head=dict(
            type="SemanticFPNplusHead",
            num_classes=num_classes,
            in_strides=in_strides,
            in_channels=in_channels,
            upscale=upscale,
            node_name=f"{task_name}_head",
        ),
        loss=dict(
            type="SegLoss",
            loss=[
                dict(
                    type="CrossEntropyLoss",
                    loss_name=f"loss_ce_stride_{head_out_stride}",
                    use_sigmoid=False,
                    class_weight=class_weight,
                    loss_weight=task_loss_weight,
                    ignore_index=ignore_index,
                )
            ],
            node_name=f"{task_name}_loss",
        )
        if mode == "train"
        else None,
        target=dict(
            type="BMSegTarget",
            ignore_index=ignore_index,
            label_name="gt_seg",
            node_name=f"{task_name}_target",
        )
        if mode == "train"
        else None,
        desc=dict(
            type="AddDesc",
            per_tensor_desc=parsing_desc,
            node_name=f"{task_name}_desc",
        )
        if mode != "train"
        else None,
        postprocess=dict(
            type="BMSegDecoder",
            out_strides=[head_out_stride],
            do_inverse_transform=True,
            node_name=f"{task_name}_decoder",
        )
        if mode != "train"
        else None,
    )

    return semanticfpn_segmentor
