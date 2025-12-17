from ...common import (
    bifpn_stride2channels,
    fcos_stacked_convs,
    feat_channels,
    rpn_out_strides,
    training_step,
    val_only,
)

classnames = ["rear"]
object_type = "_".join(classnames)
norm_target_bbox = False if training_step == "int_infer" and val_only else True

fcos_head = dict(
    type="FCOSHead",
    num_classes=1,
    in_strides=rpn_out_strides,
    out_strides=rpn_out_strides,
    stride2channels=bifpn_stride2channels,
    feat_channels=feat_channels,
    stacked_convs=fcos_stacked_convs,
    use_sigmoid=True,
    share_bn=False,
    upscale_bbox_pred=training_step != "int_infer",
    node_name="rear_head",
)
