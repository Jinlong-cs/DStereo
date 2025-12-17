import copy
from typing import Dict, List


def build_attribute_classifier(
    backbone: Dict,
    neck: Dict,
    task_name: str,
    bn_kwargs: Dict,
    num_classes: int,
    mode: str = "train",
    attr_type_list: List = None,
    attr_type_value: List = None,
    ignore_idxs: List = None,
):
    """Build the classifier model.

    Args:
        backbone: the backbone of this fcos detector, e.g., EfficientNet.
        neck: the neck of this classifier, e.g. FPN.
        task_name: the name of a specific task, e.g., vehicle_attribute_classification, used
            to distinguish nodes of different tasks in MultitaskGraphModel.
        bn_kwargs: BatchNormEx kwargs.
        num_classes: Num classes of workcondition task.
        mode: the mode of this fcos detector, e.g. "train", "val", "test".
            Different mode corresponds to different detector composition.
        attr_type_list: List of all types.
        attr_type_value: Number of categories for each type.
        ignore_idx: The index of the category to be ignored.

    Returns:
        A complete fcos detector can be used as a training model, a validation
        model or a test model.
    """
    in_channel_list = neck["stride2channels"]

    classifier_model = dict(
        type="WorkConditionClassifier",
        backbone=backbone,
        backbone_extra=copy.deepcopy(neck["neck"]),
        prediction_head=dict(
            type="WorkConditionClsHead",
            output_dim=1024,
            bn_kwargs=bn_kwargs,
            num_classes=num_classes,
            in_channel=list(in_channel_list.values())[-1],
            padding_size=1,
            use_gn=True,
            node_name=f"{task_name}_prediction_head",
        ),
        losses=dict(
            type="AttrMultiLabelLoss",
            attr_type_list=attr_type_list,
            attr_type_numcls=attr_type_value,
            ignore_idx=ignore_idxs,
            node_name=f"{task_name}_cls_loss",
        )
        if mode == "train"
        else None,
    )

    return classifier_model
