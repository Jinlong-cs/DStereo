# Copyright (c) Horizon Robotics. All rights reserved.
import collections

from hat.core.proj_spec.classification import _get_classification_labels


def reformat_tl_cls_to_aidi_eval(
    batch_data,
    batch_outputs,
    task_cfgs,
):
    if isinstance(batch_data, tuple):
        assert isinstance(
            batch_data[1], str
        ), f"Batch format error, it should be (batch, object_name), \
             but {batch_data}"
        batch_data, _ = batch_data
    rets = []
    tmp_rets = {}
    obj_ids = batch_data.get("obj_id", None)
    img_names = batch_data.get("img_name", None)
    for obj_key in task_cfgs:
        batch_objects = batch_outputs["traffic_light_multitask"][0][obj_key]
        cur_ret = collections.defaultdict(list)
        obj_rets = []
        for img_name, _, object in zip(img_names, obj_ids, batch_objects):
            object.update(obj_id=_)
            cur_ret[img_name].append(object)
        for key in cur_ret:
            obj_rets.append({"image_key": key, "objects": cur_ret[key]})
        tmp_rets[obj_key] = obj_rets
    rets.append(tmp_rets)
    return rets


def reformat_attr_det_to_aidi_eval(
    batch_data,
    batch_outputs,
    num_color_cls,
    num_type_cls,
    num_eval_color,
    num_eval_type,
):
    TYPE_DET2EVAL = _get_classification_labels(
        "traffic_lens_type_det2eval", str(num_eval_type)
    )
    COLOR_DET2EVAL = _get_classification_labels(
        "traffic_lens_color_det2eval", str(num_eval_color)
    )
    if isinstance(batch_data, tuple):
        assert isinstance(
            batch_data[1], str
        ), f"Batch format error, it should be (batch, object_name), \
             but {batch_data}"
        batch_data, _ = batch_data
    tmp_dict = collections.defaultdict(list)
    img_names = batch_data.get("img_name", None)
    for bbox, img_key in zip(batch_outputs["detection"][0], img_names):
        if bbox.shape[0] == 0:
            continue
        else:
            for pp in bbox:
                # 匹配label
                type_value = _get_classification_labels(
                    "traffic_lens_type_classification", str(num_type_cls)
                )[int(pp[6])]
                color_value = _get_classification_labels(
                    "traffic_lens_color_classification", str(num_color_cls)
                )[int(pp[7])]
                type_out_id = (
                    num_eval_color - 1
                    if type_value not in TYPE_DET2EVAL
                    else TYPE_DET2EVAL.index(type_value)
                )
                color_out_id = (
                    num_eval_color - 1
                    if color_value not in COLOR_DET2EVAL
                    else COLOR_DET2EVAL.index(color_value)
                )
                tmp_dict[img_key].append(
                    {
                        "bbox": [
                            float(pp[0]),
                            float(pp[1]),
                            float(pp[2]),
                            float(pp[3]),
                        ],
                        "bbox_score": float(pp[4]),
                        "attrs": {
                            "tl_type_prediction": type_out_id,
                            "tl_color_prediction": color_out_id,
                        },
                    }
                )
    rets = []
    for img_key, bbox in tmp_dict.items():
        rets.append({"image_key": img_key, "common_box": bbox})
    # TODO:
    # 1. 如果一张图中的roi被分到不同的batch中，会导致后面的prediction不会被写进结果中
    return rets
