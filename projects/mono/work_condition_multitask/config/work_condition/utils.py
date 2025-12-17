# Copyright (c) Horizon Robotics. All rights reserved.


def reformat_wk_cls_to_aidi_eval(
    batch_data,
    batch_outputs,
    task_cfgs,
):
    if isinstance(batch_data, tuple):
        assert isinstance(
            batch_data[1], str
        ), f"Batch format error, it should be (batch, object_name), \
             but {batch_data}"
        batch_data, batch_obj_key = batch_data
    rets = []
    tmp_rets = {}
    for obj_key in task_cfgs:
        batch_objects = batch_outputs["work_condition_multitask"][0][obj_key]
        obj_rets = []
        for img_name, objects in zip(batch_data["img_name"], batch_objects):
            ret = {
                "image_key": img_name,
                "objects": [objects],
            }
            obj_rets.append(ret)
        tmp_rets[obj_key] = obj_rets
    rets.append(tmp_rets)
    return rets
