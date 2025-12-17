import os

import torch

from hat.data.transforms.frame.resize import BPUPyramidResizer

DET_TASKS = [
    "vehicle_rear",
    "person",
    "traffic_light",
    "cyclist",
    "full",
    "ts",
    "road_arrow",
    "traffic_cone",
]

adas_eval_json_key = {
    "rear_detection": "vehicle",
    "person_detection": "person",
    "vehicle_detection": "vehicle",
    "ts": "traffic_sign",
    "traffic_light": "Traffic_light_shell",
    "cyclist_detection": "cyclist",
    "road_arrow": "road_arrow",
    "traffic_cone": "common_box",
}

adas_eval_file_name_list = {
    "vehicle_rear": "vehicle_rear.json",
    "person": "person.json",
    "full": "full.json",
    "ts": "ts.json",
    "traffic_light": "traffic_light.json",
    "cyclist": "cyclist.json",
    "parsing": "parsing.tar",
    "lane": "lane.tar",
    "road_arrow": "road_arrow.json",
    "traffic_cone": "traffic_cone.json",
}

adas_eval_category = {
    "rear_detection": 0,
    "person_detection": 0,
    "vehicle_detection": 0,
    "ts": 0,
    "traffic_light": 0,
    "cyclist_detection": 0,
    "road_arrow": 0,
    "traffic_cone": 0,
}


def format_model_outs(outputs, task_name):
    """
    Format model outputs {taskname: [{key1: out1, key2: out2}]} by converting to dict.
    """
    results = {}

    for tn, output in outputs.items():
        if task_name != tn:
            continue
        output = output[0]
        for attr in output.keys():
            # keep pred keys
            if "pred" in attr or "predict" in attr:
                # only keep predict value.
                results[attr] = output.get(attr)
    return results


def is_adaseval_key(key):
    if "pred_bboxes" in key:
        return True
    for _key in DET_TASKS:
        if _key in key:
            return True
    return False


def save_predict_result(
    batch, task_results, inverse_infos, transforms, task_name
):
    for key in task_results.keys():
        if is_adaseval_key(key):
            result = task_results[key]
    if isinstance(result, list) and len(result) == 1:
        if isinstance(result[0], list):
            result = result[0]

    results_adaseval = {}
    if task_name not in results_adaseval:
        results_adaseval[task_name] = {}

    for i, (img_name, single_result) in enumerate(
        zip(batch[0]["img_name"], result)
    ):
        if transforms is not None:
            for transform in transforms[::-1]:
                if hasattr(transform, "inverse_transform"):
                    if isinstance(transform, BPUPyramidResizer):
                        single_result = transform.inverse_transform(
                            obj=single_result
                        )
                    else:
                        single_result = transform.inverse_transform(
                            inputs=single_result,
                            task_type="detection",
                            inverse_info=inverse_infos[i],
                        )
        results_adaseval[task_name][img_name] = {}
        results_adaseval[task_name][img_name]["pred"] = single_result

    all_results = save_result(
        results=results_adaseval,
        json_key=adas_eval_json_key,
        category=adas_eval_category,
    )
    return all_results


def reformat_prediction_fn(
    batch, model_outs, inverse_transform_keys, transforms, task_name
):
    """define a function to generate prediction results"""
    inverse_infos = []
    for img_id in range(len(batch[0]["img_name"])):
        inverse_info = {}
        if inverse_transform_keys is not None:
            for key, values in batch[0].items():
                if key in inverse_transform_keys:
                    inverse_info[key] = values[img_id]
        inverse_infos.append(inverse_info)
    task_results = format_model_outs(model_outs, task_name)
    all_results = save_predict_result(
        batch, task_results, inverse_infos, transforms, task_name
    )
    return all_results


def save_result(
    results,
    json_key,
    category,
):
    all_results = []
    for task_name, outputs in results.items():
        if task_name in json_key:
            for img_name, output in outputs.items():
                one_img_data = {}
                pred = output["pred"]
                image_key = img_name
                one_img_data["image_key"] = image_key
                one_img_data[json_key[task_name]] = []
                pred = (
                    pred.cpu().numpy()
                    if isinstance(pred, torch.Tensor)
                    else pred
                )  # noqa
                bboxes = pred[:, :4]
                bboxes_scores = pred[:, 4]
                bboxes_category = pred[:, 5]
                valid_mask = bboxes_scores > 0
                bboxes = bboxes[valid_mask]
                bboxes_scores = bboxes_scores[valid_mask]
                bboxes_category = bboxes_category[valid_mask]
                for j in range(bboxes.shape[0]):
                    if not float(bboxes_category[j]) == category[task_name]:
                        continue
                    one_bbox = {}
                    one_bbox["bbox"] = bboxes[j].tolist()
                    one_bbox["bbox_score"] = float(bboxes_scores[j])
                    one_img_data[json_key[task_name]].append(one_bbox)
                all_results.append(one_img_data)
        else:
            print(f"task [{task_name}] not supported")
    return all_results


def reformat_seg_to_aidi_eval(
    batch_data,
    batch_outputs,
    decoder,
    task_name,
    task_keyword="",
):
    if isinstance(batch_data, tuple):
        assert isinstance(
            batch_data[1], str
        ), f"Batch format error, it should be (batch, object_name), \
             but {batch_data}"
        batch_data = batch_data[0]
    batch_objects = batch_outputs[task_name][task_keyword]
    batch_objects = decoder((batch_objects,), batch_data)[task_keyword]
    rets = []
    for img_name, objects in zip(batch_data["img_name"], batch_objects):
        objects = objects.to("cpu").unsqueeze(-1).numpy()
        if os.path.splitext(img_name)[-1] != ".png":
            img_postfix = os.path.splitext(img_name)[-1]
            assert img_postfix in [".jpg", ".jpeg", "bmp"]
            img_name = img_name.replace(img_postfix, ".png")
        assert img_name.endswith(
            ".png"
        ), f"Image type error! expect .png file, but get {img_name}"
        ret = {
            "image_name": img_name,
            "out_img": objects,
        }
        rets.append(ret)

    return rets
