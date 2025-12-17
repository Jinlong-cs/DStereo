import os

import torch.nn.functional as ff

from hat.core.data_struct.base_struct import DetBoxes2D, Mask
from hat.utils.apply_func import convert_numpy

task2jsonkey = {
    "rear_detection": "vehicle",
    "person_detection": "person",
    "vehicle_detection": "vehicle",
    "traffic_sign_detection": "traffic_sign",
    "traffic_light_detection": "Traffic_light_shell",
    "cyclist_detection": "cyclist",
    "road_arrow_detection": "road_arrow",
    "traffic_cone_detection": "common_box",
    "vehicle_wheel_detection": "common_box",
    "person_head_detection": "person_4pe_head",
    "person_face_detection": "face",
    "cyclist_wheel_detection": "common_box",
    "traffic_light_len_detection": "common_box",
    "cycle_detection": "cyclist",
    "vehicle_plate_detection": "plate",
    "vehicle_side_detection": "vehicle_side",
    "adb_light_detection": "Headlight",  # Headlight, Taillight, Streetlight, Reflectionpoint # noqa
}

task2categoryid = {
    "vehicle_rear_detection": 0,
    "rear_detection": 0,
    "person_detection": 0,
    "vehicle_detection": 0,
    "traffic_light_detection": 0,
    "cyclist_detection": 0,
    "road_arrow_detection": 0,
    "traffic_cone_detection": 0,
    "traffic_sign_detection": 0,
    "vehicle_wheel_detection": 0,
    "person_head_detection": 0,
    "person_face_detection": 0,
    "cyclist_wheel_detection": 0,
    "traffic_light_len_detection": 0,
    "cycle_detection": 0,
    "vehicle_plate_detection": 0,
    "vehicle_side_detection": 0,
    "adb_light_detection": 0,  # 0 for Headlight, 1 for Taillight, 2 for Streetlight, 4 for Reflectionpoint # noqa
}


def save_predict_result(batch, results, task_name, **kwargs):
    aidi_results = []
    for img_name, result in zip(batch["img_name"], results):
        if task_name.endswith("_detection"):
            if task_name == "vehicle_side_detection":
                cls_idxs = result["cls_idxs"]
                task_key = task2jsonkey[task_name]
                select_idx = cls_idxs == task2categoryid[task_name]
                boxes = result["boxes"][select_idx]
                scores = result["scores"][select_idx]
                polygons = result["polygons"][select_idx]
                box_results = []
                for box, score, polygon in zip(boxes, scores, polygons):
                    box_results.append(
                        {
                            "bbox": box.tolist(),
                            "bbox_score": float(score),
                            "polygon": polygon.tolist(),
                        }
                    )
                aidi_result = {
                    "image_key": img_name,
                    task_key: box_results,
                }
            else:
                result = getattr(result, task_name)
                assert isinstance(result, DetBoxes2D)
                task_key = task2jsonkey[task_name]
                result = result[result.cls_idxs == task2categoryid[task_name]]
                aidi_result = {
                    "image_key": img_name,
                    task_key: [
                        {
                            "bbox": result_i.box.tolist(),
                            "bbox_score": float(result_i.score),
                        }
                        for result_i in result
                    ],
                }
        elif task_name.endswith("_segmentation"):
            assert isinstance(result, Mask)
            result = convert_numpy(result.mask).astype("float32")
            aidi_result = {
                "image_name": os.path.splitext(img_name)[0] + ".png",
                "out_img": result,
            }
        elif task_name.endswith("_classification"):
            name_dict = kwargs.get("name_dict", None)
            attr_type_list = [
                item.replace("category", "type")
                for item in list(name_dict.keys())
            ]
            if "category" in name_dict:
                name_dict.update(type=name_dict.pop("category"))
            type_len = [
                len(name_dict[type_name]) for type_name in attr_type_list
            ]
            aidi_result = {type_cat: [] for type_cat in attr_type_list}

            for jdx, t_name in enumerate(attr_type_list):
                if jdx == 0:
                    final_pred_index = result[0 : type_len[0]].argmax()
                    softmax_score = ff.softmax(
                        result[0 : type_len[0]], dim=0
                    ).tolist()
                else:
                    final_pred_index = result[
                        sum(type_len[:jdx]) : sum(type_len[: jdx + 1])
                    ].argmax()
                    softmax_score = ff.softmax(
                        result[sum(type_len[:jdx]) : sum(type_len[: jdx + 1])],
                        dim=0,
                    ).tolist()

                content = {"image_key": img_name, "objects": []}
                obj = {
                    "obj_id": 0,
                    "prediction": int(final_pred_index),
                    "scores": softmax_score,
                }
                content["objects"].append(obj)
                aidi_result[t_name].append(content)
        else:
            raise NotImplementedError(f"{task_name} not implemented")

        aidi_results.append(aidi_result)
    return aidi_results


def reformat_prediction_fn(batch, model_outs, **kwargs):
    task_name = kwargs.get("task_name", None)
    name_dict = kwargs.get("name_dict", None)
    assert task_name, "`task_name` can not be None."
    if task_name.endswith("_classification"):
        task_results = model_outs[task_name]["pred_cls"]
    else:
        task_results = model_outs[task_name]

    if isinstance(batch, tuple) and isinstance(batch[0], dict):
        batch = batch[0]

    all_results = save_predict_result(
        batch, task_results, task_name, name_dict=name_dict
    )
    return all_results
