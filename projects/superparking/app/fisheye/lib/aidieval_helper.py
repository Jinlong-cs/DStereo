import os

import torch
from torch.nn import functional as F

CONFIG_DIR = os.path.dirname(__file__)
is_local = not os.path.exists("/running_package")
bucket_root = "/horizon-bucket" if is_local else "/bucket/input"

TASK2AIDI_EVAL_IDS = {
    "vehicle_detection": [6036754],
    "vehicle_detection_3d": [6037292],
    "person_detection": [6037159],
    "person_detection_3d": [6037298],
    "cyclist_detection": [6037203],
    "cyclist_detection_3d": [6037297],
    "rear_detection": [6036810],
    "parsing": [6035917],
    "sod": [6038980, 6038981, 6039139, 6038982, 6038984, 6038977],
    # rcnn
    "vehicle_category_classification": [6036753],
    "vehicle_occlusion_classification": [6036751],
    "vehicle_truncation_classification": [6036752],
    "vehicle_wheel_kps": [6036781],
    "person_occlusion_classification": [6036786],
    "person_orientation_classification": [6037156],
    "person_pose_classification": [6036788],
    "person_face_detection": [6037152],
    "rear_occlusion_classification": [6036804],
    "rear_part_classification": [6037150],
    "rear_plate_detection": [6037151],
}

aidi_val_datapath = os.path.join(
    bucket_root, "auto_eval/adas_eval/eval_platform/fs/"
)
TASK2AIDI_EVAL_DATA_PATH = {
    task_name: [
        os.path.join(aidi_val_datapath, str(id), "datasets")
        for id in dataset_id
    ]
    for task_name, dataset_id in TASK2AIDI_EVAL_IDS.items()
}


def reformat_model_outs_fn_2d_detection(
    batch,
    model_outs,
    task_name,
    class_name,
    category_ids,
    rescale_to_shape=None,
):
    pred_bboxes_batch = None
    for out in model_outs:
        for field in dir(out):
            if task_name not in field:
                continue
            if "pred_bboxes" in field:
                pred_bboxes_batch = getattr(out, field)
                break
    assert (
        pred_bboxes_batch is not None
    ), "no pred bboxes result found outputs."

    batch_results = []

    gt_labels = batch[0].get("gt_labels")
    for ii, pred_bboxes in enumerate(pred_bboxes_batch):
        layout = batch[0]["layout"][ii]
        input_h = batch[0]["img_shape"][ii][layout.index("h")]
        input_w = batch[0]["img_shape"][ii][layout.index("w")]
        if rescale_to_shape is None:
            rescale_h, rescale_w = input_h, input_w
        else:
            rescale_h, rescale_w = rescale_to_shape[:2]
        results = {"image_key": batch[0]["img_name"][ii], class_name: []}
        for pred in pred_bboxes:
            pred_category = int(pred[5])
            if pred_category not in category_ids or (
                gt_labels is not None
                and pred_category not in set(gt_labels[ii].cpu().numpy())
            ):
                continue
            current_len = len(results[class_name])
            pred[:4:2] *= rescale_w / input_w
            pred[1:4:2] *= rescale_h / input_h
            results[class_name].append(
                {
                    "id": current_len,
                    "bbox": pred[:4].detach().cpu().tolist(),
                    "bbox_score": pred[4].detach().cpu().item(),
                    "attrs": {},
                }
            )
        batch_results.append(results)
    return batch_results


def reformat_seg_to_aidi_eval(
    batch_data,
    batch_outputs,
    original_shape,
):
    if isinstance(batch_data, tuple):
        assert isinstance(
            batch_data[1], str
        ), f"Batch format error, it should be (batch, object_name), \
             but {batch_data}"
        batch_data = batch_data[0]
    batch_objects = batch_outputs[0][0]
    rets = []
    ori_hws = batch_data["orig_hw"]
    for _, (img_name, objects, h_i, w_i) in enumerate(
        zip(batch_data["img_name"], batch_objects, ori_hws[0], ori_hws[1])
    ):
        objects = objects.to("cpu")[
            None,
        ]
        objects = F.interpolate(
            objects.float(), size=(h_i, w_i), mode="bilinear"
        )
        objects = F.softmax(objects, dim=1).max(dim=1)[1][0]
        if os.path.splitext(img_name)[-1] != ".png":
            img_postfix = os.path.splitext(img_name)[-1]
            assert img_postfix in [".jpg", ".jpeg", "bmp"]
            img_name = img_name.replace(img_postfix, ".png")
        assert img_name.endswith(
            ".png"
        ), f"Image type error! expect .png file, but get {img_name}"
        ret = {
            "image_name": img_name,
            "out_img": objects.numpy(),
        }
        rets.append(ret)

    return rets


def reformat_model_outs_fn_rcnn_cls(
    batch_data,
    model_outs,
    task_name,
    class_name,
    category_ids,
    rescale_to_shape=None,
):
    batch = batch_data[0]
    bs_num = len(batch["img_name"])
    batch_results = []

    # generate preds
    cls_objects = []
    for out in model_outs:
        for field in dir(out):
            if task_name not in field:
                continue
            if "pred_cls" in field:
                cls_object = getattr(out, field)
                cls_objects.append(cls_object)
            if "pred_bboxes" in field:
                roi_boxes = getattr(out, field)

    for index in range(bs_num):
        results = {}
        results["image_key"] = batch["img_name"][index]
        cls_name = task_name.split("_")[0]
        if cls_name == "rear":
            cls_name = "vehicle"
        results[cls_name] = []
        scale_factor = batch["scale_factor"][index]
        cls_objects_per_img = cls_objects[index]

        box_num_per_img = roi_boxes[index].shape[0]
        for j in range(box_num_per_img):
            current_len = len(results[cls_name])
            bbox = roi_boxes[index][j, :]
            w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
            if w == 0 or h == 0:
                continue
            bbox = bbox / scale_factor
            bbox = bbox.detach().cpu().tolist()
            cls_obj = cls_objects_per_img[j]
            single_res = {
                "id": current_len,
                "bbox": bbox,
                "bbox_score": float(cls_obj.score.detach().cpu()),
            }
            single_res.update(cls_obj.to_aidi_eval())
            results[cls_name].append(single_res)
        batch_results.append(results)
    return batch_results


def reformat_model_outs_fn_rcnn_kps(
    batch,
    model_outs,
    task_name,
    class_name,
    category_ids,
    rescale_to_shape=None,
):
    batch = batch[0]
    bs_num = len(batch["img_name"])
    batch_results = []

    # # generate preds
    kps_objects = []
    for out in model_outs:
        for field in dir(out):
            if task_name not in field:
                continue
            if "pred_kps" in field:
                kps_object = getattr(out, field)
                kps_objects.append(kps_object)
            if "pred_bboxes" in field:
                roi_boxes = getattr(out, field)

    for index in range(bs_num):
        results = {}
        results["image_key"] = batch["img_name"][index]
        results[class_name] = []
        objects_per_img = kps_objects[index]
        scale_factor = batch["scale_factor"][index]
        objects_per_img.points0.points /= scale_factor[:2]
        objects_per_img.points1.points /= scale_factor[:2]

        objects_per_img.points0.points = objects_per_img.points0.points.cpu()
        objects_per_img.points1.points = objects_per_img.points1.points.cpu()
        objects_per_img.points0.scores = objects_per_img.points0.scores.cpu()
        objects_per_img.points1.scores = objects_per_img.points1.scores.cpu()
        objects_per_img.points0.cls_idxs = (
            objects_per_img.points0.cls_idxs.cpu()
        )
        objects_per_img.points1.cls_idxs = (
            objects_per_img.points1.cls_idxs.cpu()
        )

        box_num_per_img = roi_boxes[index].shape[0]

        for j in range(box_num_per_img):
            kps_obj = objects_per_img[j]
            bbox = roi_boxes[index][j, :]

            w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
            if w == 0 or h == 0:
                continue
            bbox = bbox / scale_factor
            bbox = bbox.detach().cpu().tolist()
            single_res = {
                "bbox": bbox,
                "bbox_score": 1.0,
            }
            single_res.update(kps_obj.to_aidi_eval())

            results[class_name].append(single_res)
        batch_results.append(results)
    return batch_results


def reformat_model_outs_fn_rcnn_det(
    batch_data,
    model_outs,
    task_name,
    class_name,
    category_ids,
    rescale_to_shape=None,
):
    batch = batch_data[0]
    bs_num = len(batch["img_name"])
    batch_results = []

    # generate preds
    det_objects = []
    for out in model_outs:
        for field in dir(out):
            if task_name not in field:
                continue
            if "pred_boxes" in field:
                det_object = getattr(out, field)
                det_objects.append(det_object)

    for i in range(bs_num):
        results = {}
        results["image_key"] = batch["img_name"][i]
        det_object = det_objects[i]
        det_object.boxes_list = [box.cpu() for box in det_object.boxes_list]
        det_object.scores_list = [
            score.cpu() for score in det_object.scores_list
        ]
        det_object.cls_idxs_list = [
            cls_id.cpu() for cls_id in det_object.cls_idxs_list
        ]
        scale_factor = batch["scale_factor"][i].cpu()
        det_object.rescale(
            torch.tensor(1 / scale_factor[::2]),
            torch.tensor(1 / scale_factor[1::2]),
        )
        results[task_name] = det_object.to_aidi_eval()
        batch_results.append(results)
    return batch_results
