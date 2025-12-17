import os

import torch
import yaml


def _dfs(ins, bucket_root):
    for _, v in ins.items():
        if isinstance(v, dict):
            _dfs(v, bucket_root)
        if isinstance(v, list):
            for i in range(len(v)):
                v[i] = os.path.join(bucket_root, v[i])


def get_data_path(bucket_root, yaml_path):
    dataset = load_yaml(yaml_path)
    _dfs(dataset, bucket_root)
    return dataset


def load_yaml(yaml_path):
    yaml_file = yaml.load(open(yaml_path, "r"), Loader=yaml.FullLoader)
    return yaml_file


def _exists(path):
    return os.path.exists(path)


def reorganize_sod_batch_results(
    batch, model_outs, size_diff, filter_pred=False, filter_gt=False
):
    out = model_outs["sod"]
    pred_bboxes_batch = [
        data for field, data in out[0].items() if "pred_bboxes" in field
    ][0]

    # filter hard predictions
    if filter_pred:
        filtered_pred_bboxes_batch = []
        for pred_bboxes in pred_bboxes_batch:
            filtered_pred = []
            for pred in pred_bboxes:
                pred_lb = int(pred[5])
                pred_height = float(pred[3] - pred[1])
                if pred_height >= size_diff[pred_lb]:
                    filtered_pred.append(pred)
            filtered_pred = (
                torch.stack(filtered_pred, dim=0)
                if len(filtered_pred) > 0
                else torch.empty((0, 6)).to(pred_bboxes.device)
            )
            filtered_pred_bboxes_batch.append(filtered_pred)
        pred_bboxes_batch = tuple(filtered_pred_bboxes_batch)

    # filter hard gts
    gt_difficults = []
    if filter_gt:
        for gt_bboxes, gt_classes in zip(
            batch[0]["gt_bboxes"], batch[0]["gt_classes"]
        ):
            gt_difficult = []
            for gt_class in gt_classes:
                gt_class = int(gt_class)
                # gt_class = 3 if gt_class >= 0 else -1
                # TODO fix single class prediction bug
                # is_small = gt_bbox[3] - gt_bbox[1] < size_diff[gt_class]
                if gt_class < 0:
                    gt_difficult.append(1)
                else:
                    gt_difficult.append(0)
            gt_difficults.append(
                torch.tensor(gt_difficult).to(gt_bboxes.device)
            )
    model_outputs = {
        "img": batch[0].get("ori_image"),
        "gt_bboxes": batch[0].get("gt_bboxes", []),
        "gt_classes": batch[0].get("gt_classes", []),
        "gt_labels": batch[0].get("gt_labels", []),
        "gt_difficult": None if len(gt_difficults) == 0 else gt_difficults,
        "ig_bboxes": batch[0].get("ig_bboxes", None),
        "pred_bboxes": pred_bboxes_batch,
    }

    return model_outputs


def get_dataset_list(
    bucket_root,
    data_root="projects/superparking/app/fisheye/datasets",
    attribution="superparking",
    model_version="v0.9.0",
    version_yaml="parsing_dataset_version.yaml",
    dataset_yaml="parsing_all_datasets.yaml",
):
    version_info = load_yaml(os.path.join(data_root, version_yaml))
    datasets_info = get_data_path(
        bucket_root, os.path.join(data_root, dataset_yaml)
    )

    train_version_info = version_info[attribution][model_version]["train"]
    val_version_info = version_info[attribution][model_version]["val"]

    train_rec_lst = []
    train_json_lst = []
    val_rec_lst = []
    val_json_lst = []

    for version in train_version_info:
        train_rec_lst.extend(
            datasets_info[attribution][version]["train"]["rec"]
        )
        train_json_lst.extend(
            datasets_info[attribution][version]["train"]["json"]
        )

    for version in val_version_info:
        val_rec_lst.extend(datasets_info[attribution][version]["val"]["rec"])
        val_json_lst.extend(datasets_info[attribution][version]["val"]["json"])

    return train_rec_lst, train_json_lst, val_rec_lst, val_json_lst


def get_list(key, task_configs, merge_list=False, wrap_fn=None):
    result = []
    for config in task_configs:
        assert config
        val = getattr(config, key)
        if wrap_fn:
            task_name = config.task_name
            if isinstance(val, list):
                for i in range(len(val)):
                    val[i] = wrap_fn(task_name, val[i])
            else:
                val = wrap_fn(task_name, val)
        if isinstance(val, list) and merge_list:
            result.extend(val)
        else:
            result.append(val)
    return result
