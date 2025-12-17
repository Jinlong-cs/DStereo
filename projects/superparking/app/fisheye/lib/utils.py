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
    pred_bboxes_batch = model_outs[0][0]

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


def get_parsing_dataset_list(
    bucket_root,
    data_root="projects/superparking/app/fisheye/datasets",
    attribution="superparking",
    model_version="v0.9.0",
    version_yaml="parsing_dataset_version.yaml",
    dataset_yaml="parsing_all_datasets.yaml",
    commit_level_ci=False,
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

    data_list = [train_rec_lst, train_json_lst, val_rec_lst, val_json_lst]

    if commit_level_ci:
        CI_BUCKET_MAP = get_ci_bucket_map(bucket_root)
        for i, d_list in enumerate(data_list):
            for j, path in enumerate(d_list):
                for rt_src, rt_dst in CI_BUCKET_MAP.items():
                    if rt_src in path:
                        path = path.replace(rt_src, rt_dst)
                d_list[j] = path
            data_list[i] = d_list

    return data_list


def get_list(key, task_configs, merge_list=False, wrap_fn=None):
    result = []
    for config in task_configs:
        assert config
        val = getattr(config, key, None)
        if wrap_fn:
            task_name = config.task_name
            if isinstance(val, list):
                for i in range(len(val)):
                    val[i] = wrap_fn(task_name, val[i])
            elif val is not None:
                val = wrap_fn(task_name, val)
        if isinstance(val, list) and merge_list:
            result.extend(val)
        elif val is not None:
            result.append(val)
    return result


def get_ci_bucket_map(bucket_root):
    # the map will be removed on aidi ci
    return {  # for commit CI
        os.path.join(bucket_root, "HDLTAlgorithm"): os.path.join(
            bucket_root, "HDLTAlgorithm"
        ),
        os.path.join(bucket_root, "auto_eval"): os.path.join(
            bucket_root, "auto_eval"
        ),
    }


def regain_dataset_auto2d_paths(
    bucket_root, datapaths, use_mini_dataset, commit_level_ci
):
    CI_BUCKET_MAP = get_ci_bucket_map(bucket_root)

    if use_mini_dataset:
        for task, path_dict in datapaths.items():
            if "mini_data_paths" in path_dict:
                path_dict["train_data_paths"] = path_dict["mini_data_paths"]
                path_dict["val_data_paths"] = path_dict["mini_data_paths"]
                datapaths[task] = path_dict

    if commit_level_ci:
        for task, path_dict in datapaths.items():
            for key, paths in path_dict.items():
                if "path" not in key:
                    continue
                for i, path in enumerate(paths):
                    for pk, pn in path.items():
                        if not isinstance(pn, str):
                            continue
                        for rt_src, rt_dst in CI_BUCKET_MAP.items():
                            if rt_src in pn:
                                pn = pn.replace(rt_src, rt_dst)
                        path[pk] = pn
                    paths[i] = path
                path_dict[key] = paths
            datapaths[task] = path_dict

    return datapaths


def get_bev_real3d_lmdb(
    version_yaml,
    spec_yaml,
    version,
    bucket_root,
    transforms=None,
    num_dist=1,
):
    version_config = yaml.load(open(version_yaml, "r"), Loader=yaml.FullLoader)
    spec_config = yaml.load(open(spec_yaml, "r"), Loader=yaml.FullLoader)

    datatypes_in_version = version_config[version]
    real3d_path_name = "multi_view_lmdb_path"
    camera_names = [
        "fisheye_front",
        "fisheye_left",
        "fisheye_right",
        "fisheye_rear",
    ]

    root_paths = []
    syncf_data_names = []
    label_data_names = []

    for data_type, names_in_version in datatypes_in_version.items():
        for plate in names_in_version:
            for name in names_in_version[plate]:
                spec = spec_config[data_type][plate][name]
                if real3d_path_name not in spec:
                    print(f"{real3d_path_name} not in lmdb yaml")
                    continue
                if spec[real3d_path_name] is None:
                    continue
                root_paths.append(os.path.join(bucket_root, spec["root"]))
                syncf_data_names.append(spec["sync_file_lmdb"])
                label_data_names.append(spec[real3d_path_name])
    dataset = dict(
        type="Real3DDatasetLMDBFromBEV",
        paths=root_paths,
        syncf_data_names=syncf_data_names,
        label_data_names=label_data_names,
        camera_names=camera_names,
        num_classes=1,
        num_dist=num_dist,
        transforms=transforms,
    )
    return dataset
