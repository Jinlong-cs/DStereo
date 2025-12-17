import os

import yaml


def _dfs(ins, bucket_root):
    for _k, v in ins.items():
        if isinstance(v, dict):
            _dfs(v, bucket_root)
        if isinstance(v, list):
            for i in range(len(v)):
                v[i] = os.path.join(bucket_root, v[i])


def get_data_path(bucket_root, yaml_name):
    dataset = load_yaml(yaml_name)
    _dfs(dataset, bucket_root)
    return dataset


def load_yaml(yaml_name):
    _dir = os.path.dirname(__file__)
    yaml_path = os.path.join(_dir, yaml_name)
    yaml_file = yaml.load(open(yaml_path, "r"), Loader=yaml.FullLoader)
    return yaml_file


def get_dataset_list(
    bucket_root,
    attribution="superparking",
    model_version="v0.3.0",
    version_yaml="depth_dataset_version.yaml",
    dataset_yaml="depth_all_datasets.yaml",
):
    version_info = load_yaml(version_yaml)
    datasets_info = get_data_path(bucket_root, dataset_yaml)

    train_version_info = version_info[attribution][model_version]["train"]
    val_version_info = version_info[attribution][model_version]["val"]

    train_version_info = train_version_info["data_version"]
    train_data_version = list(train_version_info.keys())

    val_version_info = val_version_info["data_version"]
    val_data_version = list(val_version_info.keys())

    train_lst = []
    val_lst = []

    for version in train_data_version:
        plates = train_version_info[version]
        for plate in plates:
            plate_info = datasets_info[attribution][version]["train"]["plate"]
            train_lst.extend(plate_info[plate]["rec"])

    for version in val_data_version:
        plates = val_version_info[version]
        for plate in plates:
            plate_info = datasets_info[attribution][version]["val"]["plate"]
            val_lst.extend(plate_info[plate]["rec"])

    return train_lst, val_lst


if __name__ == "__main__":
    train_lst, val_lst = get_dataset_list("/horizon-bucket")
    print("train:", train_lst)
    print("val:", val_lst)
