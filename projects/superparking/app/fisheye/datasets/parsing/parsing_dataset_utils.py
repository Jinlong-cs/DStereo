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
    model_version="v0.9.0",
    version_yaml="parsing_dataset_version.yaml",
    dataset_yaml="parsing_all_datasets.yaml",
):
    version_info = load_yaml(version_yaml)
    datasets_info = get_data_path(bucket_root, dataset_yaml)

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


if __name__ == "__main__":
    data_lst = get_dataset_list("/horizon-bucket")
    print("train:", data_lst[0], data_lst[1])
    print("val:", data_lst[2], data_lst[3])
