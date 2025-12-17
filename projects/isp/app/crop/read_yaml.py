import yaml


def load_yaml(yaml_path):
    with open(yaml_path) as f:
        data = yaml.safe_load(f)

    return data


def load_data_info(datas):
    train_data_paths = []
    val_data_paths = []
    eval_data_paths = []

    train_num_samples = []
    val_num_samples = []
    eval_num_samples = []

    for data in datas["train"]:
        train_data_paths.append(data[0])
        train_num_samples.append(data[1])

    for data in datas["val"]:
        val_data_paths.append(data[0])
        val_num_samples.append(data[1])

    for data in datas["eval"]:
        eval_data_paths.append(data[0])
        eval_num_samples.append(data[1])

    val_anno_data_path = datas["val_anno_path"]

    camera_info = datas["cam_info"]
    if "type" in camera_info:
        camera_info.pop("type")

    return (
        train_data_paths,
        val_data_paths,
        eval_data_paths,
        val_anno_data_path,
        train_num_samples,
        val_num_samples,
        eval_num_samples,
        camera_info,
    )
