import os

import yaml

CONFIG_DIR = os.path.dirname(__file__)
is_local = not os.path.exists("/running_package")
PREDICT_TAGS = ["merge", "mono_v12_hat", "resize", "single_task"]

adas_eval_datadet_id_list = {
    "vehicle_rear": [6028397],
    "vehicle3d": [6036724, 6035976],
    "person": [6029315],
    "full": [6028372],
    "ts": [6026233],
    "traffic_light": [6026482],
    "cyclist": [6029316],
    "parsing": [6031390, 6031358, 6028471, 6028384],
    # "parsing": [6031390],
    "lane": [6028550, 6028549, 6028548, 6028549, 6028598],
    # "lane": [6028550],
    "road_arrow": [6026449],
    "traffic_cone": [6026280],
}

bucket_root = "/horizon-bucket" if is_local else "/bucket/input"
aidi_val_datapath = os.path.join(
    bucket_root, "auto_eval/adas_eval/eval_platform/fs/"
)
dataset_val_img_path = {
    task_name: [
        os.path.join(aidi_val_datapath, str(id), "datasets")
        for id in dataset_id
    ]
    for task_name, dataset_id in adas_eval_datadet_id_list.items()
}


def load_mono_data_v12(bucket_root):
    data_yaml_2d_mono_v12 = (
        "mono3.0_cn_x8b-detection-v12.0.0-cfg-datasets.yaml"
    )
    url_path = os.path.join(CONFIG_DIR, data_yaml_2d_mono_v12)
    data_path = yaml.safe_load(open(url_path))

    rec_root_mono = os.path.join(bucket_root, "mono")

    tasks = {
        "det": [
            "cyclist",
            "person",
            "rear",
            "traffic_light",
            "traffic_sign",
            "vehicle",
            "road_arrow",
            "traffic_cone",
        ],
        "seg": ["lane_parsing", "semantic_parsing"],
        "others": ["ihbc"],
    }

    mono_data = {}

    for task, infos in data_path.items():
        # only use det and seg datasets
        if task in tasks["others"]:
            continue
        mono_data[task] = {"train": [], "val": []}

        for info in infos:
            if not info["train_rec"]:
                print("WARNING: empty train_rec in info")
                continue

            info["train_rec"] = (
                info["train_rec"][1:]
                if info["train_rec"].startswith("/")
                else info["train_rec"]
            )
            for split in ["train", "val"]:
                rec_name = info[f"{split}_rec"]
                if rec_name is None:
                    continue

                sample_weight = info["batch_size"]
                rec = os.path.join(rec_root_mono, rec_name)

                if not os.path.exists(rec):
                    print(f"WARNING: {rec} not exists! Skip this.")
                    continue

                pbrec_1 = rec.replace(".rec", ".anno.pb_rec")
                if os.path.exists(pbrec_1):
                    pbrec = pbrec_1
                else:
                    pbrec_2 = rec.replace(".rec", ".rec.anno.pb_rec")
                    if os.path.exists(pbrec_2):
                        pbrec = pbrec_2
                    else:
                        print(
                            f"WARNING: Neither {pbrec_1} or {pbrec_2} exists! Skip this."
                        )
                        continue

                if not os.path.exists(pbrec + ".idx"):
                    print(f"WARNING: {pbrec + '.idx'} not exist. Skip this.")
                    continue

                mono_data[task][split].append(
                    dict(
                        rec=rec,
                        pbrec=pbrec,
                        sample_weight=sample_weight,
                    )
                )

    # for task in mono_data:
    #     print(f"\t{task}:")
    #     for split in mono_data[task]:
    #         print(f"\t\t{split}: #{len(mono_data[task][split])} recs.")
    return mono_data
