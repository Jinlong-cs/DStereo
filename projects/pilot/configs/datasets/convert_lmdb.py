import argparse
import os
from collections import defaultdict
from importlib import import_module


def parse_full_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-setting",
        type=str,
        required=True,
    )
    args = parser.parse_args()
    return args


def rec_config_to_lmdb(rec_config, local_train):
    rec_root = "/horizon-bucket/matrix"
    lmdb_root = "/horizon-bucket/matrix/multicam_pilot/data/lmdb_datasets"

    remote_rec_root = "/bucket/input/matrix"
    remote_lmdb_root = "/bucket/input/matrix/multicam_pilot/data/lmdb_datasets"

    dup_3d = defaultdict(int)
    for task, dataset in rec_config.items():
        for i, path in enumerate(dataset["train_data_paths"]):
            if isinstance(path["rec_path"], str):
                if local_train:
                    rec_path = path["rec_path"].replace(
                        rec_root,
                        lmdb_root,
                    )
                else:
                    rec_path = path["rec_path"].replace(
                        remote_rec_root,
                        remote_lmdb_root,
                    )
                data_path = os.path.splitext(rec_path)[0]
                rec_config[task]["train_data_paths"][i][
                    "data_path"
                ] = data_path
            elif isinstance(path["rec_path"], list):
                data_paths = []
                for anno_path in path["anno_path"]:
                    if local_train:
                        anno_path = anno_path.replace(
                            rec_root,
                            lmdb_root,
                        )
                    else:
                        anno_path = anno_path.replace(
                            remote_rec_root,
                            remote_lmdb_root,
                        )
                    data_path = os.path.splitext(anno_path)[0]
                    dup_3d[data_path] += 1
                    if dup_3d[data_path] > 1:
                        data_path = data_path + str(dup_3d[data_path])
                    data_paths.append(data_path)
                rec_config[task]["train_data_paths"][i][
                    "data_path"
                ] = data_paths
    return rec_config


args = parse_full_args()
datapaths = import_module(f"{args.model_setting}_datasets").datapaths

new_path = rec_config_to_lmdb(datapaths, True)
for _task, dataset in new_path.items():
    for _i, path in enumerate(dataset["train_data_paths"]):
        path.pop("rec_path")
        path.pop("anno_path")
print("import os")
print("from easydict import EasyDict")
print("is_local_train = not os.path.exists('/running_package')")
print("bucket_root = '/horizon-bucket' if is_local_train else '/bucket/input'")
print("root = os.path.join(bucket_root, 'matrix')")
print("datapaths = {")
for key, item in new_path.items():
    print(f"    '{key}':" + "{")
    print("        'train_data_paths':[")
    for dataset in item["train_data_paths"]:
        if isinstance(dataset["data_path"], list):
            print("            {")
            print("                'data_path':[")
            for d in dataset["data_path"]:
                d = d.replace("/horizon-bucket/matrix", "{root}")
                print(f"                    f'{d}',")
            print("                ],")
            print(f"            'sample_weight':{dataset['sample_weight']},")
            print("            },")
        else:
            dataset["data_path"] = dataset["data_path"].replace(
                "/horizon-bucket/matrix", "{root}"
            )
            print("            {")
            print(f"                'data_path':f'{dataset['data_path']}',")
            print(
                f"                'sample_weight':{dataset['sample_weight']},"
            )
            print("            },")
    print("        ]")
    print("    },")
print("}")
print("datapaths = EasyDict(datapaths)")
