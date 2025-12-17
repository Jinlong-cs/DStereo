import argparse
import importlib
import os
import subprocess
import sys


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        help="[all_task_data_files] must be defined in config",
    )
    parser.add_argument(
        "--upload-file-path",
        type=str,
        default="./tmp_output/tmp_preheat_list.txt",
    )
    parser.add_argument(
        "--cloud-name",
        type=str,
        default="tcloud",
        choices=["acloud", "bcloud", "tcloud", "ucloud"],
    )
    args = parser.parse_args()
    return args


def load_config(cfg_path):
    cfg_dir = os.path.dirname(cfg_path)
    cfg_module = os.path.basename(cfg_path).split(".")[0]
    sys.path.append(cfg_dir)
    return importlib.import_module(cfg_module).__dict__


def preheat_from_config(config_dict, upload_file_path, cloud_name):
    file_list_path = config_dict["all_task_data_files"]
    bucket_root = "/horizon-bucket/"
    dmp_prefix = "dmpv2://"
    with open(file_list_path, "rt") as fr:
        file_list = fr.readlines()
    for i, fn in enumerate(file_list):
        assert bucket_root in fn
        file_list[i] = fn.replace(bucket_root, dmp_prefix)
    with open(upload_file_path, "wt") as fw:
        fw.writelines(file_list)
    cmd = f"hitc bkt-file prepare --c {cloud_name} --f {upload_file_path}"
    subprocess.check_call(cmd, shell=True)


if __name__ == "__main__":
    args = parse_args()
    preheat_from_config(
        load_config(args.config),
        args.upload_file_path,
        args.cloud_name,
    )
