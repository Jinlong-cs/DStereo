# -*- coding: utf-8 -*-  # noqa: D100
###################################################################
#   Copyright (C) 2022 Horizon Robotics. All rights reserved.
#
#   Filename    : compile.py
#   Author      : shengzhe.dai
#   Date        : June 2022
#   Description : the auto model compiler for HAT
###################################################################

import argparse
import getpass
import hashlib
import os
import time
import warnings
from glob import glob

import hbdk

# import hbdk_model_verifier
import horizon_plugin_pytorch as horizon
import torch
import yaml
from hbdk.torch_script.tools import perf_model, visualize_model
from horizon_plugin_pytorch.quantization import March

try:
    from hat.callbacks.aidi_expmodel import AIDIExpModel  # noqa: E402
except Exception:
    warnings.warn("Fail to load the dependencies of hat.")


THIS_PATH = os.sep.join(os.path.abspath(__file__).split("/")[:-1])
print(THIS_PATH)


def parse_args():  # noqa: D202, D401
    """The input parameters."""

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task-type",
        type=str,
        required=True,
        help="the task name.",
    )
    parser.add_argument(
        "--hat-path",
        type=str,
        required=True,
        help="the HAT repo path",
    )
    parser.add_argument(
        "--source-bpu",
        type=float,
        required=False,
        default=2.5,
        help="the source bpu version.",
    )
    parser.add_argument(
        "--target-bpu",
        type=float,
        required=False,
        default=2.5,
        help="the target bpu version.",
    )
    parser.add_argument(
        "--num-obs",
        type=int,
        required=False,
        default=16,
        help="the maximum number of obstacles in one batch.",
    )
    parser.add_argument(
        "--compile",
        type=int,
        required=False,
        default=0,
        help="whether to compile the pt.",
    )
    parser.add_argument(
        "--upload",
        type=int,
        required=False,
        default=0,
        help="whether to upload to aidi model.",
    )
    return parser.parse_args()


def cal_md5(filename: str):
    """Calculate the md5sum of the given file.

    Args:
        filename: the path of the file.

    Return:
        md5 (str): the md5sum value.
    """
    if os.path.isfile(filename):
        fp = open(filename, "rb")
        contents = fp.read()
        fp.close()
        md5 = hashlib.md5(contents).hexdigest()
        return md5
    else:
        raise FileNotFoundError(f"file {filename} not exists")


def prepare_test_input(task_type: str, source_bpu: float, num_obs: int):
    """Prepare test input.

    Args:
        source_bpu: the bpu version during training.
        target_bpu: the bpu version that the user want to compile.
        num_obs: the number of obstacles in the quantized model.

    Returns:
        ei (dict): the test input.
    """
    if "multipath" in task_type:
        if source_bpu == 2.2:
            img_homographys = torch.randn((num_obs, 8, 8, 2))
        elif source_bpu == 2.5:
            img_homographys = hbdk.torch_script.tools.placeholder(
                (num_obs, 8, 8, 2), torch_native=True
            )
        else:
            raise ValueError(f"Unsupported bpu version {source_bpu}.")

        ei = dict(
            road_map=torch.randn((1, 1, 512, 512)),
            rendered_obs=torch.randn((1, 4, 512, 512)),
            state_vectors=torch.randn((num_obs, 3, 1, 1)),
            img_homographys=img_homographys,
        )
    else:
        raise TypeError(f"Undefined task type {task_type}")
    return ei


def auto_prepare_test_input(source_bpu, input_size, input_key):
    """Prepare test input given size and key strings.

    Args:
        source_bpu: the bpu version during training.
        input_size: the input sizes connected by ",".
        input_key: the input keys connected by ",".

    Returns:
        ei (dict): the test input.
    """
    input_size = input_size.split(",")
    input_key = input_key.split(",")
    assert len(input_size) == len(
        input_key
    ), "The input size and the input key must have the same length"
    ei = {}
    for key, size in zip(input_key, input_size):
        use_ph = False
        if "p" in size and source_bpu > 2.2:
            use_ph = True
            size = size.split("p")[1]
        shape = [int(i) for i in size.split("x")]
        assert len(shape) == 4, f"The input of {key} should be dim = 4."
        if use_ph:
            tmp_ei = hbdk.torch_script.tools.placeholder(
                shape, torch_native=True
            )
        else:
            tmp_ei = torch.randn(shape)
        ei[key] = tmp_ei
    return ei


def main(
    task_type: str,
    source_bpu: float,
    target_bpu: float,
    num_obs_in_hbm: int,
    hat_path: str = None,
    if_compile: bool = False,
    if_upload: bool = False,
):
    """Main function.

    Args:
        task_type: the task type should be assigned in
            `train_model_cfg.yaml`. Also the necessary parameters
            should be changed according to the model you want to
            compile.
        source_bpu: the bpu version during training.
        target_bpu: the bpu version that the user want to compile.
        num_obs_in_hbm: the number of obstacles in the quantized model.
        hat_path: the root path of the used HAT repo. The function will
            automatically find the used config file and upload to
            aidi-model. If you do not need this, please keep it as None.
        if_compile: whether to compile the model
        if_upload: whether to upload the model to aidi.
    """
    train_model_cfg = os.path.join(THIS_PATH, "train_model_cfg.yaml")
    print("train_model_cfg: ", train_model_cfg)
    with open(train_model_cfg, "r") as f:
        task_cfg = yaml.load(f, Loader=yaml.FullLoader)

    assert (
        task_type in task_cfg
    ), f"Undefined type of training task {task_type}."
    task_name = task_cfg["task_name"]
    task_ver = task_cfg["version"]
    user_name = task_cfg["user_name"]
    tag = task_cfg["tag"]
    task = task_cfg[task_type]
    local_train = task_cfg["local_train"]
    compile_level = task_cfg["compile_level"]

    # Paths.
    if local_train:
        ckpt_dir = (
            f"/jfs-public/users/{user_name}/results/{task_name}/checkpoint"
        )
    else:
        ckpt_dir = "/job_data/models/checkpoint"

    time_struct = time.localtime(time.time())
    year = str(time_struct.tm_year).zfill(4)
    month = str(time_struct.tm_mon).zfill(2)
    day = str(time_struct.tm_mday).zfill(2)
    date = f"{year}{month}{day}"
    compile_task_name = (
        f"{task_type}_release_{task_ver}_{date}_bpu{target_bpu}"
    )
    compile_dir = os.path.join(ckpt_dir, compile_task_name)
    print("compile_dir: ", compile_dir)
    os.makedirs(compile_dir, exist_ok=True)

    model_name = task["pt2hbm"]["model_name"]
    aidi_name = task["hbm2aidi"]["model_name"]
    name = f"{aidi_name}_{task_ver}_bpu{target_bpu}_{num_obs_in_hbm}_obstacles"

    # -- parse the input size for test input generating.
    input_size = task["pt2hbm"]["input_size"]
    input_key = task["pt2hbm"]["input_key"]

    if if_compile:
        if target_bpu == 2.2:
            march = "bayes1"
        elif target_bpu == 2.5:
            march = March.BAYES
        else:
            raise ValueError(f"Unsupported bpu version {target_bpu}.")

        # Load pt and generate test inputs
        pt_path = os.path.join(ckpt_dir, "deploy-checkpoint-last.pt")
        ts = torch.jit.load(pt_path)
        # ei = prepare_test_input(task_type, source_bpu, num_obs_in_hbm)
        ei = auto_prepare_test_input(source_bpu, input_size, input_key)

        print(ei.keys())
        for i in list(ei.keys()):
            print(i, ei[i].shape)

        # Generate hbir and hbm.
        extra_args = task["pt2hbm"]["extra_args"]
        extra_args = extra_args.split(" ")
        horizon.quantization.export_hbir(
            ts.eval(),
            ei,
            hbir=os.path.join(compile_dir, f"{name}.hbir"),
            march=march,
        )
        # horizon.quantization.compile_model(
        #     ts.eval(),
        #     ei,
        #     march=march,
        #     hbm=os.path.join(compile_dir, f"{name}.hbm"),
        #     opt=compile_level,
        #     name=model_name,
        # extra_args=extra_args,
        # )

        # Copy the necessary files and rename.
        cp_pt = os.path.join(compile_dir, f"{name}.pt")
        os.system(f"cp {pt_path} {cp_pt}")
        # os.system(f"cd {compile_dir} && hbdk-perf {name}.hbm")

        perf_model(
            ts.eval(),
            ei,
            march=march,
            name=model_name,
            hbm=os.path.join(compile_dir, f"{name}.hbm"),
            opt=compile_level,
            out_dir=compile_dir,
            jobs=2,
            layer_details=True,
        )

        # If you want to draw the model graph, enable this:
        visualize_model(
            ts.eval(),
            ei,
            march=march,
            save_path=os.path.join(compile_dir, "xxx.svg"),
            show=False,
        )

        # cal md5
        files_suffix = [".hbir", ".hbm", ".pt"]
        for suffix in files_suffix:
            ori_file = os.path.join(compile_dir, f"{name}{suffix}")
            if suffix == ".hbm":
                cmd = f"cd {compile_dir} && hbdk-disas {ori_file} > desc.txt"
                os.system(cmd)
                tmp_trans_file = os.path.join(
                    compile_dir, f"tmp_{name}{suffix}"
                )
                os.system(f"mv {ori_file} {tmp_trans_file}")
                os.system(
                    f"hbdk-pack {tmp_trans_file} -o {ori_file} --tag {tag}"  # noqa: E501,
                )
                os.system(f"rm {tmp_trans_file}")

            tmp_md5 = cal_md5(ori_file)
            trans_file = os.path.join(compile_dir, f"{name}_{tmp_md5}{suffix}")
            os.system(f"mv {ori_file} {trans_file}")

        float_name = "float-checkpoint-best.pth.tar"
        qat_name = "qat-checkpoint-best.pth.tar"
        if "qat_ckpt_for_int_infer" in task:
            qat_name = task["qat_ckpt_for_int_infer"]
        if "float_ckpt_for_int_infer" in task:
            float_name = task["float_ckpt_for_int_infer"]
        qat_path = os.path.join(ckpt_dir, qat_name)
        float_path = os.path.join(ckpt_dir, float_name)
        metric_path = os.path.join(ckpt_dir, "metric.txt")
        if os.path.exists(qat_path):
            os.system(f"cp {qat_path} {compile_dir}")
        if os.path.exists(float_path):
            os.system(f"cp {float_path} {compile_dir}")
        if os.path.exists(metric_path):
            os.system(f"cp {metric_path} {compile_dir}")

    if if_upload:
        # cluster = os.getenv("CLUSTER", None)
        job_id = os.getenv("JOB_ID", None)
        # Here are the example values, if you want to manually upload.
        # cluster = "idc-v2"
        # job_id = "2301970"
        if job_id is not None:
            job_id = int(job_id)
        if not local_train:
            user_name = getpass.getuser()
        cfg_file = task["train_config_file"]
        if hat_path is not None:
            cfg_file = [os.path.join(hat_path, cfg_file)]
        else:
            cfg_file = None

        aidi_expmodel = AIDIExpModel(
            model_name=f"{aidi_name}_{user_name}",
            task_type="multitask",
            save_model="last",
            desc="",
            tags=None,
            model_version=task_ver,
            # basic_versions=None,
            # job_id=job_id,
            # cluster_info=cluster,
            platforms=["J5"],
            # config_paths=cfg_file,
            attachments=None,
        )
        for key in ["pt", "hbm", "hbir"]:
            print("编译文件夹：", compile_dir)
            file = glob(os.path.join(compile_dir, f"{name}*.{key}"))
            print("file:", file)
            assert (
                len(file) == 1
            ), f"The compile path must have an unique {key} file, pls check."
            ret = aidi_expmodel._create_exp_model_version_stage(
                file[0], f"{task_type}_{key}"
            )
            if len(ret) == 0:
                warnings.warn(
                    f"Cannot overwrite the {key} file, may be has been "
                    "uploaded before."
                )
        html_file = glob(os.path.join(compile_dir, f"{model_name}.html"))
        if len(html_file):
            ret = aidi_expmodel._create_exp_model_version_stage(
                html_file[0], f"{task_type}_html"
            )

        float_name = "float-checkpoint-best.pth.tar"
        qat_name = "qat-checkpoint-best.pth.tar"
        if "qat_ckpt_for_int_infer" in task:
            qat_name = task["qat_ckpt_for_int_infer"]
        if "float_ckpt_for_int_infer" in task:
            float_name = task["float_ckpt_for_int_infer"]
        qat_path = os.path.join(compile_dir, qat_name)
        float_path = os.path.join(compile_dir, float_name)
        metric_path = os.path.join(compile_dir, "metric.txt")
        if os.path.exists(qat_path):
            ret = aidi_expmodel._create_exp_model_version_stage(
                qat_path, "qat"
            )
        if os.path.exists(float_path):
            ret = aidi_expmodel._create_exp_model_version_stage(
                float_path, "float"
            )
        if os.path.exists(metric_path):
            ret = aidi_expmodel._create_exp_model_version_stage(
                metric_path, "metric"
            )


if __name__ == "__main__":
    # python3 projects/prediction/tools/compile.py \
    # --task-type densetnt_traj --compile 1 --upload 0 \
    # --hat-path /home/users/zifan.li/master/HAT
    args = parse_args()
    main(
        task_type=args.task_type,
        hat_path=args.hat_path,
        source_bpu=args.source_bpu,
        target_bpu=args.target_bpu,
        num_obs_in_hbm=args.num_obs,
        if_compile=args.compile,
        if_upload=args.upload,
    )
