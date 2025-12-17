import glob
import json
import os
import random

import torch

from projects.halo.cv.tools.pccr_calibration.data_utils import (
    gather_all_infos,
    padding_pbs,
)
from projects.halo.cv.tools.pccr_calibration.fitting import Fitting


def start_fitting(
    eye_params_init,
    lr,
    step_size,
    gamma,
    epochs,
    save_dir,
    loss_save_path,
    loss_weights,
    device,
    fitting_tags,
    all_info,
    idp,
    stage="train",
    mode=None,
):
    batch_size = 0
    fitting_infos = []
    max_pb_nums = 0
    min_pb_nums = 100
    for tag_d in fitting_tags:
        batch_size += len(all_info[tag_d]["img_path"])
        fitting_infos.append(all_info[tag_d])
        tag_info = all_info[tag_d]
        for pb in tag_info["pupil_boundary"]:
            max_pb_nums = max(max_pb_nums, len(pb))
            min_pb_nums = min(min_pb_nums, len(pb))
    # padding
    fitting_infos = padding_pbs(fitting_infos, max_pb_nums)

    if stage == "train" or mode is None or len(mode) < 1:
        fitting_save_path = os.path.join(
            save_dir,
            f"{idp}_tag_{fitting_tags}_imgnum_{batch_size}_{stage}.ckpt",
        )
    elif loss_weights.get("angle") is None:
        fitting_save_path = os.path.join(
            save_dir,
            f"{idp}_tag_{fitting_tags}_imgnum_{batch_size}_{stage}_{mode}_wo_angle.ckpt",
        )
    else:
        fitting_save_path = os.path.join(
            save_dir,
            f"{idp}_tag_{fitting_tags}_imgnum_{batch_size}_{stage}_{mode}.ckpt",
        )
    print("-----------------")
    print(fitting_save_path)
    print(max_pb_nums, min_pb_nums)
    print("-----------------")

    fitting = Fitting(
        batch_size=batch_size,
        device=device,
        eye_params_init=eye_params_init,
        lr=lr,
        step_size=step_size,
        gamma=gamma,
        epochs=epochs,
        save_path=fitting_save_path,
        data_infos=fitting_infos,
        loss_weights=loss_weights,
        mode=mode,
    )
    loss, state_dict = fitting()
    save_loss = {
        "save_ckpt": fitting_save_path,
    }
    for k, v in loss.items():
        save_loss[k] = float(v.item())
    with open(loss_save_path, "a") as f:
        f.writelines(json.dumps(save_loss) + "\n")

    return state_dict


def check_calib(loss_save_path, idp, train_state_dict, val_state_dict):
    dis_r = abs(float(train_state_dict["R"]) - float(val_state_dict["R"]))
    dis_k = abs(float(train_state_dict["K"]) - float(val_state_dict["K"]))
    dis_a = abs(
        float(train_state_dict["alpha"]) - float(val_state_dict["alpha"])
    )
    dis_b = abs(
        float(train_state_dict["beta"]) - float(val_state_dict["beta"])
    )
    save_calib_error = {
        "id": idp,
        "error_R": dis_r,
        "error_K": dis_k,
        "error_alpha": dis_a,
        "error_beta": dis_b,
    }
    with open(loss_save_path, "a") as f:
        f.writelines(json.dumps(save_calib_error) + "\n")


def calib_ids(
    img_infos_file,
    modelbase_res_file,
    lr_train,
    lr_val,
    step_size,
    gamma,
    epochs,
    save_dir,
    loss_weights_train,
    loss_weights_val,
    id_list=None,
    device="cuda:0",
    mode=None,
):
    os.makedirs(save_dir, exist_ok=True)
    loss_save_path = os.path.join(save_dir, "0_fitting_loss.txt")
    all_id_infos = gather_all_infos(img_infos_file, modelbase_res_file)
    for idp, all_info in all_id_infos.items():
        if id_list is not None and idp not in id_list:
            continue
        # total_tag_nums = len(all_info["all_tags"])
        # train_tag_num = int(
        #     min(total_tag_nums - 1, max(5, total_tag_nums * 0.5))
        # )
        # random.shuffle(all_info["all_tags"])
        # train_tags = all_info["all_tags"][:train_tag_num]

        train_tags_tmp = ["0", "1", "2", "3", "4"]
        train_tags = []
        val_tags = []
        for tt in all_info["all_tags"]:
            if tt in train_tags_tmp:
                train_tags.append(tt)
            else:
                val_tags.append(tt)

        train_save_path = glob.glob(
            os.path.join(save_dir, f"{idp}_tag_*_train.ckpt")
        )
        if len(train_save_path) > 0:
            train_state_dict = torch.load(train_save_path[0])
        else:
            init_eye = all_info["eye_params_modelbase"]

            train_state_dict = start_fitting(
                init_eye,
                lr_train,
                step_size,
                gamma,
                epochs,
                save_dir,
                loss_save_path,
                loss_weights_train,
                device,
                train_tags,
                all_info,
                idp,
                stage="train",
                mode=mode,
            )

        train_state_dict.pop("pitch")
        train_state_dict.pop("yaw")
        train_state_dict.pop("kq_result")

        _ = start_fitting(
            train_state_dict,
            lr_val,
            step_size,
            gamma,
            epochs,
            save_dir,
            loss_save_path,
            loss_weights_val,
            device,
            val_tags,
            all_info,
            idp,
            stage="val",
            mode="refitting_wo_angle",
        )


if __name__ == "__main__":
    img_infos_file = (
        "/home/users/yisu.zhou/gaze_all/1007_pccr/yisu0922input.txt"
    )
    modelbase_res_file = (
        "/home/users/yisu.zhou/gaze_all/1007_pccr/yisu0922resdict.txt"
    )
    id_list = [
        "ho0186-3",
        "ho1525-2",
        "ho1786-3",
        "ho1825-2",
        "ho4154-2",
        "ho5130-5",
        "ho6315-2",
        "ho6361-1",
    ]

    device = "cuda:0"

    mode = ""  # support2] #'','refitting'
    lr_train = [0.2, 1, 1, 0.1]
    lr_val = [0, 0.1, 0.01, 0]

    step_size = 3000
    gamma = 0.75
    epochs = 30000
    save_dir = "pccr_fitting_exp9"

    loss_weights_train = {
        "regularization": {
            "R": 0.001,
            "K": 0.001,
            "alpha": 0.002,
            "beta": 0.002,
        },
        "in_range": {
            "R": 1000,
            "K": 1000,
            "alpha": 1000,
            "beta": 1000,
            "pitch": 1000,
            "yaw": 1000,
            "kq": 1000,
            "r_pb": 100,
        },
        "pupil_dis": 5,  # center, K, pitch, yaw
        "center_dis": 50,  # kq R
        "eye3d_center_dis": 1,  # center_dis
        "eye3d_allow_dis": 30,
        "angle": 1,  # kappa, opt
        # "gaze_point_dis": 0.01,  # all
        # "gaze_point_dis_thresh": 10,
    }

    loss_weights_val = {
        "regularization": {
            "pitch": 0.0001,
            "yaw": 0.001,
        },
        "in_range": {
            "R": 0,
            "K": 0,
            "alpha": 0,
            "beta": 0,
            "pitch": 1000,
            "yaw": 1000,
            "kq": 1000,
            "r_pb": 100,
        },
        "pupil_dis": 1,  # center, K, pitch, yaw
        "center_dis": 50,  # kq R
        "eye3d_center_dis": 1,  # center_dis
        "eye3d_allow_dis": 30,
    }

    calib_ids(
        img_infos_file,
        modelbase_res_file,
        lr_train,
        lr_val,
        step_size,
        gamma,
        epochs,
        save_dir,
        id_list=id_list,
        device=device,
        loss_weights_train=loss_weights_train,
        loss_weights_val=loss_weights_val,
        mode=mode,
    )
