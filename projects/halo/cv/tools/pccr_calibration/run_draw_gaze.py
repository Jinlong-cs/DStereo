import glob
import os

import cv2
import numpy as np
import seaborn as sns
import torch

from projects.halo.cv.tools.pccr_calibration.data_utils import (
    gather_all_infos,
    padding_pbs,
)
from projects.halo.cv.tools.pccr_calibration.fitting import Fitting


def save_pog_show(
    info_draw, save_dir, idp, stage="train", mode="replace", error=None
):
    sns.set()
    palette = sns.color_palette("CMRmap", n_colors=20)
    bgr_color = []
    for color in palette:
        bgr_color.append(tuple(int(c * 255) for c in color))

    if mode is None or len(mode) < 1:
        img_save_path = f"{save_dir}/calib_{idp}.jpg"
    else:
        img_save_path = f"{save_dir}/calib_{idp}_{mode}.jpg"

    if stage == "train":
        rendering = np.zeros((1080, 1920, 3))
        gt_pogs, pred_pogs = info_draw
        text_loc = (50, 50)
        for gt_pog, pred_pog in zip(gt_pogs, pred_pogs):
            cv2.circle(
                rendering, pred_pog, 3, color=bgr_color[-1], thickness=2
            )
            cv2.circle(
                rendering, gt_pog, 10, color=bgr_color[-1], thickness=-1
            )
    else:
        assert os.path.exists(img_save_path)
        rendering = cv2.imread(img_save_path)
        text_loc = (800, 50)
        for i, info in enumerate(info_draw):
            gt_pogs, pred_pogs = info
            for gt_pog, pred_pog in zip(gt_pogs, pred_pogs):
                cv2.circle(
                    rendering,
                    pred_pog,
                    3,
                    color=bgr_color[i * 2],
                    thickness=2,
                )
                cv2.circle(
                    rendering, gt_pog, 10, color=bgr_color[i * 2], thickness=-1
                )
    if error is not None:
        cv2.putText(
            rendering,
            "{} gaze error: {:.4f}".format(stage, error),
            text_loc,
            cv2.FONT_HERSHEY_DUPLEX,
            1,
            (255, 255, 255),
        )
    cv2.imwrite(img_save_path, rendering)


def infer_pccr(
    batch_size,
    device,
    ckpt_dict,
    fitting_infos,
):
    R = ckpt_dict["R"]
    K = ckpt_dict["K"]
    alpha = ckpt_dict["alpha"]
    beta = ckpt_dict["beta"]
    pitch = ckpt_dict["pitch"]
    yaw = ckpt_dict["yaw"]
    kq_result = ckpt_dict["kq_result"]

    fitting = Fitting(
        batch_size=batch_size,
        device=device,
        eye_params_init={},
        lr=0.1,
        step_size=100,
        gamma=1,
        epochs=1,
        save_path="",
        data_infos=fitting_infos,
        loss_weights=None,
    )
    with torch.no_grad():
        fitting.eval()
        in_pccr = {
            "R": R.repeat(batch_size, 1),
            "K": K.repeat(batch_size, 1),
            "alpha": alpha.repeat(batch_size, 1),
            "beta": beta.repeat(batch_size, 1),
            "pitch": pitch,
            "yaw": yaw,
            "kq_result": kq_result,
            "light_left": fitting.light_left.repeat(batch_size, 1),
            "light_right": fitting.light_right.repeat(batch_size, 1),
            "cam_o": fitting.cam_o.repeat(batch_size, 1),
            "glint_ccs_list": fitting.data["glint_ccs"],
            "pupil_boundary_ccs": fitting.data["pupil_boundary_ccs"],
            "pb_real_mask": fitting.data["pb_real_mask"],
            "pixel_width": fitting.pixel_width.repeat(batch_size, 1),
            "pixel_height": fitting.pixel_height.repeat(batch_size, 1),
            "scr_left_top_3d": fitting.scr_left_top_3d.repeat(batch_size, 1),
            "scr_left_bottom_3d": fitting.scr_left_bottom_3d.repeat(
                batch_size, 1
            ),
            "scr_right_bottom_3d": fitting.scr_right_bottom_3d.repeat(
                batch_size, 1
            ),
        }
        pccr_pred = fitting.pccr(in_pccr)
        error = torch.linalg.norm(
            pccr_pred["screen_coords"] - fitting.data["gaze_point"], 2, -1
        ).mean()
        pred = pccr_pred["screen_coords"].to(torch.int32).cpu().numpy()
        gt = fitting.data["gaze_point"].to(torch.int32).cpu().numpy()
    return pred, gt, error.item()


def draw_gaze(
    save_dir,
    fitting_tags,
    all_info,
    idp,
    stage="train",
    mode="replace",
    device="cuda:0",
):
    batch_size = 0
    fitting_infos = []
    tag_list = [
        0,
    ]
    max_pb_nums = 0
    min_pb_nums = 100
    for tag_d in fitting_tags:
        batch_size += len(all_info[tag_d]["img_path"])
        tag_list.append(batch_size)
        fitting_infos.append(all_info[tag_d])
        tag_info = all_info[tag_d]
        for pb in tag_info["pupil_boundary"]:
            max_pb_nums = max(max_pb_nums, len(pb))
            min_pb_nums = min(min_pb_nums, len(pb))
    # padding
    fitting_infos = padding_pbs(fitting_infos, max_pb_nums)

    if stage == "val" and "refitting" in mode:
        fitting_save_path = glob.glob(
            os.path.join(
                save_dir,
                f"{idp}_tag_*_val_{mode}*.ckpt",
            )
        )[0]
    else:
        fitting_save_path = os.path.join(
            save_dir,
            f"{idp}_tag_{fitting_tags}_imgnum_{batch_size}_{stage}.ckpt",
        )
    ckpt_dict = torch.load(fitting_save_path, map_location=device)

    if stage == "val" and mode == "replace":
        fitting_save_path_train = glob.glob(
            os.path.join(save_dir, f"{idp}_tag_*_train.ckpt")
        )[0]
        ckpt_dict_train = torch.load(fitting_save_path_train)
        ckpt_dict["R"] = ckpt_dict_train["R"].to(device)
        ckpt_dict["K"] = ckpt_dict_train["K"].to(device)
        ckpt_dict["alpha"] = ckpt_dict_train["alpha"].to(device)
        ckpt_dict["beta"] = ckpt_dict_train["beta"].to(device)

    pred, gt, error = infer_pccr(batch_size, device, ckpt_dict, fitting_infos)

    if stage == "train":
        res = [gt.tolist(), pred.tolist()]
    else:
        res = []
        for i in range(len(fitting_tags)):
            gt_tmp = gt[tag_list[i] : tag_list[i + 1], :].tolist()
            pred_tmp = pred[tag_list[i] : tag_list[i + 1], :].tolist()
            res.append([gt_tmp, pred_tmp])

    save_pog_show(res, save_dir, idp, stage=stage, mode=mode, error=error)
    return error


def infer_for_draw(
    img_infos_file,
    modelbase_res_file,
    save_dir,
    id_list=None,
    mode="replace",
):
    all_id_infos = gather_all_infos(img_infos_file, modelbase_res_file)
    train_error_sum = 0
    val_error_sum = 0
    idp_num = 0
    for idp, all_info in all_id_infos.items():
        if id_list is not None and idp not in id_list:
            continue
        idp_num += 1
        train_tags_tmp = ["0", "1", "2", "3", "4"]
        train_tags = []
        val_tags = []
        for tt in all_info["all_tags"]:
            if tt in train_tags_tmp:
                train_tags.append(tt)
            else:
                val_tags.append(tt)

        error_train = draw_gaze(
            save_dir, train_tags, all_info, idp, stage="train", mode=mode
        )
        train_error_sum += error_train

        error_val = draw_gaze(
            save_dir,
            val_tags,
            all_info,
            idp,
            stage="val",
            mode=mode,
        )
        val_error_sum += error_val
    print("train_error: ", train_error_sum / idp_num)
    print("val_error: ", val_error_sum / idp_num)


if __name__ == "__main__":
    img_infos_file = (
        "/home/users/yisu.zhou/gaze_all/1007_pccr/yisu0922input.txt"
    )
    modelbase_res_file = (
        "/home/users/yisu.zhou/gaze_all/1007_pccr/yisu0922resdict.txt"
    )
    save_dir = "pccr_fitting_exp8"
    mode = "refitting_wo_angle"  # support '','refitting', 'replace','refitting_wo_angle'

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

    infer_for_draw(
        img_infos_file,
        modelbase_res_file,
        save_dir,
        id_list=id_list,
        mode=mode,
    )
