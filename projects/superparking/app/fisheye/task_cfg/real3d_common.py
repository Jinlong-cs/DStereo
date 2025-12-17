import os
import time

import numpy as np
import torch

from hat.callbacks.metric_updater import update_metric_using_regex
from hat.core.virtual_camera import (
    CameraBase,
    CylindricalCamera,
    FisheyeCamera,
)
from hat.data.collates.collates import collate_real3d
from hat.metrics.real3d import Real3dEval
from hat.models.task_modules.mtfcos3d.decoder import limit_period
from hat.visualize.real3d import compute_box_3d
from ..common import (
    aidi_eval,
    config_file_root,
    get_aidi_eval_info_common,
    input_size,
    is_int_infer,
    real3d_num_workers,
    task_common_transforms,
    use_mini_dataset,
    val_gpu_transforms,
)
from ..lib.utils import _exists, load_yaml
from .vdvru_common import get_aidi_eval_dataset

data_version = "v1.1.6" if not use_mini_dataset else "ci_test"
view = "round"
focal_length_default = 605.0
depth_type = "Cylindrical"  # Cartesian
camera_type = "Cylindrical"
is_virtual = False
resize = True  # default 1/2 resize
keep_org_img = False
visual = False
rescale = False  # pred rescale to org image
num_classes = 1
input_sequence_length = 1
num_dist = 4
use_bev_lmdb = False
fisheye = True
use_dynamic_loss_weight = False
save_val_results = True
pred_json_name = "pred_vehicle.json"
save_pred_json = False
calib_json = "your camera calibration json path from xin06.zhang"
camera_names = (
    "fisheye_front",
    "fisheye_back",
    "fisheye_rear",
    "fisheye_left",
    "fisheye_right",
)
current_time = time.strftime("%Y%m%d-%H%M", time.localtime())

visual_save_base = "/horizon-bucket/auto_tmp/xudong.he/temp/debug_inferencev2/"
dataset_root = os.path.join(config_file_root, "datasets/real3d")
base_data_version = load_yaml(os.path.join(dataset_root, "data_version.yaml"))

lmdb_version = "4DGT_SD"


def get_input(mode, keep_org_img=False, rescale=False):
    assert mode in ["train", "val", "test"]
    inputs = dict(
        image_name=[
            "default.jpg",
        ],
        color_space=["bgr"],
        image_id=torch.Tensor(
            [
                [0],
            ]
        ),
        image_height=torch.Tensor(
            [
                [input_size[0]],
            ]
        ),
        image_width=torch.Tensor(
            [
                [input_size[1]],
            ]
        ),
        calibration=[torch.zeros((3, 3))],
        dist_coeffs=[torch.zeros((8,))],
        # ignore_mask=None,
        view=[
            "round",
        ],
        Tr_vel2cam=[
            torch.eye(
                4,
            )
        ],
        Tr_vcs2cam=[
            torch.eye(
                4,
            )
        ],
        # fcos3d add
        gt_bboxes=[torch.zeros((1, 4))],
        gt_classes=[torch.zeros((1,))],
        gt_bboxes_3d=[torch.zeros((1, 7))],
        gt_classes_3d=[torch.zeros((1,))],
        centers2d_prj=[torch.zeros((1, 2))],
        depths=[torch.ones((1,))],
        source_cam=[
            FisheyeCamera(image_size=[input_size[1], input_size[0]]),
        ],
        virtual_cam=[
            FisheyeCamera(image_size=[input_size[1], input_size[0]]),
        ],
        depth_type=["Cartesian"],
    )

    if aidi_eval:
        if keep_org_img:
            inputs["org_image"] = [np.zeros((*input_size[:2], 3))]

        pop_keys = [
            "gt_bboxes",
            "gt_classes",
            "gt_bboxes_3d",
            "gt_classes_3d",
            "centers2d_prj",
            "depths",
            "depth_type",
            "view",
            "Tr_vel2cam",
            "Tr_vcs2cam",
        ]

        for key in pop_keys:
            inputs.pop(key)

    elif mode == "val":
        inputs["annotations"] = dict()
        if keep_org_img:
            inputs["org_image"] = [np.zeros((input_size[0], input_size[1], 3))]

        if rescale:
            pop_keys = [
                "gt_bboxes",
                "gt_classes",
                "gt_bboxes_3d",
                "gt_classes_3d",
                "centers2d_prj",
                "depths",
                "depth_type",
            ]
            for key in pop_keys:
                inputs.pop(key)

    elif mode == "test" and is_int_infer:
        inputs = dict()
    return inputs


def get_inputs(mode):
    inputs_ = dict()
    if mode == "train":
        inputs_ = get_input("train", keep_org_img)
    elif mode == "val":
        inputs_ = get_input("val", keep_org_img, rescale)
    else:
        inputs_ = get_input("test", keep_org_img)
    return inputs_


def get_real3d_transforms(
    mode,
    max_depth=50,
    num_classes=3,
    depth_type="Cartesian",
    category_id_dict=None,
    repeat_times=1,
    rescale=False,
    keep_org_img=False,
    camera_type="Cylindrical",
    is_virtual=False,
    resize=False,
    flip=False,
    translate=False,
    warp=False,
):

    assert mode in ["train", "val", "test"]
    del_keys = ["imgs", "ignore_mask"]
    if mode == "train":
        del_keys.append("annotations")
    transforms = [
        dict(
            type="CreateVirtualCamera",
            task_type="real3d" if mode != "test" else "inference",
            source_cam=camera_type,
            virtual_cam="Cylindrical",
            image_size=[704 * 2, 576 * 2],  # W * H
            cameraMatrix=np.array(
                [[220 * 2, 0, 352 * 2], [0, 220 * 2, 288 * 2], [0, 0, 1]]
            ),
            is_virtual=is_virtual,
            calib_json=calib_json,
        ),
        dict(
            type="FCOS3DCameraPlugin",
            is_train=True if mode == "train" else False,
            num_classes=num_classes,
            category_id_dict=category_id_dict,
            bbox_ct=True,
            rescale=rescale,
            max_depth=max_depth,
            depth_type=depth_type,  # "Cylindrical",
            camera_names=camera_names,
            cache_static_map=True,
            is_warp_image=warp,
            keep_org_img=keep_org_img,
        ),
        dict(
            type="ImageToTensor",
            from_numpy=True,
        ),
        dict(
            type="ConvertLayout",
            hwc2chw=True,
            keys=["img"],
        ),
    ]
    if flip:
        transforms.insert(
            1,
            dict(
                type="CameraParamHorizontalFlip2D",
                flip_ratio=0.0 if mode != "train" else 0.5,
            ),
        )
    if resize:
        transforms.insert(
            1,
            dict(
                type="CameraParamResize",
                scale_x=0.5,  # 1/2 resize
                scale_y=0.5,  # 1/2 resize
            ),
        )
    if translate:
        transforms.insert(
            1,
            dict(
                type="CameraParamVerticalTranslation",
                designated_alignmnet_depth=10.0,
                norm_camera_z=0.75,
                random_trans=True,
                random_type="uniform",
                random_upper_boundary=2,
                random_lower_boundary=20,
            ),
        )
    if repeat_times > 1:
        transforms.append(
            dict(type="RepeatImage", times=repeat_times, output_key="img"),
        )
    transforms.append(
        dict(type="DeleteKeys", keys=del_keys),
    )
    return transforms


def get_single_result_for_aidi_eval_real3d(
    output,
    img_idx,
    input_meta,
    rescale,
    prefix,
    category_id_to_name,
    category_id_dict,
    merge_class=True,
):
    try:
        camera_name = "source_cam" if rescale else "virtual_cam"
        camera: CameraBase = input_meta[camera_name][img_idx]
    except KeyError:
        camera = CylindricalCamera.init_cam_param_by_matrix(
            image_size=[704, 576],  # W * H
            camera_matrix=np.array([[220, 0, 352], [0, 220, 288], [0, 0, 1]]),
            is_virtual=True,
        )

    model_output_class_to_actual_class = {}
    assert (
        category_id_dict is not None
    ), "visual mode need set category_id_dict"

    for key, value in category_id_dict.items():
        if value >= 0:
            model_output_class_to_actual_class[value] = key
    image_name = input_meta["image_name"][img_idx]

    pred_dict = {"image_key": image_name}
    cid_idx = 1 if merge_class else 0
    for _, value in model_output_class_to_actual_class.items():
        pred_dict[category_id_to_name[value][cid_idx]] = []

    def getattr_from_tensor_dict(name):
        return (
            getattr(output, f"{prefix}_{name}")[img_idx].clone().cpu().numpy()
        )

    dim = getattr_from_tensor_dict("dim")
    location = getattr_from_tensor_dict("location")
    category_id = getattr_from_tensor_dict("category_id")
    score2d = getattr_from_tensor_dict("score2d")
    score3d = getattr_from_tensor_dict("score")
    # attention, special op if is_train_3d_branch
    keep = getattr_from_tensor_dict("nms_keep")

    rotation_y = getattr(output, f"{prefix}_rotation_y")[img_idx]
    rotation_y = limit_period(rotation_y, 0.5, 2 * np.pi)
    rotation_y = rotation_y.clone().cpu().numpy()

    num_pred_obj = score3d.shape[0]
    for i in range(num_pred_obj):
        if not keep[i]:
            continue
        corner_3d = compute_box_3d(
            dim[i],
            location[i],
            rotation_y[i],
            pitch=0,
        )
        corner_2dp = camera.project_cam2pixel(corner_3d)
        # aidi metrics is projected 2d bbox
        bbox2d_prj = (
            np.min(corner_2dp, axis=0).tolist()
            + np.max(corner_2dp, axis=0).tolist()
        )
        sample = {
            "bbox_2d": bbox2d_prj,
            "score": np.float32(score3d[i]),
            "score2d": np.float32(score2d[i]),
            "depth": np.float32(location[i][-1]),
            "dimensions": dim[i].tolist(),
            "rotation_y": np.float32(rotation_y[i]),
            "location": location[i].tolist(),
        }
        actual_class = model_output_class_to_actual_class[category_id[i]]
        class_name = category_id_to_name[actual_class][cid_idx]
        pred_dict[class_name].append(sample)
    return pred_dict


def real3d_reformat_aidi_eval_out(
    batch,
    model_outs,
    category_id_to_name,
    category_id_dict,
):
    batch, name = batch
    output = model_outs[0]
    bs_num = len(batch["image_name"])
    batch_results = []
    prefix = "{}_{}".format(name, 0)

    for img_idx in range(bs_num):
        input_meta = batch
        pred_dict = get_single_result_for_aidi_eval_real3d(
            output,
            img_idx,
            input_meta,
            rescale,
            prefix,
            category_id_to_name,
            category_id_dict,
            merge_class=True,
        )
        batch_results.append(pred_dict)
    return batch_results


def _get_metric_update_func(mode, task_name):
    assert mode in ["train", "val"]

    def _train(metrics, batch, model_outs):
        for metric, loss in zip(metrics, model_outs[0]):
            metric.update(loss)

    def _val(metrics, batch, model_outs):
        pred_dict = {}
        image_id = []
        result_keys = [
            "dim",
            "category_id",
            "score",
            "center",
            "bbox",
            "dep",
            "alpha",
            "location",
            "rotation_y",
            "track_offset",
        ]

        index_real3d = -1
        for i in range(len(metrics)):
            if isinstance(metrics[i], Real3dEval):
                index_real3d = i
        assert index_real3d != -1, "Real3dEval is not in metrics list!"

        if task_name in model_outs[0]._fields[0]:
            for k, v in zip(model_outs[0]._fields, model_outs[0]):
                if "image_id" in k:
                    # flat image_id
                    image_id.extend(list(v))
                else:
                    for key in result_keys:
                        if "0_" + key in k:
                            pred_dict[key] = v

            pred_dict["image_id"] = image_id
            metrics[index_real3d].update(batch[0], pred_dict)

    if mode == "val":
        return _val
    if mode == "train":
        return _train


def get_train_metric_updater(
    task_name,
    log_freq,
    head_channels,
    use_dynamic_weight=False,
    disentangled_corner3D=False,
    use_multibin=False,
    multibin_margin=0.0,
):
    loss_names = []
    for name in head_channels.keys():
        if name == "rot_3d_group_reg":
            continue
        loss_names.append("loss_" + name)
        if use_dynamic_weight:
            loss_names.append(name + "_weight_loss")
    if disentangled_corner3D:
        loss_names.append("loss_corners_reg")
    if use_multibin:
        if multibin_margin > 0:
            loss_names.append("loss_rot_mask_l1")
        else:
            loss_names.append("loss_rot_3d_group_reg")
    else:
        loss_names.append("loss_rot_3d_group_reg")
    # loss_names.extend(["ce_loss2", "lovasz_loss2"])

    metric_updater = dict(
        type="MetricUpdater",
        metrics=[dict(type="LossShow", name=name) for name in loss_names],
        metric_update_func=update_metric_using_regex(
            per_metric_patterns=[
                {"label_pattern": None, "pred_pattern": f"^.*{n}*"}
                for n in loss_names
            ]
        ),
        filter_condition=lambda x: task_name
        in x[1],  # lambda x: x[1] == task_name,
        step_log_freq=log_freq,
        epoch_log_freq=1,
        log_prefix=task_name,
        reset_metrics_by="log",
    )
    return metric_updater


def get_val_metric_updater(
    need_eval_categories,
    task_name,
    eval_camera_names,
    depth_intervals=(20, 50, 80, 120),
    iou_threshold=0.2,
    score_threshold=0.1,
    gt_max_depth=300,
    save_path=None,
    num_dist=4,
    fisheye=True,
    metrics=("dxyp", "drot"),
    match_basis="det2d",
    save_name="vehicle.xlsx",
):
    val_metric_updater = dict(
        type="MetricUpdater",
        metrics=[
            dict(
                type="Real3dEval",
                need_eval_categories=need_eval_categories,
                eval_camera_names=eval_camera_names,
                metrics=metrics,
                depth_intervals=depth_intervals,
                iou_threshold=iou_threshold,
                score_threshold=score_threshold,
                gt_max_depth=gt_max_depth,
                save_path=save_path,
                num_dist=num_dist,
                fisheye=fisheye,
                match_basis=match_basis,
                save_name=save_name,
            )
        ],
        metric_update_func=_get_metric_update_func("val", task_name),
        log_prefix="Validation " + task_name,
        step_log_freq=-1,
    )
    return val_metric_updater


def get_data_info_fn(data_version, train_ls, val_ls):
    def _help(inf, ls):
        for i in ls:
            assert i.endswith(".rec") and _exists(i) and _exists(i + ".idx")
            inf["data"].append({"rec": i, "idx": i + ".idx"})

    def callback_fn(phase, verbose=False):
        info = {"version": data_version, "data": []}
        assert phase in ["train", "validation", "test"]
        if verbose:
            if phase == "train":
                _help(info, train_ls)
            elif phase == "validation":
                _help(info, val_ls)
            else:
                pass
            return info
        else:
            return info["version"]

    return callback_fn


EVAL_TYPE_3D_DET = "detection"


def get_real3d_aidi_eval_info(
    task_name, category_id_to_name, category_id_dict
):
    reformat_kwargs = dict(
        category_id_to_name=category_id_to_name,
        category_id_dict=category_id_dict,
    )
    return get_aidi_eval_info_common(
        task_name,
        EVAL_TYPE_3D_DET,
        real3d_reformat_aidi_eval_out,
        reformat_kwargs,
    )


def get_real3d_aidi_eval_loaders(batch_size, task_name, real3d_val_transform):
    pre_val_transform = [
        dict(
            type="RenameKeys",
            keys=[
                "img_name|image_name",
                "img_id|image_id",
                "img_height|image_height",
                "img_width|image_width",
            ],
        ),
    ]
    post_val_transform = (
        val_gpu_transforms
        + task_common_transforms
        + [
            dict(type="DeleteKeys", keys=["img_buf"]),
        ]
    )
    real3d_val_transform = (
        pre_val_transform + real3d_val_transform + post_val_transform
    )
    aidi_eval_loaders = [
        dict(
            type=torch.utils.data.DataLoader,
            dataset=ds,
            sampler=dict(type=torch.utils.data.DistributedSampler),
            collate_fn=collate_real3d,
            batch_size=batch_size,
            shuffle=False,
            num_workers=real3d_num_workers["val"],
            pin_memory=False,
            drop_last=False,
        )
        for ds in get_aidi_eval_dataset(
            task_name,
            real3d_val_transform,
            return_orig_hw=False,
            infer_model_type="real3d",
        )
    ]
    return aidi_eval_loaders
