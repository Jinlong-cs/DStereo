# Copyright (c) Horizon Robotics. All rights reserved.
import copy
import json
import logging
import os
import re
from collections import OrderedDict
from dataclasses import fields
from typing import Dict, List, Optional, Sequence, Union

import cv2
import horizon_plugin_pytorch
import numpy as np
import torch
from horizon_plugin_pytorch.qtensor import QTensor

from hat.callbacks.save_eval_results.save_eval_result import SaveEvalResult
from hat.core.data_struct.base import BaseDataList
from hat.core.data_struct.base_struct import (
    ClsLabel,
    DetBox2D,
    DetBox3D,
    Line2D,
    Mask,
    MultipleBox2D,
    Point2D_2,
)
from hat.core.virtual_camera import (
    CylindricalCamera,
    FisheyeCamera,
    PinholeCamera,
    SphericalCamera,
)
from hat.registry import OBJECT_REGISTRY
from hat.utils.apply_func import _as_list, convert_numpy

__all__ = [
    "SaveDetConsistencyResult",
    "SaveSegConsistencyResult",
    "SaveBevDetConsistencyResult",
]

logger = logging.getLogger(__name__)


def align_ceil(para: int, interval: int, per_bytes: int, bias: List[str]):
    align_para = 0
    k = 0
    while align_para < para * per_bytes:
        for b in bias:
            align_para = interval * k + b
            if align_para >= para * per_bytes:
                return align_para // per_bytes
        k += 1


def align_padding(
    data: np.array,
    layout: str,
    per_bytes: int,
    dtype: np.dtype,
    input_layout: bool = True,
):
    """
    Specify the data layout for model input and output.

    Details see:
        https://horizonrobotics.feishu.cn/wiki/wikcnrUODfinotcrer3apT6xlib.

    Args:
        data: The data of numpy array.
        layout: Specify layout of array, support "NCHW" and "NHWC".
        per_bytes: Bytes occupied by each value.
        dtype: Output dtype.
        input_layout: The save directory of images.
    """

    assert layout in ["NCHW", "NHWC"], "only support 'NCHW' and 'NHWC'."

    if layout == "NCHW":
        n, c, h, w = data.shape
        new_w = align_ceil(w, 256, per_bytes, bias=[0, 16, 32, 64, 128])
        new_data = np.zeros((n, c, h, new_w), dtype=dtype)
        new_data[:, :, :, :w] = data
    else:
        n, h, w, c = data.shape
        if input_layout and c <= 4:
            new_h = align_ceil(c, 2, per_bytes, bias=[0])
            new_w = align_ceil(w, 32, per_bytes, bias=[0])
            new_data = np.zeros((n, new_h, new_w, c), dtype=dtype)
            new_data[:, :h, :w, :] = data
        else:
            new_c = align_ceil(c, 256, per_bytes, bias=[0, 16, 32, 64, 128])
            new_data = np.zeros((n, h, w, new_c), dtype=dtype)
            new_data[:, :, :, :c] = data

    return new_data


def dump_bbox(det_box: DetBox2D):
    box = det_box.box.numpy().astype(float)
    return {
        "conf": det_box.score.item() * 10,
        "x1": box[0],
        "x2": box[2],
        "y1": box[1],
        "y2": box[3],
    }


def dump_kps_2(kps: Point2D_2):
    p0 = kps.point0.point.numpy().astype(float)
    p1 = kps.point1.point.numpy().astype(float)
    return [
        {
            "conf": kps.point0.score.item() * 10,
            "type": 0,
            "x1": p0[0],
            "y1": p0[1],
        },
        {
            "conf": kps.point1.score.item() * 10,
            "type": 1,
            "x1": p1[0],
            "y1": p1[1],
        },
    ]


def dump_multi_box(multi_box: MultipleBox2D):
    boxes = multi_box.boxes.numpy().astype(float)

    return [
        {
            "conf": score.item() * 10,
            "x1": box[0],
            "x2": box[2],
            "y1": box[1],
            "y2": box[3],
        }
        for box, score in zip(boxes, multi_box.scores)
    ]


def dump_line(line: Line2D):
    res = dump_kps_2(line)
    res[0]["type"] = 6
    res[1]["type"] = 7

    return res


def dump_cls_label(cls_label: ClsLabel):
    return [
        {
            "conf": cls_label.score.item() * 10,
            "property_id": cls_label.cls_idx.item(),
            "property_name": cls_label.cls_name,
        }
    ]


def dump_bbox_3d(box_3d: DetBox3D):
    ret = {
        "width": box_3d.w.item(),
        "height": box_3d.h.item(),
        "length": box_3d.l.item(),
        "x": box_3d.x.item(),
        "y": box_3d.y.item(),
        "z": box_3d.z.item(),
        "yaw": box_3d.yaw.item(),
    }
    if isinstance(box_3d.cls_name, str):
        ret.update(
            {
                "conf": box_3d.score.item(),
                "type": box_3d.cls_idx.item(),
            }
        )
    else:
        ret.update(
            {
                "type": -1,
            }
        )
    return ret


DUMP_FUNC_MAPS = {
    ClsLabel: dump_cls_label,
    DetBox2D: dump_bbox,
    MultipleBox2D: dump_multi_box,
    Point2D_2: dump_kps_2,
    Line2D: dump_line,
    DetBox3D: dump_bbox_3d,
}

DUMP_KEY_MAPS = {
    DetBox2D: "box2d",
    ClsLabel: "category",
    DetBox3D: "box3d",
}


@OBJECT_REGISTRY.register
class SaveDetConsistencyResult(SaveEvalResult):
    """
    Save consistency results of bev-detection tasks.

    Args:
        output_dir: output directory.
        task_type_mapping: mapper for task type.
        task_name_mapping: mapper for task name.
        dump_extra_key: dump extra key for model result.
        bev_mode: whether eval is bev mode.
    """

    def __init__(
        self,
        output_dir: str,
        task_type_mapping: Optional[Dict] = None,
        task_name_mapping: Optional[Dict] = None,
        dump_extra_key: Optional[str] = None,
        bev_mode: Optional[bool] = False,
        default_camera_model: str = "pinhole",
        to_vcs: bool = False,
    ):
        super().__init__(output_dir)
        self.task_type_mapping = task_type_mapping
        self.task_name_mapping = task_name_mapping
        self.dump_extra_key = dump_extra_key
        self.bev_mode = bev_mode
        self.CAMERA_MODELS = {
            "cylindrical": CylindricalCamera,
            "fisheye": FisheyeCamera,
            "pinhole": PinholeCamera,
            "spherical": SphericalCamera,
        }
        self.default_camera_model = default_camera_model
        self.to_vcs = to_vcs

    def bbox3d_cam_to_vcs(self, cam, bbox3d):
        assert isinstance(bbox3d, DetBox3D)
        location = bbox3d.location.cpu().numpy()
        yaw = bbox3d.yaw.cpu().numpy()
        rot_vector = np.stack(
            [np.cos(yaw), np.zeros_like(yaw), -np.sin(yaw)],
            axis=-1,
        )
        location = np.expand_dims(location, axis=0)
        location = cam.project_cam2vcs(location)
        poseMat_cam2vcs = np.linalg.inv(cam.poseMat_vcs2cam)
        rot_vector = np.dot(rot_vector, poseMat_cam2vcs[:3, :3].T)
        yaw = np.arctan2(rot_vector[..., 1], rot_vector[..., 0])

        location = bbox3d.x.new_tensor(location)
        yaw = bbox3d.x.new_tensor(yaw)

        bbox3d.x = location[0, 0]
        bbox3d.y = location[0, 1]
        bbox3d.z = location[0, 2]
        bbox3d.yaw = yaw
        return bbox3d

    def get_calib_all_cpu(self, calib_all, idx):
        calib_cpu = OrderedDict()
        for key, value in calib_all.items():
            if isinstance(value, list):
                if isinstance(value[0], torch.Tensor):
                    calib_cpu[key] = [x[idx].to("cpu") for x in value]
                else:
                    calib_cpu[key] = value[idx]
            elif isinstance(value, dict):
                calib_cpu[key] = OrderedDict()
                for subkey, subvalue in value.items():
                    if isinstance(subvalue, list):
                        if isinstance(subvalue[0], torch.Tensor):
                            calib_cpu[key][subkey] = [
                                x[idx].to("cpu") for x in subvalue
                            ]
                        else:
                            calib_cpu[key][subkey] = subvalue[idx].to("cpu")
                    else:
                        calib_cpu[key][subkey] = subvalue[idx].to("cpu")
            else:
                calib_cpu[key] = value[idx].to("cpu")
        return calib_cpu

    def save_result(self, batch, result):
        if self.bev_mode:
            img_ids = [i[0] for i in batch["img_id"]]
        else:
            img_ids = batch["img_id"]

        for i, img_id in enumerate(img_ids):

            if self.to_vcs:
                calib_all = self.get_calib_all_cpu(batch["calib_all"], i)
                camera = self.CAMERA_MODELS[self.default_camera_model]()
                camera = camera.init_cam_param_by_dict(calib_all)

            res_dict = {}
            for k, name in self.task_type_mapping.items():
                data = result[k][i].to("cpu")
                data = data.to("cpu")
                if isinstance(data, BaseDataList):
                    res_dict[name] = []
                    for item in iter(data):
                        dd = {}
                        for f in fields(item):
                            attr = getattr(item, f.name)
                            if self.to_vcs:
                                if isinstance(attr, DetBox3D):
                                    attr = self.bbox3d_cam_to_vcs(camera, attr)
                            dump_func = DUMP_FUNC_MAPS.get(type(attr), None)
                            if dump_func is not None:
                                assert (
                                    type(attr) in DUMP_KEY_MAPS
                                    or f.name in self.task_name_mapping
                                )
                                dump_key = self.task_name_mapping.get(
                                    f.name, None
                                )
                                if dump_key is None:
                                    dump_key = DUMP_KEY_MAPS[type(attr)]

                                if dump_key in dd:
                                    dd[dump_key] += dump_func(attr)
                                else:
                                    dd[dump_key] = dump_func(attr)

                        res_dict[name].append(dd)

            if self.dump_extra_key is not None:
                res_dict = {self.dump_extra_key: res_dict}

            res_dict = {img_id: res_dict}

            with open(
                os.path.join(self.output_dir, f"{img_id}.json"), "w"
            ) as wf:
                json.dump(res_dict, wf, indent=2)

    def on_batch_end(self, batch, model_outs, **kwargs):
        if self.match_task_output:
            batch_output, match_output = self.match_task_output(
                batch, model_outs
            )
        else:
            batch_output, match_output = batch, model_outs

        self.save_result(batch_output, match_output)

    def __repr__(self):
        return "SaveDetConsistencyResult"


@OBJECT_REGISTRY.register
class SaveSegConsistencyResult(SaveEvalResult):
    """
    Save consistency results of segmentation tasks.

    Args:
        output_dir: output directory.
        task_list: list of tasks.
        dump_extra_key: dump extra key for model result.
        bev_mode: whether eval is bev mode.
        task_name_map: mapper for task name.
    """

    def __init__(
        self,
        output_dir: str,
        task_list: Optional[List[str]] = None,
        dump_extra_key: Optional[str] = None,
        bev_mode: Optional[bool] = False,
        task_name_map: Optional[Dict] = None,
    ):
        super().__init__(output_dir)
        self.task_list = task_list
        self.dump_extra_key = dump_extra_key
        self.bev_mode = bev_mode
        self.task_name_map = task_name_map

    def save_result(self, batch, result):
        if self.bev_mode:
            img_ids = [i[0] for i in batch["img_id"]]
        else:
            img_ids = batch["img_id"]

        for i, img_id in enumerate(img_ids):

            data = (
                {k: result[k][i] for k in self.task_list if k in result}
                if self.task_list is not None
                else {None: result[i]}
            )

            for k, task_data in data.items():
                assert isinstance(task_data, Mask)
                task_data = task_data.to("cpu")

                output_names = []
                if self.dump_extra_key is not None:
                    output_names.append(self.dump_extra_key)
                if k is not None:
                    if self.task_name_map:
                        task_name = self.task_name_map.get(k, k)
                    else:
                        task_name = k
                    output_names.append(task_name)
                output_names.append(img_id)

                out_name = "_".join(output_names) + ".bin"

                with open(os.path.join(self.output_dir, out_name), "wb") as wf:
                    task_data.mask.numpy().astype(np.uint8).tofile(wf)

    def on_batch_end(self, batch, model_outs, **kwargs):
        if self.match_task_output:
            batch_output, match_output = self.match_task_output(
                batch, model_outs
            )
        else:
            batch_output, match_output = batch, model_outs

        self.save_result(batch_output, match_output)

    def __repr__(self):
        return "SaveSegConsistencyResult"


@OBJECT_REGISTRY.register
class SaveBevDetConsistencyResult(SaveEvalResult):
    """
    Save consistency results of bev-detection tasks.

    Args:
        output_dir: output directory.
        task_type_mapping: mapper for task type.
        dump_extra_key: dump extra key for model result.
    """

    def __init__(
        self,
        output_dir: str,
        task_type_mapping: Optional[Dict] = None,
        dump_extra_key: Optional[str] = None,
    ):
        super().__init__(output_dir)
        self.task_type_mapping = task_type_mapping
        self.dump_extra_key = dump_extra_key

    def save_result(self, batch, result):
        res_dict = {}
        for i, img_id in enumerate(batch["sample_id"]):
            for k, name in self.task_type_mapping.items():
                if k not in result:
                    continue
                data = result[k][i].to("cpu")
                data = data.to("cpu")
                if isinstance(data, BaseDataList):
                    res_dict[name] = []
                    for item in iter(data):
                        dd = {}
                        dump_func = DUMP_FUNC_MAPS.get(type(item), None)
                        if dump_func is not None:
                            assert type(item) in DUMP_KEY_MAPS

                            dump_key = DUMP_KEY_MAPS[type(item)]

                            if dump_key in dd:
                                dd[dump_key] += dump_func(item)
                            else:
                                dd[dump_key] = dump_func(item)

                        res_dict[name].append(dd)

            if self.dump_extra_key is not None:
                res_dict = {self.dump_extra_key: res_dict}

            res_dict = {img_id: res_dict}

            with open(
                os.path.join(self.output_dir, f"{img_id}.json"), "w"
            ) as wf:
                json.dump(res_dict, wf, indent=2)

    def on_batch_end(self, batch, model_outs, **kwargs):
        if self.match_task_output:
            batch_output, match_output = self.match_task_output(
                batch, model_outs
            )
        else:
            batch_output, match_output = batch, model_outs

        self.save_result(batch_output, match_output)

    def __repr__(self):
        return "SaveBevDetConsistencyResult"


@OBJECT_REGISTRY.register
class SaveBEV3DConsistencyResult(SaveEvalResult):
    """
    Save consistency results of bev3d tasks.

    Args:
        output_dir: output directory.
        prefix: prefix of predict results.
    """

    def __init__(
        self,
        output_dir: str,
        prefix: str,
    ):
        super().__init__(output_dir)
        self.prefix = prefix

        self.PACK_CATEGORY_MAP = {
            "bev_3d_vehicle": {i: "bev_3d_vehicle" for i in range(7)},
            "bev_3d_vrumerge": {0: "bev_3d_cyclist", 1: "bev_3d_pedestrian"},
        }
        self.vehicle_cls_id_map = {
            4: 3,
            3: 4,
        }

    def on_batch_end(self, batch, model_outs, **kwargs):

        model_outs = convert_numpy(model_outs)

        batch_size = batch["img"][0].shape[0]

        task_name_list = [
            k for k in self.PACK_CATEGORY_MAP.keys() if k in model_outs
        ]

        category_list = copy.copy(task_name_list)
        if "bev_3d_vrumerge" in category_list:
            category_list.remove("bev_3d_vrumerge")
            category_list.extend(["bev_3d_cyclist", "bev_3d_pedestrian"])

        for batch_i in range(batch_size):
            name = int(convert_numpy(batch["timestamp"][batch_i]) * 1000)
            result = {f"{name}": {f"{task}": [] for task in category_list}}
            for task in task_name_list:
                task_results = model_outs[task]
                if "obj" in self.prefix:
                    obj = task.split("_")[-1]
                    prefix = self.prefix.replace("obj", obj)
                else:
                    prefix = self.prefix

                bev3d_dim = task_results[prefix + "_bev3d_dim"]
                bev3d_ct = task_results[prefix + "_bev3d_ct"]
                bev3d_loc_z = task_results[prefix + "_bev3d_loc_z"]
                bev3d_rot = task_results[prefix + "_bev3d_rot"]
                bev3d_score = task_results[prefix + "_bev3d_score"]
                bev3d_cls_id = task_results[prefix + "_bev3d_cls_id"]
                bev3d_occlusion_id = task_results[
                    prefix + "_bev3d_occlusion_id"
                ]

                bev3d_dim_ = bev3d_dim[batch_i]
                bev3d_ct_ = bev3d_ct[batch_i]
                bev3d_loc_z_ = bev3d_loc_z[batch_i]
                bev3d_rot_ = bev3d_rot[batch_i]
                bev3d_score_ = bev3d_score[batch_i]
                bev3d_cls_id_ = bev3d_cls_id[batch_i]
                bev3d_occlusion_id_ = bev3d_occlusion_id[batch_i]

                num_bboxes = bev3d_dim.shape[1]
                for idx in range(num_bboxes):
                    if bev3d_score_[idx] > 0:
                        dim = bev3d_dim_[idx]  # h w l
                        ct = bev3d_ct_[idx]  # x y
                        rot = bev3d_rot_[idx]
                        z = bev3d_loc_z_[idx]
                        cls_id = bev3d_cls_id_[idx]
                        if int(cls_id) in self.vehicle_cls_id_map:
                            cls_id = self.vehicle_cls_id_map[int(cls_id)]
                        occlusion_id = bev3d_occlusion_id_[idx]
                        result[f"{name}"][
                            f"{self.PACK_CATEGORY_MAP[task][cls_id]}"
                        ].append(
                            {
                                "conf": np.float64(bev3d_score_[idx]),
                                "height": np.float64(dim[0]),
                                "length": np.float64(dim[2]),
                                "width": np.float64(dim[1]),
                                "x": np.float64(ct[0]),
                                "y": np.float64(ct[1]),
                                "yaw": np.float64(rot),
                                "z": np.float64(z),
                                "sub_type": int(cls_id)
                                if task == "bev_3d_vehicle"
                                else None,
                                "occlusion_id": int(occlusion_id),
                            }
                        )

            with open(
                os.path.join(self.output_dir, f"{name}.json"), "w"
            ) as fw:
                json.dump(result, fw, ensure_ascii=False, indent=4)

    def __repr__(self):
        return "SaveBEV3DConsistencyResult"


@OBJECT_REGISTRY.register
class SaveDiscObjConsistencyResult(SaveEvalResult):
    """
    Save consistency results of disc-object tasks.

    Args:
        output_dir: output directory.
        prefix: prefix of predict results.
        task_res_key_cfg: task key in model outputs.
        scor_thr_cfg: score threshold of each task.
        merge_crosswalk_to_junction: whether crosswalk in junction task.
    """

    def __init__(
        self,
        output_dir: str,
        prefix: str,
        task_res_key_cfg: Dict,
        scor_thr_cfg: Dict,
        merge_crosswalk_to_junction: bool = False,
    ):
        super().__init__(output_dir)
        self.prefix = prefix
        self.task_res_key_cfg = task_res_key_cfg
        self.scor_thr_cfg = scor_thr_cfg
        # model class id to pack proto type id
        # pls refer to https://gitlab.hobot.cc/ptd/ap/middleware/protocol/-/blob/master/hobot-auto-protocol/raw/perception/object.proto # noqa
        self.DISCOBJ_CLS_PACk_TYPE = {
            "roadmarking": {
                0: 5,
                3: 6,
            },
            "junction": {
                0: 16,
                1: 8,
            },
            "cementcolumn": {
                0: 14,
            },
            "parkinglock": {
                0: 13,
                1: 12,
            },
            "sod3d": {
                0: 11,
            },
        }
        self.DISCOBJ_CLS_PACk_SUBTYPE = {
            "arrow": {
                2: 1,
                1: 2,
                3: 3,
                13: 4,
                14: 5,
                15: 6,
                4: 7,
                17: 8,
                16: 9,
                5: 10,
                6: 11,
                8: 15,
                9: 16,
                10: 17,
                18: 18,
                7: 28,
                11: 29,
                12: 30,
            },
            "roadmarking": {
                1: 12,  # 32 is ok
                2: 25,  # 33 is ok
                4: 31,
                5: 36,
            },
        }
        if not merge_crosswalk_to_junction:
            self.DISCOBJ_CLS_PACk_TYPE["roadmarking"] = {0: 8, 1: 5, 4: 6}
            self.DISCOBJ_CLS_PACk_TYPE["junction"] = {0: 16}
            self.DISCOBJ_CLS_PACk_SUBTYPE["roadmarking"] = {
                2: 12,  # 32 is ok
                3: 25,  # 33 is ok
                5: 31,
                6: 36,
            }

    def get_vertices_from_box(self, wh, ct, yaw, is_right_hand_coor):
        w, h = wh[:2]
        ctx, cty = ct

        p0 = [0.5 * w, 0.5 * h, 1]
        p1 = [0.5 * w, -0.5 * h, 1]
        p2 = [-0.5 * w, -0.5 * h, 1]
        p3 = [-0.5 * w, 0.5 * h, 1]
        points = np.array([p0, p1, p2, p3])
        if is_right_hand_coor:
            R = np.array(
                [
                    [np.cos(yaw), -np.sin(yaw), ctx],
                    [np.sin(yaw), np.cos(yaw), cty],
                    [0, 0, 1],
                ]
            )
        else:
            R = np.array(
                [
                    [np.cos(yaw), np.sin(yaw), ctx],
                    [-np.sin(yaw), np.cos(yaw), cty],
                    [0, 0, 1],
                ]
            )
        points = R.dot(points.T).T
        return points[:, :2]

    def on_batch_end(self, batch, model_outs, **kwargs):

        model_outs = convert_numpy(model_outs)
        batch_size = batch["img"][0].shape[0]

        task_name_list = [
            k for k in self.task_res_key_cfg.keys() if k in model_outs
        ]

        for batch_i in range(batch_size):
            name = int(convert_numpy(batch["timestamp"][batch_i]) * 1000)
            result = {
                "header": {"time_stamp": np.float64(name)},
                "objects": [],
            }
            for task in task_name_list:
                scor_thr = self.scor_thr_cfg.get(task, 0.2)
                for category in _as_list(self.task_res_key_cfg[task]):
                    task_result = model_outs[task]
                    for out_key, out_res in task_result.items():
                        if re.compile("^.*_pred_bev_discobj_ct").match(
                            out_key
                        ):
                            ct = out_res[batch_i]

                        if re.compile("^.*_pred_bev_discobj_wh").match(
                            out_key
                        ):
                            wh = out_res[batch_i]

                        if re.compile("^.*_pred_bev_discobj_rot").match(
                            out_key
                        ):
                            rot = out_res[batch_i]

                        if re.compile("^.*_pred_bev_discobj_score").match(
                            out_key
                        ):
                            score = out_res[batch_i]

                        if re.compile("^.*_pred_bev_discobj_cls_id").match(
                            out_key
                        ):
                            cls_id = out_res[batch_i]

                    for j in range(len(score)):
                        # filter fake data
                        if cls_id[j] == -1:
                            continue
                        score_ = score[j]
                        if score_ >= scor_thr:
                            if category in self.DISCOBJ_CLS_PACk_TYPE:
                                pack_cls_type = (
                                    pack_cls_type
                                ) = self.DISCOBJ_CLS_PACk_TYPE[category].get(
                                    cls_id[j], 4
                                )
                            else:
                                pack_cls_type = 4
                            if pack_cls_type == 4:
                                pack_cls_subtype = (
                                    self.DISCOBJ_CLS_PACk_SUBTYPE[category][
                                        cls_id[j]
                                    ]
                                )
                            else:
                                pack_cls_subtype = 0
                            vcs_points = self.get_vertices_from_box(
                                np.float64(wh[j]),
                                np.float64(ct[j]),
                                np.float64(rot[j]),
                                is_right_hand_coor=True,
                            )
                            pred_obj = {
                                "border": {
                                    "points": [
                                        list(vcs_points[pt_idx])
                                        for pt_idx in range(len(vcs_points))
                                    ]
                                },
                                "type": pack_cls_type,
                                "sub_type": pack_cls_subtype,
                                "conf": np.float64(score_),
                            }
                            result["objects"].append(pred_obj)

            with open(
                os.path.join(self.output_dir, f"{name}.json"), "w"
            ) as fw:
                json.dump(result, fw, ensure_ascii=False, indent=4)

    def __repr__(self):
        return "SaveDiscObjConsistencyResult"


@OBJECT_REGISTRY.register
class SaveSegmentationConsistencyResult(SaveEvalResult):
    """
    Save consistency results of segmentation tasks.

    Args:
        output_dir: output directory.
        prefix: prefix of predict results.
        task_list: list of tasks.
    """

    def __init__(
        self,
        output_dir: str,
        prefix: str,
        task_list: Optional[List[str]] = None,
    ):
        super().__init__(output_dir)
        self.task_list = task_list
        self.prefix = prefix

    def _trans_results(self, objs_res):
        if isinstance(objs_res, dict):
            objs_res = {k: self._trans_results(v) for k, v in objs_res.items()}
        elif isinstance(objs_res, list):
            objs_res = [self._trans_results(i) for i in objs_res]
        elif isinstance(objs_res, torch.Tensor):
            objs_res = objs_res.cpu().numpy().tolist()
        elif isinstance(objs_res, np.ndarray):
            objs_res = objs_res.tolist()
        elif isinstance(objs_res, np.floating):
            objs_res = float(objs_res)
        elif isinstance(objs_res, np.integer):
            objs_res = int(objs_res)
        return objs_res

    def on_batch_end(self, batch, model_outs, **kwargs):
        model_outs = convert_numpy(model_outs)
        batch_size = batch["img"][0].shape[0]
        task_name_list = [k for k in self.task_list if k in model_outs]
        for batch_i in range(batch_size):
            timestamp = int(convert_numpy(batch["timestamp"][batch_i]) * 1000)
            res = {}
            for task in task_name_list:
                task_result = model_outs[task]
                batch_result = task_result[
                    self.prefix + f"_pred_{task}_frame0"
                ][0][batch_i : batch_i + 1]

                objs_res = self._trans_results(batch_result)
                if not isinstance(objs_res, dict):
                    res.update({task: objs_res})
                else:
                    res.update(objs_res)
            with open(
                os.path.join(self.output_dir, f"{timestamp}.json"), "w"
            ) as fw:
                json.dump(res, fw, ensure_ascii=False, indent=4)

    def __repr__(self):
        return "SaveSegmentationConsistencyResult"


@OBJECT_REGISTRY.register
class SaveOnlineMappingConsistencyResult(SaveEvalResult):
    """
    Save consistency results of online-mapping tasks.

    Args:
        output_dir: output directory.
        output_key: output_key of predict results.
    """

    def __init__(
        self,
        output_dir: str,
        output_key: str,
    ):
        super().__init__(output_dir)
        self.output_key = output_key
        self.init_cls_remap()

    def init_cls_remap(self):
        self.subtype_key = {
            "lane": [
                "direction",
                "color",
                "wide",
                "double",
                "property",
                "dashed",
            ],
            "roadedge": [
                "subtype",
                "direction",
            ],
        }

        self.category2id = {
            "lane": 0,
            "roadedge": 1,
        }

        self.OMSubType = {
            "sub_type_unknown": 0,
            "lane_other": 32,
            "lane_unknown": 32,
            "lane_general": 33,
            "lane_stay": 34,
            "lane_tide": 35,
            "lane_three_color": 36,
            "lane_bus": 37,
            "roadedge_other": 64,
            "roadedge_groundside": 65,
            "roadedge_roadside": 66,
            "roadedge_cone": 67,
            "roadedge_water_horse": 68,
            "roadedge_guardrail": 69,
            "crosswalk_unknown": 96,
            # for unused subtype index, in case of the model has unexpected output, we set is as other (unknown).  # noqa
            "roadedge_cls_placeholder": 64,
            "roadedge_cone_water_horse": 67,
        }

        self.OMDirection = {
            "direction_unknown": 0,
            "direction_up": 1,
            "direction_down": 2,
            "direction_left": 3,
            "direction_right": 4,
            "direction_bidirection": 5,
            "direction_num": 6,
        }

        self.OMLaneColor = {
            "color_unknown": 0,
            "color_white": 1,
            "color_yellow": 2,
            "color_blue": 3,
            "color_orange": 4,
            "color_red": 5,
            "color_other": 6,
            "color_num": 7,
        }

        self.OMLaneAttribute = {
            "attr_non": 0,
            "attr_known": 1 << 0,
            "attr_dashed": 1 << 1,
            "attr_fishbone": 1 << 2,
            "attr_directional": 1 << 3,
            "attr_double": 1 << 4,
            "attr_wide": 1 << 5,
            "attr_left_solid_right_dash": 1 << 8,
            "attr_left_dash_right_solid": 1 << 9,
        }

    def on_batch_end(self, batch, model_outs, **kwargs):
        model_outs = convert_numpy(model_outs)

        batch_size = batch["img"][0].shape[0]
        for batch_i in range(batch_size):
            total_pts_stats = []
            name = int(convert_numpy(batch["timestamp"][batch_i]) * 1000)
            om_result = model_outs["online_mapping"][self.output_key][batch_i]
            pred_stats = om_result["pred_stats"]
            group_thrs = om_result["aux_info"]["group_thrs"]
            top, left = om_result["aux_info"]["vcs_range"][2:]
            h, w = om_result["aux_info"]["out_size"]
            res_h, res_w = om_result["aux_info"]["out_res"]
            head_groups = copy.deepcopy(om_result["aux_info"]["head_groups"])
            unit_res = max(res_h, res_w)

            # reuse lane direction as roadedge direction
            if "lane_direction_cls" in pred_stats["lane"]:
                channel = pred_stats["roadedge"]["prob"].shape[0]
                pred_stats["roadedge"]["roadedge_direction_cls"] = pred_stats[
                    "lane"
                ]["lane_direction_cls"][:channel, ...]
                head_groups["roadedge_direction"] = head_groups[
                    "lane_direction"
                ]

            for group, group_stats in pred_stats.items():
                pos_mask = group_stats["prob"] > group_thrs[group]
                pos_prob = group_stats["prob"][pos_mask]
                pos_cls = np.full_like(pos_prob, self.category2id[group])
                hw_array = np.array(np.meshgrid(np.arange(h), np.arange(w))).T
                hw_array = np.repeat(
                    hw_array[None], group_stats["prob"].shape[0], 0
                )
                h_array = hw_array[..., 0][pos_mask]
                w_array = hw_array[..., 1][pos_mask]
                pos_offset = group_stats["offset"][pos_mask]

                x_vcs_origin = top - h_array * res_h - res_h / 2
                y_vcs_origin = left - w_array * res_w - res_w / 2
                x_vcs = x_vcs_origin + pos_offset[:, 0] * unit_res
                y_vcs = y_vcs_origin + pos_offset[:, 1] * unit_res

                sin = group_stats["sin"][pos_mask]
                cos = group_stats["cos"][pos_mask]
                embedding = group_stats["embedding"][pos_mask]

                subtype_stats = {}
                subtype_cls = {}
                for subtype in self.subtype_key[group]:
                    key_name = f"{group}_{subtype}_cls"
                    if key_name not in group_stats:
                        continue
                    subtype_stats[subtype] = group_stats[key_name][pos_mask]
                    cls_name = []
                    cls_list = head_groups[f"{group}_{subtype}"]["cls_list"]
                    for i in subtype_stats[subtype]:
                        cls_name.append(cls_list[i - 1])
                    subtype_cls[subtype] = cls_name

                (
                    subtype,
                    lane_attribute,
                    lane_direction,
                    lane_color,
                ) = self.reamp_attribute(subtype_cls, group)
                pts_id = np.arange(pos_cls.shape[0])
                group_pts_stats = np.concatenate(
                    [
                        pos_cls[:, None],
                        pts_id[:, None],
                        w_array[:, None],
                        h_array[:, None],
                        np.where(pos_mask.astype(int) == 1)[0][:, None],
                        x_vcs[:, None],
                        y_vcs[:, None],
                        sin[:, None],
                        cos[:, None],
                        pos_prob[:, None],
                        embedding,
                        subtype[:, None],
                        lane_attribute[:, None],
                        lane_direction[:, None],
                        lane_color[:, None],
                    ],
                    1,
                )
                total_pts_stats.append(group_pts_stats)

            total_pts_stats = np.concatenate(total_pts_stats, 0)

            with open(os.path.join(self.output_dir, f"{name}.txt"), "w") as fw:
                for row in total_pts_stats:
                    line = " ".join(str(element) for element in row)
                    fw.write(line + "\n")

    def reamp_attribute(self, subtype_cls, group):
        first_key = list(subtype_cls.keys())[0]
        subtype = np.zeros([len(subtype_cls[first_key])], np.uint8)
        lane_direction = np.zeros_like(subtype)
        lane_color = np.zeros_like(subtype)

        # reuse lane direction as roadedge direction
        for idx, cls_name in enumerate(subtype_cls["direction"]):
            lane_direction[idx] = self.OMDirection[f"direction_{cls_name}"]

        if group == "lane":
            lane_attribute = np.ones_like(subtype, dtype=np.uint16)
            for idx, cls_name in enumerate(subtype_cls["property"]):
                subtype[idx] = self.OMSubType[f"{group}_{cls_name}"]
        else:
            lane_attribute = np.zeros_like(subtype)
            for idx, cls_name in enumerate(subtype_cls["subtype"]):
                subtype[idx] = self.OMSubType[f"{group}_{cls_name}"]

        if group == "lane":
            for idx, cls_name in enumerate(subtype_cls["color"]):
                lane_color[idx] = self.OMLaneColor[f"color_{cls_name}"]
            for idx, cls_name in enumerate(subtype_cls["dashed"]):
                if cls_name == "dashed":
                    lane_attribute[idx] += self.OMLaneAttribute[
                        f"attr_{cls_name}"
                    ]
                elif cls_name == "left_solid_right_dash":
                    lane_attribute[idx] += self.OMLaneAttribute[
                        f"attr_{cls_name}"
                    ]
                elif cls_name == "left_dash_right_solid":
                    lane_attribute[idx] += self.OMLaneAttribute[
                        f"attr_{cls_name}"
                    ]
            for idx, cls_name in enumerate(subtype_cls["double"]):
                if cls_name == "double":
                    lane_attribute[idx] += self.OMLaneAttribute[
                        f"attr_{cls_name}"
                    ]
            for idx, cls_name in enumerate(subtype_cls["wide"]):
                if cls_name == "wide":
                    lane_attribute[idx] += self.OMLaneAttribute[
                        f"attr_{cls_name}"
                    ]

        return subtype, lane_attribute, lane_direction, lane_color

    def __repr__(self):
        return "SaveOnlineMappingConsistencyResult"


@OBJECT_REGISTRY.register
class SaveCrossPointConsistencyResult(SaveEvalResult):
    """
    Save consistency results of crosspoint tasks.

    Args:
        output_dir: output directory.
        output_key: output_key of predict results.
        cls_score_threshold: score threshold of different classes.
    """

    def __init__(
        self,
        output_dir: str,
        output_key: str,
        cls_score_threshold: Optional[Dict] = None,
    ):
        super().__init__(output_dir)
        self.output_key = output_key
        self.cls_thresh = cls_score_threshold
        self.init_cls_remap()

    def init_cls_remap(self):
        self.crosspoint_remap = {
            "other": 1 << 0,
            "split": 1 << 1,
            "merge": 1 << 2,
            "start": 1 << 3,
            "stop": 1 << 4,
            "u_turn": 1 << 5,
            "changepoint": 1 << 6,
        }

        self.id2name = {
            0: "merge_start",
            1: "merge_stop",
            2: "split_start",
            3: "split_stop",
            4: "u_turn",
            5: "other",
            6: "changepoint",
        }

    def on_batch_end(self, batch, model_outs, **kwargs):
        model_outs = convert_numpy(model_outs)
        batch_size = batch["img"][0].shape[0]
        for batch_i in range(batch_size):
            result = {}
            extra_points = []
            extra_points_type = []
            name = int(convert_numpy(batch["timestamp"][batch_i]) * 1000)
            task_result = model_outs["bev_crosspoint"]
            cp_result = task_result[self.output_key][batch_i]
            for pt in cp_result:
                # find key from value
                changept_cls_id = [
                    k for k, v in self.id2name.items() if v == "changepoint"
                ][0]
                group = (
                    "crosspoints"
                    if pt[2] != changept_cls_id
                    else "changepoints"
                )
                if pt[3] <= self.cls_thresh[group]:
                    continue
                extra_points.append(
                    {
                        "x": pt[0],
                        "y": pt[1],
                    }
                )
                pt_type = 0
                cls_name = self.id2name[pt[2]]
                for k, v in self.crosspoint_remap.items():
                    if k in cls_name:
                        pt_type += v
                extra_points_type.append(pt_type)

            result["extra_points"] = extra_points
            result["extra_points_type"] = extra_points_type
            with open(
                os.path.join(self.output_dir, f"{name}.json"), "w"
            ) as fw:
                json.dump(result, fw, ensure_ascii=False, indent=4)

    def __repr__(self):
        return "SaveCrossPointConsistencyResult"


def get_data_vcs(
    data_bev,
    vcs_range,
    bev_size,
):
    bottom, right, top, left = np.abs(vcs_range)
    AffinePointsDst = np.array(
        [[-bottom, -right], [top, -right], [top, left], [-bottom, left]],
        dtype=np.float32,
    )

    h, w = bev_size
    AffinePointsSrc = np.array(
        [[w, h], [w, 0], [0, 0], [0, h]],
        dtype=np.float32,
    )
    T_vcs2bevpixel = cv2.getPerspectiveTransform(
        AffinePointsSrc, AffinePointsDst
    )
    a = np.ones(data_bev.shape[0])
    data_bev = np.insert(data_bev, 2, a, axis=1)
    return (data_bev @ T_vcs2bevpixel.T)[:, :2].tolist()


@OBJECT_REGISTRY.register
class SavePsdConsistencyResult(SaveEvalResult):
    """
    Save consistency results of psd task.

        output_dir: output directory.
        output_key: output_key of predict results.
        vcs_range: vcs range of psd task.
        bev_psd_gt_size: gt size of psd task.
    """

    def __init__(
        self,
        output_dir: str,
        output_key: str,
        vcs_range: Union[tuple, List] = (-12.8, -12.8, 25.6, 12.8),
        bev_psd_gt_size: Union[tuple, List] = (192, 128),
    ):
        super().__init__(output_dir)
        self.output_key = output_key
        self.vcs_range = vcs_range
        self.bev_psd_gt_size = bev_psd_gt_size

    def on_batch_end(self, batch, model_outs, **kwargs):
        model_outs = convert_numpy(model_outs)
        batch_size = batch["img"][0].shape[0]
        for batch_i in range(batch_size):
            name = int(convert_numpy(batch["timestamp"][batch_i]) * 1000)
            result = {
                "header": {"time_stamp": np.float64(name)},
                "slots": [],
            }
            for slot_info in model_outs["bev_psd"][self.output_key][
                "decode_slots"
            ][batch_i]:
                slot = {}
                slot_info = np.array(slot_info)
                slot_vcs = get_data_vcs(
                    slot_info[:8].reshape(-1, 2),
                    self.vcs_range,
                    self.bev_psd_gt_size,
                )
                slot["type"] = int(slot_info[17])
                slot["slot_points"] = slot_vcs
                slot["available"] = int(slot_info[16])
                slot["score"] = slot_info[-5]
                result["slots"].append(slot)

            with open(
                os.path.join(self.output_dir, f"{name}.json"), "w"
            ) as fw:
                json.dump(result, fw, ensure_ascii=False, indent=4)

    def __repr__(self):
        return "SavePsdConsistencyResult"


@OBJECT_REGISTRY.register
class SaveParkingRodConsistencyResult(SaveEvalResult):
    """
    Save consistency results of parkingrod task.

        output_dir: output directory.
        output_key: output_key of predict results.
        vcs_range: vcs range of parkingrod task.
        bev_parkingrod_gt_size: gt size of parkingrod task.
    """

    def __init__(
        self,
        output_dir: str,
        output_key: str,
        vcs_range: Union[tuple, List] = (-12.8, -12.8, 25.6, 12.8),
        bev_parkingrod_gt_size: Union[tuple, List] = (192, 128),
    ):
        super().__init__(output_dir)
        self.output_key = output_key
        self.vcs_range = vcs_range
        self.bev_parkingrod_gt_size = bev_parkingrod_gt_size

    def on_batch_end(self, batch, model_outs, **kwargs):
        model_outs = convert_numpy(model_outs)
        batch_size = batch["img"][0].shape[0]
        for batch_i in range(batch_size):
            name = int(convert_numpy(batch["timestamp"][batch_i]) * 1000)
            result = {
                "header": {"time_stamp": np.float64(name)},
                "objects": [],
            }
            for rod_info in model_outs["bev_parkingrod"][self.output_key][
                batch_i
            ]:
                obj = {}
                rod_info = np.array(rod_info)
                rod_vcs = get_data_vcs(
                    rod_info[:4].reshape(-1, 2),
                    self.vcs_range,
                    self.bev_parkingrod_gt_size,
                )
                obj["pnts_vcs"] = rod_vcs
                result["objects"].append(obj)

            with open(
                os.path.join(self.output_dir, f"{name}.json"), "w"
            ) as fw:
                json.dump(result, fw, ensure_ascii=False, indent=4)

    def __repr__(self):
        return "SaveParkingRodConsistencyResult"


@OBJECT_REGISTRY.register
class ANCSaveEachViewFeature(SaveEvalResult):
    """
    Save each view outputs after quantize and padding.

    Args:
        output_dir (str): The save directory of features.
        views (int): view numbers.
        task_key (str): Key of tasks results.
    """

    def __init__(
        self,
        output_dir: str,
        views: Union[int, Sequence],
        task_key: str,
    ):
        super().__init__(output_dir)
        self.output_dir = output_dir
        self.views = [n for n in views if n > 0]
        self.task_key = task_key

    def on_batch_end(self, batch, model_outs, **kwargs):
        data_batch_list = [
            torch.split(cur_data, cur_view, dim=0)
            for cur_data, cur_view in zip(
                model_outs[self.task_key], self.views
            )
        ]

        for batch_idx, cur_data_batch_list in enumerate(zip(*data_batch_list)):
            data_batch_split = []
            for cur_data_batch in cur_data_batch_list:
                data_batch_split = data_batch_split + list(
                    torch.split(cur_data_batch, 1, dim=0)
                )
            for idx, view_data in enumerate(data_batch_split):
                name = int(
                    list(batch["timestamp"].cpu().numpy())[batch_idx] * 1000
                )
                save_path = os.path.join(self.output_dir, f"{idx}_{name}.bin")
                if isinstance(
                    view_data, horizon_plugin_pytorch.qtensor.QTensor
                ):
                    res, scale = convert_numpy(view_data)
                    bev_stage1_qtensor = QTensor(
                        torch.tensor(res),
                        torch.tensor([scale]),
                        dtype=view_data.dtype,
                    )
                    bev_stage1_int = bev_stage1_qtensor.int_repr().numpy()
                    bev_stage1_int = align_padding(
                        bev_stage1_int, "NCHW", 1, np.int8
                    )
                    bev_stage1_int.tofile(save_path)
                else:
                    convert_numpy(view_data).tofile(save_path)

    def __repr__(self):
        return "ANCSaveEachViewFeature"


@OBJECT_REGISTRY.register
class ANCSaveFusionFeature(SaveEvalResult):
    """
    Save BEV stage two fusion outputs after quantize and padding.

    Args:
        output_dir (str): The save directory of features.
        task_key (str): Key of task outputs.
        result_key (str): Key of model results.
    """

    def __init__(
        self,
        output_dir: str,
        task_key: str,
        result_key: str,
    ):
        super().__init__(output_dir)
        self.output_dir = output_dir
        self.task_key = task_key
        self.result_key = result_key

    def on_batch_end(self, batch, model_outs, **kwargs):
        task_results = model_outs[self.task_key][self.result_key]

        for batch_idx, cur_data_batch in enumerate(task_results):
            name = int(
                list(batch["timestamp"].cpu().numpy())[batch_idx] * 1000
            )
            save_path = os.path.join(self.output_dir, f"{name}.bin")
            if isinstance(
                cur_data_batch, horizon_plugin_pytorch.qtensor.QTensor
            ):
                res, scale = convert_numpy(cur_data_batch)
                bev_stage1_qtensor = QTensor(
                    torch.tensor(res),
                    torch.tensor([scale]),
                    dtype=cur_data_batch.dtype,
                )
                bev_stage1_int = bev_stage1_qtensor.int_repr().numpy()
                bev_stage1_int = bev_stage1_int.reshape(
                    1, *bev_stage1_int.shape
                )
                bev_stage1_int = align_padding(
                    bev_stage1_int, "NCHW", 1, np.int8
                )
                bev_stage1_int.tofile(save_path)
            else:
                convert_numpy(cur_data_batch).tofile(save_path)

    def __repr__(self):
        return "ANCSaveFusionFeature"


@OBJECT_REGISTRY.register
class ANCSaveHomoOffset(SaveEvalResult):
    """
    Save HomoOffset after quantize and padding.

    Args:
        output_dir: The save directory of features.
        vcs_plane_nums: The number of vcs planes.
        num_views: The number of views.
        vcs_plane_heights: The heights of multi-height vcs planes.
        block_warp_padding: Order is (left,right,up,bottom).
        grid_quant_scale: Quantize scale.
    """

    def __init__(
        self,
        output_dir: str = "",
        vcs_plane_nums: int = 1,
        num_views: int = 1,
        block_warp_padding: List[tuple] = None,
        grid_quant_scale: float = None,
    ):
        super().__init__(output_dir)
        self.output_dir = output_dir
        self.vcs_plane_nums = vcs_plane_nums
        self.num_views = num_views
        self.block_warp_padding = block_warp_padding
        self.grid_quant_scale = grid_quant_scale
        self.saved_homo_offset = False

    def _save_offset(
        self,
        homo_offset_list,
        block_warp_padding,
        vcs_plane_nums,
        grid_quant_scale,
    ):
        for idx, offset in enumerate(homo_offset_list):

            _, h, w, _ = offset.shape
            pad_l, pad_r, pad_u, pad_b = block_warp_padding[
                idx // vcs_plane_nums
            ]
            block_warp_offset = torch.clone(
                offset[
                    :,
                    pad_u : h - pad_b,
                    pad_l : w - pad_r,
                    :,
                ]
            )
            block_warp_offset[:, :, :, 0] += pad_l
            block_warp_offset[:, :, :, 1] += pad_u
            # offset quanti
            offset_qtensor = QTensor(
                block_warp_offset.to("cpu"),
                torch.tensor([grid_quant_scale]),
                dtype="qint16",
            )
            offset_quanti = (
                offset_qtensor.int_repr().numpy().transpose(0, 3, 1, 2)
            )
            offset_quanti = align_padding(offset_quanti, "NCHW", 2, np.int16)
            save_name = os.path.join(self.output_dir, f"homo_offset_{idx}.bin")
            offset_quanti.tofile(save_name)

    def on_batch_end(self, batch, model_outs, **kwargs):

        if self.saved_homo_offset:
            return

        if not self.saved_homo_offset:
            if "meta_info" in batch:
                assert "homo_offset" in batch["meta_info"]

                homo_offset_list = list(
                    torch.split(
                        batch["meta_info"]["homo_offset"][
                            : self.num_views * self.vcs_plane_nums
                        ],
                        1,
                        dim=0,
                    )
                )

                self._save_offset(
                    homo_offset_list,
                    self.block_warp_padding,
                    self.vcs_plane_nums,
                    self.grid_quant_scale,
                )

        self.saved_homo_offset = True

    def __repr__(self):
        return "SaveHomoOffset"
