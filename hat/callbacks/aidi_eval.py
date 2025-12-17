# Copyright (c) Horizon Robotics. All rights reserved.

import json
import logging
import os
import pickle
import tarfile
import uuid
from abc import ABC
from collections import defaultdict
from enum import Enum
from functools import partial
from typing import Callable, Dict, List, Optional

import cv2
import torch

from hat.registry import OBJECT_REGISTRY
from hat.utils.aidi import (
    AIDIClient,
    AIDIExperimentLogger,
    Artifact,
    ModelEvalDataset,
    get_aidi_client,
)
from hat.utils.apply_func import _as_list
from hat.utils.distributed import get_dist_info, get_global_out, rank_zero_only
from hat.utils.logger import MSGColor, format_msg
from hat.utils.package_helper import require_packages
from .callbacks import CallbackMixin
from .local_eval import (
    LocalEvalDetHandler,
    LocalEvalHandler,
    LocalEvalSegHandler,
)

__all__ = ["AIDIEvalTaskType", "AIDIEval", "AIDIReport"]

logger = logging.getLogger(__name__)


class AIDIEvalTaskType(Enum):
    """AIDIEval Task Type.

    The name of task should reference from:
    "https://horizonrobotics.feishu.cn/wiki/wikcnJhclfzMmFYEPNiIe0zhKsh".

    """

    Detection_2D = 26
    Semantic_Segmentation = 22
    Real3D = 53
    Detection_3D = 36
    Keypoints_Detection_Oks = 32
    DepthEstimation = 50
    Keypoints_Ground_Line = 57
    Vehicle_Flank_Corners = 59
    BevSeg = 55
    BevDet = 61
    IPMPSD = 60
    ADAS_Classification_V2 = 24
    Bev_Rotbox = 63


def _get_data_desc_save_key(data_desc):
    if data_desc.dataset_name is None:
        assert data_desc.dataset_id is not None
        return data_desc.dataset_id
    else:
        return data_desc.dataset_name


class AIDIEvalHandler(ABC):
    """Basic AIDI Eval Handler.

    Args:
        root: Root dir path.
        data_desc: Dataset description.
        project_id: Project id.
        prediction_name: Prediction name.
        prediction_tags: Prediction tags.
        client: Evaluation client.
        task_cfgs: Task special config.
        cpu: evaluation cpu number.
            value in [1, 2, 4, 6, 8], default None.
        cpu_mem_ratio: memory for per cpu.
            value in [2, 4, 6, 8], default None.
        task_num: num of aidieval datasets to submit for single task.
    """

    @require_packages("aidisdk")
    def __init__(
        self,
        root: str,
        data_desc: ModelEvalDataset,
        project_id: str,
        prediction_name: str,
        prediction_tags: str,
        client: AIDIClient,
        task_cfgs: Optional[List] = None,
        cpu: int = None,
        cpu_mem_ratio: int = None,
        task_num: int = 1,
    ):
        self.root = root
        self.data_desc = data_desc

        self.project_id = project_id
        self.prediction_name = prediction_name
        self.prediction_tags = prediction_tags
        self.client = client
        self.task_cfg = task_cfgs
        self.result_file = None
        self.dir_name = None
        self.cpu = cpu
        self.cpu_mem_ratio = cpu_mem_ratio
        self.task_num = task_num

        _, global_world_size = get_dist_info()
        self.global_world_size = global_world_size
        os.makedirs(self.root, exist_ok=True)

    def create_prediction(self):
        """Submit prediction to aidi platform."""
        assert self.result_file is not None and len(self.result_file) == len(
            self.data_desc
        ), (
            "`result_file` can not be None and should be"
            "len(self.result_file) == len(self.data_desc)."
        )
        evaluation_list = []
        for index in range(len(self.data_desc)):
            data_desc = self.data_desc[index]
            assert data_desc.dataset_id, "`dataset_id` can not be None."
            # 创建预测集对象
            model_eval_prediction = (
                self.client.model_eval.create_model_eval_prediction(
                    prediction_name=self.prediction_name,
                    dataset_id=data_desc.dataset_id,
                    prediction_file_path=self.result_file[index],
                    tags=self.prediction_tags,
                )
            )
            # 发起异步评测任务
            model_eval_task = self.client.model_eval.create_model_eval(
                project_id=self.project_id,
                model_eval_dataset_id=data_desc.dataset_id,
                model_eval_prediction_ids=str(
                    model_eval_prediction.prediction_id
                ),
                cpu=self.cpu,
                cpu_mem_ratio=self.cpu_mem_ratio,
            )
            evaluation_list += model_eval_task.evaluation_list
        return evaluation_list


class AIDIEvalDetHandler(AIDIEvalHandler, LocalEvalDetHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._cache = set()

    def write(self, batch, model_outs, reformat_output_fn=None):
        if (
            reformat_output_fn
            and reformat_output_fn.keywords.get("task_name", None)
            and "bev_3d" in reformat_output_fn.keywords["task_name"]
        ):
            self.aidieval_task_name = reformat_output_fn.keywords["task_name"]
            self.task_num = reformat_output_fn.keywords.get("task_num", 1)
            os.makedirs(self.dir_name, exist_ok=True)
            rets = reformat_output_fn(batch, model_outs)
            rets = pickle.dumps(rets)
            global_rank, global_out_rets = get_global_out(rets)
            if global_rank == 0:
                try:
                    if hasattr(self, "aidieval_task_name"):
                        for ret in global_out_rets:
                            with open(self.pkl_file, "ab") as f:
                                f.write(ret)
                except PermissionError as e:
                    raise PermissionError(
                        f"Failed: {str(e)}. Make sure you have the write "
                        f"permission of the file {self.pkl_file}"
                    )
        else:
            super().write(batch, model_outs, reformat_output_fn)

    def merge_file(self):
        if hasattr(self, "aidieval_task_name"):
            pred_front, pred_side = self._load_bev3d_pkl(self.pkl_file)
            front_pred = []
            front_dump_path = os.path.join(self.root, "pred_front.json")
            # Process front
            front_reformat_dict = defaultdict(list)
            for front_ori in pred_front:  # rm duplicates
                image_key = front_ori["image_key"]
                front_reformat_dict[image_key].append(front_ori)
            for _, front_frame in front_reformat_dict.items():
                front = front_frame[0]
                front_pred.append(json.dumps(front) + "\n")
            all_sample_num = len(front_pred)
            if all_sample_num > 0:
                with open(front_dump_path, "w") as w:
                    for one in front_pred:
                        w.write(one)
            # Process side
            side_pred = []
            side_dump_path = os.path.join(self.root, "pred_side.json")
            side_reformat_dict = defaultdict(list)
            for side_ori in pred_side:  # rm duplicates
                image_key = side_ori["image_key"]
                side_reformat_dict[image_key].append(side_ori)
            for _, side_frame in side_reformat_dict.items():
                side = side_frame[0]
                side_pred.append(json.dumps(side) + "\n")
            all_sample_num = len(side_pred)
            if all_sample_num > 0:
                with open(side_dump_path, "w") as w:
                    for one in side_pred:
                        w.write(one)
            self.result_file = []
            for _ in range(self.task_num):
                self.result_file.extend([front_dump_path, side_dump_path])
        else:
            super().merge_file()

    def _load_bev3d_pkl(self, pkl_path):
        pred_front = []
        pred_side = []
        f = open(pkl_path, "rb")
        while True:
            try:
                preds = pickle.load(f)
                pred_front.extend(preds["front_outputs"])
                pred_side.extend(preds["side_outputs"])
            except EOFError:
                break
        f.close()
        return pred_front, pred_side


class AIDIEvalSegHandler(AIDIEvalHandler, LocalEvalSegHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._cache = set()
        self.dir_name = f"{self.root}/tmp_result_{str(uuid.uuid4())}"
        self.result_file = f"{self.root}/result.tar"

    def write(self, batch, model_outs, reformat_output_fn=None):
        ret_names = super().write(batch, model_outs, reformat_output_fn)
        global_rank, global_ret_names = get_global_out(ret_names)
        if global_rank == 0:
            tar = tarfile.open(self.result_file, "a")
            for ret_names in global_ret_names:
                assert os.path.exists(self.dir_name)
                for file_name in ret_names:
                    full_path = os.path.join(self.dir_name, file_name)
                    tar.add(full_path, file_name)
            tar.close()

    def merge_file(self):
        self.result_file = _as_list(self.result_file)


class AIDIEvalDepthHandler(AIDIEvalHandler, LocalEvalHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._cache = set()

    def write(self, batch, model_outs, reformat_output_fn=None):
        # TODO (hongyu.xie): support async
        if reformat_output_fn:
            rets = reformat_output_fn(batch, model_outs)
            assert isinstance(
                rets, List
            ), f"`rets` should be List[Dict], but get {type(rets)}"
        else:
            # TODO(mengyang.duan): Refactor this after using Structure
            raise NotImplementedError(
                "Not support now, please provide `reformat_output_fn`"
                "to handle outputs by yourself."
            )

        for ret in rets:
            assert isinstance(
                ret, Dict
            ), f"`ret` should be Dict, but get {type(ret)}"
            image_key = ret.get("image_name", None)
            assert image_key, "`image_name` can not be None"
            if image_key in self._cache:
                continue
            self._cache.add(image_key)
            cv2.imwrite(
                os.path.join(self.dir_name_depth, ret["image_name"]),
                ret["depth"],
            )
            cv2.imwrite(
                os.path.join(self.dir_name_conf, ret["image_name"]),
                ret["conf"],
            )

        if self.global_world_size > 1:
            torch.distributed.barrier()

    def check_file(self):
        """Rename file or establish file."""
        result_root = f"{self.root}/result_tmp_{str(uuid.uuid4())}"
        self.result_file = f"{result_root}/result_depth.tar"  # noqa E501
        self.dir_name_depth = f"{result_root}/pred_depth"  # noqa E501
        self.dir_name_conf = f"{result_root}/pred_conf"  # noqa E501
        os.makedirs(result_root, exist_ok=True)
        os.makedirs(self.dir_name_depth, exist_ok=True)
        os.makedirs(self.dir_name_conf, exist_ok=True)

    def merge_file(self):
        _, global_dir_depth_name = get_global_out(self.dir_name_depth)
        _, global_dir_conf_name = get_global_out(self.dir_name_conf)
        tar = tarfile.open(self.result_file, "a")
        for rank_dir in global_dir_depth_name:
            assert os.path.exists(rank_dir)
            for file_name in os.listdir(rank_dir):
                full_path = os.path.join(rank_dir, file_name)
                tar.add(
                    full_path,
                    f"{self.prediction_name}/pred_depth/" + file_name,
                )
        for rank_dir in global_dir_conf_name:
            assert os.path.exists(rank_dir)
            for file_name in os.listdir(rank_dir):
                full_path = os.path.join(rank_dir, file_name)
                tar.add(
                    full_path, f"{self.prediction_name}/pred_conf/" + file_name
                )
        tar.close()
        self.result_file = _as_list(self.result_file)


class AIDIEvalReal3DHandler(AIDIEvalHandler, LocalEvalHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._cache = set()

        self.result_file = f"{self.root}/result.tar"
        self.dir_name = f"{self.root}/tmp_result"
        self.pkl_file = f"{self.dir_name}/result.pkl"

    def write(self, batch, model_outs, reformat_output_fn=None):
        # TODO (hongyu.xie): support async
        if reformat_output_fn:
            ret = reformat_output_fn(batch, model_outs)
        else:
            # TODO(mengyang.duan): Refactor this after using Structure
            raise NotImplementedError(
                "Not support now, please provide `reformat_output_fn`"
                "to handle outputs by yourself."
            )
        ret = pickle.dumps(ret)
        global_rank, global_out_rets = get_global_out(ret)
        if global_rank == 0:
            try:
                for ret in global_out_rets:
                    with open(self.pkl_file, "ab") as f:
                        f.write(ret)
            except PermissionError as e:
                raise PermissionError(
                    f"Failed: {str(e)}. Make sure you have the write "
                    f"permission of the file {self.pkl_file}"
                )


class AIDIEvalBevSegHandler(AIDIEvalHandler, LocalEvalHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._cache = set()

        self.result_file = f"{self.root}/result.tar"
        self.dir_name = f"{self.root}/tmp_result"
        self.pkl_file = f"{self.dir_name}/result.pkl"

    def write(self, batch, model_outs, reformat_output_fn=None):
        # TODO (hongyu.xie): support async
        if reformat_output_fn:
            ret = reformat_output_fn(batch, model_outs)
        else:
            # TODO(mengyang.duan): Refactor this after using Structure
            raise NotImplementedError(
                "Not support now, please provide `reformat_output_fn`"
                "to handle outputs by yourself."
            )
        # save img
        img_name = ret["outputs"]["img_name"]
        for idx in range(len(img_name)):
            cv2.imwrite(
                os.path.join(self.dir_name, "bev_seg_" + img_name[idx]),
                ret["predict_results"][idx].astype("uint8"),
            )

        ret.pop("predict_results")
        ret = pickle.dumps(ret)
        global_rank, global_out_rets = get_global_out(ret)
        if global_rank == 0:
            try:
                for ret in global_out_rets:
                    with open(self.pkl_file, "ab") as f:
                        f.write(ret)
            except PermissionError as e:
                raise PermissionError(
                    f"Failed: {str(e)}. Make sure you have the write "
                    f"permission of the file {self.pkl_file}"
                )


class AIDIEvalBev3DHandler(AIDIEvalHandler, LocalEvalHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._cache = set()

    def write(self, batch, model_outs, reformat_output_fn=None):
        # TODO (hongyu.xie): support async
        if reformat_output_fn:
            ret = reformat_output_fn(batch, model_outs)
        else:
            # TODO(mengyang.duan): Refactor this after using Structure
            raise NotImplementedError(
                "Not support now, please provide `reformat_output_fn`"
                "to handle outputs by yourself."
            )
        ret = pickle.dumps(ret)
        global_rank, global_out_rets = get_global_out(ret)
        if global_rank == 0:
            try:
                for ret in global_out_rets:
                    with open(self.pkl_file, "ab") as f:
                        f.write(ret)
            except PermissionError as e:
                raise PermissionError(
                    f"Failed: {str(e)}. Make sure you have the write "
                    f"permission of the file {self.pkl_file}"
                )

    def _load_bev3d_pkl(self, pkl_path):
        pred_front = []
        pred_side = []
        f = open(pkl_path, "rb")
        while True:
            try:
                preds = pickle.load(f)
                pred_front.extend(preds["front_outputs"])
                pred_side.extend(preds["side_outputs"])
            except EOFError:
                break
        f.close()
        return pred_front, pred_side

    def merge_file(self):
        pred_front, pred_side = self._load_bev3d_pkl(self.pkl_file)

        front_pred = []
        front_dump_path = os.path.join(self.root, "pred_front.json")
        # Process front
        for front in pred_front:
            front_pred.append(json.dumps(front) + "\n")
        all_sample_num = len(front_pred)
        if all_sample_num > 0:
            with open(front_dump_path, "w") as w:
                for one in front_pred:
                    w.write(one)
        # Process side
        side_pred = []
        side_dump_path = os.path.join(self.root, "pred_side.json")
        for side in pred_side:
            side_pred.append(json.dumps(side) + "\n")
        all_sample_num = len(side_pred)
        if all_sample_num > 0:
            with open(side_dump_path, "w") as w:
                for one in side_pred:
                    w.write(one)

        self.result_file = [front_dump_path, side_dump_path]


class AIDIEvalBevDetHandler(AIDIEvalHandler, LocalEvalHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._cache = set()
        self.aidieval_task_type = None

    def write(self, batch, model_outs, reformat_output_fn=None):
        # TODO (hongyu.xie): support async
        if reformat_output_fn:
            self.aidieval_task_type = reformat_output_fn.keywords.get(
                "aidieval_task_type", None
            )
            self.task_num = reformat_output_fn.keywords.get("task_num", 1)
            if self.aidieval_task_type is not None:
                rets = reformat_output_fn(
                    batch,
                    model_outs,
                    aidieval_task_type=self.aidieval_task_type,
                )
            else:
                rets = reformat_output_fn(batch, model_outs)
            assert isinstance(
                rets, List
            ), f"`rets` should be List[Dict], but get {type(rets)}"
        else:
            # TODO(mengyang.duan): Refactor this after using Structure
            raise NotImplementedError(
                "Not support now, please provide `reformat_output_fn`"
                "to handle outputs by yourself."
            )
        global_rank, global_out_rets = get_global_out(rets)
        if global_rank == 0:
            try:
                with open(self.result_file, "a") as fwrite:
                    for rets in global_out_rets:
                        for ret in rets:
                            assert isinstance(
                                ret, Dict
                            ), f"`ret` should be Dict, but get {type(ret)}"
                            sample_key = ret.get("timestamp", None)
                            assert sample_key, "`timestamp` can not be None"
                            if sample_key in self._cache:
                                continue
                            self._cache.add(sample_key)
                            fwrite.write(json.dumps(ret) + "\n")
            except PermissionError as e:
                raise PermissionError(
                    f"Failed: {str(e)}. Make sure you have the write "
                    f"permission of the file {self.json_file}"
                )

    def check_file(self):
        self.result_file = (
            f"{self.root}/result_{str(uuid.uuid4())}.json"  # noqa E501
        )

    def merge_file(self):
        self.result_file = _as_list(self.result_file) * self.task_num


class AIDIEvalIPMPSDHandler(AIDIEvalHandler, LocalEvalHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._cache = set()

    def write(self, batch, model_outs, reformat_output_fn=None):
        if reformat_output_fn:
            rets = reformat_output_fn(batch, model_outs)
            assert isinstance(
                rets, List
            ), f"`rets` should be List[Dict], but get {type(rets)}"
        else:
            raise NotImplementedError(
                "Not support now, please provide `reformat_output_fn`"
                "to handle outputs by yourself."
            )
        global_rank, global_out_rets = get_global_out(rets)
        if global_rank == 0:
            try:
                with open(self.result_file, "a") as fwrite:
                    for rets in global_out_rets:
                        for ret in rets:
                            assert isinstance(
                                ret, Dict
                            ), f"`ret` should be Dict, but get {type(ret)}"
                            image_key = ret.get("image_key", None)
                            assert image_key, "`image_key` can not be None"
                            if image_key in self._cache:
                                continue
                            self._cache.add(image_key)
                            fwrite.write(json.dumps(ret) + "\n")
            except PermissionError as e:
                raise PermissionError(
                    f"Failed: {str(e)}. Make sure you have the write "
                    f"permission of the file {self.json_file}"
                )

    def check_file(self):
        self.result_file = f"{self.root}/result_{str(uuid.uuid4())}.json"

    def merge_file(self):
        self.result_file = _as_list(self.result_file)


class AIDIEvalClsHandler(AIDIEvalHandler, LocalEvalHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._cache = set()

    def write(self, batch, model_outs, reformat_output_fn=None):
        # TODO (hongyu.xie): support async
        if reformat_output_fn:
            rets = reformat_output_fn(batch, model_outs)
            assert isinstance(rets, List) or isinstance(
                rets, Dict
            ), f"`rets` should be List[Dict], but get {type(rets)}"
        else:
            # TODO(mengyang.duan): Refactor this after using Structure
            raise NotImplementedError(
                "Not support now, please provide `reformat_output_fn`"
                "to handle outputs by yourself."
            )
        global_rank, global_out_rets = get_global_out(rets)
        if global_rank == 0:
            try:
                for obj_key, result_file in self.result_file.items():
                    with open(result_file, "a") as fwrite:
                        for rets in global_out_rets:
                            for rett in rets:
                                for ret in rett[obj_key]:
                                    assert isinstance(
                                        ret, Dict
                                    ), f"`ret` should be Dict, but get {type(ret)}"  # noqa
                                    image_key = ret.get("image_key", None)
                                    assert (
                                        image_key
                                    ), "`image_key` can not be None"
                                    fwrite.write(json.dumps(ret) + "\n")
            except PermissionError as e:
                raise PermissionError(
                    f"Failed: {str(e)}. Make sure you have the write "
                    f"permission of the file {self.json_file}"
                )

    def check_file(self):
        self.result_file = {}
        if "category" in self.prediction_tags:
            self.prediction_tags.update(
                type=self.prediction_tags.pop("category")
            )
        for obj_key in self.prediction_tags.keys():
            self.result_file[
                obj_key
            ] = f"{self.root}/result_{str(uuid.uuid4())}_{self.prediction_name}_{obj_key}.json"  # noqa

    def merge_file(self):
        self.result_file = _as_list(self.result_file)

    def create_prediction(self):
        """Submit prediction to aidi platform."""
        assert self.result_file is not None and len(self.result_file) == len(
            self.data_desc
        ), (
            "`result_file` can not be None and should be"
            "len(self.result_file) == len(self.data_desc)."
        )
        evaluation_list = []
        for index in range(len(self.data_desc)):
            data_desc = self.data_desc[index]
            assert (
                data_desc.dataset_id is not None
            ), "`dataset_id` can not be None."
            for obj_key, res_file in self.result_file[0].items():
                # 创建预测集对象
                model_eval_prediction = (
                    self.client.model_eval.create_model_eval_prediction(
                        prediction_name=self.prediction_name,
                        dataset_id=data_desc.dataset_id,
                        prediction_file_path=res_file,
                        tags=["test", obj_key],
                    )
                )
                # 发起异步评测任务
                model_eval_task = self.client.model_eval.create_model_eval(
                    project_id=self.project_id,
                    model_eval_dataset_id=data_desc.dataset_id,
                    model_eval_prediction_ids=str(
                        model_eval_prediction.prediction_id
                    ),
                    cpu=self.cpu,
                    cpu_mem_ratio=self.cpu_mem_ratio,
                )
                evaluation_list += model_eval_task.evaluation_list
        return evaluation_list


class AIDIEvalBevRotboxtHandler(AIDIEvalHandler, LocalEvalHandler):
    """AIDI Rotate box Detection Evaluation Callback."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._cache = set()
        self.result_file = f"{self.root}/result.json"
        self.aidieval_task_type = None

    def write(self, batch, model_outs, reformat_output_fn=None):
        # TODO (hongyu.xie): support async
        if reformat_output_fn:
            self.aidieval_task_type = reformat_output_fn.keywords.get(
                "aidieval_task_type", None
            )
            self.task_num = reformat_output_fn.keywords.get("task_num", 1)
            if self.aidieval_task_type is not None:
                rets = reformat_output_fn(
                    batch,
                    model_outs,
                    aidieval_task_type=self.aidieval_task_type,
                )
            else:
                rets = reformat_output_fn(batch, model_outs)
            assert isinstance(
                rets, List
            ), f"`rets` should be List[Dict], but get {type(rets)}"
        else:
            # TODO(mengyang.duan): Refactor this after using Structure
            raise NotImplementedError(
                "Not support now, please provide `reformat_output_fn`"
                "to handle outputs by yourself."
            )
        global_rank, global_out_rets = get_global_out(rets)
        if global_rank == 0:
            try:
                with open(self.result_file, "a") as fwrite:
                    for rets in global_out_rets:
                        for ret in rets:
                            assert isinstance(
                                ret, Dict
                            ), f"`ret` should be Dict, but get {type(ret)}"
                            timestamp = ret.get("timestamp", None)
                            assert timestamp, "`timestamp` can not be None"
                            if timestamp in self._cache:
                                continue
                            self._cache.add(timestamp)
                            fwrite.write(json.dumps(ret) + "\n")
            except PermissionError as e:
                raise PermissionError(
                    f"Failed: {str(e)}. Make sure you have the write "
                    f"permission of the file {self.json_file}"
                )

    def check_file(self):
        self.result_file = (
            f"{self.root}/result_{str(uuid.uuid4())}.json"  # noqa E501
        )

    def merge_file(self):
        self.result_file = _as_list(self.result_file) * self.task_num


TASK_TO_HANDLER = {
    AIDIEvalTaskType.Detection_2D: AIDIEvalDetHandler,
    AIDIEvalTaskType.Semantic_Segmentation: AIDIEvalSegHandler,
    AIDIEvalTaskType.Real3D: AIDIEvalReal3DHandler,
    AIDIEvalTaskType.Detection_3D: AIDIEvalDetHandler,
    AIDIEvalTaskType.Bev_Rotbox: AIDIEvalBevRotboxtHandler,
    AIDIEvalTaskType.Keypoints_Detection_Oks: AIDIEvalDetHandler,
    AIDIEvalTaskType.Keypoints_Ground_Line: AIDIEvalDetHandler,
    AIDIEvalTaskType.Vehicle_Flank_Corners: AIDIEvalDetHandler,
    AIDIEvalTaskType.BevSeg: AIDIEvalBevSegHandler,
    AIDIEvalTaskType.BevDet: AIDIEvalBevDetHandler,
    AIDIEvalTaskType.DepthEstimation: AIDIEvalDepthHandler,
    AIDIEvalTaskType.IPMPSD: AIDIEvalIPMPSDHandler,
    AIDIEvalTaskType.ADAS_Classification_V2: AIDIEvalClsHandler,
}

TASK_TO_HANDLER.update({k.value: v for k, v in TASK_TO_HANDLER.items()})


@OBJECT_REGISTRY.register
class AIDIEval(CallbackMixin):
    """AIDI Evaluation Callback.

    Args:
        output_root: Evaluation result output dir.
        prediction_name: Prediction name.
        project_id: Project id.
        prediction_tags: Prediction tags.
        aidi_eval_dataset_name: Dataset name.
        aidi_eval_dataset_id: Dataset id, `aidi_eval_dataset_id` and
        `aidi_eval_dataset_name` should have and only have one is not None.
        aidi_eval_token: AIDI Eval user token.
        reformat_output_fn: Callabel function to reformat model outpus.
        reformat_out_fn_kwargs: Custom params used in `reformat_output_fn`.
        reformat_input_fn: Callabel function to reformat input batch data.
        overwrite: Whether to overwrite existing prediction file.
        task_cfgs: Task special config.
        cpu: evaluation cpu number.
            value in [1, 2, 4, 6, 8], default None.
        cpu_mem_ratio: memory for per cpu.
            value in [2, 4, 6, 8], default None.
    """

    @require_packages("aidisdk")
    def __init__(
        self,
        output_root: str,
        prediction_name: str,
        project_id: str,
        prediction_tags: Optional[List[str]] = None,
        aidi_eval_dataset_name: Optional[List[str]] = None,
        aidi_eval_dataset_id: Optional[List[int]] = None,
        aidi_eval_token: Optional[str] = None,
        reformat_output_fn: Optional[Callable] = None,
        reformat_out_fn_kwargs: Optional[Dict] = None,
        reformat_input_fn: Optional[Callable] = None,
        with_epoch_id: bool = False,
        overwrite: Optional[bool] = True,
        task_cfgs: Optional[List] = None,
        cpu: Optional[int] = None,
        cpu_mem_ratio: Optional[int] = None,
    ):
        super().__init__()
        self.output_root = output_root
        # aidi_eval
        self.project_id = project_id
        self.prediction_name = prediction_name
        self.prediction_tags = prediction_tags
        self.task_cfgs = task_cfgs
        self.cpu = cpu
        self.cpu_mem_ratio = cpu_mem_ratio

        assert (
            aidi_eval_dataset_id or aidi_eval_dataset_name
        ), "`aidi_eval_dataset_id` and `aidi_eval_dataset_name` can not both be None."  # noqa E501
        assert not (aidi_eval_dataset_id and aidi_eval_dataset_name), (
            f"`aidi_eval_dataset_id` and `aidi_eval_dataset_name` should have"
            f"and only have one is not None, but get `aidi_eval_dataset_id`: "
            f"{aidi_eval_dataset_id}, `aidi_eval_dataset_name`: {aidi_eval_dataset_name}."  # noqa E501
        )

        self.aidi_eval_dataset_id = (
            _as_list(aidi_eval_dataset_id) if aidi_eval_dataset_id else []
        )
        self.aidi_eval_dataset_name = (
            _as_list(aidi_eval_dataset_name) if aidi_eval_dataset_name else []
        )

        self.reformat_outputs_fn = reformat_output_fn
        self.aidi_eval_token = aidi_eval_token
        self._handler = {}
        self._missing_datasets = set()
        self._prediction_ids = []
        self.overwrite = overwrite
        self.reformat_input_fn = reformat_input_fn
        self.reformat_out_fn_kwargs = reformat_out_fn_kwargs
        self.with_epoch_id = with_epoch_id

        self._init_data_desc()

    def _init_data_desc(self):
        self._eval_client = get_aidi_client()
        self.data_desc = []
        task_id = None

        if self.aidi_eval_dataset_name and not self.aidi_eval_dataset_id:
            for dataset_name in self.aidi_eval_dataset_name:
                model_eval_datasets = (
                    self._eval_client.model_eval.query_model_eval_dataset(
                        dataset_name=str(dataset_name),
                    )
                )
                model_eval_datasets = model_eval_datasets[0]
                self.aidi_eval_dataset_id.append(
                    model_eval_datasets.dataset_id
                )
                if task_id is None:
                    task_id = model_eval_datasets.task_id
                else:
                    assert task_id == model_eval_datasets.task_id, (
                        f"dataset_name in {self.aidi_eval_dataset_name} "
                        f"should correspond to the same task."
                    )
                self.data_desc.append(model_eval_datasets)
        else:
            for dataset_id in self.aidi_eval_dataset_id:
                model_eval_datasets = (
                    self._eval_client.model_eval.query_model_eval_dataset(
                        dataset_id=int(dataset_id)
                    )
                )
                model_eval_datasets = model_eval_datasets[0]
                self.aidi_eval_dataset_name.append(
                    model_eval_datasets.dataset_name
                )
                if task_id is None:
                    task_id = model_eval_datasets.task_id
                else:
                    assert task_id == model_eval_datasets.task_id, (
                        f"dataset_id in {self.aidi_eval_dataset_id} "
                        f"should correspond to the same task."
                    )
                self.data_desc.append(model_eval_datasets)

    def _get_handler(self, data_desc: dict):
        key_list = [
            _get_data_desc_save_key(data_desc) for data_desc in self.data_desc
        ]
        key = "+".join([str(key).strip() for key in key_list])
        if key not in self._handler:
            obj = TASK_TO_HANDLER.get(data_desc[0].task_id, None)
            if obj is None:
                raise ValueError(
                    f"Does not support aidi eval task type {data_desc[0].task_id}"  # noqa
                )

            self.output_root = "{}/{}/aidi_eval".format(self.output_root, key)
            handler = obj(
                root=self.output_root,
                data_desc=data_desc,
                project_id=self.project_id,
                prediction_name=self.prediction_name,
                prediction_tags=self.prediction_tags,
                client=self._eval_client,
                task_cfgs=self.task_cfgs,
                cpu=self.cpu,
                cpu_mem_ratio=self.cpu_mem_ratio,
            )
            self._handler[key] = handler
        return self._handler[key]

    def on_loop_begin(self, **kwargs):
        handler = self._get_handler(self.data_desc)
        handler.check_file()

    def on_batch_begin(self, batch, **kwargs):
        if self.reformat_input_fn:
            self.reformat_input_fn(batch)

    def on_batch_end(self, batch, model_outs, **kwargs):
        if model_outs is None:
            self._missing_datasets.add(
                [data_desc.dataset_id for data_desc in self.data_desc]
            )
            return

        handler = self._get_handler(self.data_desc)
        handler.write(
            batch,
            model_outs,
            partial(self.reformat_outputs_fn, **self.reformat_out_fn_kwargs),
        )

    @rank_zero_only
    def on_loop_end(self, **kwargs):
        for data_desc in self.data_desc:
            if data_desc.dataset_id in self._missing_datasets:
                return
        handler = self._get_handler(self.data_desc)
        handler.merge_file()
        evaluation_ids = handler.create_prediction()
        prediction_ids = set()
        for element in evaluation_ids:
            evaluation_id = element["evaluation_id"]
            prediction_id = element["prediction_id"]
            setting_id = element["setting_id"]
            dataset = [d.dataset_id for d in self.data_desc]
            logger.info(
                format_msg(
                    f"Create evaluation_id: {evaluation_id}, "
                    f"prediction_id: {prediction_id}, "
                    f"setting_id: {setting_id} "
                    f"for dataset {dataset}",
                    MSGColor.RED,
                )
            )
            prediction_ids.add(prediction_id)
        # aidi tracking prediction
        if self.enable_tracking:
            for prediction_id in list(prediction_ids):
                self._tracking_prediction(prediction_id)

    def _tracking_prediction(self, prediction_id):
        """Tracking prediction."""
        try:
            model_eval_predictions = (
                self._eval_client.model_eval.query_model_eval_prediction(
                    prediction_id=int(prediction_id)
                )
            )

            self._eval_client.experiment.log_artifact(
                Artifact.from_model_eval_prediction(model_eval_predictions[0])
            )
            logger.info(
                format_msg(
                    f"AIDI Tracking: log_artifact prediction_id: {prediction_id}",  # noqa E501
                    MSGColor.GREEN,
                )
            )
        except Exception as e:
            logger.info(
                format_msg(
                    f"AIDI Tracking: log_artifact prediction_id: {prediction_id} failed: {e}",  # noqa E501
                    MSGColor.RED,
                )
            )

    @property
    def enable_tracking(self):
        return AIDIExperimentLogger.enabled_tracking()


@OBJECT_REGISTRY.register
class AIDIReport(CallbackMixin):
    """AIDI Report Callback.

    Args:
        report_name: The report name, cannot be repeated.
        page_label_list: The name used to show in the Report tab
        column and cannot be repeated.
        type_list: Task type of dataset.
        old_prediction_list: The diff model name is used by description.
        new_prediction_list: The eval model name.
        datasets_id_list: The aidieval dataset ids.
        report_type_name: Methods of integrating evaluation results.
        Currently supported "Multi-Task", "Report Group".
    """

    @require_packages("aidisdk")
    def __init__(
        self,
        report_name: str,
        project_id: str,
        page_label_list: Optional[List[str]] = None,
        type_list: Optional[List[str]] = None,
        old_prediction_list: Optional[List[str]] = None,
        new_prediction_list: Optional[List[str]] = None,
        datasets_id_list: Optional[List[int]] = None,
        description: Optional[str] = "",
        report_type_name: Optional[str] = "Multi-Task",
    ):
        super().__init__()
        self.report_name = report_name
        self.project_id = project_id
        self.page_label_list = page_label_list
        self.type_list = type_list
        self.old_prediction_list = old_prediction_list
        self.new_prediction_list = new_prediction_list
        self.datasets_id_list = datasets_id_list
        self.description = description
        self.report_type_name = report_type_name
        self._report_client = None
        self._report_type = None

    @property
    def client(self):
        self._report_client = get_aidi_client()
        return self._report_client

    @rank_zero_only
    def on_epoch_end(self, **kwargs):
        report_config = {"modules": []}
        for idx, page_label in enumerate(self.page_label_list):
            sub_report_config = {
                "name": page_label.strip(),
                "type": self.type_list[idx].strip(),
                "datasets": _as_list(self.datasets_id_list[idx]),
                "eval_model_name": self.new_prediction_list[idx].strip(),
                "diff_model_name": self.old_prediction_list[idx],
            }
            report_config["modules"].append(sub_report_config)
        response = self.client.model_eval.create_eval_report(
            project_id=self.project_id,
            report_type_name=self.report_type_name,
            report_name=self.report_name.strip(),
            config=report_config,
            description=self.description,
        )
        logger.info(response)
