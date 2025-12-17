# Copyright (c) Horizon Robotics. All rights reserved.

import json
import logging
import os
import shutil
import tarfile
import uuid
from abc import ABC
from functools import partial
from typing import Callable, Dict, List, Optional

import cv2

from hat.registry import OBJECT_REGISTRY
from hat.utils.apply_func import _as_list
from hat.utils.distributed import get_dist_info, get_global_out, rank_zero_only
from .callbacks import CallbackMixin

__all__ = ["LocalEval"]

logger = logging.getLogger(__name__)


class LocalEvalHandler(ABC):
    """Basic Local Eval Handler.

    因为时间线上先实现了aidievalhandler, 所以部分handler没有localevalhandler,
    但建议函数放在LocalEvalHandler, aidi相关属性和创建测评任务函数放在
    AIDIEvalHandler.

    Args:
        root: Root dir path.
        kwargs: Custom params used in subclass.
    """

    def __init__(self, root: str, **kwargs):
        self.root = root
        self.result_file = None
        self.dir_name = None
        _, global_world_size = get_dist_info()
        self.global_world_size = global_world_size
        os.makedirs(self.root, exist_ok=True)

    def write(self, batch, model_outs):
        """Save predict result to file.

        Args:
            batch: The model's input.
            model_outs: The model's output.
        """
        raise NotImplementedError

    def create_prediction(self):
        raise NotImplementedError

    def check_file(self):
        """Rename file or establish file."""
        self.result_file = (
            f"{self.root}/result_{str(uuid.uuid4())}.tar"  # noqa E501
        )
        self.dir_name = (
            f"{self.root}/tmp_result_{str(uuid.uuid4())}"  # noqa E501
        )
        self.pkl_file = f"{self.dir_name}/result.pkl"
        os.makedirs(self.dir_name, exist_ok=True)

    def merge_file(self):
        """Merge result to json or tar img result and so on."""
        assert os.path.exists(self.dir_name)
        tar = tarfile.open(self.result_file, "w")
        for file_name in os.listdir(self.dir_name):
            full_path = os.path.join(self.dir_name, file_name)
            tar.add(
                full_path, arcname=os.path.basename(full_path), recursive=False
            )
        tar.close()
        shutil.rmtree(self.dir_name, ignore_errors=True)
        self.result_file = _as_list(self.result_file)


class LocalEvalDetHandler(LocalEvalHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._cache = set()
        self.gt_file = kwargs.get("gt_file")
        self.setting_file = _as_list(kwargs.get("setting_file"))

    def create_prediction(self):
        from hat.evaluation.detection2d.single import evaluate

        for setting_file_i in self.setting_file:
            setting_file_i_name = os.path.basename(setting_file_i)
            setting_file_i_name = os.path.splitext(setting_file_i_name)[0]
            save_dir = os.path.join(self.root, setting_file_i_name)
            evaluate(
                self.gt_file, self.result_file[0], setting_file_i, save_dir
            )

    def write(self, batch, model_outs, reformat_output_fn=None):
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
                    f"permission of the file {self.result_file}"
                )

    def check_file(self):
        self.result_file = (
            f"{self.root}/result_{str(uuid.uuid4())}.json"  # noqa E501
        )
        self.dir_name = (
            f"{self.root}/tmp_result_{str(uuid.uuid4())}"  # noqa E501
        )
        self.pkl_file = f"{self.dir_name}/result.pkl"

    def merge_file(self):
        self.result_file = _as_list(self.result_file)


class LocalEvalSegHandler(LocalEvalHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._cache = set()
        self.dir_name = f"{self.root}/tmp_result_{str(uuid.uuid4())}"
        self.images_dir = kwargs.get("images_dir", None)
        self.gt_dir = kwargs.get("gt_dir", None)
        self.pred_dir = kwargs.get("pred_dir", None)
        self.config_file = _as_list(kwargs.get("config_file", None))
        self.images_json = kwargs.get("images_json", None)
        self.attr_path = kwargs.get("attr_path", None)

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
        ret_names = []
        if global_rank == 0:
            for rets in global_out_rets:
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
                        os.path.join(self.dir_name, ret["image_name"]),
                        ret["out_img"],
                    )
                    ret_names.append(ret["image_name"])
        return ret_names

    def check_file(self):
        """Rename file or establish file."""
        rank, _ = get_dist_info()
        if rank == 0:
            os.makedirs(self.dir_name, exist_ok=True)

    def merge_file(self):
        pass

    def create_prediction(self):
        from hat.evaluation.semantic_segmentation import evaluate

        for setting_file_i in self.config_file:
            setting_file_i_name = os.path.basename(setting_file_i)
            setting_file_i_name = os.path.splitext(setting_file_i_name)[0]
            save_dir = os.path.join(self.root, setting_file_i_name)
            evaluate(
                images_dir=self.images_dir,
                gt_dir=self.gt_dir,
                pred_dir=self.dir_name,
                config_file=setting_file_i,
                outfile_dir=save_dir,
                images_json=self.images_json,
                attr_path=self.attr_path,
            )


Task2Handler = {
    "Detection2D": LocalEvalDetHandler,
    "Segmentaion": LocalEvalSegHandler,
}


@OBJECT_REGISTRY.register
class LocalEval(CallbackMixin):
    """local eval pipeline, similar to AIDIEval.

    Args:
        output_root: Evaluation result output dir.
        reformat_output_fn: Callabel function to reformat model outpus.
        reformat_out_fn_kwargs: Custom params used in `reformat_output_fn`.
        reformat_input_fn: Callabel function to reformat input batch data.
        task_name: Name of task, used to confirm handler type.
        eval_handler_kwargs: Parameters for init of handler.
    """

    def __init__(
        self,
        output_root: str,
        reformat_output_fn: Optional[Callable] = None,
        reformat_out_fn_kwargs: Optional[Dict] = None,
        reformat_input_fn: Optional[Callable] = None,
        task_name=None,
        eval_handler_kwargs=None,
    ):
        super().__init__()
        self.output_root = output_root
        self.reformat_outputs_fn = reformat_output_fn
        self._missing_datasets = set()
        self.reformat_input_fn = reformat_input_fn
        self.reformat_out_fn_kwargs = reformat_out_fn_kwargs
        self.handler = Task2Handler[task_name](
            root=output_root,
            **eval_handler_kwargs,
        )

    def on_loop_begin(self, **kwargs):
        self.handler.check_file()

    def on_batch_begin(self, batch, **kwargs):
        if self.reformat_input_fn:
            self.reformat_input_fn(batch)

    def on_batch_end(self, batch, model_outs, **kwargs):
        handler = self.handler
        handler.write(
            batch,
            model_outs,
            partial(self.reformat_outputs_fn, **self.reformat_out_fn_kwargs),
        )

    @rank_zero_only
    def on_loop_end(self, **kwargs):
        handler = self.handler
        handler.merge_file()
        handler.create_prediction()
