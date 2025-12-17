# Copyright (c) Horizon Robotics. All rights reserved.

import copy
import logging
from typing import List, Sequence, Union

try:
    import imutils
except ImportError:
    imutils = object

import numpy as np
import torch

from hat.core.affine import get_vcs2bev_img_mat
from hat.metrics.metric import EvalMetric
from hat.registry import OBJECT_REGISTRY
from hat.utils.apply_func import _as_list
from hat.utils.package_helper import require_packages

__all__ = ["BevSegInstanceEval"]

logger = logging.getLogger(__name__)


@OBJECT_REGISTRY.register_module
class BevSegInstanceEval(EvalMetric):
    """The bev seg instance eval metrics.

    Args:
        seg_class(list(str)): A list of classes the segmentation dataset
            includes, the order should be the same as the label.
        metrics (list(str)): A list of eval metrics
        target_categorys(list(str)) : The output category to validation
        target_x_intervals (list(float)): X-coordinate range to validation
        target_y_intervals(list(float)): y-coordinate range to validation
        vcs_range(list(float)):vcs range.(order is (bottom,right,top,left))
        bev_seg_target_size (sequence):bev_seg_target_size,
                (order is (height,width))
        gt_min_x(float): Max depth for gts.
        gt_max_x(float): Max depth for gts.
        max_distance(float): maximum distance threshold.
        verbose(bool):  Whether to return verbose value for aidi eval, default
            is False.
        name(str): Name of this metric instance for display
        enable_occlusion(bool):Whether to use occlusion. default False
    """

    @require_packages("imutils")
    def __init__(
        self,
        seg_class: List[str],
        metirc_key: List[str],
        target_categorys: Sequence[str],
        target_x_intervals: Sequence[float],
        target_y_intervals: Sequence[float],
        vcs_range: Sequence[float],
        bev_seg_target_size: Sequence[float],
        gt_min_x: float = -30.0,
        gt_max_x: float = 72.4,
        max_distance: float = 1.5,
        verbose: bool = False,
        enable_occlusion: bool = False,
        name: str = "BevSegInstanceEval",
    ):
        self.seg_class = seg_class
        self.verbose = verbose
        self.name = name
        self.max_distance = max_distance
        self.metirc_key = metirc_key
        self.enable_occlusion = enable_occlusion
        self.target_categorys = target_categorys
        target_x_intervals = [gt_min_x] + list(target_x_intervals) + [gt_max_x]
        self.full_target_x_intervals = [
            "({},{})".format(start, end)
            for start, end in zip(
                target_x_intervals[:-1], target_x_intervals[1:]
            )
        ] + ["({},{})".format(gt_min_x, gt_max_x)]

        self.full_target_y_intervals = [
            "({},{})".format(start, end)
            for start, end in zip(
                [0 - item for item in target_y_intervals], target_y_intervals
            )
        ]
        scope_H = vcs_range[2] - vcs_range[0]
        scope_W = vcs_range[3] - vcs_range[1]
        bev_height, bev_width = bev_seg_target_size
        vcs_origin_coord = [
            vcs_range[2] / scope_H * bev_height,
            vcs_range[3] / scope_W * bev_width,
        ]
        self.vcs_origin_coord = vcs_origin_coord
        self.x_perpixel = (
            abs(vcs_range[2] - vcs_range[0]) / bev_seg_target_size[0]
        )
        self.x_perpixel_half = self.x_perpixel / 2

        self.vcs2bev = get_vcs2bev_img_mat(
            vcs_range=vcs_range,
            bev_size=bev_seg_target_size,
        )
        super(BevSegInstanceEval, self).__init__("BevSegInstanceEval")

    def collect_bev_seg_data(
        self,
        label: torch.Tensor,
        preds: Union[Sequence[torch.Tensor], torch.Tensor],
    ):
        if self.enable_occlusion:
            assert "occlusion" in label
        # If there is no occlusion in the label,
        # the occlusion mask will be all zeros
        occlusion = (
            label.pop("occlusion")
            if "occlusion" in label
            else torch.zeros_like(_as_list(preds)[0])
        )
        pred_label = _as_list(preds)[0].detach().cpu().numpy()
        occlusion_label = _as_list(occlusion)[0].detach().cpu().numpy()
        batch_size = pred_label.shape[0]
        output_label_list = []
        output_pred_list = []
        occlusion_label_list = []
        for bs in range(batch_size):
            output_pred_list.append(self.skeletonize_pred(pred_label[bs]))
            occlusion_label_list.append(occlusion_label[bs])
            label_dict = {}
            for category in self.target_categorys:
                _current_category_list = (
                    label[category][bs].detach().cpu().numpy().tolist()
                )
                label_dict[category] = _current_category_list
            output_label_list.append(label_dict)
        return output_pred_list, output_label_list, occlusion_label_list

    def split_intervals(self, intervals_str):
        str_list = intervals_str[1:-1].split(",")
        float_list = [float(_item) for _item in str_list]
        return float_list

    def skeletonize_pred(self, pred):
        output = np.zeros_like(pred)
        process_index = [
            self.seg_class.index(category)
            for category in self.target_categorys
        ]
        for skeletonize_index in process_index:
            pred_copy = copy.deepcopy(pred)
            pred_copy = pred_copy.astype("uint8")
            for i in range(len(self.seg_class)):
                if i == skeletonize_index:
                    continue
                pred_copy[pred_copy == i] = 0
            pred_copy[pred_copy == skeletonize_index] = 1
            pred_copy_skele = imutils.skeletonize(pred_copy, size=(3, 3))
            # pred[pred == skeletonize_index] = 0
            output[pred_copy_skele != 0] = skeletonize_index
        return output

    def convert_to_dict(self, x_origin, y_origin):
        def merge(values, thresh=0.2):
            values.sort()
            values = [round(_item, 2) for _item in values]
            values = np.array(values)
            res = []
            cur_res = [values[0]]
            for i in range(1, len(values)):
                if abs(values[i] - values[i - 1]) <= (self.x_perpixel + 0.01):
                    cur_res.append(values[i])
                else:
                    res.append(np.mean(cur_res))
                    cur_res = [values[i]]
            res.append(np.mean(cur_res))
            return res

        x_y_pred = [
            [
                (self.vcs_origin_coord[0] - y) * self.x_perpixel
                - self.x_perpixel_half,
                (self.vcs_origin_coord[1] - x) * self.x_perpixel
                - self.x_perpixel_half,
            ]
            for (x, y) in zip(list(x_origin), list(y_origin))
        ]
        x_y_pred_dict = {}
        for x_pred, y_pred in x_y_pred:
            x_pred = round(x_pred, 2)
            if x_pred in x_y_pred_dict.keys():
                x_y_pred_dict[x_pred].append(y_pred)
            else:
                x_y_pred_dict[x_pred] = []
                x_y_pred_dict[x_pred].append(y_pred)
        output_dict = {}
        for key in x_y_pred_dict.keys():
            y_pred_list = x_y_pred_dict[key]
            _new_key = str(round(key, 2))
            output_dict[_new_key] = merge(y_pred_list)
            # x_y_pred_dict[key]=merge(y_pred_list)
        return output_dict

    def update_pred_num(self, x_y_pred_dict, category):
        for target_x_interval in self.full_target_x_intervals:
            (
                target_x_begin,
                target_x_end,
            ) = self.split_intervals(target_x_interval)
            for x in x_y_pred_dict.keys():
                _str_x = float(x)
                if _str_x > target_x_begin and _str_x < target_x_end:
                    y_list = x_y_pred_dict[x]
                    for y in y_list:
                        for target_y_interval in self.full_target_y_intervals:
                            (
                                target_y_begin,
                                target_y_end,
                            ) = self.split_intervals(target_y_interval)
                            if y > target_y_begin and y < target_y_end:
                                prefix = "{}_{}_{}".format(
                                    target_y_interval,
                                    category,
                                    target_x_interval,
                                )
                                setattr(
                                    self,
                                    prefix + "_pred_num",
                                    getattr(self, prefix + "_pred_num", None)
                                    + 1,
                                )

    def in_occlusion(self, x, y, occlusion):
        """Whether x y in occlusion.

        Args:
            x,y: the vcs coordinate, (x,y).
            occlusion: occlusion mask
        """
        height, width = occlusion.shape
        pt = self.vcs2bev @ np.array([x, y, 1.0], dtype=np.float64).reshape(
            (3, 1)
        )
        v = int(pt[0] + 0.5)
        u = int(pt[1] + 0.5)
        if u >= height or u < 0 or v >= width or v < 0:
            return False
        if occlusion[u, v] == 1:
            return True
        else:
            return False

    def update(
        self,
        label: torch.Tensor,
        preds: Union[Sequence[torch.Tensor], torch.Tensor],
    ):

        matched_num_list = {}
        # save matched distance
        for target_y_interval in self.full_target_y_intervals:
            current_target_y_interval = {}
            for category in self.target_categorys:
                current_category = {}
                for target_x_interval in self.full_target_x_intervals:
                    current_category[target_x_interval] = []
                current_target_y_interval[category] = current_category
            matched_num_list[target_y_interval] = current_target_y_interval

        (
            pred_list,
            label_list,
            occlusion_label_list,
        ) = self.collect_bev_seg_data(label, preds)

        for pred, label, occlusion in zip(
            pred_list, label_list, occlusion_label_list
        ):
            for category in self.target_categorys:
                pred = (
                    pred * (1 - occlusion) if self.enable_occlusion else pred
                )
                y_origin, x_origin = np.where(
                    pred == self.seg_class.index(category)
                )
                x_y_pred_dict = self.convert_to_dict(x_origin, y_origin)

                self.update_pred_num(x_y_pred_dict, category)

                if label[category] is not None:
                    for target_y_interval in self.full_target_y_intervals:
                        (
                            target_y_begin,
                            target_y_end,
                        ) = self.split_intervals(target_y_interval)
                        x_y_pred_dict_copy = copy.deepcopy(x_y_pred_dict)
                        for [x, y] in list(label[category]):
                            if x == 1000.0 and y == 1000.0:
                                break
                            if self.enable_occlusion and self.in_occlusion(
                                x, y, occlusion
                            ):
                                continue

                            if y <= target_y_begin or y >= target_y_end:
                                continue

                            # get gt num
                            for (
                                target_x_interval
                            ) in self.full_target_x_intervals:
                                (
                                    target_x_begin,
                                    target_x_end,
                                ) = self.split_intervals(target_x_interval)

                                if x > target_x_begin and x < target_x_end:
                                    prefix = "{}_{}_{}".format(
                                        target_y_interval,
                                        category,
                                        target_x_interval,
                                    )
                                    setattr(
                                        self,
                                        prefix + "_gt_num",
                                        getattr(self, prefix + "_gt_num", None)
                                        + 1,
                                    )

                            # matching stat
                            _distance = []
                            _str_key = str(round(x, 2))
                            if _str_key not in x_y_pred_dict_copy.keys():
                                continue
                            y_pred_list = x_y_pred_dict_copy[_str_key]
                            _distance = [
                                [abs(y_pred - y), y_pred]
                                for y_pred in y_pred_list
                                if abs(y_pred - y) < self.max_distance
                            ]
                            if len(_distance) == 0:
                                continue
                            _distance.sort()
                            x_y_pred_dict_copy[_str_key].remove(
                                _distance[0][1]
                            )
                            for (
                                target_x_interval
                            ) in self.full_target_x_intervals:
                                (
                                    target_x_begin,
                                    target_x_end,
                                ) = self.split_intervals(target_x_interval)
                                if x > target_x_begin and x < target_x_end:
                                    prefix = "{}_{}_{}".format(
                                        target_y_interval,
                                        category,
                                        target_x_interval,
                                    )
                                    matched_num_list[target_y_interval][
                                        category
                                    ][target_x_interval].append(
                                        _distance[0][0]
                                    )
        for target_y_interval in self.full_target_y_intervals:
            for category in self.target_categorys:
                for target_x_interval in self.full_target_x_intervals:
                    prefix = "{}_{}_{}".format(
                        target_y_interval,
                        category,
                        target_x_interval,
                    )
                    match_list = getattr(self, prefix + "_matched_num")
                    curren_list = matched_num_list[target_y_interval][
                        category
                    ][target_x_interval]
                    current_tensor_list = torch.tensor(
                        np.array(curren_list),
                        device=self._device_key.device,
                        dtype=torch.float16,
                    )

                    match_list.extend([current_tensor_list])
                    setattr(
                        self,
                        prefix + "_matched_num",
                        match_list,
                    )

    def get_per_metric(
        self, current_calculate_list, current_pred_num, current_gt_num
    ):
        current_metirc = {}
        _current_len = len(current_calculate_list)
        if _current_len == 0:
            _current_mean = 0.0
            _current_max = 0.0
            _current_Percentile_0_9967 = 0.0
            _current_Percentile_0_996 = 0.0
            _current_std = 0.0
            _current_expression_3std = 0.0
            _current_expression_0_075 = 0.0
            _recall = 0.0
            _precision = 0.0
        else:
            current_calculate_list.sort()
            _recall = _current_len / current_gt_num
            _precision = _current_len / current_pred_num
            _current_mean = np.mean(current_calculate_list)
            _current_max = max(current_calculate_list)
            _current_Percentile_0_9967 = current_calculate_list[
                int(0.9967 * _current_len)
            ]
            _current_Percentile_0_996 = current_calculate_list[
                int(0.96 * _current_len)
            ]
            _current_std = np.std(np.array(current_calculate_list), ddof=1)
            _current_expression_3std = (
                np.count_nonzero(
                    np.array(current_calculate_list) < 3 * _current_std
                )
                / _current_len
            )
            _current_expression_0_075 = (
                np.count_nonzero(np.array(current_calculate_list) < 0.075)
                / _current_len
            )

        current_metirc["Recall"] = round(_recall, 4)
        current_metirc["Precision"] = round(_precision, 4)
        current_metirc["Num"] = round(_current_len, 4)
        current_metirc["Mean"] = round(_current_mean, 4)
        current_metirc["Max"] = round(_current_max, 4)
        current_metirc["Percentile:0.9976"] = round(
            _current_Percentile_0_9967, 4
        )
        current_metirc["Percentile:0.96"] = round(_current_Percentile_0_996, 4)
        current_metirc["Expression:[VALUE<3*Std]"] = round(
            _current_expression_3std, 4
        )
        current_metirc["Expression:[VALUE<0.075]"] = round(
            _current_expression_0_075, 4
        )
        return current_metirc

    def get_metric_names(self):
        metric_names = []
        for target_y_interval in self.full_target_y_intervals:
            for category in self.target_categorys:
                for target_x_interval in self.full_target_x_intervals:
                    prefix = "{}_{}_{}".format(
                        target_y_interval, category, target_x_interval
                    )
                    metric_names += [prefix + "_gt_num"]
                    metric_names += [prefix + "_pred_num"]
                    metric_names += [prefix + "_matched_num"]
        metric_names += ["_device_key"]
        return metric_names

    def _init_states(self):
        for name in self.get_metric_names():
            if "matched" in name:
                self.add_state(
                    name,
                    default=[],
                    dist_reduce_fx="cat",
                )
            else:
                self.add_state(
                    name,
                    default=torch.tensor([0.0]),
                    dist_reduce_fx="sum",
                )

    def compute(self):
        """Get evaluation metrics."""

        result_metric = {}
        for target_y_interval in self.full_target_y_intervals:
            _category_metric = {}
            for category in self.target_categorys:
                _intervals_metric = {}
                for target_x_interval in self.full_target_x_intervals:
                    prefix = "{}_{}_{}".format(
                        target_y_interval, category, target_x_interval
                    )
                    current_calculate_list = getattr(
                        self, prefix + "_matched_num"
                    )

                    current_calculate_array = (
                        torch.cat(_as_list(current_calculate_list))
                        .cpu()
                        .numpy()
                        .astype(np.float64)
                    )
                    delete_index = np.where(
                        np.equal(current_calculate_array, self.max_distance)
                    )
                    current_calculate_list = np.delete(
                        current_calculate_array, delete_index
                    ).tolist()
                    current_pred_num = getattr(self, prefix + "_pred_num")
                    current_pred_num = (
                        current_pred_num.cpu().numpy().tolist()[0]
                    )

                    current_gt_num = getattr(self, prefix + "_gt_num")
                    current_gt_num = current_gt_num.cpu().numpy().tolist()[0]

                    current_metirc = self.get_per_metric(
                        current_calculate_list,
                        current_pred_num,
                        current_gt_num,
                    )
                    _intervals_metric.update(
                        {target_x_interval: current_metirc}
                    )
                _category_metric.update({category: _intervals_metric})
            result_metric.update({target_y_interval: _category_metric})
        if self.verbose:
            return result_metric

        summary_str = "~~~~ %s Summary metrics ~~~~\n" % (self.name)
        dividing_line = "-" * 180 + "\n"
        line_format = "{:^8} {:^15} {:^20} {:^8} {:^8} {:^8} {:^8} {:^8} {:^20} {:^20} {:^20} {:^20} \n"  # noqa
        summary_str += line_format.format(
            "Range Y",
            "class",
            "Range X",
            "Recall",
            "Precision",
            "Num",
            "Mean",
            "Max",
            "Percentile:0.9976",
            "Percentile:0.96",
            "Expression:[VALUE<3*Std]",
            "Expression:[VALUE<0.075]",
        )
        for target_y_interval in self.full_target_y_intervals:
            for category in self.target_categorys:
                for target_x_interval in self.full_target_x_intervals:
                    _show_metric = []
                    _show_metric.append(target_y_interval)
                    _show_metric.append(category)
                    _show_metric.append(target_x_interval)
                    _current_metirc = result_metric[target_y_interval][
                        category
                    ][target_x_interval]
                    for metric_name in self.metirc_key:
                        assert metric_name in _current_metirc.keys()
                        _show_metric.append(_current_metirc[metric_name])

                    summary_str += line_format.format(*_show_metric)
                summary_str += dividing_line
        logger.info(summary_str)
