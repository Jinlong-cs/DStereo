# Copyright (c) Horizon Robotics. All rights reserved.
from typing import Mapping, Optional, Sequence

import cv2
import numpy as np

from hat.core.bev_elevation_utils import (
    crop_roi_vcs_range,
    get_area_threshod_mask,
    get_center_coord,
    get_vcsrange_bbox_data,
    max_freespace_contain_ego,
)
from hat.core.bev_occupancy_utils import generate_bev_weight_map
from hat.data.transforms.auto_3dv import ANCClassRemap
from hat.registry import OBJECT_REGISTRY

__all__ = [
    "ANCBEVElevationGenerator",
    "ANCBEVVisMaskGenerator",
    "ANCBEVFreespaceGenerator",
]


@OBJECT_REGISTRY.register
class ANCBEVElevationGenerator(object):
    """Generate and remap bev elevation gt with specific remap dict.

    This class generate elevation gt from 'gt_bev_elevation' file,
    which include 3 chanel, default pixel value is 255.
    In channel 2, pixel value stand for lidar seg class. see detail in
    https://horizonrobotics.feishu.cn/docs/doccn29lbVRDqv0bR54gFIF4Aid.
    In channel 1, the value range of pixels is 0 to 19, which stand
    for bin of height.
    In channel 0, the value stand for height value calculated from
    (h+bias)*ratio, which can be used to repartition bins. Default bias
    is 1, default ratio is 50.

    In channel 0, 1 stand for Occluded areas (such as the area after car).

    Args:
        vcs_range: current roi vcs range during training.
        raw_gt_res_vcsrange_cfg: a dict, raw offline gt resolution
            as key, and corresponding vcs range as value.
        target_size: bev elevation gt size, order (h, w).
        remap_dict dict: remap dict[str: dict].
        height_to_bin_config: config for dividing the height into bins.
        gt_elevation_code_ratio: code ratio of bev elevation gt.
        gt_elevation_code_bias:  code bias of bev elevation gt.
        min_gt_elevation: min value to clip gt, default is -0.5m.
        max_gt_elevation: max value to set ignore for gt, default is 4m.
        use_continuous_gt: whether use conyinuous gt, default is
            false (discrete gt).
        ignore_index: ignore index, default value is 255.
        gt_name: return gt name.

    Returns:
        ndarray: processed height gt for bev elevation.
    """

    def __init__(
        self,
        vcs_range: Sequence[float],
        raw_gt_res_vcsrange_cfg: dict,
        target_size: Sequence[int],
        remap_dict: Optional[Mapping] = None,
        height_to_bin_config: Optional[Mapping] = None,
        gt_elevation_code_ratio: int = 50,
        gt_elevation_code_bias: int = 1,
        min_gt_elevation: float = -0.5,
        max_gt_elevation: float = 4,
        use_continuous_gt: bool = False,
        ignore_index: int = 255,
        gt_name: str = "gt_bev_elevation",
    ):
        self.remap_dict = remap_dict if remap_dict else {}
        self.height_to_bin_config = height_to_bin_config
        self.gt_elevation_code_ratio = gt_elevation_code_ratio
        self.gt_elevation_code_bias = gt_elevation_code_bias
        self.min_gt_elevation = min_gt_elevation
        self.max_gt_elevation = max_gt_elevation
        self.use_continuous_gt = use_continuous_gt
        if self.use_continuous_gt is False:
            assert (
                height_to_bin_config is not None
            ), "'height_to_bin_config' must be provide when use_continuous_gt is False"  # noqa
        self.target_size = target_size
        self.ignore_index = ignore_index
        self.vcs_range = vcs_range
        self.raw_gt_res_vcsrange_cfg = raw_gt_res_vcsrange_cfg
        self.gt_name = gt_name

    def _remap(self, data, remap_dict):
        if isinstance(data, Sequence):
            return [self._remap(_, remap_dict) for _ in data]
        else:
            data = np.array(data)

            # generator gt
            # 第0通道原始高度, 编码方式 (h+bias)*ratio
            height = data[..., 0].copy()
            # 用 h/50-1 可以直接解码成连续的浮点数高度
            height = (
                height.astype(np.float) / self.gt_elevation_code_ratio
                - self.gt_elevation_code_bias
            )
            height[height <= self.min_gt_elevation] = self.min_gt_elevation
            height[height > self.max_gt_elevation] = self.ignore_index

            # generator mask
            # 第2个通道是lidar seg的类别
            seg = data[..., 2].copy()
            # 生成高于2米的植物的掩码, 后面ignore用, seg类别为11, bin的类别为9 # noqa
            mask = (height >= 2) * (seg == 11)
            height[mask] = self.ignore_index

            if self.use_continuous_gt:
                return height
            else:
                # step1: 按照bins_num将高度等间隔划分
                # step2: 通过remap将step1得到的bins映射为非线性
                # example:
                #   remap_dict: {
                #       0: 0,
                #       1: 1,
                #       2: 1,
                #   }
                #   before step1: height= [0.6, 1.5, 2.2 ]
                #   after step1: height = [0, 1, 2]
                #   after step2: height = [0, 1, 1]
                bins = np.linspace(
                    self.height_to_bin_config["min_height_value"],
                    self.height_to_bin_config["max_height_value"],
                    self.height_to_bin_config["bins_num"],
                )
                # 不乘scale, 会有精度损失
                height = np.digitize(
                    height * self.height_to_bin_config["scale"],
                    bins * self.height_to_bin_config["scale"],
                ).astype(np.int64)
                return np.vectorize(remap_dict.get)(height)

    def __call__(self, data):
        assert "gt_bev_elevation_raw" in data

        gt_bev_elevation = np.array(data["gt_bev_elevation_raw"])
        raw_gt_res = gt_bev_elevation.shape[:2]
        assert raw_gt_res in self.raw_gt_res_vcsrange_cfg
        raw_gt_vcs_range = self.raw_gt_res_vcsrange_cfg[raw_gt_res]
        gt_bev_elevation, vcs_range_bbox = get_vcsrange_bbox_data(
            gt_bev_elevation,
            self.vcs_range,
            raw_gt_vcs_range,
            self.ignore_index,
        )
        gt_bev_elevation = crop_roi_vcs_range(
            gt_bev_elevation, self.vcs_range, vcs_range_bbox
        )

        gt_bev_elevation = self._remap(
            gt_bev_elevation, self.remap_dict["gt_bev_elevation"]
        )

        data[self.gt_name] = cv2.resize(
            gt_bev_elevation,
            (int(self.target_size[-1]), int(self.target_size[-2])),
            interpolation=cv2.INTER_NEAREST,
        )
        return data

    def __repr__(self):
        return "BEVElevationGenerator"


@OBJECT_REGISTRY.register
class ANCBEVVisMaskGenerator(object):
    """Generate vis mask for bev elevation.

    .. note:: This class generate vismask from 'gt_bev_elevation_vismask' file,
        which include 3 chanel, default pixel value is 255.
        In channel 0, 1 stand for visible areas in current lane, 2 stand for
        opposing lane.
        In channel 1, pixel value stand for parsing class. see detail in
        http://wiki.hobot.cc/pages/viewpage.action?pageId=248420446.
        In channel 2, 0 stand for car or people, 1 stand for the area
        after car or people, 2 stand for the area after tree, 3 stand
        for the area after fence.

    Args:
        vcs_range: current roi vcs range during training.
        raw_vcs_range: vcs range of raw offline freespace gt, used for crop
            current vcs_range from it in small range model.
        target_size: bev vismask gt size, order (h, w).
        ignore_index: ignore index, default value is 255.
        global_dilate_cfg: dict, global dilate configs to dilate.

            .. code-block:: none

                # freespace region for elevation.
                {
                    "global_dilate_cfg": {
                        "use_dilate": bool,

                        "kernel": int,

                        "iterations": int,

                    },

                }

        gt_name: return gt name.

    Returns:
        ndarray: gt for bev vismask, o is invisuble, 1 is visible.
    """

    def __init__(
        self,
        vcs_range: Sequence[float],
        raw_gt_res_vcsrange_cfg: dict,
        target_size: Sequence[int],
        ignore_index: int = 255,
        global_dilate_cfg: dict = None,
        gt_name: str = "gt_bev_elevation_vismask",
    ):
        self.target_size = target_size
        self.ignore_index = ignore_index
        self.vcs_range = vcs_range
        self.raw_gt_res_vcsrange_cfg = raw_gt_res_vcsrange_cfg
        self.global_dilate_cfg = global_dilate_cfg
        self.gt_name = gt_name

    def get_island_mask(self, vismask):
        # Get island region in vismask.
        vismask = vismask.astype(np.uint8)
        island_mask = np.zeros_like(vismask)
        contours = cv2.findContours(
            vismask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE
        )[0]
        for contour in contours:
            if 0 < len(contour) < 300:
                cv2.drawContours(island_mask, [contour], -1, 255, -1)
        return island_mask

    def process_mask(self, vis_mask):
        # see detail in https://horizonrobotics.feishu.cn/docx/Ya7xd6u7doWwJbxaAOFcJeoFnMf # noqa
        mask = vis_mask[..., 0].copy()
        obj = vis_mask[..., 2].copy()
        mask[mask == self.ignore_index] = 0
        # mask=2 代表对向车道.
        mask[mask == 2] = 0

        # obj=0 代表 车或人自身区域
        # obj=1 代表 车或人遮挡区域
        # obj=2 代表 植物遮挡区域
        # obj=3 代表 栏杆遮挡区域
        mask[(obj == 1) | (obj == 2) | (obj == 3)] = 0
        mask[obj == 0] = 1

        use_dilate = False
        if self.global_dilate_cfg is not None:
            use_dilate = self.global_dilate_cfg["use_dilate"]
            kernel = self.global_dilate_cfg["kernel"]
            iterations = self.global_dilate_cfg["iterations"]

        if use_dilate:
            mask = cv2.dilate(
                mask,
                np.ones((kernel, kernel), np.uint8),
                iterations,
            )

        agent = np.zeros_like(obj)
        agent[obj == 0] = 1
        island_mask = self.get_island_mask(mask)
        mask[island_mask > 0] = 0
        agent[island_mask > 0] = 0

        return mask, agent

    def __call__(self, data):
        assert "gt_bev_elevation_vismask_raw" in data

        bev_vismask = np.array(data["gt_bev_elevation_vismask_raw"])
        raw_gt_res = bev_vismask.shape[:2]
        assert raw_gt_res in self.raw_gt_res_vcsrange_cfg
        raw_gt_vcs_range = self.raw_gt_res_vcsrange_cfg[raw_gt_res]
        bev_vismask, vcs_range_bbox = get_vcsrange_bbox_data(
            bev_vismask,
            self.vcs_range,
            raw_gt_vcs_range,
            self.ignore_index,
        )
        bev_vismask = crop_roi_vcs_range(
            bev_vismask, self.vcs_range, vcs_range_bbox
        )

        bev_vismask = cv2.resize(
            bev_vismask,
            (int(self.target_size[-1]), int(self.target_size[-2])),
            interpolation=cv2.INTER_NEAREST,
        )
        bev_vismask, agent = self.process_mask(bev_vismask)

        data[self.gt_name] = {
            "vismask": bev_vismask.astype(np.int64),
            "agent": agent.astype(np.int64),
        }
        return data

    def __repr__(self):
        return "BEVVisMaskGenerator"


@OBJECT_REGISTRY.register
class ANCBEVFreespaceGenerator(object):
    """Generate gt for bev freespace.

    Generate bev freespace gt for freespace, elevation task.
    Note that, currently, at most one of global_dilate_cfg and
    smallobj_dilate_cfg is applied to data.

    Args:
        vcs_range: current roi vcs range during training.
        raw_gt_res_vcsrange_cfg: a dict, raw offline gt resolution
            as key, and corresponding vcs range as value.
        target_size: freespace gt size, order (h, w).
        lidar_seg_remap_dict: a remap dict for lidar seg to label
            dynamic obstacles.
        freespace_mannual_remap_dict: a remap dict, map mannual
            freespace label to freespace 0-1 label.
        freespace_auto_remap_dict: a remap dict, map auto freespace
            label to freespace 0-1 label.
        ele_freespace_mannual_remap_dict: a remap dict, map mannual
            freespace label to get freespace, obstructions and
            constructions for elevation.
        ele_freespace_auto_remap_dict: a remap dict, map auto freespace
            label to get freespace and objects in freespace region from
            lidar seg and lidar det, for elevation.
        occupancy_output_freespace: whether to output freespace gt .
        occupancy_output_elevation: whether to output mask according freespace
            static annotation, for elevation task.
        output_extra_obstacle: whether to output extra freespace obstacle.
        output_lidardet:  whether to output vehicle mask from lidardet.
        ignore_index: ignore index, default value is 255.
        lidar_det_conf_thresh: confidence thresh for lidar det bbox.
        global_dilate_cfg: dict, global dilate configs to dilate
            freespace region for elevation.

            .. code-block:: none

                {
                    "global_dilate_cfg": {
                        "use_dilate": bool,

                        "kernel": int,

                        "iterations": int,

                    },

                }

        smallobj_dilate_cfg: dict, small object dilate configs to
            reduce small object disappear when gt is downsampled.

            .. code-block:: none

                {
                    "smallobj_dilate_cfg": {
                        "use_dilate": bool, whether dilate small obj.

                        "kernel": int, dilate kernel.

                        "autolabel_area_range": (4, 100), tuple with
                            min_obj_area_thresh and max_obj_area_thresh,
                            e.g, (4, 100) for small object, measured in
                            pixel nums. usually auto labeling data use
                            larger min_obj_area_thresh, due to noise existence.

                        "manuallabel_area_range": (1, 100), tuple with
                            min_obj_area_thresh and max_obj_area_thresh,
                            e.g, (1, 100) for small object, measured in
                            pixel nums.

                    },

                }

        mannualabel_use_lidarseg: whether add lidar seg to mannual
            static annotation to generate final freespace gt.
        ego_region_range: float values denotes ego region in vcs,
            same with vcs_range order (bottom, right, top, left).
        median_blur_seg_height: whether to median blur lidar seg
            and height, to filter salt-pepper noise.
        freespace_gt_name: freespace gt name.
        ele_freespace_gt_name: ele freespace gt name.
        extra_obstacle_gt_name: gt name for fine-grained freespace obstacle.
        lidardet_veh_mask_gt_name: veh mask gt name from lidardet.
        output_bev_weight_map: whether use bev weight mask.
    """

    def __init__(
        self,
        vcs_range: Sequence[float],
        raw_gt_res_vcsrange_cfg: dict,
        target_size: Sequence[int],
        lidar_seg_remap_dict: dict,
        freespace_mannual_remap_dict: dict,
        freespace_auto_remap_dict: dict,
        ele_freespace_mannual_remap_dict: dict,
        ele_freespace_auto_remap_dict: dict,
        occupancy_output_freespace: bool = True,
        occupancy_output_elevation: bool = False,
        output_extra_obstacle: bool = False,
        output_lidardet: bool = False,
        ignore_index: int = 255,
        lidar_det_conf_thresh: float = 0.4,
        global_dilate_cfg: dict = None,
        smallobj_dilate_cfg: dict = None,
        mannualabel_use_lidar_seg: bool = False,
        ego_region_range: Optional[Sequence[float]] = None,
        median_blur_seg_height: bool = False,
        use_extra_mask: bool = False,
        freespace_gt_name: str = "gt_bev_freespace",
        ele_freespace_gt_name: str = "gt_bev_ele_freespace",
        extra_obstacle_gt_name: str = "gt_bev_extra_obstacle",
        lidardet_veh_mask_gt_name: str = "gt_lidardet_veh_mask",
        output_bev_weight_map: bool = False,
    ):
        assert (
            occupancy_output_freespace or occupancy_output_elevation
        ), "occupancy_output_freespace or occupancy_output_elevation at least one is True!"  # noqa
        assert not (
            global_dilate_cfg and smallobj_dilate_cfg
        ), "global_dilate_cfg and smallobj_dilate_cfg at most one is used!"

        self.output_freespace = occupancy_output_freespace
        self.output_elevation = occupancy_output_elevation
        self.output_extra_obstacle = output_extra_obstacle
        self.output_lidardet = output_lidardet
        self.output_bev_weight_map = output_bev_weight_map
        self.target_size = target_size
        self.ignore_index = ignore_index
        self.vcs_range = vcs_range
        self.raw_gt_res_vcsrange_cfg = raw_gt_res_vcsrange_cfg
        self.lidar_det_conf_thresh = lidar_det_conf_thresh
        self.lidar_seg_remap_dict = lidar_seg_remap_dict
        self.freespace_mannual_remap_dict = freespace_mannual_remap_dict
        self.freespace_auto_remap_dict = freespace_auto_remap_dict
        self.ele_freespace_mannual_remap_dict = (
            ele_freespace_mannual_remap_dict
        )
        self.ele_freespace_auto_remap_dict = ele_freespace_auto_remap_dict
        self.global_dilate_cfg = global_dilate_cfg
        self.smallobj_dilate_cfg = smallobj_dilate_cfg
        self.mannualabel_use_lidar_seg = mannualabel_use_lidar_seg
        self.ego_region_range = ego_region_range
        self.median_blur_seg_height = median_blur_seg_height
        self.freespace_gt_name = freespace_gt_name
        self.ele_freespace_gt_name = ele_freespace_gt_name
        self.extra_obstacle_gt_name = extra_obstacle_gt_name
        self.lidardet_veh_mask_gt_name = lidardet_veh_mask_gt_name
        self.use_extra_mask = use_extra_mask

    def fill_ego_region_freespace(self, gt, freespace_label=0):
        """Fill ego region with freespace label 0.

        Note that usually ego is detected by lidar det, and
        detected vehicles are taken as non freespace. But ego
        region is always freespace. So need to fill ego region
        to freespace region.

        Args:
            gt: freespace gt label data, (h, w)
            freespace_label: class label of freespace.
        """
        ego_center_v, ego_center_u = [
            int(
                self.vcs_range[2]
                / (self.vcs_range[2] - self.vcs_range[0])
                * gt.shape[0]
            ),
            int(
                self.vcs_range[3]
                / (self.vcs_range[3] - self.vcs_range[1])
                * gt.shape[1]
            ),
        ]  # (v, u)

        # convert to binary image, better accuracy
        gt4contour = gt.copy()
        gt4contour[gt4contour == self.ignore_index] = 0

        contours, _ = cv2.findContours(
            gt4contour.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE
        )
        for _, contour in enumerate(contours):
            x, y, w, h = cv2.boundingRect(contour)
            if x == 0 or y == 0:
                continue
            if (x <= ego_center_u <= x + w) and (y <= ego_center_v <= y + h):
                gt = cv2.fillConvexPoly(
                    gt.astype(np.uint8), contour, freespace_label
                )
                break
        return gt

    def dilate_small_object(self, data: dict, is_auto_label: bool):
        """Dilate small object in freespace gt.

        Args:
            data: freespace target dict. e.g.

                .. code-block:: none

                    {
                        "gt_bev_ele_freespace": gt_bev_ele_freespace,

                        "gt_bev_freespace": {
                            "gt_bev_freespace": gt_bev_freespace,
                        },

                    }

            is_auto_label: gt type from auto label or mannual label.

        Returns: freespace target dict.
        """

        # dilate small object area between lower_area_thresh
        # and upper_area_thresh. auto labeling data use larger
        # lower_area_thresh, due to noise existence.
        if (
            self.output_freespace
            and self.smallobj_dilate_cfg is not None
            and self.smallobj_dilate_cfg["use_dilate"]
        ):
            gt_bev_freespace = data[self.freespace_gt_name][
                self.freespace_gt_name
            ]
            if is_auto_label:
                (
                    lower_area_thresh,
                    upper_area_thresh,
                ) = self.smallobj_dilate_cfg["autolabel_area_range"]
            else:
                (
                    lower_area_thresh,
                    upper_area_thresh,
                ) = self.smallobj_dilate_cfg["manuallabel_area_range"]
            small_obj_mask = get_area_threshod_mask(
                gt_bev_freespace,
                upper_area_thresh,
                lower_area_thresh,
                self.ignore_index,
            )
            kernel = self.smallobj_dilate_cfg["kernel"]
            small_obj_mask = cv2.dilate(
                small_obj_mask,
                np.ones((kernel, kernel), np.uint8),
            )
            # freespace label: 0 for freespace_yes, 1 for freespace_no,
            # 255 for ignore label.
            gt_bev_freespace[small_obj_mask == 1] = 1
            data[self.freespace_gt_name][
                self.freespace_gt_name
            ] = gt_bev_freespace

        for key, value in data.items():
            if (
                self.global_dilate_cfg is not None
                and self.global_dilate_cfg["use_dilate"]
            ):
                kernel = self.global_dilate_cfg["kernel"]
                iterations = self.global_dilate_cfg["iterations"]
                if isinstance(value, dict):
                    for sub_key, sub_value in value.items():
                        sub_value = cv2.dilate(
                            sub_value.astype(np.uint8),
                            np.ones((kernel, kernel), np.uint8),
                            iterations,
                        )
                        data[key][sub_key] = sub_value
                else:
                    value = cv2.dilate(
                        value.astype(np.uint8),
                        np.ones((kernel, kernel), np.uint8),
                        iterations,
                    )
                    data[key] = value.astype(np.int64)

        return data

    def parse_freespace_from_autolabel(
        self, gt: np.array, scene_type: str
    ) -> np.ndarray:
        """Process raw autolabel gt to 0-1 class for bev freespace.

        Generate 0-1 freespace gt from raw gt_bev_freespace.

        .. note:: Raw gt_bev_freespace includes 3 channel, e.g.
            channel 0: lidar seg label, label id range in {0, 1, 2, ..., 38}.

            channel 1: elevation of objects.

            channel 2: lidar det box, all labeled by id 50.

            see detail in
            https://horizonrobotics.feishu.cn/docx/HGZLdBCCuoId9DxJHyVcxd2anOb

        Args:
            gt: 3 channels includes  lidar seg, elevation,
                lidar det, described as above.

        Returns:
            processed 0-1 gt for bev freespace.
        """
        data = {}
        # filter salt-pepper noise for lidar seg and height
        if self.output_elevation:
            if self.median_blur_seg_height:
                gt_bev_ele_freespace = cv2.medianBlur(gt[..., 0].copy(), 3)
            else:
                gt_bev_ele_freespace = gt[..., 0].copy()
            obj = gt[..., 2].copy()
            mask_obj = obj == 50  # 50 is saved for all lidar det box
            gt_bev_ele_freespace[mask_obj] = 22  # 22是车

            gt_bev_ele_freespace = ANCClassRemap.remap(
                gt_bev_ele_freespace,
                self.ele_freespace_auto_remap_dict,
            )
            data[self.ele_freespace_gt_name] = gt_bev_ele_freespace

        if self.output_freespace:
            data[self.freespace_gt_name] = {}
            if self.median_blur_seg_height:
                gt_bev_freespace = cv2.medianBlur(gt[..., 0].copy(), 3)
                height_raw = cv2.medianBlur(gt[..., 1].copy(), 3)
                height = height_raw / 50 - 1
            else:
                gt_bev_freespace = gt[..., 0].copy()
                height_raw = gt[..., 1].copy()
                height = height_raw / 50 - 1
            obj = gt[..., 2].copy()

            # TODO:
            # 障碍物保存的最低高度就是15cm，可以验证下20，25的模型
            # 25cm much robust for auto label data, because of noise
            mask_height = (height_raw != self.ignore_index) * (height >= 0.25)
            mask_obj = obj == 50  # 50 is saved for all lidar det box
            # fill lidar-det detected ego bbox mask
            mask_obj = self.fill_ego_region_freespace(
                mask_obj.astype(np.uint8), freespace_label=0
            ).astype(np.bool)

            gt_bev_freespace[mask_height] = 19  # 19是其他障碍物
            gt_bev_freespace[mask_obj] = 22  # 22是车

            gt_bev_freespace = ANCClassRemap.remap(
                gt_bev_freespace.copy(),
                self.freespace_auto_remap_dict,
            )
            # seg[height!=255] = 50
            # seg[obj==50] = 51 # 根据bbox找的障碍物
            # seg_freespace = seg.copy()

            # fill ego region with freespace because some
            # ego regions annotated by ignore index for autolabel gt
            # freespace_yes label is 0
            if self.ego_region_range is not None:
                top, left, crop_h, crop_w = crop_roi_vcs_range(
                    gt_bev_freespace,
                    vcs_range=self.ego_region_range,
                    raw_vcs_range=self.vcs_range,
                    return_crop_coord=True,
                )
                gt_bev_freespace[top : top + crop_h, left : left + crop_w] = 0

            data[self.freespace_gt_name][
                self.freespace_gt_name
            ] = gt_bev_freespace

        data = self.dilate_small_object(data, True)

        if self.output_extra_obstacle:
            assert self.output_freespace is True
            gt_bev_freespace = data[self.freespace_gt_name][
                self.freespace_gt_name
            ]
            gt_bev_obstacle_extra = np.zeros_like(gt_bev_freespace)
            # 通过原始高程真值补充最大可行驶区域内的通用障碍物
            gt_bev_elevation = gt[..., 1].copy() / 50 - 1
            gt_bev_elevation_mask = (gt[..., 1] != self.ignore_index) * (
                gt_bev_elevation >= 0.25
            )

            if scene_type == "driving":
                max_region_mask = max_freespace_contain_ego(
                    gt_bev_freespace,
                    self.vcs_range,
                )
                gt_bev_obstacle_extra[
                    gt_bev_elevation_mask * max_region_mask
                ] = 1
            elif scene_type == "parking":
                gt_bev_obstacle_extra[gt_bev_elevation_mask] = 1
            else:
                raise ValueError("Invaild scenc type.")
            # 通过lidarseg语义补充限位器等障碍物
            gt_lidar_seg = gt[..., 0]
            gt_bev_obstacle_extra[gt_lidar_seg == 19] = 1

            data[self.freespace_gt_name][
                self.extra_obstacle_gt_name
            ] = gt_bev_obstacle_extra

        if self.output_lidardet:
            assert self.output_freespace is True
            lidar_det_veh_mask = np.zeros_like(gt[..., 2])
            lidar_det_veh_mask[gt[..., 2] == 50] = 1
            data[self.freespace_gt_name][
                self.lidardet_veh_mask_gt_name
            ] = lidar_det_veh_mask

        return data

    def parse_freespace_from_mannuallabel(self, gt: np.ndarray) -> np.ndarray:
        """Process raw gt to 0-1 class for bev freespace with mannual label.

        .. note:: Generate 0-1 freespace gt from raw gt_bev_freespace,
            Raw gt_bev_freespace includes 3 channel, e.g.

            channel 0: static annotation, 0, 1, 2, 3, 255 denotes
                freespaces_yes, freespaces_no, obstructions,
                constructions, ignores, respectively.

            channel 1: lidar seg label, label id start from 20.

            channel 2: lidar det confidence encoding for obj,
                encoded value equals to int(10*confidence).

            see detail in
            https://horizonrobotics.feishu.cn/docs/doccnDteHTj1EHdkm1aRhQyWf5c#
            For freespace auto label format details, please refer to
            https://horizonrobotics.feishu.cn/docs/doccnDteHTj1EHdkm1aRhQyWf5c

        Args:
            gt: 3 channels includes static anno,
                lidar seg, lidar det, described as above.

        Returns:
            processed 0-1 gt for bev freespace.
        """
        data = {}
        if self.output_elevation:
            gt_bev_ele_freespace = ANCClassRemap.remap(
                gt[..., 0].copy(),
                self.ele_freespace_mannual_remap_dict,
            )
            # lidar box confidence encoded by int(10 * confidence)
            mask_box = (gt[..., 2].copy() / 10) >= self.lidar_det_conf_thresh
            gt_bev_ele_freespace[mask_box] = 1
            data[self.ele_freespace_gt_name] = gt_bev_ele_freespace

        if self.output_freespace:
            data[self.freespace_gt_name] = {}
            gt_bev_freespace = ANCClassRemap.remap(
                gt[..., 0].copy(),
                self.freespace_mannual_remap_dict,
            )
            if self.mannualabel_use_lidar_seg:
                # dynamic object map to label == 1
                seg = (
                    cv2.medianBlur(gt[..., 1].copy(), 3)
                    if self.median_blur_seg_height
                    else gt[..., 1].copy()
                )
                mask_seg = (
                    ANCClassRemap.remap(
                        seg,
                        self.lidar_seg_remap_dict,
                    )
                    > 0
                )
                gt_bev_freespace[mask_seg] = 1

            # lidar box confidence encoded by int(10 * confidence)
            mask_box = (gt[..., 2].copy() / 10) >= self.lidar_det_conf_thresh
            # fill lidar-det detected ego bbox mask
            mask_box = self.fill_ego_region_freespace(
                mask_box.astype(np.uint8), freespace_label=0
            ).astype(np.bool)
            gt_bev_freespace[mask_box] = 1

            data[self.freespace_gt_name][
                self.freespace_gt_name
            ] = gt_bev_freespace

        data = self.dilate_small_object(data, False)

        if self.output_extra_obstacle:
            assert self.output_freespace is True
            gt_bev_freespace = data[self.freespace_gt_name][
                self.freespace_gt_name
            ]
            gt_bev_obstacle_extra = np.zeros_like(gt_bev_freespace)

            # 通过lidarseg语义补充限位器等障碍物
            gt_lidar_seg = gt[..., 0]
            gt_bev_obstacle_extra[gt_lidar_seg == 19] = 1

            data[self.freespace_gt_name][
                self.extra_obstacle_gt_name
            ] = gt_bev_obstacle_extra

        if self.output_lidardet:
            assert self.output_freespace is True
            lidar_det_veh_mask = np.zeros_like(gt[..., 2])
            lidar_det_veh_mask[gt[..., 2] == 50] = 1
            data[self.freespace_gt_name][
                self.lidardet_veh_mask_gt_name
            ] = lidar_det_veh_mask

        return data

    def get_freespace_gt_type(self, gt: np.ndarray) -> np.ndarray:
        """Get type of raw freespace gt, autolabel or mannual label.

        Get the type of raw freespace gt, i.e., mannual label or auto label.
        If gt is from mannual labeling,  it includes 3 channels such as
        mannual static anno, lidar seg, lidar det. If gt from auto labeling,
        it includes 3 channels such as lidar seg, elevation, lidar det.
        see detail in
        https://horizonrobotics.feishu.cn/docs/doccnDteHTj1EHdkm1aRhQyWf5c#
        For freepace auto label and mannual label format details, refer to
        https://horizonrobotics.feishu.cn/docx/HGZLdBCCuoId9DxJHyVcxd2anOb,
        https://horizonrobotics.feishu.cn/docs/doccnDteHTj1EHdkm1aRhQyWf5c,
        respectively.

        Args:
            gt_raw: 3 channels data from mannual labeling or auto
                labeling, described as above.

        Returns:
            bool, true for auto label while false for mannual label.
        """

        # gt可能来自自动标注或者人工标注, 如果来自自动标注, 第1个通道存储的是高度信息,
        # 如果来自人工标注数据，第1个通道存储的lidar seg的label.
        # 高度的编码方式为：int((h+bias)*ratio), 其中bias=1, ratio=50; 可以参考
        # class BEVElevationGenerator 中关于高度存储的详细说明.
        # 路面上总有高度>0的物体，所以高度信息编码的最大值>50; 而lidar seg的类别除掉
        # ignore_index=255后，最大label为38 < 50; 利用这个信息, 判断数据来自人工标注
        # 还是自动标注.

        gt_chann_1 = gt[..., 1].copy()
        gt_chann_1[gt_chann_1 == self.ignore_index] = 0
        seg_labels = set(self.freespace_auto_remap_dict.keys())
        seg_labels.discard(self.ignore_index)

        if gt_chann_1.max() > max(seg_labels):
            is_auto_label = True
        else:
            is_auto_label = False

        return is_auto_label

    def estimate_data_scene(
        self,
        gt_bev_freespace: np.ndarray,
        raw_gt_vcs_range: Sequence,
        is_auto_label: bool,
        depthwise_ignore_ratio_thresh: float = 0.5,
        front_depth_thresh: float = 72.4,
        far_ignore_ratio: float = 0.5,
    ) -> str:
        """Estimate data scene type from freespace gt.

        Args:
            gt_bev_freespace: 3 channel freespace gt.
            raw_gt_vcs_range: vcs range of raw gt.
            is_auto_label: gt type from auto label or mannual label.
            depthwise_ignore_ratio_thresh: ignore ratio between far
                and nearby region.
            front_depth_thresh: depth thresh to divide front region.
            far_ignore_ratio: ignore ratio for front region.
        Returns:
            scene type, "driving" or "parking".
        """
        if is_auto_label:
            seg = gt_bev_freespace[..., 0]
        else:
            seg = gt_bev_freespace[..., 1]

        vcs_origin_coord_v, _ = get_center_coord(raw_gt_vcs_range, seg.shape)

        vcs_front_depth_thresh_v = int(
            (raw_gt_vcs_range[2] - front_depth_thresh)
            / (raw_gt_vcs_range[2] - raw_gt_vcs_range[0])
            * seg.shape[0]
        )
        # Compatible with historical data versions
        assert (
            0 < front_depth_thresh <= 153.6
        ), "Max front vcs range of freespace gt is 153.6!"
        if front_depth_thresh >= raw_gt_vcs_range[2]:
            return "driving"  # strong related to historical freespace gt
        else:
            remote_front_region = seg[:vcs_front_depth_thresh_v, :] >= 0
            remote_front_region_wo_ignore = (
                seg[:vcs_front_depth_thresh_v, :] != self.ignore_index
            )
            close_front_region_wo_ignore = (
                seg[vcs_front_depth_thresh_v:vcs_origin_coord_v, :]
                != self.ignore_index
            )
            if (
                remote_front_region_wo_ignore.sum()
                / close_front_region_wo_ignore.sum()
                < depthwise_ignore_ratio_thresh
                and remote_front_region_wo_ignore.sum()
                / remote_front_region.sum()
                < far_ignore_ratio
            ):
                return "parking"
            else:
                return "driving"

    def process_freespace(
        self, gt: np.ndarray, is_auto_label: bool, scene_type: str
    ) -> np.ndarray:
        """Process raw gt to 0-1 class for bev freespace.

        .. note:: Generate 0-1 freespace gt from mannual labeling or
            auto labeling.
            If gt is from mannual labeling,  it includes 3 channels such as
            mannual static anno, lidar seg, lidar det. If gt from auto
            labeling, it includes 3 channels such as lidar seg, elevation,
            lidar det. see detail in
            https://horizonrobotics.feishu.cn/docs/doccnDteHTj1EHdkm1aRhQyWf5c#
            For freepace auto label and mannual label format details, refer to
            https://horizonrobotics.feishu.cn/docx/HGZLdBCCuoId9DxJHyVcxd2anOb,
            https://horizonrobotics.feishu.cn/docs/doccnDteHTj1EHdkm1aRhQyWf5c,
            respectively.

        Args:
            gt: 3 channels data from mannual labeling or auto
                labeling, described as above.
            is_auto_label: gt type from auto label or mannual label.

        Returns:
            processed 0-1 gt for bev freespace.
        """

        if is_auto_label:
            data = self.parse_freespace_from_autolabel(gt, scene_type)
        else:
            data = self.parse_freespace_from_mannuallabel(gt)

        for key, value in data.items():
            if isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    sub_value = cv2.resize(
                        sub_value,
                        self.target_size[::-1],
                        interpolation=cv2.INTER_NEAREST,
                    ).astype(np.int64)
                    data[key][sub_key] = sub_value
            else:
                value = cv2.resize(
                    value,
                    self.target_size[::-1],
                    interpolation=cv2.INTER_NEAREST,
                )
                data[key] = value.astype(np.int64)

        if self.output_bev_weight_map:
            if not hasattr(self, "bev_weight_map"):
                self.bev_weight_map = generate_bev_weight_map(
                    self.target_size, self.vcs_range
                )
            data[self.freespace_gt_name][
                "bev_weight_map"
            ] = self.bev_weight_map

        return data

    def __call__(self, data):
        assert "gt_bev_freespace_raw" in data

        gt_bev_freespace = np.array(data["gt_bev_freespace_raw"])

        # get freespace gt type from raw saved gt
        is_auto_label = self.get_freespace_gt_type(gt_bev_freespace)

        raw_gt_res = gt_bev_freespace.shape[:2]
        assert raw_gt_res in self.raw_gt_res_vcsrange_cfg
        raw_gt_vcs_range = self.raw_gt_res_vcsrange_cfg[raw_gt_res]
        scene_type = self.estimate_data_scene(
            gt_bev_freespace, raw_gt_vcs_range, is_auto_label
        )
        gt_bev_freespace, vcs_range_bbox = get_vcsrange_bbox_data(
            gt_bev_freespace,
            self.vcs_range,
            raw_gt_vcs_range,
            self.ignore_index,
        )
        gt_bev_freespace = crop_roi_vcs_range(
            gt_bev_freespace, self.vcs_range, vcs_range_bbox
        )

        output = self.process_freespace(
            gt_bev_freespace, is_auto_label, scene_type
        )

        if self.use_extra_mask:
            assert (
                "bev_freespace_mask" in data
            ), "Need extra mask for bev_freespace"
            bev_freespace_gt = output[self.freespace_gt_name][
                self.freespace_gt_name
            ]
            bev_freespace_mask = np.array(data["bev_freespace_mask"])
            bev_freespace_mask, vcs_range_bbox = get_vcsrange_bbox_data(
                bev_freespace_mask,
                self.vcs_range,
                self.raw_gt_res_vcsrange_cfg[bev_freespace_mask.shape],
                0,
            )
            bev_freespace_mask = crop_roi_vcs_range(
                bev_freespace_mask, self.vcs_range, vcs_range_bbox
            )
            bev_freespace_gt[bev_freespace_mask == 1] = self.ignore_index
            output[self.freespace_gt_name][
                self.freespace_gt_name
            ] = bev_freespace_gt

        for key, value in output.items():
            data[key] = value

        return data

    def __repr__(self):
        return "ANCBEVFreespaceGenerator"
