# Copyright (c) Horizon Robotics. All rights reserved.

import math
from typing import Mapping, Optional, Sequence, Union

import numpy as np
import torch

from hat.data.transforms.auto_3dv import (
    PIL_INTERP_CODES,
    ANCApplyMaskOnImg,
    ANCBev3dTargetGenerator,
    ANCCrop3DV,
    ANCMultiViewTargetGenerator,
    ANCPrepareTempoDataBEV,
    ANCResize3DV,
    ANCToTensor3DV,
    get_ctoff_map,
)
from hat.data.transforms.real3d import (
    draw_heatmap,
    get_gaussian2D,
    get_reg_map,
)
from hat.registry import OBJECT_REGISTRY
from hat.utils.package_helper import require_packages

__all__ = [
    "ANCTemporalResize3DV",
    "ANCTemporalCrop3DV",
    "ANCToTensorTemporal3DV",
    "ANCTemporalBev3dTargetGenerator",
    "ANCPrepareTempoALLFrameDataBEV",
    "ANCTemporalMultiViewTargetGenerator",
    "ANCTemporalApplyMaskOnImg",
]


@OBJECT_REGISTRY.register
class ANCTemporalResize3DV(ANCResize3DV):
    """Resize PIL Images for all frame in a temporal clips.

    Args:
        size: Desired output size. If size is a sequence like
            (h, w), output size will be matched to this.
        interpolation:Desired interpolation. Default is 'nearest'.
        resize_depth: whether resize gt depth.
    """

    @require_packages("torchvision")
    def __init__(
        self,
        size: Union[Sequence, Sequence[Sequence]],
        interpolation: str = "nearest",
        resize_depth: bool = True,
    ):
        super().__init__(size, interpolation, resize_depth)

    def __call__(self, data: Mapping):
        gt_multi_view_values = data.pop("gt_multi_view", None)
        super().__call__(data)
        if gt_multi_view_values is not None:
            data["gt_multi_view"] = gt_multi_view_values
        if "gt_multi_view" in data and data["gt_multi_view"]:
            if data["gt_multi_view"] and isinstance(
                data["gt_multi_view"][0], Sequence
            ):
                # 时序长度大于1，处理该时序所有ignore_mask_2d
                for gt_multi_view in data["gt_multi_view"]:
                    for idx, multi_view_info in enumerate(gt_multi_view):
                        if multi_view_info:
                            ignore_mask_2d = multi_view_info["meta"][
                                "ignore_mask"
                            ]["ignore_mask_2d"]
                            ignore_mask_2d = self._resize(
                                ignore_mask_2d,
                                self.size[idx],
                                PIL_INTERP_CODES["nearest"],
                            )
                            multi_view_info["meta"]["ignore_mask"][
                                "ignore_mask_2d"
                            ] = ignore_mask_2d
            else:
                for idx, multi_view_info in enumerate(data["gt_multi_view"]):
                    if multi_view_info:
                        ignore_mask_2d = multi_view_info["meta"][
                            "ignore_mask"
                        ]["ignore_mask_2d"]
                        ignore_mask_2d = self._resize(
                            ignore_mask_2d,
                            self.size[idx],
                            PIL_INTERP_CODES["nearest"],
                        )
                        multi_view_info["meta"]["ignore_mask"][
                            "ignore_mask_2d"
                        ] = ignore_mask_2d

        return data

    def __repr__(self):
        return "TemporalResize3DV"


@OBJECT_REGISTRY.register
class ANCTemporalCrop3DV(ANCCrop3DV):  # noqa: D205,D400
    """Crop the of images in a tmpooral clips. \

        The image can be a PIL Image or a Tensor,
        in which case it is expected to have [..., H, W] shape,
        where ... means an arbitrary number of leading dimensions.

    Args:
        heights: Height of the crop boxs in each view.
        widths: Width of the crop boxs in each view.
        top: Vertical component of the top left corner \
            of the crop box in each view. Setting None means random crop.
        left: Horizontal component of the top left corner \
            of the crop box in each view. Setting None means random crop.
    """

    @require_packages("torchvision")
    def __init__(
        self,
        height: Union[Sequence, int],
        width: Union[Sequence, int],
        top: Union[Sequence, int],
        left: Union[Sequence, int],
    ):
        super().__init__(height, width, top, left)

    def __call__(self, data: Mapping):
        assert "pil_imgs" in data, 'input data must has "pil_imgs"'

        nums = len(data["size"])
        assert (
            len(data["pil_imgs"][0])
            == len(self.top)
            == len(self.left)
            == len(self.height)
            == len(self.width)
        )

        for i in range(nums):
            assert data["size"][i][0] >= self.height[i]
            assert data["size"][i][1] >= self.width[i]

        # generate a random integer if self.top is None
        top = [
            np.random.randint(data["size"][i][0] - self.height[i] + 1)
            if self.top[i] is None
            else self.top[i]
            for i in range(nums)
        ]
        left = [
            np.random.randint(data["size"][i][1] - self.width[i] + 1)
            if self.left[i] is None
            else self.left[i]
            for i in range(nums)
        ]

        data["pil_imgs"] = [
            self._crop(pil_img, top, left, self.height, self.width)
            for pil_img in data["pil_imgs"]
        ]

        if "imgs" in data:
            data["imgs"] = [
                self._crop(pil_img, top, left, self.height, self.width)
                for pil_img in data["imgs"]
            ]

        if "gt_seg" in data:
            data["gt_seg"] = [
                self._crop(gt_seg, top, left, self.height, self.width)
                for gt_seg in data["gt_seg"]
            ]

        if "obj_mask" in data:
            data["obj_mask"] = self._crop(
                data["obj_mask"],
                top[0],
                left[0],
                self.height[0],
                self.width[0],
            )

        if "gt_depth" in data:
            for i in range(len(data["gt_depth"])):
                depth_w, depth_h = data["gt_depth"][i].size
                scale_h, scale_w = (
                    depth_h // data["size"][i][0],
                    depth_w // data["size"][i][1],
                )
                data["gt_depth"][i] = self._crop(
                    data["gt_depth"][i],
                    top[i] * scale_h,
                    left[i] * scale_w,
                    self.height[i] * scale_h,
                    self.width[i] * scale_w,
                )

        if "color_imgs" in data:
            data["color_imgs"] = [
                self._crop(color_img, top, left, self.height, self.width)
                for color_img in data["color_imgs"]
            ]

        if "front_mask" in data:
            data["front_mask"] = self._crop(
                data["front_mask"],
                top[0],
                left[0],
                self.height[0],
                self.width[0],
            )

        if "intrinsics" in data:
            # modify intrinsics matrix
            data["intrinsics"][0, 2] -= left[0]
            data["intrinsics"][1, 2] -= top[0]
        if "gt_multi_view" in data and data["gt_multi_view"]:
            if data["gt_multi_view"] and isinstance(
                data["gt_multi_view"][0], Sequence
            ):
                # 时序长度大于1，处理该时序所有ignore_mask_2d
                for gt_multi_view in data["gt_multi_view"]:
                    for idx, multi_view_info in enumerate(gt_multi_view):
                        if multi_view_info:
                            ignore_mask_2d = multi_view_info["meta"][
                                "ignore_mask"
                            ]["ignore_mask_2d"]
                            ignore_mask_2d = self._crop(
                                ignore_mask_2d,
                                top[idx],
                                left[idx],
                                self.height[idx],
                                self.width[idx],
                            )
                            multi_view_info["meta"]["ignore_mask"][
                                "ignore_mask_2d"
                            ] = ignore_mask_2d
            else:
                for idx, multi_view_info in enumerate(data["gt_multi_view"]):
                    if multi_view_info:
                        ignore_mask_2d = multi_view_info["meta"][
                            "ignore_mask"
                        ]["ignore_mask_2d"]
                        ignore_mask_2d = self._crop(
                            ignore_mask_2d,
                            top[idx],
                            left[idx],
                            self.height[idx],
                            self.width[idx],
                        )
                        multi_view_info["meta"]["ignore_mask"][
                            "ignore_mask_2d"
                        ] = ignore_mask_2d

        return data

    def __repr__(self):
        return "TemporalCrop3DV"


@OBJECT_REGISTRY.register
class ANCToTensorTemporal3DV(ANCToTensor3DV):
    """Convert a ``PIL Image`` or ``numpy.ndarray`` to torch.tensor.

    Args:
        with_color_imgs: whether return original color imags.
    """

    @require_packages("torchvision")
    def __init__(
        self,
        with_color_imgs: bool = True,
    ):
        super().__init__(with_color_imgs)

    def __call__(self, data: Mapping):
        gt_multi_view_values = data.pop("gt_multi_view", None)
        super().__call__(data)
        if gt_multi_view_values is not None:
            data["gt_multi_view"] = gt_multi_view_values
        if "gt_multi_view" in data and data["gt_multi_view"]:
            if data["gt_multi_view"] and isinstance(
                data["gt_multi_view"][0], Sequence
            ):
                # 时序长度大于1，处理该时序所有ignore_mask_2d
                for gt_multi_view in data["gt_multi_view"]:
                    for multi_view_info in gt_multi_view:
                        if multi_view_info:
                            ignore_mask_2d = multi_view_info["meta"][
                                "ignore_mask"
                            ]["ignore_mask_2d"]
                            ignore_mask_2d = self._map_to_tensor(
                                ignore_mask_2d
                            )
                            ignore_mask_2d = ignore_mask_2d.squeeze(0)
                            ignore_mask_2d = ignore_mask_2d > 0
                            multi_view_info["meta"]["ignore_mask"][
                                "ignore_mask_2d"
                            ] = ignore_mask_2d
            else:
                for multi_view_info in data["gt_multi_view"]:
                    if multi_view_info:
                        ignore_mask_2d = multi_view_info["meta"][
                            "ignore_mask"
                        ]["ignore_mask_2d"]
                        ignore_mask_2d = self._map_to_tensor(ignore_mask_2d)
                        ignore_mask_2d = ignore_mask_2d.squeeze(0)
                        ignore_mask_2d = ignore_mask_2d > 0
                        multi_view_info["meta"]["ignore_mask"][
                            "ignore_mask_2d"
                        ] = ignore_mask_2d
        return data

    def __repr__(self):
        return "TensorTemporal3DV"


@OBJECT_REGISTRY.register
class ANCTemporalBev3dTargetGenerator(ANCBev3dTargetGenerator):
    """Generate gound truth labels for temporal bev_3d.

    Args:
        **kwargs: see :py:class:`ANCBev3dTargetGenerator`
        use_trackid_attribute (bool): return track id.
        reverse_anno: reverse the anno.
    """

    def __init__(
        self,
        num_classes: int,
        bev_size: Sequence[int],
        vcs_range: Sequence[float],
        cls_hm_kernel: Mapping,
        category2id_map: Mapping,
        cls_dimension: np.ndarray = None,
        max_objs: int = 100,
        enable_ignore: bool = True,
        filter_vcs_range: Sequence[float] = None,
        ego_ignore_range: Sequence[float] = None,
        use_category_decouple: bool = False,
        ignore_obj_cls: bool = False,
        roi_label_seq: Sequence[str] = (),
        enable_roi_label_seq: Sequence[str] = (),
        use_occlusion_attribute: bool = False,
        occlusion_attribute_seq: Sequence[str] = (),
        occlusion_ignore_id: float = -99.0,
        category_class_weight: Optional[Mapping] = None,
        length_limitation: Optional[Mapping] = None,
        bigobj_hm_kernel_cfg: Optional[Mapping] = None,
        use_psc_rot: bool = False,
        N_steps_PSC_rot: int = None,
        use_dual_freq: bool = False,
        background_reweight_cfg: Optional[Mapping] = None,
        gt_name: str = "gt_bev_3d",
        anno_name: str = "annos_bev_3d",
        roi_background_weight_cfg: Optional[Mapping] = None,
        roi_weight_cfg: Optional[Mapping] = None,
        bigobj_length_thresh_cfg: Optional[Mapping] = None,
        use_trackid_attribute: Optional[bool] = False,
        reverse_anno: Optional[bool] = False,
    ):
        super().__init__(
            num_classes,
            bev_size,
            vcs_range,
            cls_hm_kernel,
            category2id_map,
            cls_dimension,
            max_objs,
            enable_ignore,
            filter_vcs_range,
            ego_ignore_range,
            use_category_decouple,
            ignore_obj_cls,
            roi_label_seq,
            enable_roi_label_seq,
            use_occlusion_attribute,
            occlusion_attribute_seq,
            occlusion_ignore_id,
            category_class_weight,
            length_limitation,
            bigobj_hm_kernel_cfg,
            use_psc_rot,
            N_steps_PSC_rot,
            use_dual_freq,
            background_reweight_cfg,
            gt_name,
            anno_name,
            roi_background_weight_cfg,
            roi_weight_cfg,
            bigobj_length_thresh_cfg,
        )
        self.use_trackid_attribute = use_trackid_attribute
        self.reverse_anno = reverse_anno

    def gt(self, annotations, gt_multi_view, object_tag_info=None):
        # build bev3d gt heatmaps
        # Now(20230303), bev3d_hm have three roles,
        # 1. When enable_vehicle_cls is False, the model don't
        # output vehicle category info, bev_hm(1-c) takes the
        # role on distinguishing the fore/background.
        # When enable_vehicle_cls is True, three are two schemes
        # to judge vehicle category(classification task), so:
        # 2. use_category_decouple is False, bev3d_hm(n-c, n is
        # the num of vehicle category) distinguish the
        # back/foreground and judge the vehicle category. By the way,
        # when all scores of n category are lower than threshshold,
        # it is background.
        # 3. use_category_decouple is True, the model output a new
        # tensor named ben3d_cls_hm(n-c), bev3d_hm(1-c) only
        # distinguish the fore/background, and bev3d_cls_hm judge
        # the vehicle category. we name this scheme as <category_decouple>.
        if self.use_category_decouple:
            bev3d_hm = np.zeros((*self.bev_size, 1), dtype=np.float32)
            bev3d_cls_hm = np.zeros(
                (*self.bev_size, self.num_classes), dtype=np.float32
            )
        else:
            bev3d_hm = np.zeros(
                (*self.bev_size, self.num_classes), dtype=np.float32
            )
        if self.use_occlusion_attribute:
            if gt_multi_view:
                multi_view_infos_exist_flag = True
                occlusion_multi_view = gt_multi_view["occlusion_multi_view"]
                occlusion_id_map = (
                    np.ones(self.bev_size, dtype=np.float32)
                    * self.occlusion_ignore_id
                )
            else:
                multi_view_infos_exist_flag = False
            occlusion_num_classes = len(self.occlusion_attribute_seq)
            occlusion_cls_hm = np.zeros(
                (*self.bev_size, occlusion_num_classes), dtype=np.float32
            )
        class_id_map = np.ones(self.bev_size, dtype=np.float32) * -99
        valid_rot_reweight_mask = np.zeros(self.bev_size, dtype=np.bool)
        category_class_weight_hm = np.ones(self.bev_size, dtype=np.float32)
        roi_weight_hm = np.ones(self.bev_size, dtype=np.float32)
        bev3d_roi_weight_hm = np.ones(self.bev_size, dtype=np.float32)
        bev3d_dim = np.zeros((*self.bev_size, 3), dtype=np.float32)  # h, w, l
        bev3d_rot = np.zeros(
            (*self.bev_size, self.rot_encode_size), dtype=np.float32
        )

        bev3d_ct_offset = np.zeros(
            (*self.bev_size, 2), dtype=np.float32
        )  # bev center offest
        bev3d_loc_z = np.zeros((self.bev_size), dtype=np.float32)
        bev3d_weight_hm = np.zeros((self.bev_size), dtype=np.float32)
        bev3d_point_pos_mask = np.zeros((self.bev_size), dtype=np.float32)
        bev3d_ignore_mask = np.zeros((self.bev_size), dtype=np.float32)
        bev3d_ignore_obj_cls = np.array([self.ignore_obj_cls], dtype=np.bool)
        background_reweight_hm = np.ones(self.bev_size, dtype=np.float32)
        background_gaussian_hm = np.zeros((self.bev_size), dtype=np.float32)
        bev3d_rot_reweight_mask = np.ones((self.bev_size), dtype=np.float32)
        bev3d_length_map = np.ones((self.bev_size), dtype=np.float32)
        bev3d_aspect_ratio_map = np.ones((self.bev_size), dtype=np.float32)

        # used for eval
        vcs_loc_ = np.zeros((self.max_objs, 3), dtype=np.float32)
        vcs_dim_ = np.zeros((self.max_objs, 3), dtype=np.float32)
        vcs_rot_z_ = np.zeros((self.max_objs), dtype=np.float32)
        vcs_cls_ = np.zeros((self.max_objs), dtype=np.float32) - 99
        vcs_ignore_ = np.zeros((self.max_objs), dtype=np.bool_)
        vcs_visible_ = np.zeros((self.max_objs), dtype=np.float32)
        if self.use_occlusion_attribute:
            vcs_occlusion_ = np.zeros((self.max_objs), dtype=np.float32) - 99
        if self.use_trackid_attribute:
            obj_idxes = np.zeros((self.max_objs), dtype=np.int64) - 99
        object_tags = {}
        if object_tag_info is not None:
            for object_tag in object_tag_info:
                object_tags[object_tag] = np.zeros(
                    (self.max_objs), dtype=np.bool_
                )
        # In our train/val data, tow bbox3ds having different category
        # maybe overlap heavily. This will affect the training process,
        # so we filter the overlapped bbox3d.
        # {overlap_thresh} is a hyper-param, for:
        # bbox3d1(x1, y1, z1), bbox3d2(x2, y2, z2),
        # when [abs(x1 - x2) + abs(y1 - y2)] <  {overlap_thresh},
        # they will be set as ignore, not involved in training.
        overlap_thresh = 0.1
        for i in range(len(annotations)):
            for j in range(i + 1, len(annotations)):
                loci = np.array(annotations[i]["location"][:2])
                locj = np.array(annotations[j]["location"][:2])
                loc_diff = np.abs(loci - locj).sum()
                if loc_diff <= overlap_thresh:
                    annotations[i]["ignore"] = True
                    annotations[j]["ignore"] = True

        count_id = 0
        for ann_idx, anno in enumerate(annotations):  # noqa [B007]
            if count_id > (self.max_objs):
                break
            label = anno["label"]
            # 2D detailed classification label in data from
            # 4D GT link production
            need_background_reweight = False
            if (
                label.lower() in self.background_reweight_cfg
                and not anno["ignore"]
            ):
                need_background_reweight = True
            if (
                self.enable_roi_label_seq
                and label in self.enable_roi_label_seq
                and self.category2id_map.get(label, -99) > -99
            ):
                roi_label = anno.get("roi_2d_type_label", None)
                if roi_label is not None:
                    if roi_label in self.roi_label_seq:
                        label = roi_label
                    else:
                        anno["ignore"] = True
            # The classification of occlusion attributes are obtained
            # from multi view lmdb, while annos are recorded in bev3d lmdb.
            # So use uid to match the two.
            if self.use_occlusion_attribute and multi_view_infos_exist_flag:
                uid = anno["track_id"]
                if uid in occlusion_multi_view:
                    occlusion_id = occlusion_multi_view[uid]
                else:
                    occlusion_id = occlusion_num_classes - 1
                    anno["ignore"] = True
            cls_id = int(self.category2id_map.get(label, -99))
            if (cls_id <= -99 and not need_background_reweight) or (
                not self.enable_ignore and anno.get("ignore", False)
            ):
                continue
            vcs_loc = anno["location"]
            vcs_length = anno["dimension"][-1]
            vcs_width = anno["dimension"][1]
            if self.filter_vcs_range:
                if not (
                    self.filter_vcs_range[0]
                    < vcs_loc[0]
                    < self.filter_vcs_range[2]
                    and self.filter_vcs_range[1]
                    < vcs_loc[1]
                    < self.filter_vcs_range[3]
                ):
                    continue

            if self.ego_ignore_range:
                if (
                    self.ego_ignore_range[0]
                    < vcs_loc[0]
                    < self.ego_ignore_range[2]
                    and self.ego_ignore_range[1]
                    < vcs_loc[1]
                    < self.ego_ignore_range[3]
                ):
                    continue

            bev_ct = (
                ((self.vcs_range[3] - vcs_loc[1]) / self.m_perpixel[1]),
                ((self.vcs_range[2] - vcs_loc[0]) / self.m_perpixel[0]),
            )  # (u, v)
            bev_ct_int = (int(bev_ct[0]), int(bev_ct[1]))
            if (
                0 <= bev_ct_int[0] < self.bev_size[1]
                and 0 <= bev_ct_int[1] < self.bev_size[0]
            ):
                if self.background_reweight_cfg:
                    (
                        background_gaussian_hm,
                        background_reweight_hm,
                    ) = self.reweight_background(
                        need_background_reweight,
                        label.lower(),
                        cls_id,
                        bev_ct_int,
                        background_gaussian_hm,
                        background_reweight_hm,
                    )
                # For one 3d bbox of background category,
                # it don't need to generate foreground target.
                if need_background_reweight:
                    continue

                length_limitation = self.length_limitation.get(
                    cls_id, (0.0, np.inf)
                )
                if not (
                    length_limitation[0]
                    < anno["dimension"][-1]
                    < length_limitation[1]
                ):
                    anno["ignore"] = True
                vcs_loc_[count_id] = vcs_loc
                vcs_cls_[count_id] = cls_id
                vcs_dim_[count_id] = anno["dimension"]
                if self.use_trackid_attribute:
                    obj_idxes[count_id] = anno["track_id"]
                # transfer the yaw to value [-pi,pi]
                vcs_yaw = np.arctan2(np.sin(anno["yaw"]), np.cos(anno["yaw"]))
                vcs_rot_z_[count_id] = vcs_yaw
                if object_tag_info is not None:
                    for object_tag in object_tag_info:
                        if (
                            "track_id" in anno
                            and anno["track_id"] in object_tag_info[object_tag]
                        ):
                            object_tags[object_tag][count_id] = 1
                if anno.get("ignore", False):
                    vcs_ignore_[count_id] = 1
                # Visibility reflects the occlusion degree of the target
                if anno.get("visibility", False):
                    vcs_visible_[count_id] = anno["visibility"]
                if (
                    self.use_occlusion_attribute
                    and multi_view_infos_exist_flag
                ):
                    vcs_occlusion_[count_id] = occlusion_id

                # draw bev3d heatmap
                kernel_size = self.cls2kernel[cls_id]
                # ugly code, optimize below code seg in the future
                # TODO @zihao.lu
                if self.bigobj_hm_kernel_cfg is not None:
                    big_object_length_thresh = self.bigobj_hm_kernel_cfg[
                        "length_thresh"
                    ]
                    big_object_kernel_size = self.bigobj_hm_kernel_cfg[
                        "kernel_size"
                    ]
                    if anno["dimension"][-1] > big_object_length_thresh:
                        kernel_size = np.array(
                            [
                                big_object_kernel_size,
                                big_object_kernel_size,
                            ],
                            dtype=np.float32,
                        )

                insert_bev_hm = get_gaussian2D(kernel_size, alpha=1)
                insert_bev_hm_wh = insert_bev_hm.shape[:2][::-1]
                if anno.get("ignore", False):
                    insert_ignore_mask = np.ones_like(insert_bev_hm)
                else:
                    insert_ignore_mask = np.zeros_like(insert_bev_hm)
                if self.cls_dimension is not None:
                    ori_dim = np.array(anno["dimension"])
                    avg_dim = self.cls_dimension[cls_id]
                    residual_dim = np.log(abs(ori_dim) / avg_dim)
                    ann_dim = list(residual_dim)
                else:
                    ann_dim = anno["dimension"]

                if self.use_psc_rot:
                    phase_shift_targets = tuple(
                        np.cos(vcs_yaw + 2 * np.pi * x / self.N_steps_PSC_rot)
                        for x in range(self.N_steps_PSC_rot)
                    )
                    if self.use_dual_freq:
                        raise NotImplementedError
                else:
                    phase_shift_targets = (np.cos(vcs_yaw), np.sin(vcs_yaw))

                insert_bev_reg_map_list = [
                    get_reg_map(insert_bev_hm_wh, ann_dim),
                    get_reg_map(
                        insert_bev_hm_wh,
                        phase_shift_targets,
                    ),
                    get_reg_map(insert_bev_hm_wh, vcs_loc[-1]),
                    get_ctoff_map(insert_bev_hm_wh, bev_ct),
                ]
                bev_reg_map_list = [
                    bev3d_dim,
                    bev3d_rot,
                    bev3d_loc_z,
                    bev3d_ct_offset,
                ]
                insert_bev_reg_map_list.append(
                    get_reg_map(insert_bev_hm_wh, cls_id)
                )
                bev_reg_map_list.append(class_id_map)

                # Variant of length/width ratio to highlight bigObj
                aspect_ratio = (
                    vcs_length / vcs_width + vcs_width / vcs_length
                ) / 2
                if not math.isnan(aspect_ratio):
                    insert_bev_reg_map_list.append(
                        get_reg_map(insert_bev_hm_wh, vcs_length)
                    )
                    bev_reg_map_list.append(bev3d_length_map)
                    insert_bev_reg_map_list.append(
                        get_reg_map(insert_bev_hm_wh, aspect_ratio)
                    )
                    bev_reg_map_list.append(bev3d_aspect_ratio_map)

                insert_bev_reg_map_list.append(
                    get_reg_map(
                        insert_bev_hm_wh,
                        self.category_class_weight.get(cls_id, 1.0),
                    )
                )
                bev_reg_map_list.append(category_class_weight_hm)

                if self.use_category_decouple:
                    draw_heatmap(bev3d_hm[:, :, 0], insert_bev_hm, bev_ct_int)
                else:
                    draw_heatmap(
                        bev3d_hm[:, :, cls_id], insert_bev_hm, bev_ct_int
                    )
                if (
                    self.use_occlusion_attribute
                    and multi_view_infos_exist_flag
                ):
                    insert_bev_reg_map_list.append(
                        get_reg_map(insert_bev_hm_wh, occlusion_id)
                    )
                    bev_reg_map_list.append(occlusion_id_map)

                draw_heatmap(
                    bev3d_weight_hm,
                    insert_bev_hm,
                    bev_ct_int,
                    bev_reg_map_list,
                    insert_bev_reg_map_list,
                )
                draw_heatmap(bev3d_ignore_mask, insert_ignore_mask, bev_ct_int)

                bev3d_point_pos_mask[bev_ct_int[1], bev_ct_int[0]] = 1
                count_id += 1

        if self.roi_background_weight_cfg:
            roi_weight_hm = self.finetune_roi_background_weight(
                roi_weight_hm, bev3d_weight_hm
            )
        if self.roi_weight_cfg:
            bev3d_roi_weight_hm = self.finetune_roi_weight(
                bev3d_roi_weight_hm, bev3d_length_map
            )
        bev3d_background_weight = np.maximum(
            roi_weight_hm, background_reweight_hm
        )
        one_hot_pool = np.eye(self.num_classes)
        valid_class_mask = class_id_map >= 0
        valid_class_id = class_id_map[valid_class_mask].astype(np.int32)
        if self.use_category_decouple:
            bev3d_cls_hm[valid_class_mask] = one_hot_pool[valid_class_id]
        else:
            bev3d_hm[valid_class_mask] *= one_hot_pool[valid_class_id]
        if self.use_occlusion_attribute and multi_view_infos_exist_flag:
            occlusion_one_hot_pool = np.eye(occlusion_num_classes)
            valid_occlusion_cls_mask = occlusion_id_map >= 0
            valid_occlusion_id = occlusion_id_map[
                valid_occlusion_cls_mask
            ].astype(np.int32)
            occlusion_cls_hm[
                valid_occlusion_cls_mask
            ] = occlusion_one_hot_pool[valid_occlusion_id]

        if self.bigobj_length_thresh_cfg:
            for (
                class_id,
                length_thresh,
            ) in self.bigobj_length_thresh_cfg.items():
                # get bigObjs based on length_thresh for rot reweight
                valid_rot_reweight_mask = np.logical_or(
                    valid_rot_reweight_mask,
                    np.logical_and(
                        class_id_map == class_id,
                        bev3d_length_map >= length_thresh,
                    ),
                )
                if (
                    self.bigobj_hm_kernel_cfg is not None
                    and self.bigobj_hm_kernel_cfg.get("rot_reweight", False)
                ):
                    valid_rot_reweight_mask = np.logical_or(
                        valid_rot_reweight_mask,
                        bev3d_length_map
                        >= self.bigobj_hm_kernel_cfg["length_thresh"],
                    )
            bev3d_rot_reweight_mask[
                valid_rot_reweight_mask
            ] = bev3d_aspect_ratio_map[valid_rot_reweight_mask]

        gt_bev_3d = {
            "bev3d_hm": bev3d_hm,
            "bev3d_dim": bev3d_dim,
            "bev3d_rot": bev3d_rot,
            "bev3d_ct_offset": bev3d_ct_offset,
            "bev3d_loc_z": bev3d_loc_z[:, :, np.newaxis],
            "bev3d_weight_hm": bev3d_weight_hm[:, :, np.newaxis],
            "bev3d_point_pos_mask": bev3d_point_pos_mask[:, :, np.newaxis],
            "bev3d_ignore_mask": bev3d_ignore_mask[:, :, np.newaxis],
            "bev3d_ignore_obj_cls": bev3d_ignore_obj_cls,
            "bev3d_category_class_weight": category_class_weight_hm[
                :, :, np.newaxis
            ],
            "bev3d_background_weight": bev3d_background_weight[
                :, :, np.newaxis
            ],
            "bev3d_roi_weight": bev3d_roi_weight_hm[:, :, np.newaxis],
            "bev3d_rot_reweight_mask": bev3d_rot_reweight_mask[
                :, :, np.newaxis
            ],
        }
        if self.use_category_decouple:
            gt_bev_3d["bev3d_cls_hm"] = bev3d_cls_hm
        annos_bev_3d = {
            "vcs_loc_": vcs_loc_,
            "vcs_cls_": vcs_cls_,
            "vcs_rot_z_": vcs_rot_z_,
            "vcs_dim_": vcs_dim_,
            "vcs_ignore_": vcs_ignore_,
            "vcs_visible_": vcs_visible_,
        }
        if self.use_trackid_attribute:
            annos_bev_3d["obj_idxes"] = obj_idxes
        if self.use_occlusion_attribute:
            gt_bev_3d["bev3d_occlusion_hm"] = occlusion_cls_hm
            gt_bev_3d["bev3d_ignore_occlusion"] = np.array(
                [not multi_view_infos_exist_flag], dtype=np.bool
            )
            annos_bev_3d["vcs_occlusion_"] = vcs_occlusion_
        for object_tag in object_tags:
            annos_bev_3d["tag_" + object_tag] = object_tags[object_tag]
        return gt_bev_3d, annos_bev_3d

    def __call__(self, data: dict) -> dict:
        """Generate bev_3d labels.

        Args:
            data: The dict contains at leaset annotations

        """
        assert "gt_bev_dynamic_anno" in data
        if (
            isinstance(data["gt_bev_dynamic_anno"], Sequence)
            and data["gt_bev_dynamic_anno"]
            and isinstance(data["gt_bev_dynamic_anno"][0], Sequence)
        ):
            # 时序长度大于1，获取该时序所有帧的GT和anno
            gt_bev_3d = []
            annos_bev_3d = []
            for idx, annotations in enumerate(data["gt_bev_dynamic_anno"]):
                gt_bev_3d_i, annos_bev_3d_i = self.gt(
                    annotations,
                    data["gt_multi_view"][idx],
                    data.get(
                        "object_tag_info",
                        [None] * len(data["gt_bev_dynamic_anno"]),
                    )[idx],
                )
                gt_bev_3d.append(gt_bev_3d_i)
                annos_bev_3d.append(annos_bev_3d_i)
            if self.reverse_anno:
                gt_bev_3d.reverse()
                annos_bev_3d.reverse()
        else:
            annotations = data["gt_bev_dynamic_anno"]
            gt_bev_3d, annos_bev_3d = self.gt(
                annotations,
                data["gt_multi_view"],
                data.get("object_tag_info", None),
            )
        data[self.gt_name] = gt_bev_3d
        data[self.anno_name] = annos_bev_3d
        return data

    def __repr__(self):
        return "TemporalBev3dTargetGenerator"


@OBJECT_REGISTRY.register
class ANCPrepareTempoALLFrameDataBEV(ANCPrepareTempoDataBEV):
    """Build input temporal data with all GT for BEV task.

    NOTE: PrepareTempoDataBEV used for BEV temporal task,
    e.g. bev_motion, bev_multitask contains (bev_motion
    bev_seg and bev_3d. etc.).

    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def _prepare_common_data(self, data: Mapping):
        data.pop("imgs")
        data.pop("object_tag_info", None)
        bevseg_gt_names = ["gt_bev_seg", "gt_bev_seg_small"]
        for bevseg_gt_name in bevseg_gt_names:
            if bevseg_gt_name in data:
                data[bevseg_gt_name] = data[bevseg_gt_name].long()

        if "occlusion" in data:
            data["occlusion"] = data["occlusion"].long()

        bev3d_rm_names = [
            "gt_bev_dynamic_anno",
            "gt_multi_view",
            "image_id",
            "image_name",
        ]
        bev3d_gt_group_name = [
            ["gt_bev_3d", "annos_bev_3d"],
            ["gt_bev_3d_small", "annos_bev_3d_small"],
        ]
        for one_group_name in bev3d_gt_group_name:
            gt_name, anno_name = one_group_name
            if gt_name in data:
                for bev3d_rm_name in bev3d_rm_names:
                    if bev3d_rm_name in data:
                        if bev3d_rm_name == "gt_multi_view":
                            if isinstance(data["gt_multi_view"], Sequence):
                                for gt_multi_view in data["gt_multi_view"]:
                                    gt_multi_view.pop(
                                        "occlusion_multi_view", None
                                    )
                            else:
                                data[bev3d_rm_name].pop(
                                    "occlusion_multi_view", None
                                )
                        else:
                            data.pop(bev3d_rm_name, None)
                if isinstance(data[gt_name], Sequence):
                    # 返回时序所有帧的GT
                    keys = data[gt_name][0].keys()
                    gts = data[gt_name]
                    data[gt_name] = {}
                    for k in keys:
                        gt = []
                        for gt_data in gts:
                            if (
                                k == "bev3d_ignore_obj_cls"
                                or k == "bev3d_ignore_occlusion"
                            ):
                                gt.append(torch.from_numpy(gt_data[k]))
                            else:
                                gt.append(
                                    torch.from_numpy(
                                        gt_data[k].transpose(2, 0, 1)
                                    )
                                )
                        data[gt_name][k] = torch.stack(gt, 0)
                else:
                    for k in data[gt_name].keys():
                        if (
                            k == "bev3d_ignore_obj_cls"
                            or k == "bev3d_ignore_occlusion"
                        ):
                            data[gt_name][k] = torch.from_numpy(
                                data[gt_name][k]
                            )
                        else:
                            data[gt_name][k] = torch.from_numpy(
                                data[gt_name][k].transpose(2, 0, 1)
                            )
                assert anno_name in data
                if isinstance(data[anno_name], Sequence):
                    # 返回时序所有帧的anno
                    keys = data[anno_name][0].keys()
                    annos = data[anno_name]
                    data[anno_name] = {}
                    for k in keys:
                        anno = []
                        for anno_data in annos:
                            anno.append(torch.from_numpy(anno_data[k]))
                        data[anno_name][k] = torch.stack(anno, 0)
                else:
                    for k in data[anno_name].keys():
                        data[anno_name][k] = torch.from_numpy(
                            data[anno_name][k]
                        )

        if "size" in data:
            data.pop("size")


@OBJECT_REGISTRY.register
class ANCTemporalMultiViewTargetGenerator(ANCMultiViewTargetGenerator):
    """Multi view data processer for temporal bev 3d.

    Currently, it is only used to process data for occlusion attributes

    Args:
        occlusion_attribute: Whether use occlusion attributes.
        occlusion_attribute_dict: occlusion category to id dict.
            Specifically, if the category id is smaller, it indicates
            a lower degree of occlusion.
        use_ignore_mask_img: Whether use ignore mask.
    """

    def __init__(
        self,
        occlusion_attribute: bool = False,
        occlusion_attribute_dict: Mapping[str, int] = None,
        use_ignore_mask_img: bool = False,
    ):
        super().__init__(
            occlusion_attribute,
            occlusion_attribute_dict,
            use_ignore_mask_img,
        )

    def __call__(self, data: Mapping):
        assert "gt_multi_view" in data, 'input data must has "gt_multi_view"'
        if not data["gt_multi_view"] or not (
            data["gt_multi_view"]
            and isinstance(data["gt_multi_view"][0], Sequence)
        ):
            return super().__call__(data)
        if self.occlusion_attribute:
            occlusion_multi_view_dict = {}
        img_mask_exists = True
        if "img_mask" not in data and self.use_ignore_mask_img:
            data["img_mask"] = []
            img_mask_exists = False
        # 返回时序所有帧的multi view信息
        for data_idx, gt_multi_view in enumerate(data["gt_multi_view"]):
            if self.occlusion_attribute:
                occlusion_multi_view_dict = {}
            if self.use_ignore_mask_img:
                ignore_mask = [
                    torch.zeros(img.size()[1:], dtype=torch.bool)
                    for img in data["imgs"][0]
                ]
            if not gt_multi_view:
                data["gt_multi_view"][data_idx] = {}
                continue
            assert len(gt_multi_view) == len(data["imgs"][data_idx])
            for idx, multi_view_info in enumerate(gt_multi_view):
                if multi_view_info:
                    for bbox in multi_view_info["objects"]:
                        if self.occlusion_attribute and bbox["bbox2d"]:
                            occlusion_attribute = self.get_occlusion_attribute(
                                bbox, occlusion_multi_view_dict
                            )
                            occlusion_multi_view_dict.update(
                                {bbox["uid"]: occlusion_attribute}
                            )
                    if self.use_ignore_mask_img:
                        ignore_mask[idx] = multi_view_info["meta"][
                            "ignore_mask"
                        ]["ignore_mask_2d"]

            data["gt_multi_view"][data_idx] = {}

            if self.use_ignore_mask_img:
                if not img_mask_exists:
                    data["img_mask"].append(ignore_mask)
                else:
                    for index, mask in data["img_mask"][data_idx]:
                        assert mask.size() == ignore_mask[index].size()
                        data["img_mask"][data_idx][index] = torch.logical_or(
                            mask, ignore_mask[index]
                        )
            if self.occlusion_attribute:
                data["gt_multi_view"][data_idx][
                    "occlusion_multi_view"
                ] = occlusion_multi_view_dict
        if self.use_ignore_mask_img and not data["img_mask"]:
            data.pop("img_mask")
        return data

    def __repr__(self):
        return "TemporalMultiViewTargetGenerator"


@OBJECT_REGISTRY.register
class ANCTemporalApplyMaskOnImg(ANCApplyMaskOnImg):
    """Apply Mask on temporal original image for.

        Mask is a torch.Tensor bool format data, the size of mask equal to
        the img w,h size.
        If one index of mask is 'True',
        means the index of img will be blackened.
        else, the index of img will be retained.

    Args:
        use_yuv_format: Whether the format of the image is yuv.
    """

    def __init__(
        self,
        use_yuv_format: bool = False,
    ):
        super().__init__(use_yuv_format)

    def __call__(self, data: Mapping):
        if "img_mask" in data:
            img_masks = data["img_mask"]
            if not (
                len(img_masks) == len(data["imgs"])
                and isinstance(img_masks[0], Sequence)
            ):
                return super().__call__(data)
            img_masks = data.pop("img_mask")
            # 处理时序所有图像
            for idx in range(len(img_masks)):
                assert len(img_masks[idx]) == len(data["imgs"][idx])
                for img_mask, img in zip(img_masks[idx], data["imgs"][idx]):
                    self._apply_mask_on_img(img, img_mask)
        return data

    def __repr__(self):
        return "TemporalApplyMaskOnImg"
