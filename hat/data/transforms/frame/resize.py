# Copyright (c) Horizon Robotics. All rights reserved.

import math
from typing import Optional

import cv2
import numpy as np
from horizon_plugin_pytorch.march import March
from torch import Tensor

from hat.core.data_struct.base_struct import Mask
from hat.data.transforms.detection import imresize
from hat.registry import OBJECT_REGISTRY
from hat.utils.package_helper import raise_error_if_import_failed

try:
    from pyramid_resizer import pyramid_resizer
except ImportError:
    pyramid_resizer = None

try:
    import hat_sim
except ImportError:
    hat_sim = None


__all__ = ["BPUPyramidResizer", "CV2AdptiveResolutionInput"]


@OBJECT_REGISTRY.register
class BPUPyramidResizer(object):
    """BPU Pyramid Resizer.

    mark: The output is YUV444 while the input is YUVi420.

    Args:
        scale_wh: Pyramid resize scale. Valid scale is 0.5^n, n=0,1,2,3,4,5.
        target_hw: Pyramid resize target size. One and only one of 'scale_wh'
            and 'target_hw' should be given.
        pyramid_type: Pyramid type, "ips" or "ipu".
        pyramid_idx: Index of pyramid layer, only work while march is J5.
            support to be int(math.log(src_wh / dst_wh, 2.0)),
            Range is 0 to 5, while 0 for 1/2, 5 for 1/64.
        auto_pyramid_idx: Whether use _get_pyramid_layer_id to get pyramid idx.
        resize_gt: Whether to resize gt labels according to scale wh.
            Default is False.
        task_type: Ground True Type for forward transform.
        inverse_by: Currently "transform_meta" or "inverse_info".
            decide which inverse_transform will be used.
        ratio_range: If scale image randomly.
        is_del_img_buf: If del img_buf in dict data.
        march: BPU platform.
    """

    def __init__(
        self,
        scale_wh: Optional[tuple] = None,
        target_hw: Optional[tuple] = None,
        pyramid_type: str = "ips",
        pyramid_idx: Optional[int] = None,
        auto_pyramid_idx: Optional[bool] = False,
        resize_gt: Optional[bool] = False,
        task_type: Optional[str] = None,
        inverse_by: str = "transform_meta",
        ratio_range: Optional[tuple] = None,
        is_del_img_buf: Optional[bool] = False,
        march: March = None,
    ):
        super().__init__()

        assert inverse_by in ["transform_meta", "inverse_info"]
        assert pyramid_type in [
            "ips",
            "ipu",
        ], f"`pyramid_type` should be 'ips' or 'ipu', but get {pyramid_type}."

        if march == March.BAYES:
            raise_error_if_import_failed(hat_sim, "hat_sim")
            assert pyramid_type == "ips", "j5 only support ips."
        else:
            raise_error_if_import_failed(pyramid_resizer, "pyramid_resizer")
        assert not (scale_wh is None and target_hw is None) and not (
            scale_wh is not None and target_hw is not None
        ), "One and only one of scale_wh and target_wh should be provided."

        self.scale_wh = scale_wh
        self.target_hw = target_hw
        self.pyramid_type = pyramid_type
        self.pyramid_idx = pyramid_idx
        self.pyramid = self._get_pyramid_class(self.pyramid_type)
        self.resize_gt = resize_gt
        self.task_type = task_type
        self.is_del_img_buf = is_del_img_buf
        inverse_transform = dict(  # noqa
            transform_meta=self.inverse_transform_transform_meta,
            inverse_info=self.inverse_transform_inverse_info,
        )
        self.inverse_transform = inverse_transform[inverse_by]

        self.ratio_range = ratio_range
        self.march = march
        if self.ratio_range:
            if not (self.pyramid_type == "ips" and self.march == March.BAYES):
                raise NotImplementedError("not available yet")

        self.auto_pyramid_idx = auto_pyramid_idx
        assert np.any([not auto_pyramid_idx, pyramid_idx is None])

    def _get_pyramid_class(self, pyramid_type):
        if pyramid_type == "ips":
            pyramid = pyramid_resizer.IPSPyramid()
        elif pyramid_type == "ipu":
            pyramid = pyramid_resizer.IpuPyramid()
        else:
            raise NotImplementedError(
                "`pyramid_type` only support `ips` and `ipu`."
            )
        return pyramid

    def __getstate__(self):
        state = self.__dict__.copy()
        state["pyramid"] = None
        return state

    def __setstate__(self, state):
        self.__dict__ = state.copy()

        self.pyramid = self._get_pyramid_class(self.pyramid_type)

    def __call__(self, data):
        data, scale_wh, resize_hw, origin_hw = self._resize_image(data)
        if self.is_del_img_buf:
            del data["img_buf"]
        if self.resize_gt:
            if "gt_bboxes" in data or "gt_boxes" in data:
                self._resize_bbox(data, resize_hw)
            if "gt_seg" in data:
                self._resize_seg(data, resize_hw)

        for key in ["calib_all", "standardized_calib_all"]:
            if key in data:
                data[key] = self._resize_calib_all(data[key])
        transform_meta = dict(  # noqa
            origin_hw=origin_hw,
            resize_hw=resize_hw,
            scale_wh=scale_wh,
        )
        if "transform_meta" in data:
            data["transform_meta"].append(transform_meta)
        else:
            data["transform_meta"] = [transform_meta]

        return data

    def _resize_calib_all(self, calib_all):
        if calib_all is None:
            return calib_all
        calib_all["focal_u"] *= self.scale_wh[0]
        calib_all["focal_v"] *= self.scale_wh[1]
        calib_all["center_u"] *= self.scale_wh[0]
        calib_all["center_v"] *= self.scale_wh[1]
        calib_all["image_width"] = int(
            self.scale_wh[0] * calib_all["image_width"]
        )
        calib_all["image_height"] = int(
            self.scale_wh[1] * calib_all["image_height"]
        )
        return calib_all

    def _resize_seg(self, data, resize_hw):
        h_scale, w_scale = resize_hw
        resized_seg = imresize(
            data["gt_seg"],
            w_scale,
            h_scale,
            "hw",
            keep_ratio=False,
            return_scale=False,
            interpolation="nearest",
        )
        data["gt_seg"] = resized_seg
        return data

    def _resize_bbox(self, data, resize_hw):
        w_scale, h_scale = self.scale_wh
        scale_factor = np.array(
            [w_scale, h_scale, w_scale, h_scale], dtype=np.float32
        )
        data["scale_factor"] = scale_factor
        data["keep_ratio"] = False
        data["scale"] = resize_hw
        data["scale_idx"] = 0

        h, w = resize_hw

        def _det_task():
            if data["gt_bboxes"].any():
                bboxes = data["gt_bboxes"] * data["scale_factor"]
                bboxes[:, 0::2] = np.clip(bboxes[:, 0::2], 0, w - 1)
                bboxes[:, 1::2] = np.clip(bboxes[:, 1::2], 0, h - 1)
                data["gt_bboxes"] = bboxes

        def _rcnn_kps_task():
            if data["gt_boxes"].any():
                bboxes = data["gt_boxes"]
                bboxes[:, [0, 2, 4, 7]] = np.clip(
                    bboxes[:, [0, 2, 4, 7]] * data["scale_factor"][0], 0, w - 1
                )
                bboxes[:, [1, 3, 5, 8]] = np.clip(
                    bboxes[:, [1, 3, 5, 8]] * data["scale_factor"][1], 0, h - 1
                )
                data["gt_boxes"] = bboxes

        def _rcnn_cls_task():
            if data["gt_boxes"].any():
                bboxes = data["gt_boxes"]
                bboxes[:, [0, 2]] = np.clip(
                    bboxes[:, [0, 2]] * data["scale_factor"][0], 0, w - 1
                )
                bboxes[:, [1, 3]] = np.clip(
                    bboxes[:, [1, 3]] * data["scale_factor"][1], 0, h - 1
                )
                data["gt_boxes"] = bboxes

        def _rcnn_det_task():
            if data["gt_boxes"].any():
                bboxes = data["gt_boxes"]
                parent_bboxes = data["parent_gt_boxes"]
                bboxes[:, [0, 2]] = np.clip(
                    bboxes[:, [0, 2]] * data["scale_factor"][0], 0, w - 1
                )
                bboxes[:, [1, 3]] = np.clip(
                    bboxes[:, [1, 3]] * data["scale_factor"][1], 0, h - 1
                )
                parent_bboxes[:, [0, 2]] = np.clip(
                    parent_bboxes[:, [0, 2]] * data["scale_factor"][0],
                    0,
                    w - 1,
                )
                parent_bboxes[:, [1, 3]] = np.clip(
                    parent_bboxes[:, [1, 3]] * data["scale_factor"][1],
                    0,
                    h - 1,
                )
                data["gt_boxes"] = bboxes
                data["parent_gt_boxes"] = parent_bboxes

        map_dict = {
            "detection": _det_task,
            "rcnn_kps": _rcnn_kps_task,
            "rcnn_classification": _rcnn_cls_task,
            "rcnn_detection": _rcnn_det_task,
        }
        if self.task_type:
            task_type = self.task_type
        else:
            task_type = data.get("task_type", "detection")
        map_dict[task_type]()

    def random_sample_ratio(self, img_scale, ratio_range):
        assert isinstance(img_scale, (tuple, list)) and len(img_scale) == 2
        min_ratio, max_ratio = ratio_range
        assert min_ratio <= max_ratio
        ratio = np.random.random_sample() * (max_ratio - min_ratio) + min_ratio
        scale = int(img_scale[0] * ratio), int(img_scale[1] * ratio)
        return scale

    def _resize_image(self, data):
        yuvi420_buf = data["img_buf"]
        img_hw = (data["img_height"], data["img_width"])

        if self.target_hw is not None:
            resize_hw = self.target_hw
            scale_wh = (
                resize_hw[1] / img_hw[1],
                resize_hw[0] / img_hw[0],
            )
        else:
            scale_wh = self.scale_wh
            resize_hw = (
                int(img_hw[0] * self.scale_wh[1]),
                int(img_hw[1] * self.scale_wh[0]),
            )

        if self.ratio_range is not None:
            resize_hw = self.random_sample_ratio(resize_hw, self.ratio_range)
            scale_wh = (
                resize_hw[1] / img_hw[1],
                resize_hw[0] / img_hw[0],
            )

        if self.pyramid_idx is None:
            pyramid_idx = self._get_pyramid_layer_id(img_hw, resize_hw)
        else:
            pyramid_idx = self.pyramid_idx

        if self.pyramid_type == "ips":
            # pyramid
            target_w = img_hw[1]
            target_h = img_hw[0]
            for _ in range(pyramid_idx + 1):
                target_w = (target_w >> 1) & (~0x1)
                target_h = (target_h >> 1) & (~0x1)
            tgt_rois = [0, 0, target_w, target_h]
            if self.march in (
                None,
                March.BERNOULLI2,
                March.BERNOULLI,
                # None means default setting in plugin.
                # current default to use J2/J3.
            ):
                pyramid = pyramid_resizer.IPSPyramid()
                pyramid.build_pyramid(yuvi420_buf, img_hw[1], img_hw[0], 1)
                yuv420sp = pyramid.get_image_data_yuv420(pyramid_idx + 1)
                yuvi420_buf = self._yuv420sp_to_yuvi420(
                    yuv420sp, [target_h, target_w]
                )
                yuv444 = pyramid_resizer.yuvi420_str2yuv444_np(
                    yuvi420_buf, target_w, target_h
                )
                data["layout"] = "hwc"
            elif self.march == March.BAYES:
                yuvi420 = np.fromstring(yuvi420_buf, dtype="uint8")
                pyr_rois = [0, 0, target_w, target_h]

                pyramid = hat_sim.IPSPyramid(
                    yuvi420, img_hw[1], img_hw[0], 0, 0, pyr_rois
                )
                pyr_tgt = np.random.randint(
                    0,
                    255,
                    size=(int(3 * target_w * target_h / 2),),
                    dtype=np.uint8,
                )

                pyramid.build_pyramid()
                pyramid.crop_resize(pyr_tgt, pyramid_idx)

                # resizer
                pyr_tgt_yuv444 = np.random.randint(
                    0, 255, size=(3, target_h, target_w), dtype=np.uint8
                )
                hat_sim.yuvi4202yuv444(
                    pyr_tgt, pyr_tgt_yuv444, target_w, target_h
                )
                yuv444 = np.random.randint(
                    0,
                    255,
                    size=(3, resize_hw[0], resize_hw[1]),
                    dtype=np.uint8,
                )

                resizer = hat_sim.RoiResize(pyr_tgt_yuv444, target_w, target_h)
                resizer.crop_resize(
                    yuv444, resize_hw[0], resize_hw[1], tgt_rois
                )
                data["layout"] = "chw"

        elif self.pyramid_type == "ipu":
            yuv420sp = self.pyramid.get_resize_420sp_from_i420bytes(
                yuvi420_buf, tuple(img_hw), tuple(resize_hw)
            )
            h_padding = int(resize_hw[0])
            w_padding = int(len(yuv420sp) / 1.5 / resize_hw[0])
            padding_hw = (h_padding, w_padding)
            yuv420sp_np = np.fromstring(yuv420sp, dtype="uint8").reshape(
                (-1, w_padding)
            )
            yuvi420_buf = self._yuv420sp_to_yuvi420(
                yuv420sp_np, tuple(padding_hw)
            )
            yuvi420_buf = self._unpad_image_yuvi420(
                yuvi420_buf, tuple(padding_hw), tuple(resize_hw)
            )
            yuv444 = pyramid_resizer.yuvi420_str2yuv444_np(
                yuvi420_buf, resize_hw[1], resize_hw[0]
            )
            data["layout"] = "hwc"

        data["img"] = yuv444
        data["img_shape"] = yuv444.shape
        data["img_height"], data["img_width"] = resize_hw
        return data, scale_wh, resize_hw, img_hw

    def _yuv420sp_to_yuvi420(self, yuv420sp, img_hw, to_string=True):
        y_img = yuv420sp[: img_hw[0], :]
        u_img = yuv420sp[img_hw[0] :, ::2]
        v_img = yuv420sp[img_hw[0] :, 1::2]
        if to_string:
            return y_img.tobytes() + u_img.tobytes() + v_img.tobytes()
        else:
            return y_img, u_img, v_img

    def _get_pyramid_layer_id(self, src_hw, dst_hw):
        layer = int(math.log(src_hw[1] / dst_hw[1], 2.0)) - 1
        assert layer in [0, 1, 2, 3, 4, 5, 6]
        return layer

    def _unpad_image_yuvi420(self, img_yuvi420, src_hw, dst_hw):
        img_y_str = img_yuvi420[: src_hw[0] * src_hw[1]]
        u_offset = int(src_hw[0] * src_hw[1] + int(src_hw[0] * src_hw[1] / 4))
        img_u_str = img_yuvi420[src_hw[0] * src_hw[1] : u_offset]
        img_v_str = img_yuvi420[u_offset:]

        img_y = np.fromstring(img_y_str, dtype="uint8").reshape(src_hw)
        src_uv_hw = (int(src_hw[0] / 2), int(src_hw[1] / 2))
        img_u = np.fromstring(img_u_str, dtype="uint8").reshape(src_uv_hw)
        img_v = np.fromstring(img_v_str, dtype="uint8").reshape(src_uv_hw)
        img_y_crop = img_y[: dst_hw[0], : dst_hw[1]]
        img_u_crop = img_u[: int(dst_hw[0] / 2), : int(dst_hw[1] / 2)]
        img_v_crop = img_v[: int(dst_hw[0] / 2), : int(dst_hw[1] / 2)]

        return (
            img_y_crop.tobytes() + img_u_crop.tobytes() + img_v_crop.tobytes()
        )

    def inverse_transform_transform_meta(self, obj, transform_meta=None):
        if isinstance(obj, Tensor):
            obj[..., [0, 2]] /= self.scale_wh[0]
            obj[..., [1, 3]] /= self.scale_wh[1]
        else:
            obj.rescale(1 / self.scale_wh[0], 1 / self.scale_wh[1])
        return obj

    def inverse_transform_inverse_info(self, inputs, task_type, inverse_info):
        if task_type == "segmentation":
            transform_meta = inverse_info["transform_meta"][0]
            origin_hw = transform_meta["origin_hw"]
            if isinstance(inputs, Tensor):
                inputs = inputs.cpu().numpy()
            inputs = cv2.resize(
                inputs, dsize=origin_hw[::-1], interpolation=cv2.INTER_NEAREST
            )
            return inputs


@OBJECT_REGISTRY.register
class CV2InverseTransform(object):
    def __init__(
        self,
        scale_wh,
        padding_wh=None,
        original_wh=None,
    ):
        self.scale_wh = scale_wh
        self.padding_wh = padding_wh
        self.original_wh = original_wh

    def __call__(self, batch_outputs, batch_data):

        w = self.scale_wh[0]
        h = self.scale_wh[1]

        for _, task_objs in batch_outputs.items():
            # task_rescale_objs = []
            for obj in task_objs:
                obj.rescale(1 / w, 1 / h)
                # Cut off padding in seg output mask.
                # Since det and attribute results are not affected by padding, only masks contain padding part.  # noqa
                if isinstance(obj, Mask):
                    if self.padding_wh:
                        obj.inv_pad(
                            0,
                            self.padding_wh[0] - self.original_wh[0],
                            0,
                            self.padding_wh[1] - self.original_wh[1],
                        )
        return batch_outputs, batch_data


@OBJECT_REGISTRY.register
class CV2AdptiveResolutionInput(object):
    """CV2 resizer transformer.

    Args:
        model_input_hw : tuple of int,
            The size of model input data.
        scale_type : str, default="MIN"
            The way to transfrom scale. Possivle value:
            "W": scale = w_scale = float(target_wh[0])/img_wh[0]
            "H": scale = h_scale = float(target_wh[1])/img_wh[1]
            "MIN": scale = min(w_scale, h_scale)
            "MAX": scale = max(w_scale, h_scale)
    """

    def __init__(self, model_input_hw: tuple, scale_type="MIN"):
        super().__init__()
        self.model_input_hw = model_input_hw

        assert scale_type in ["W", "H", "MIN", "MAX"]
        self.scale_type = scale_type

    def _get_scale(self, model_input_hw, img_hw, scale_type):
        h_scale = float(model_input_hw[0]) / img_hw[0]
        w_scale = float(model_input_hw[1]) / img_hw[1]
        if scale_type == "W":
            res_scale = w_scale
        elif scale_type == "H":
            res_scale = h_scale
        elif scale_type == "MIN":
            res_scale = min(w_scale, h_scale)
        elif scale_type == "MAX":
            res_scale = max(w_scale, h_scale)
        else:
            raise ValueError("Unknow scale_type:{}".format(scale_type))

        scale_side_idx = 0 if res_scale == h_scale else 1
        return res_scale, scale_side_idx

    def _cal_trans_param(self, img_hw):
        resize_scale, scale_side_idx = self._get_scale(
            self.model_input_hw, img_hw, self.scale_type
        )
        padd_size_idx = 1 - scale_side_idx
        padding_len = (
            int(self.model_input_hw[padd_size_idx] / resize_scale)
            - img_hw[padd_size_idx]
        )  # noqa

        padding_side = padd_size_idx
        scale_wh = [resize_scale, resize_scale]

        return padding_len, padding_side, scale_wh

    def _crop_img(self, bgr_img, crop_hw):
        img_h = bgr_img.shape[0]
        img_w = bgr_img.shape[1]

        return bgr_img[: img_h - crop_hw[0], : img_w - crop_hw[1], :]

    def _cv2_padding_zero(self, bgr_img, padding_len):
        assert len(padding_len) == 4
        padding_left = padding_len[0]
        padding_top = padding_len[1]
        padding_right = padding_len[2]
        padding_bottom = padding_len[3]

        padding_param = (
            (padding_top, padding_bottom),  # height
            (padding_left, padding_right),  # width
            (0, 0),
        )
        padding_img = np.pad(bgr_img, padding_param, "constant")
        return padding_img

    def __call__(self, img_meta):
        bgr_img = img_meta["img"]
        org_img_hw = (img_meta["img_height"], img_meta["img_width"])

        # 获取变换参数
        padding_len, padding_side, scale_wh = self._cal_trans_param(org_img_hw)
        if padding_len >= 0:
            padding_param = [0, 0, 0, 0]
            padding_param[-(padding_side + 1)] = padding_len
            padding_img = self._cv2_padding_zero(bgr_img, padding_param)
        else:
            crop_hw = [0, 0]
            crop_hw[padding_side] = -padding_len
            padding_img = self._crop_img(bgr_img, crop_hw)
        padding_img_hw = padding_img.shape[:2]

        resize_img_hw = [
            int(self.model_input_hw[0]),
            int(self.model_input_hw[1]),
        ]

        resize_img = cv2.resize(
            padding_img, (resize_img_hw[1], resize_img_hw[0])
        )

        transform_meta = {
            "original_img_hw": org_img_hw,
            "padding_img_hw": padding_img_hw,
            "transform_hw": resize_img_hw,
            "scale_wh": scale_wh,
            "padding_len": padding_len,
            "padding_side": padding_side,
        }

        self.transform_meta = transform_meta

        if "transform_meta" not in img_meta:
            img_meta["transform_meta"] = []
        img_meta["transform_meta"].append(transform_meta)

        img_meta.update(
            {
                "img": resize_img,
                "img_height": resize_img_hw[0],
                "img_width": resize_img_hw[1],
            }
        )

        return img_meta

    def inverse_transform(self, obj, transform_meta=None):
        transform_meta = (
            self.transform_meta if not transform_meta else transform_meta
        )

        w = transform_meta["scale_wh"][0]
        if isinstance(w, Tensor):
            w = w.item()
        h = transform_meta["scale_wh"][1]
        if isinstance(h, Tensor):
            h = h.item()

        if transform_meta:
            obj.rescale(1 / w, 1 / h)
            # Cut off padding in seg output mask.
            # Since det and attribute results are not affected by padding, only masks contain padding part.  # noqa
            if isinstance(obj, Mask):
                obj.inv_pad(
                    0,
                    transform_meta["padding_img_hw"][1]
                    - transform_meta["original_img_hw"][1],
                    0,
                    transform_meta["padding_img_hw"][0]
                    - transform_meta["original_img_hw"][0],
                )
        return obj
