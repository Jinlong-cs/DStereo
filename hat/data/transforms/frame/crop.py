# Copyright (c) Horizon Robotics. All rights reserved.

import copy

import numpy as np

from hat.registry import OBJECT_REGISTRY

try:
    from crop_roi.merge.utils import get_camera, get_fanishing_point

    crop_roi_error_msg = None
except ImportError as e:
    crop_roi_error_msg = e
    get_camera = None
    get_fanishing_point = None

__all__ = ["CropImgPatch", "DynamicCropImgPatch"]


@OBJECT_REGISTRY.register
class CropImgPatch(object):
    """
    Cropping an image patch.

    Args:
        static_roi : list/tuple of 4 int, optional
            Required when roi_source is static_roi or dynamic_vanishing_point
        img_color : str, optional
            Image color, by default yuvi420
        is_buf: bool, optional
            Image dtype, if is_buf=True, get raw string, else get array.
    """

    def __init__(self, static_roi=None, img_color="yuvi420", is_buf=True):
        super().__init__()
        self.static_roi = static_roi
        self.img_color = img_color
        self.is_buf = is_buf
        self.transform_meta = {}

    def __call__(self, img_meta):
        img_meta = copy.copy(img_meta)
        img_hw = (img_meta["img_height"], img_meta["img_width"])
        if self.is_buf:
            img_crop = self._crop_roi(
                img_meta["img_buf"], img_hw, self.static_roi
            )
        else:
            img_crop = self._crop_roi(
                img_meta["img"], img_hw, self.static_roi, img_meta["layout"]
            )
        crop_hw = [
            self.static_roi[3] - self.static_roi[1],
            self.static_roi[2] - self.static_roi[0],
        ]

        transform_meta = dict(  # noqa
            original_hw=img_hw,
            transform_hw=crop_hw,
            eval_roi=self.static_roi,
        )
        # change calib params for 3d task
        if "calib" in img_meta:
            original_calib = copy.deepcopy(img_meta["calib"])
            img_meta["calib"][0][2] -= self.static_roi[0]
            img_meta["calib"][1][2] -= self.static_roi[1]
            transform_meta.update(
                dict(  # noqa
                    original_calib=original_calib,
                )
            )

        if "transform_meta" in img_meta:
            img_meta["transform_meta"].append(transform_meta)
        else:
            img_meta["transform_meta"] = [transform_meta]
        img_meta.update(
            dict(  # noqa
                img_height=crop_hw[0],
                img_width=crop_hw[1],
            )
        )
        if self.is_buf:
            img_meta.update(
                dict(  # noqa
                    img_buf=img_crop,
                )
            )
        else:
            img_meta.update(
                dict(  # noqa
                    img=img_crop,
                )
            )
        return img_meta

    def _crop_roi(self, img, img_hw, roi, layout="hwc"):
        crop_start_w, crop_start_h, crop_end_w, crop_end_h = roi
        if self.is_buf:
            if self.img_color == "yuvi420":
                y_img, u_img, v_img = self._decode_yuvi420_buf(img, img_hw)
                y_crop = y_img[
                    crop_start_h:crop_end_h, crop_start_w:crop_end_w
                ]
                # divide 2 because yuvi420
                u_crop = u_img[
                    int(crop_start_h / 2) : int(crop_end_h / 2),
                    int(crop_start_w / 2) : int(crop_end_w / 2),
                ]
                v_crop = v_img[
                    int(crop_start_h / 2) : int(crop_end_h / 2),
                    int(crop_start_w / 2) : int(crop_end_w / 2),
                ]
                return (
                    y_crop.tobytes() + u_crop.tobytes() + v_crop.tobytes()
                )  # noqa
            elif self.img_color in ["rgb", "bgr"]:
                raise NotImplementedError
            else:
                raise ValueError
        else:
            if layout == "hwc":
                img_crop = img[
                    crop_start_h:crop_end_h, crop_start_w:crop_end_w, :
                ]
            elif layout == "chw":
                img_crop = img[
                    :, crop_start_h:crop_end_h, crop_start_w:crop_end_w
                ]
            return img_crop

    def _decode_yuvi420_buf(self, yuvi420_buf, img_hw):
        h, w = img_hw
        slice_step = int(w * h / 4)
        y = np.frombuffer(yuvi420_buf[: w * h], dtype="uint8").reshape((h, w))
        u = np.frombuffer(
            yuvi420_buf[w * h : (w * h + slice_step)], dtype="uint8"
        ).reshape((int(h / 2), int(w / 2)))
        v = np.frombuffer(
            yuvi420_buf[(w * h + slice_step) :], dtype="uint8"
        ).reshape((int(h / 2), int(w / 2)))
        return y, u, v

    def inverse_transform(self, results, transform_meta=None):
        if transform_meta is not None:
            origin_h, origin_w = (
                transform_meta["original_hw"][0],
                transform_meta["original_hw"][1],
            )
            bx1, by1, bx2, by2 = transform_meta["eval_roi"]
            inv_results = results.inv_crop(
                crop_roi=(bx1, by1, bx2, by2), image_hw=(origin_h, origin_w)
            )
            return inv_results
        else:
            return results


@OBJECT_REGISTRY.register
class DynamicCropImgPatch(CropImgPatch):
    """
    Cropping an image patch using different roi.

    Args:
        static_roi : list/tuple of 4 int, optional
            Required when roi_source is static_roi or dynamic_vanishing_point.
            While roi_source is "static_roi",
                static_roi should be the crop region.
            While roi_source is "img_meta", static_roi is the default region,
                will be used when crop_roi is not in img_meta.
            While roi_source is "dynamic_vanishing_point", static_roi should be
                [fp_x, fp_y, crop_w, crop_h]
        img_color : str, optional
            Image color, by default yuvi420
        is_buf : bool, optional
            Image dtype, if is_buf=True, get raw string, else get array.
        roi_source : str, optional
            Where to get the roi,
                should be in [dynamic_vanishing_point, static_roi, img_meta].
        bpu_align : bool, optional
            Align the roi to adapt the bpu limitation.
    """

    def __init__(
        self,
        static_roi=None,
        img_color="yuvi420",
        is_buf=True,
        roi_source="dynamic_vanishing_point",
        bpu_align=True,
    ):
        super().__init__(static_roi, img_color, is_buf)
        self.roi_source = roi_source
        self.bpu_align = bpu_align
        assert roi_source in [
            "dynamic_vanishing_point",
            "static_roi",
            "img_meta",
        ]
        if roi_source == "dynamic_vanishing_point":
            assert not crop_roi_error_msg, "Please install crop_roi"

    def __call__(self, img_meta):
        img_meta = copy.copy(img_meta)
        if self.roi_source == "dynamic_vanishing_point":
            fp_x = self.static_roi[0]
            fp_y = self.static_roi[1]
            w = self.static_roi[2]
            h = self.static_roi[3]
            cam = get_camera(img_meta["calib_all"])
            vp = get_fanishing_point(cam.gnd2img)
            if vp[0] >= fp_x and vp[1] >= fp_y:
                static_roi = [
                    vp[0] - fp_x,
                    vp[1] - fp_y,
                ]
            elif vp[0] < fp_x and vp[1] >= fp_y:
                static_roi = [
                    0,
                    vp[1] - fp_y,
                ]
            elif vp[0] >= fp_x and vp[1] < fp_y:
                static_roi = [
                    vp[0] - fp_x,
                    0,
                ]
            else:
                static_roi = [
                    0,
                    0,
                ]
            # avoid the roi out of the image!
            static_roi[0] = max(
                0, min(static_roi[0], img_meta["img_width"] - w)
            )
            static_roi[1] = max(
                0, min(static_roi[1], img_meta["img_height"] - h)
            )
            static_roi += [static_roi[0] + w, static_roi[1] + h]
        elif self.roi_source == "img_meta":
            static_roi = img_meta.get("crop_roi", self.static_roi)
        else:
            static_roi = self.static_roi
        # alignment for bpu
        if self.bpu_align:
            quo_x = static_roi[0] // 16
            mod_x = static_roi[0] % 16
            mod_y = static_roi[1] % 2
            if mod_x != 0:
                if mod_x < 8:
                    static_roi[0] = quo_x * 16
                else:
                    static_roi[0] = (quo_x + 1) * 16
                static_roi[2] = static_roi[0] + self.static_roi[2]
            if mod_y != 0:
                if static_roi[1] < img_meta["img_height"] - 1:
                    static_roi[1] += 1
                else:
                    static_roi[1] -= 1
                static_roi[3] = static_roi[1] + self.static_roi[3]
        static_roi = tuple([int(v) for v in static_roi])
        img_hw = (img_meta["img_height"], img_meta["img_width"])
        if self.is_buf:
            img_crop = self._crop_roi(img_meta["img_buf"], img_hw, static_roi)
        else:
            img_crop = self._crop_roi(
                img_meta["img"], img_hw, static_roi, img_meta["layout"]
            )

        crop_hw = [
            static_roi[3] - static_roi[1],
            static_roi[2] - static_roi[0],
        ]

        transform_meta = dict(  # noqa
            original_hw=img_hw,
            transform_hw=crop_hw,
            eval_roi=static_roi,
        )
        # change calib params for 3d task
        if "calib" in img_meta:
            original_calib = copy.deepcopy(img_meta["calib"])
            img_meta["calib"][0][2] -= static_roi[0]
            img_meta["calib"][1][2] -= static_roi[1]
            transform_meta.update(
                dict(  # noqa
                    original_calib=original_calib,
                )
            )

        if "transform_meta" in img_meta:
            img_meta["transform_meta"].append(transform_meta)
        else:
            img_meta["transform_meta"] = [transform_meta]
        img_meta.update(
            dict(  # noqa
                img_height=crop_hw[0],
                img_width=crop_hw[1],
            )
        )
        if self.is_buf:
            img_meta.update(
                dict(  # noqa
                    img_buf=img_crop,
                )
            )
        else:
            img_meta.update(
                dict(  # noqa
                    img=img_crop,
                )
            )
        return img_meta
