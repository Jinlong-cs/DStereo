# Copyright (c) Horizon Robotics. All rights reserved.
import numpy as np

from hat.core.box_utils import bbox_overlaps
from hat.data.transforms.functional_img import imresize
from hat.registry import OBJECT_REGISTRY

__all__ = [
    "RawPack",
    "RawPad",
    "DEC",
    "DGain",
    "BLC",
    "NNLogLut",
]


@OBJECT_REGISTRY.register
class RawPack(object):  # noqa: D205,D400
    """Pack 1 channel raw to 4 channel feature or unpack 4 channel feature
        to 1 channel raw.

    ..note::
        Affected keys: 'img', 'img_shape', 'gt_bboxes', 'gt_classes', 'gt_seg',
        'cur_pattern'.

    Args:
        method (str): should be 'pack' or 'unpack'.
        min_area (Optional[int]): If min_area > 0, boxes whose areas are
                less than min_area will be ignored.
        min_iou (Optional[float]): If min_iou > 0, boxes whose iou between
                before and after truncation < min_iou will be ignored.
        discriminate_ignore_classes (Optional[bool]): if True, ignored area
            by min_iou retain the original class info. And class id should
            greater than 0.
            Default False. Support class id starts from 0.
        resize_gt (Optional[bool]): If True, resize bbox or seg mask.
    """

    def __init__(
        self,
        method,
        min_area=-1,
        min_iou=-1,
        discriminate_ignore_classes=False,
        resize_gt=False,
    ):
        assert method in ["pack", "unpack"]
        self.method = method
        self.min_area = min_area
        self.min_iou = min_iou
        self.discriminate_ignore_classes = discriminate_ignore_classes
        self.resize_gt = resize_gt

    def _pack_raw(self, data):
        img = data["img"]
        layout = data["layout"]
        cur_pattern = data.get("cur_pattern")
        assert layout in ["hwc", "chw"], (
            "layout of img must be `chw` or " "`hwc`"
        )
        assert cur_pattern in ["RGGB", "BGGR", "GBRG", "GRBG"]

        if layout == "hwc":
            assert img.shape[2] == 1
            left_top = img[0::2, 0::2]
            right_top = img[0::2, 1::2]
            left_bottom = img[1::2, 0::2]
            right_bottom = img[1::2, 1::2]

            packed_img = np.concatenate(
                (left_top, right_top, left_bottom, right_bottom), axis=2
            )
        elif layout == "chw":
            assert img.shape[0] == 1
            left_top = img[:, 0::2, 0::2]
            right_top = img[:, 0::2, 1::2]
            left_bottom = img[:, 1::2, 0::2]
            right_bottom = img[:, 1::2, 1::2]

            packed_img = np.concatenate(
                (left_top, right_top, left_bottom, right_bottom), axis=0
            )

        data["img"] = packed_img
        data["img_shape"] = packed_img.shape
        data["img_height"] = packed_img.shape[0]
        data["img_width"] = packed_img.shape[1]
        data["cur_pattern"] = None

    def _pack_bbox(self, data):
        img_shape = data["img_shape"]
        if data["gt_bboxes"].any():
            bboxes = data["gt_bboxes"] / 2
            boxes_real = bboxes.copy()
            classes = data["gt_classes"].copy()
            bboxes[:, 0::2] = np.clip(bboxes[:, 0::2], 0, img_shape[1] - 1)
            bboxes[:, 1::2] = np.clip(bboxes[:, 1::2], 0, img_shape[0] - 1)
            # default setttings, ignore the tiny boxes
            area = np.maximum(0, bboxes[:, 2] - bboxes[:, 0]) * np.maximum(
                0, bboxes[:, 3] - bboxes[:, 1]
            )
            area_ignore_index = (area < 1e-3) & (classes >= 0)
            if self.discriminate_ignore_classes:
                classes[area_ignore_index] = -abs(classes[area_ignore_index])
            else:
                classes[area_ignore_index] = -1
            # user settings, ignore boxes whose area is less than min_area
            if self.min_area > 0:
                area_ignore_index = (area < self.min_area) & (classes >= 0)
                if self.discriminate_ignore_classes:
                    classes[area_ignore_index] = -abs(
                        classes[area_ignore_index]
                    )
                else:
                    classes[area_ignore_index] = -1
            # ignore the regions where the iou between before and after
            # truncation < min_iou
            if self.min_iou > 0:
                for i in range(boxes_real.shape[0]):
                    iou = bbox_overlaps(
                        bboxes[i].reshape((1, -1)),
                        boxes_real[i].reshape((1, -1)),
                    )[0][0]
                    if iou < self.min_iou and classes[i] >= 0:
                        if self.discriminate_ignore_classes:
                            classes[i] = -1 * abs(classes[i])
                        else:
                            classes[i] = -1

            # filter out the gt bboxes that are completely cropped
            valid_inds = (bboxes[:, 2] > bboxes[:, 0]) & (
                bboxes[:, 3] > bboxes[:, 1]
            )
            # if no gt bbox remains after cropping, set bboxes shape (0, 4)
            if not np.any(valid_inds):
                bboxes = np.zeros((0, 4), dtype=np.float32)
                classes = np.zeros((0,), dtype=np.int64)
            data["gt_bboxes"] = bboxes
            data["gt_classes"] = classes

    def _pack_seg(self, data):
        gt_seg = data["gt_seg"]
        h, w = gt_seg.shape[:2]
        h = int(h / 2.0)
        w = int(w / 2.0)

        resized_seg = imresize(
            gt_seg,
            w,
            h,
            "hw",
            keep_ratio=False,
            return_scale=False,
            interpolation="nearest",
        )
        data["gt_seg"] = resized_seg

    def _unpack_raw(self, data):
        img = data["img"]
        layout = data["layout"]
        raw_pattern = data.get("raw_pattern")
        assert layout in ["hwc", "chw"], (
            "layout of img must be `chw` or " "`hwc`"
        )
        assert raw_pattern in ["RGGB", "BGGR", "GBRG", "GRBG"]

        if layout == "hwc":
            assert img.shape[2] == 4
            unpack_img = np.zeros(
                (img.shape[0] * 2, img.shape[1] * 2, 1)
            ).astype(img.dtype)

            unpack_img[0::2, 0::2, 0] = img[:, :, 0]
            unpack_img[0::2, 1::2, 0] = img[:, :, 1]
            unpack_img[1::2, 0::2, 0] = img[:, :, 2]
            unpack_img[1::2, 1::2, 0] = img[:, :, 3]
        elif layout == "chw":
            assert img.shape[0] == 4
            unpack_img = np.zeros(
                (1, img.shape[1] * 2, img.shape[2] * 2)
            ).astype(img.dtype)

            unpack_img[0, 0::2, 0::2] = img[0, :, :]
            unpack_img[0, 0::2, 1::2] = img[1, :, :]
            unpack_img[0, 1::2, 0::2] = img[2, :, :]
            unpack_img[0, 1::2, 1::2] = img[3, :, :]

        data["img"] = unpack_img
        data["img_shape"] = unpack_img.shape
        data["img_height"] = unpack_img.shape[0]
        data["img_width"] = unpack_img.shape[1]
        data["cur_pattern"] = None

    def _unpack_bbox(self, data):
        raise NotImplementedError

    def _unpack_seg(self, data):
        raise NotImplementedError

    def inverse_transform(self, inputs, task_type, inverse_info):
        """Inverse option of transform to map the prediction to the original image.

        Args:
            inputs (array|Tensor): Prediction.
            task_type (str): `detection` or `segmentation`.
            inverse_info (dict): The transform keyword is the key,
                and the corresponding value is the value.

        """
        if task_type == "detection":
            if self.method == "pack":
                inputs = inputs * 2
            else:
                inputs = inputs / 2
            return inputs
        else:
            raise Exception(
                "error task_type, your task_type[{}],"
                " we need detection".format(task_type)
            )

    def __call__(self, data):
        if self.method == "pack":
            self._pack_raw(data)
            if self.resize_gt:
                if "gt_bboxes" in data:
                    self._pack_bbox(data)
                elif "gt_seg" in data:
                    self._pack_seg(data)
        elif self.method == "unpack":
            self._unpack_raw(data)
            if self.resize_gt:
                if "gt_bboxes" in data:
                    self._unpack_bbox(data)
                elif "gt_seg" in data:
                    self._unpack_seg(data)
        return data

    def __repr__(self):
        repr_str = self.__class__.__name__ + ": "
        repr_str += f"method={self.method}, "
        repr_str += f"min_area={self.min_area}, "
        repr_str += f"min_iou={self.min_iou}, "
        repr_str += (
            f"discriminate_ignore_classes={self.discriminate_ignore_classes}, "
        )
        repr_str += f"resize_gt={self.resize_gt}, "

        return repr_str


@OBJECT_REGISTRY.register
class RawPad(object):
    """Pad raw img to 3 channels img.

    ..note::
    Affected keys: 'img', 'img_shape'.

    Args:
        method (str): should be 'padself' of 'padzero'.
    """

    def __init__(self, method):
        self.method = method

    def _pad_raw(self, data):
        img = data["img"]
        layout = data["layout"]
        assert layout in ["hwc", "chw"], (
            "layout of img must be `chw` or " "`hwc`"
        )
        if self.method == "padself":
            if layout == "hwc":
                pad_img = np.concatenate((img, img, img), axis=2)
            elif layout == "chw":
                pad_img = np.concatenate((img, img, img), axis=0)
        elif self.method == "padzero":
            if layout == "hwc":
                h = img.shape[0]
                w = img.shape[1]
                pad_img = np.zeros((h, w, 3))
                if data["cur_pattern"] == "BGGR":
                    pad_img[:, :, 0][1:h:2, 1:w:2] = img[1:h:2, 1:w:2, 0]  # r
                    pad_img[:, :, 1][0:h:2, 1:w:2] = img[0:h:2, 1:w:2, 0]  # g
                    pad_img[:, :, 1][1:h:2, 0:w:2] = img[1:h:2, 0:w:2, 0]  # g
                    pad_img[:, :, 2][0:h:2, 0:w:2] = img[0:h:2, 0:w:2, 0]  # b
                elif data["cur_pattern"] == "RGGB":
                    pad_img[:, :, 0][0:h:2, 0:w:2] = img[0:h:2, 0:w:2, 0]  # r
                    pad_img[:, :, 1][0:h:2, 1:w:2] = img[0:h:2, 1:w:2, 0]  # g
                    pad_img[:, :, 1][1:h:2, 0:w:2] = img[1:h:2, 0:w:2, 0]  # g
                    pad_img[:, :, 2][1:h:2, 1:w:2] = img[1:h:2, 1:w:2, 0]  # b
                elif data["cur_pattern"] == "GBRG":
                    pad_img[:, :, 0][1:h:2, 0:w:2] = img[1:h:2, 0:w:2, 0]  # r
                    pad_img[:, :, 1][0:h:2, 0:w:2] = img[0:h:2, 0:w:2, 0]  # g
                    pad_img[:, :, 1][1:h:2, 1:w:2] = img[1:h:2, 1:w:2, 0]  # g
                    pad_img[:, :, 2][0:h:2, 1:w:2] = img[0:h:2, 1:w:2, 0]  # b
                elif data["cur_pattern"] == "GRBG":
                    pad_img[:, :, 0][0:h:2, 1:w:2] = img[0:h:2, 1:w:2, 0]  # r
                    pad_img[:, :, 1][0:h:2, 0:w:2] = img[0:h:2, 0:w:2, 0]  # g
                    pad_img[:, :, 1][1:h:2, 1:w:2] = img[1:h:2, 1:w:2, 0]  # g
                    pad_img[:, :, 2][1:h:2, 0:w:2] = img[1:h:2, 0:w:2, 0]  # b
            elif layout == "chw":
                h = img.shape[1]
                w = img.shape[2]
                pad_img = np.zeros((3, h, w))
                if data["cur_pattern"] == "BGGR":
                    pad_img[0, :, :][1:h:2, 1:w:2] = img[0, 1:h:2, 1:w:2]  # r
                    pad_img[1, :, :][0:h:2, 1:w:2] = img[0, 0:h:2, 1:w:2]  # g
                    pad_img[1, :, :][1:h:2, 0:w:2] = img[0, 1:h:2, 0:w:2]  # g
                    pad_img[2, :, :][0:h:2, 0:w:2] = img[0, 0:h:2, 0:w:2]  # b
                elif data["cur_pattern"] == "RGGB":
                    pad_img[0, :, :][0:h:2, 0:w:2] = img[0, 0:h:2, 0:w:2]  # r
                    pad_img[1, :, :][0:h:2, 1:w:2] = img[0, 0:h:2, 1:w:2]  # g
                    pad_img[1, :, :][1:h:2, 0:w:2] = img[0, 1:h:2, 0:w:2]  # g
                    pad_img[2, :, :][1:h:2, 1:w:2] = img[0, 1:h:2, 1:w:2]  # b
                elif data["cur_pattern"] == "GBRG":
                    pad_img[0, :, :][1:h:2, 0:w:2] = img[0, 1:h:2, 0:w:2]  # r
                    pad_img[1, :, :][0:h:2, 0:w:2] = img[0, 0:h:2, 0:w:2]  # g
                    pad_img[1, :, :][1:h:2, 1:w:2] = img[0, 1:h:2, 1:w:2]  # g
                    pad_img[2, :, :][0:h:2, 1:w:2] = img[0, 0:h:2, 1:w:2]  # b
                elif data["cur_pattern"] == "GRBG":
                    pad_img[0, :, :][0:h:2, 1:w:2] = img[0, 0:h:2, 1:w:2]  # r
                    pad_img[1, :, :][0:h:2, 0:w:2] = img[0, 0:h:2, 0:w:2]  # g
                    pad_img[1, :, :][1:h:2, 1:w:2] = img[0, 1:h:2, 1:w:2]  # g
                    pad_img[2, :, :][1:h:2, 0:w:2] = img[0, 1:h:2, 0:w:2]  # b
        data["img"] = pad_img
        data["img_shape"] = pad_img.shape

    def __call__(self, data):
        self._pad_raw(data)

        return data

    def __repr__(self):
        repr_str = self.__class__.__name__ + ": "
        repr_str += f"method={self.method}, "

        return repr_str


@OBJECT_REGISTRY.register
class DEC(object):
    def __init__(self, sensor="ovx8b", input_bit=12, output_bit=20) -> None:
        self.input_bit = input_bit
        self.output_bit = output_bit
        self.sensor = sensor
        if sensor == "ovx8b":
            if input_bit == 12:
                if output_bit == 20:
                    self.x = [
                        0,
                        256,
                        319,
                        383,
                        447,
                        511,
                        575,
                        639,
                        703,
                        767,
                        831,
                        895,
                        959,
                        1023,
                        1087,
                        1151,
                        1215,
                        1279,
                        1343,
                        1471,
                        1599,
                        1855,
                        2111,
                        2367,
                        2623,
                        2879,
                        3135,
                        3391,
                        3519,
                        3647,
                        3775,
                        3903,
                        4031,
                        4095,
                    ]
                    self.y = [
                        0,
                        255,
                        511,
                        767,
                        1023,
                        1535,
                        2047,
                        2559,
                        3071,
                        4095,
                        5119,
                        6143,
                        8191,
                        10239,
                        12287,
                        16383,
                        24575,
                        32767,
                        40959,
                        49151,
                        65535,
                        81919,
                        98303,
                        114687,
                        131071,
                        163839,
                        196607,
                        262143,
                        327679,
                        393215,
                        524287,
                        655359,
                        786431,
                        1048575,
                    ]
                elif output_bit == 24:
                    self.x = [
                        0,
                        512,
                        767,
                        1023,
                        1279,
                        1535,
                        1663,
                        1791,
                        1919,
                        2047,
                        2175,
                        2303,
                        2431,
                        2559,
                        2687,
                        2815,
                        2943,
                        3071,
                        3135,
                        3199,
                        3263,
                        3327,
                        3391,
                        3455,
                        3519,
                        3583,
                        3647,
                        3711,
                        3775,
                        3839,
                        3903,
                        3967,
                        4031,
                        4095,
                    ]
                    self.y = [
                        0,
                        511,
                        1023,
                        2047,
                        3071,
                        4095,
                        6143,
                        8191,
                        12287,
                        16383,
                        24575,
                        32767,
                        49151,
                        65535,
                        98303,
                        131071,
                        196607,
                        262143,
                        393215,
                        524287,
                        786431,
                        1048575,
                        1310719,
                        1572863,
                        2097151,
                        2621439,
                        3145727,
                        4194303,
                        5242879,
                        6291455,
                        8388607,
                        10485759,
                        12582911,
                        16777215,
                    ]
                else:
                    raise NotImplementedError(
                        "output_bit %d not supported" % output_bit
                    )
        elif sensor == "ar0820":
            if input_bit == 12:
                if output_bit == 20:
                    self.x = [
                        0,
                        512,
                        837,
                        1162,
                        1487,
                        1812,
                        2137,
                        2462,
                        2787,
                        3112,
                        3437,
                        3762,
                        4087,
                    ]
                    self.y = [
                        0,
                        512,
                        1024,
                        2048,
                        4096,
                        8192,
                        16384,
                        32768,
                        65536,
                        131072,
                        262144,
                        524288,
                        1048576,
                    ]
                else:
                    raise NotImplementedError(
                        "output_bit %d not supported" % output_bit
                    )
        else:
            raise NotImplementedError("sensor %s not supported" % sensor)

    def __call__(self, data):
        if "img" in data:
            img = data["img"]
            img = np.interp(img, self.x, self.y)
            img = np.clip(img, 0, 2 ** self.output_bit - 1)
            img = img.astype(np.uint32)
            data["img"] = img
        return data

    def __repr__(self):
        repr_str = self.__class__.__name__ + ": "
        repr_str += f"sensor={self.sensor}, input_bit={self.input_bit}, \
            output_bit={self.output_bit}"

        return repr_str


@OBJECT_REGISTRY.register
class DGain(object):
    def __init__(self, output_bit):
        self.output_bit = output_bit

    def __call__(self, data):
        if "img" in data and "dgain" in data:
            data["img"] = np.clip(
                data["img"] * data["dgain"], 0.0, 2 ** self.output_bit - 1
            )

        return data

    def __repr__(self):
        repr_str = self.__class__.__name__ + ": "
        repr_str += f"output_bit={self.output_bit}"
        return repr_str


def get_blc_value(sensor):
    if sensor == "ovx8b":
        blc = 64
    elif sensor == "ar0820":
        blc = 168
    else:
        raise ValueError()

    return blc


@OBJECT_REGISTRY.register
class BLC(object):
    def __init__(self, sensor, output_bit):
        self.sensor = sensor
        self.blc_val = get_blc_value(sensor)
        self.output_bit = output_bit

    def __call__(self, data):
        if "img" in data:
            data["img"] = np.clip(
                data["img"] - self.blc_val, 0.0, 2 ** self.output_bit - 1
            )
        return data

    def __repr__(self):
        repr_str = self.__class__.__name__ + ": "
        repr_str += f"sensor={self.sensor}, output_bit={self.output_bit}, \
            blc_val={self.blc_val}"

        return repr_str


@OBJECT_REGISTRY.register
class NNLogLut(object):
    def __init__(
        self,
        input_bit,
        output_bit,
        lut_x,
        lut_y,
        pregamma=None,
    ) -> None:
        self.lut_x = lut_x
        self.lut_y = lut_y
        self.input_bit = input_bit
        self.output_bit = output_bit
        self.pregamma = pregamma

    def __call__(self, data):
        if "img" in data:
            img = data["img"].astype(np.float32)
            img = img / (2 ** self.input_bit - 1)

            if self.pregamma is not None:
                img = np.power(img, self.pregamma)

            img = np.interp(img, self.lut_x, self.lut_y)
            img = img * (2 ** self.output_bit - 1)
            img = np.clip(img, 0, 2 ** self.output_bit - 1).round()

            if self.output_bit <= 8:
                img = img.astype(np.uint8)
            elif self.output_bit <= 16 and self.output_bit > 8:
                img = img.astype(np.uint16)
            elif self.output_bit > 16 and self.output_bit <= 24:
                img = img.astype(np.uint32)
            else:
                raise NotImplementedError(
                    "output bit %d is not implemented" % (self.output_bit)
                )

            data["img"] = img

        return data

    def __repr__(self):
        repr_str = self.__class__.__name__ + ": "
        repr_str += f"input_bit={self.input_bit}, \
            output_bit={self.output_bit}, \
            pregamma={self.pregamma}"

        return repr_str
