# Copyright (c) Horizon Robotics. All rights reserved.
from typing import Tuple

from init_path import assert_func
from init_path import get_data as _get_data

from hat.data.transforms.gesture.affine import (
    ActionFlipKpsFrames,
    ActionRotateKpsFrames,
)
from hat.data.transforms.landmark import CropRecROI as ActionCropRecROI


class ActionRotateKpsFramesForTest(ActionRotateKpsFrames):
    def __init__(
        self,
        p_rotate: float = 1,
        rotate_range: int = 15,
        feat_ch: int = 3,
        use_rgb_branch: bool = False,
    ):
        super().__init__(p_rotate, rotate_range, feat_ch, use_rgb_branch)

    def __call__(self, data):
        # raw angle
        # angle = random.randint(-self._rotate_range, self._rotate_range)
        angle = data["angle"]
        new_img_shape = self._get_rotate_img_shape(
            angle, data["img_shape"][:2]
        )
        # kps branch
        data["clip_keypoints"] = self._rotate_kps_branch(
            data["clip_keypoints"],
            angle,
            data["img_shape"],
            new_img_shape,
            feat_ch=self.feat_ch,
        )

        if self._use_rgb_branch:
            if data["frames"] is not None:
                (
                    data["frames"],
                    data["clip_boxes"],
                ) = self._rotate_rgb_branch(
                    data["frames"],
                    data["clip_boxes"],
                    data["img_shape"],
                    angle,
                    new_img_shape,
                )
        data["img_shape"] = new_img_shape
        return data


class ActionCropRecROIForTest(ActionCropRecROI):
    def __init__(
        self,
        crop_type: str,
        target_shape: Tuple[int, int, int],
        base_roi: Tuple[float, float, float, float],
        crop_jitter_range: float,
        center_shift_range: float = 0.0,
        random_type: str = "gaussian",
        crop_frames_prob_dist: Tuple[float, float, float] = (0.3, 0.7, 0),
        use_crop_limit: bool = False,
    ):
        super().__init__(
            crop_type,
            target_shape,
            base_roi,
            crop_jitter_range,
            center_shift_range,
            random_type,
            crop_frames_prob_dist,
            use_crop_limit,
        )

    def _get_random_num(self):
        return 0.5


def get_data(filename_input, filename_output):
    # Intermediate results from gluonperson
    input_data, output_data = _get_data(filename_input, filename_output)
    return input_data, output_data


def test_rotate():
    rotate_input, gt_rotate_output = get_data(
        filename_input="rotate_input.pkl", filename_output="rotate_output.pkl"
    )

    arkf = ActionRotateKpsFramesForTest(
        p_rotate=1, rotate_range=15, feat_ch=3, use_rgb_branch=True
    )
    pred_rotate_output = arkf(rotate_input)
    assert_func(pred_rotate_output, gt_rotate_output)


def test_flip():
    flip_input, gt_flip_output = get_data(
        filename_input="flip_all_input3.pkl",
        filename_output="flip_all_output3.pkl",
    )

    afkf = ActionFlipKpsFrames(
        p_flip=1, feat_ch=3, flip_label_dict=flip_input["flip_label_dict"]
    )
    pred_data = afkf(flip_input)
    assert_func(pred_data, gt_flip_output)


def test_crop():
    crop_input, gt_crop_output = get_data(
        filename_input="crop_input.pkl",
        filename_output="crop_random_0.5_output.pkl",
    )
    crop_params = {
        "crop_type": "",
        "target_shape": (128, 128, 3),
        # norm ratio: 1.25, shape of img in rec: 256
        # 48 = (256 - 128*1.25) // 2
        "base_roi": [48, 48, 208, 208],
        "crop_jitter_range": 0.5,
        "center_shift_range": 0,
        "random_type": "gaussian",
        "use_crop_limit": False,
        "crop_frames_prob_dist": (1, 0, 0),
    }
    acrr = ActionCropRecROIForTest(**crop_params)
    pred_data = acrr(crop_input)
    assert_func(pred_data, gt_crop_output)
