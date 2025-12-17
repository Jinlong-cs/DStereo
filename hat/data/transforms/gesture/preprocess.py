from typing import Optional

import numpy as np

from hat.registry import OBJECT_REGISTRY

__all__ = ["ActionClipDataPreProcess", "ActionClipDataPostProcess"]


@OBJECT_REGISTRY.register
class ActionClipDataPreProcess(object):
    """Data preprocess for clip data.

    Including replenish missing data. If use rgb branch,
    sample img data according to act_img_seq_len.

    Args:
        act_kps_num_classes: num of action class.
            Defaults to 59.
        act_kps_seq_len: length of kps clip.
            Defaults to 32.
        num_kps: num of kps, such as 21 for hand.
            Defaults to 21.
        use_3d_kps: whether to use 3d kps of hand.
            Defaults to False.
        use_kps_score: whether to use score of kps.
            Defaults to True.
        act_img_seq_len: length of img clip.
            Defaults to None.
        use_rgb_branch: whether to use rgb branch.
            Defaults to False.
        use_replenish: whether to replenish missing data.
            Defaults to True.
        repenish_max_frame: Maximum drop frame allowed when repenishing.
            Defaults to 10000.
    """

    def __init__(
        self,
        act_kps_num_classes: int = 59,
        act_kps_seq_len: int = 32,
        num_kps: int = 21,
        use_3d_kps: bool = False,
        use_kps_score: bool = True,
        act_img_seq_len: Optional[int] = None,
        use_rgb_branch: bool = False,
        use_replenish: bool = True,
        repenish_max_frame: int = 10000,
    ):
        # network params
        self._act_kps_num_classes = act_kps_num_classes
        # for kps branch
        self._act_kps_seq_len = act_kps_seq_len
        self._num_kps = num_kps
        self._use_3d_kps = use_3d_kps
        self._use_kps_score = use_kps_score

        # for rgb branch
        self._use_rgb_branch = use_rgb_branch
        if act_img_seq_len is None:
            act_img_seq_len = act_kps_seq_len
        self._act_img_seq_len = act_img_seq_len
        # preprocess params
        self._replenish = use_replenish
        self._repenish_max_frame = repenish_max_frame

        self.train_crop_image = False
        if self._use_rgb_branch:
            if act_img_seq_len != act_kps_seq_len:
                assert (
                    int(act_kps_seq_len) % int(act_img_seq_len) == 0
                ), f"seq_len:{act_kps_seq_len} should be divisible  \
                    by img_seq_len:{act_img_seq_len}"

            img_stride = int(act_kps_seq_len / act_img_seq_len)
            if img_stride < act_kps_seq_len:
                self.img_idxs = np.arange(act_kps_seq_len)[0::img_stride]
            else:
                # for one frame and multiframe kps
                self.img_idxs = np.array([act_kps_seq_len / 2], dtype=np.int64)

            self.train_crop_image = True

    def _downsample_img_data(self, clip_img, clip_kps_meta_info, rec_labels):
        # todo: split this func to class
        # for cropped hand img, unsupported full img
        if clip_img is not None and len(clip_img) > self._act_img_seq_len:
            # sample seqlen of img from clip img
            if self.train_crop_image:
                seq_track_ids = clip_kps_meta_info["track_ids"]
                roi_ids = [
                    rec_labels[idx]["track_id"].index(seq_track_ids[idx])
                    if seq_track_ids[idx] != -1
                    else 0
                    for idx in self.img_idxs
                ]
                select_imgs, handboxes, cropboxes = [], [], []
                raw_shapes, full_image_shapes = [], []
                for idx, jdx in zip(self.img_idxs, roi_ids):
                    select_imgs.append(
                        clip_img[idx][jdx] if len(clip_img[idx]) != 0 else None
                    )
                    handboxes.append(
                        rec_labels[idx]["hand_boxes"][jdx]
                        if len(rec_labels[idx]["hand_boxes"]) != 0
                        else None
                    )
                    cropboxes.append(
                        rec_labels[idx]["crop_boxes"][jdx]
                        if len(rec_labels[idx]["crop_boxes"]) != 0
                        else None
                    )
                    raw_shapes.append(
                        rec_labels[idx]["raw_shape"][jdx]
                        if len(rec_labels[idx]["raw_shape"]) != 0
                        else None
                    )
                    full_image_shapes.append(rec_labels[idx]["image_shape"])
                clip_img = select_imgs
            else:
                raise NotImplementedError(
                    "The rgb branch only supports crop-img training mode"
                )
        return clip_img, handboxes, cropboxes, raw_shapes, full_image_shapes

    def _discard_null_data(
        self,
        clip_kps_meta_info,
        clip_img,
        clip_keypoints,
        miss_data_num,
        seq_label,
        act_label,
        full_image_shapes,
    ):
        if self._use_rgb_branch:
            if self.train_crop_image:
                # for rgb branch: crop hand img
                img_shape = (
                    clip_kps_meta_info["img_shape"]
                    if clip_img is None
                    else full_image_shapes[0]
                )
            else:
                raise NotImplementedError(
                    "The rgb branch only supports crop-img training mode"
                )
        else:
            img_shape = clip_kps_meta_info["img_shape"]

        label_weight = 1.0
        if (
            img_shape[0] < 1
            or img_shape[1] < 1
            or (clip_keypoints == -1).all(axis=1).any()
            or (
                clip_img is not None and any([img is None for img in clip_img])
            )
            or miss_data_num > self._repenish_max_frame
        ):
            act_label = -1
            seq_label[:] = -1
            label_weight = 0.0
        return label_weight, act_label, seq_label, img_shape

    def _reconstruct_kps(self, keypoints):
        assert keypoints.shape[0] == self._act_kps_seq_len
        num_kps_t3, num_kps_t4 = self._num_kps * 3, self._num_kps * 4
        if self._use_3d_kps:
            assert keypoints.shape[1] >= num_kps_t4
            if self._use_kps_score:
                keypoints = keypoints[:, :num_kps_t4]
            else:
                s_idxs = list(range(3, num_kps_t4, 4))
                keypoints = np.delete(keypoints, s_idxs, axis=1)[
                    :, :num_kps_t3
                ]
        else:  # 2d
            if keypoints.shape[1] >= num_kps_t4:
                z_idxs = list(range(2, num_kps_t4, 4))
                keypoints = np.delete(keypoints, z_idxs, axis=1)
            keypoints = keypoints[:, :num_kps_t3]
            if not self._use_kps_score:
                s_idxs = list(range(2, num_kps_t3, 3))
                keypoints = np.delete(keypoints, s_idxs, axis=1)
        return keypoints

    def __call__(self, data):
        clip_img = data["frames"]
        rec_labels = data["clip_rec_label"]
        clip_kps_meta_info = data["clip_kps_meta_info"]

        # step1: get label and check seqlabel/label size
        act_label = clip_kps_meta_info["act_label"]
        assert act_label.size == 1, "only support 1 class for now"
        act_label = int(act_label[0])
        seq_label = clip_kps_meta_info["seq_label"]
        assert act_label < self._act_kps_num_classes
        hand_attrs = clip_kps_meta_info["attrs"]["hand_left_right_attr"][:, -1]
        # hand_attrs > 0.5 -> right, 0 <= hand_attrs <= 0.5 -> left.
        is_left_hand = int(
            (hand_attrs > 0.5).sum() < ((hand_attrs >= 0).sum() * 0.5)
        )

        # step2: downsample, clip img is list, may be empty
        clip_img = None if not bool(clip_img) else clip_img
        if self._use_rgb_branch:
            (
                clip_img,
                handboxes,
                cropboxes,
                raw_shapes,
                full_image_shapes,
            ) = self._downsample_img_data(
                clip_img, clip_kps_meta_info, rec_labels
            )
        else:
            full_image_shapes = None

        # step3: replenish missing data
        # keypoints shape: (seq_len, num_kps*feat_ch+1)
        clip_keypoints = clip_kps_meta_info["clip_keypoints"]
        clip_boxes = clip_kps_meta_info["clip_boxes"]
        if self._replenish:
            # replenish missdata by preframe info
            miss_data_num = 0
            for kidx in range(1, self._act_kps_seq_len):
                if (clip_keypoints[kidx] == -1).all() and (
                    clip_keypoints[kidx - 1] != -1
                ).any():
                    clip_keypoints[kidx] = clip_keypoints[kidx - 1].copy()
                    clip_boxes[kidx] = clip_boxes[kidx - 1].copy()
                    miss_data_num += 1

            if self._use_rgb_branch:
                if clip_img is not None:
                    for jidx in range(1, len(self.img_idxs)):
                        if (
                            clip_img[jidx] is None
                            and clip_img[jidx - 1] is not None
                        ):
                            clip_img[jidx] = clip_img[jidx - 1].copy()
                            if self.train_crop_image:
                                handboxes[jidx] = handboxes[jidx - 1].copy()
                                cropboxes[jidx] = cropboxes[jidx - 1].copy()
                                raw_shapes[jidx] = raw_shapes[jidx - 1].copy()
                                full_image_shapes[jidx] = full_image_shapes[
                                    jidx - 1
                                ].copy()

        # step4: determine whether to discard the sample
        (
            label_weight,
            act_label,
            seq_label,
            img_shape,
        ) = self._discard_null_data(
            clip_kps_meta_info,
            clip_img,
            clip_keypoints,
            miss_data_num,
            seq_label,
            act_label,
            full_image_shapes,
        )

        # step5: determine to keep kps info: 3d, score
        clip_keypoints = self._reconstruct_kps(clip_keypoints)

        data.update(
            {
                "act_label": act_label,
                "seq_label": seq_label,
                "label_weight": label_weight,
                "img_shape": img_shape,
                "clip_keypoints": clip_keypoints,
                "clip_boxes": clip_boxes,
                "frames": clip_img,
                "is_left_hand": is_left_hand,
            }
        )
        if self._use_rgb_branch:
            data.update(
                {
                    "handboxes": handboxes,
                    "cropboxes": cropboxes,
                    "raw_shapes": raw_shapes,
                    "full_image_shapes": full_image_shapes,
                    "rec_shape": clip_img[0].shape[0],
                }
            )
        return data

    def __repr__(self):
        repr_str = self.__class__.__name__ + ": "
        repr_str += f"act_kps_num_classes={self._act_kps_num_classes}"
        repr_str += f"act_kps_seq_len={self._act_kps_seq_len}"
        repr_str += f"num_kps={self._num_kps}"
        repr_str += f"use_3d_kps={self._use_3d_kps}"
        repr_str += f"use_kps_score={self._use_kps_score}"
        repr_str += f"use_rgb_branch={self._use_rgb_branch}"
        repr_str += f"act_img_seq_len={self._act_img_seq_len}"
        repr_str += f"act_img_seq_len={self._act_img_seq_len}"
        repr_str += f"use_replenish={self._replenish}"
        repr_str += f"repenish_max_frame={self._repenish_max_frame}"
        return repr_str


@OBJECT_REGISTRY.register
class ActionClipDataPostProcess(object):
    """Data postprocess for clip data.

    Including delete some data.

    Args:
        keep_keynames: The name of the data to be retained.
            Defaults to None.
    """

    def __init__(self, keep_keynames: list = None):
        self.keep_keynames = keep_keynames

    def __call__(self, data):
        if self.keep_keynames is not None:
            return {
                key: value
                for key, value in data.items()
                if key in self.keep_keynames
            }
        return data

    def __repr__(self):
        repr_str = self.__class__.__name__ + ": "
        repr_str += f"keep_keynames={self.keep_keynames}"
