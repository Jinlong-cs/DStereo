# Copyright (c) Horizon Robotics. All rights reserved.

from typing import Dict

from hat.data.transforms.detection import ToTensor
from hat.registry import OBJECT_REGISTRY

__all__ = [
    "ActionLabelMap",
    "ActionLabelToTensor",
]


@OBJECT_REGISTRY.register
class ActionLabelMap(object):
    """Label mapping for action task.

    Args:
        convert_label_map: Label mapping dictionary.
            Defaults to None.
    """

    def __init__(self, convert_label_map: Dict = None):
        self.convert_label_map = convert_label_map

    def __call__(self, data):
        if "act_label" in data and self.convert_label_map is not None:
            act_label = data["act_label"]
            data["act_label"] = self.convert_label_map.get(
                str(act_label), act_label
            )
        return data

    def __repr__(self):
        repr_str = self.__class__.__name__ + ": "
        repr_str += f"convert_label_map={self.convert_label_map}"
        return repr_str


@OBJECT_REGISTRY.register
class ActionLabelToTensor(ToTensor):
    """Convert objects of label(python type) to torch.Tensor.

    Args:
        seq_len: Sequence length for clip.
            Default to 32.
        use_weight_box_vis_ratio: whether to use ratio of vis to label weight.
            Default to True.
        visible_ratio_weight: label weight depend on visible ratio.
            Default to None.
    """

    def __init__(
        self,
        seq_len: int = 32,
        use_weight_box_vis_ratio: bool = True,
        visible_ratio_weight: Dict = None,
    ):
        super(ActionLabelToTensor, self).__init__()
        self.seq_len = seq_len
        self.use_weight_box_vis_ratio = use_weight_box_vis_ratio
        if visible_ratio_weight is None:
            self.visible_ratio_weight = {
                "full_visible": 1.0,
                "occluded": 0.7,
                "heavily_occluded": 0.0,
                "invisible": 0.0,
            }
        else:
            self.visible_ratio_weight = visible_ratio_weight

    def __call__(self, data):
        # step1: set weight -> 0 if is_ignore is True or act_label = -1.
        if data["is_ignore"]:
            data["act_label"] = int(-1)
        if data["act_label"] == -1:
            data["seq_label"][:] = -1
            data["label_weight"] = 0.0
        ignore = data["clip_kps_meta_info"].get("ignore", "no")
        if isinstance(ignore, list):
            ignore = ignore[int(self.seq_len / 2)]
        if ignore == "yes":
            data["label_weight"] = 0.0

        # step2: uing weight_box_vis_ratio to scale label weight
        if self.use_weight_box_vis_ratio:
            vis_ratio_weight = self.visible_ratio_weight.get(
                (
                    data["clip_kps_meta_info"]["occlusion"][
                        int(self.seq_len / 2)
                    ]
                ),
                0.0,
            )
            data["label_weight"] *= vis_ratio_weight

        # step3: totensor
        for data_key in ["act_label", "seq_label", "label_weight"]:
            data[data_key] = self._to_tensor(data[data_key])
        return data

    def __repr__(self):
        repr_str = self.__class__.__name__ + ": "
        repr_str += f"seq_len={self.seq_len}"
        repr_str += (
            f"use_weight_box_vis_ratio={self.use_weight_box_vis_ratio}"  # noqa
        )
        repr_str += f"visible_ratio_weight={self.visible_ratio_weight}"
        return repr_str
