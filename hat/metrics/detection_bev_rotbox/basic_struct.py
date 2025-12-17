import logging
from collections import defaultdict
from typing import Any

from hat.metrics.detection3d.basic_struct import Instance3dObj

logger = logging.getLogger(__name__)

__all__ = [
    "Instance3dObj",
]


class InstanceObj(object):
    """Instance object for bev detection.

    Args:
        timestamp: timestamp key
        meta: camera information from various perspectives.
    """

    def __init__(
        self,
        timestamp: str,
        meta: Any = None,
    ):
        self.timestamp = timestamp
        self.meta = (meta,)
        self.anno_info = defaultdict(list)

    def add_obj(self, anno: dict):
        self.anno_info["timestamp"].append(self.timestamp)
        for key, value in anno.items():
            if key in self.anno_info:
                self.anno_info[key].append(value)
            else:
                self.anno_info[key] = [value]


def parse_to_struct(
    infos_dict: dict,
    eval_class: str,
    info_type: str = "gt",
):
    """Parse gt.json or pred.json to aidi eval struct.

    Args:
        infos_dict: gt_info or pred_info, gt_info:
        {
            "timestamp": str, time stamp.
            "imgs_meta": dict, camera information from various perspectives,
                including image_key, shape, calib and camera_model.
            eval_class: dict, each instance info.
        }
        eval_class: eval class name.
        info_type: "gt" or "pred".
        more info:
            https://horizonrobotics.feishu.cn/wiki/wikcnBkQbEaDOeqwzmK661cPCpC
    """
    assert info_type in ["gt", "pred"]

    if info_type == "gt":
        timestamp = infos_dict["timestamp"]
        imgs_meta = infos_dict.get("imgs_meta", [])
        instance_objs = InstanceObj(
            timestamp=timestamp,
            meta={
                "eval_vis_cfg": infos_dict["eval_vis_cfg"],
                "imgs_meta": imgs_meta,
            },
        )
        assert eval_class in infos_dict, f"{eval_class} not in gt file!"
        annos = infos_dict[eval_class]

        for obj_info in annos["objects"]:
            anno = {
                "vcs_discobj_wh": obj_info["in_vcs"]["wh"],
                "vcs_discobj_loc": obj_info["in_vcs"]["loc"],
                "vcs_discobj_yaw": obj_info["in_vcs"]["yaw"],
                "vcs_discobj_cls": obj_info["cls_id"],
                "vcs_points": obj_info["in_vcs"]["vcs_points"],
                "vcs_discobj_rate_visible": obj_info.get("rate_visible", None),
                "score": obj_info.get("score", None),
                "vcs_discobj_ignore": obj_info.get("ignore", False),
                "meta": {
                    "eval_vis_cfg": infos_dict["eval_vis_cfg"],
                    "imgs_meta": imgs_meta,
                },
                "uid": obj_info["uid"],
            }

            instance_objs.add_obj(anno)
    else:
        objs_infos = infos_dict.get(eval_class)
        assert (
            objs_infos is not None
        ), f"{eval_class} not in prediction file!!!"
        timestamp = infos_dict["timestamp"]
        instance_objs = InstanceObj(
            timestamp=timestamp,
            meta={
                "image_keys": infos_dict["image_keys"],
            },
        )
        for obj_info in objs_infos:
            output = {
                "pred_loc": obj_info["pred_loc"],
                "pred_wh": obj_info["pred_wh"],
                "pred_score": obj_info["pred_score"],
                "pred_yaw": obj_info["pred_yaw"],
                "pred_bev_discobj_cls_id": obj_info["pred_bev_discobj_cls_id"],
            }
            instance_objs.add_obj(output)

    return instance_objs.anno_info
