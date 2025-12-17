import logging
from collections import defaultdict
from typing import Any, Dict, List

import numpy as np
from scipy.spatial.transform import Rotation as R

from hat.metrics.detection3d.basic_struct import Instance3dObj

logger = logging.getLogger(__name__)

__all__ = [
    "Instance3dObj",
    "ImgObj",
    "ScenceObj",
]


class ImgObj(object):
    """Single image perception object for detection bev.

    Args:
        image_key: Image name.
        objects_3d: Ground truth or prediction objects
        width: Image width. Defaults to None.
        height: Image height. Defaults to None.
        calib: Calibration. Defaults to None.
        ignore_mask: Ignore mask in gt label. Defaults to None.
        meta: Image meta info.
    """

    def __init__(
        self,
        image_key: str,
        objects_3d: List[Instance3dObj] = None,
        width: int = None,
        height: int = None,
        calib: Dict[str, Any] = None,
        ignore_mask: dict = None,
        meta: Any = None,
    ):

        self.image_key = image_key
        self.calib = calib
        self.width = width
        self.height = height
        self.ignore_mask = ignore_mask
        self.meta = meta
        self.objects_3d = (
            {id(obj): obj for obj in objects_3d} if objects_3d else {}
        )
        self.objects_3d_info = (
            {id(obj): {} for obj in objects_3d} if objects_3d else {}
        )

    def add_obj3d(self, object_3d: Instance3dObj):
        """Add 3d object and build obj info.

        Args:
            object_3d: base 3d object
        """
        assert isinstance(object_3d, Instance3dObj)
        self.objects_3d[id(object_3d)] = object_3d
        self.objects_3d_info[id(object_3d)] = {}

    def pop_obj3d(self, object_3d: Instance3dObj):
        """Pop 3d object and info.

        Args:
            object_3d: base 3d object
        """
        self.objects_3d.pop(id(object_3d))
        self.objects_3d_info.pop(id(object_3d))

    def update_id(self):
        """Update obj key by newest obj uid.

        Args:
            None
        """
        new_objects_3d = {}
        new_objects_3d_info = {}
        for old_key, obj3d in self.objects_3d.items():
            new_objects_3d[id(obj3d)] = obj3d
            new_objects_3d_info[id(obj3d)] = self.objects_3d_info[old_key]
        self.objects_3d = new_objects_3d
        self.objects_3d_info = new_objects_3d_info

    def save_obj3d_in_img_info(self, object_3d: Instance3dObj, info: Dict):
        """Save info of 3d obj, like 2d projection in image.

        Args:
            object_3d: base 3d object
            info: any info of 3d obj
        """
        assert id(object_3d) in self.objects_3d_info
        self.objects_3d_info[id(object_3d)].update(info)

    def clear_obj3d_in_img_info(self, objects_3d: Instance3dObj):
        """Clear info of 3d obj.

        Args:
            object_3d: base 3d object
        """
        assert id(objects_3d) in self.objects_3d_info
        self.objects_3d_info[id(objects_3d)] = {}

    def to_dict(self):
        """Cover self image object to dict, usually required when dump json."""
        res_obj = {}
        for obj3d_id in self.objects_3d.keys():
            obj3d = self.objects_3d[obj3d_id]
            obj3d_info = self.objects_3d_info[obj3d_id]
            obj3d_dict = obj3d.to_dict()
            obj3d_dict.update(obj3d_info)
            res_obj[obj3d_id] = obj3d_dict
        results = dict(  # noqa
            image_key=self.image_key,
            objects_3d=res_obj,
            width=self.width,
            height=self.height,
            calib=self.calib,
            ignore_mask=self.ignore_mask,
            meta=self.meta,
        )
        return results


class ScenceObj(object):
    """Single scence(multi-sensor, only image now) object for bev detection.

    Args:
        scence_key: scence key
        instances: Ground truth or prediction 3d objects
        image_objects: The image objects contained in scence
        meta: Scence meta info.
    """

    def __init__(
        self,
        scence_key: str,
        objects_3d: List[Instance3dObj] = None,
        image_objects: List[ImgObj] = None,
        meta: Any = None,
    ):
        self.scence_key = scence_key
        self.objects_3d = (
            {id(obj): obj for obj in objects_3d} if objects_3d else {}
        )
        self.objects_3d_info = (
            {id(obj): {} for obj in objects_3d} if objects_3d else {}
        )
        self.image_objects = (
            {id(obj): obj for obj in image_objects} if image_objects else {}
        )
        self.meta = meta

    def add_obj3d(self, object_3d):
        """Add 3d object and build obj info.

        Args:
            object_3d: base 3d object
        """
        # TODO: add obj3d in sub image obj
        assert isinstance(object_3d, Instance3dObj)
        self.objects_3d[id(object_3d)] = object_3d
        self.objects_3d_info[id(object_3d)] = {}

    def pop_obj3d(self, object_3d):
        """Pop 3d object and info, also pop same 3d obj from image obj.

        Args:
            object_3d: base 3d object
        """
        self.objects_3d.pop(id(object_3d))
        self.objects_3d_info.pop(id(object_3d))
        for img_obj in self.image_objects.values():
            if id(object_3d) in img_obj.objects_3d:
                img_obj.pop_obj3d(object_3d)

    def save_obj3d_in_scence_info(self, object_3d, info):
        """Save scence info of 3d obj.

        Args:
            object_3d: base 3d object
            info: any info of 3d obj
        """
        assert id(object_3d) in self.objects_3d_info
        self.objects_3d_info[id(object_3d)].update(info)

    def clear_obj3d_in_scence_info(self, objects_3d):
        """Clear scence info of 3d obj.

        Args:
            object_3d: base 3d object
        """
        assert id(objects_3d) in self.objects_3d_info
        self.objects_3d_info[id(objects_3d)] = {}

    def update_id(self):
        """Update obj key by newest obj uid.

        Args:
            None
        """
        for img_obj in self.image_objects.values():
            img_obj.update_id()
        new_objects_3d = {}
        new_objects_3d_info = {}
        for old_key, obj3d in self.objects_3d.items():
            new_objects_3d[id(obj3d)] = obj3d
            new_objects_3d_info[id(obj3d)] = self.objects_3d_info[old_key]
        self.objects_3d = new_objects_3d
        self.objects_3d_info = new_objects_3d_info

    def get_obj3d_by_uid(self, uid, return_id=False):
        """Get 3d obj ( and its id if return_id=True) by uid.

        Args:
            uid: 3d object uid
        """
        for key, obj in self.objects_3d.items():
            if obj.meta["uid"] == uid:
                if not return_id:
                    return self.objects_3d[key]
                else:
                    return self.objects_3d[key], key

    def add_image_object(self, image_object):
        """Add image object in scence.

        Args:
            image_object: image object
        """
        self.image_objects[image_object.image_key] = image_object

    def pop_image_object(self, image_object=None, image_key=None):
        """Pop image object in scence, support by image_key.

        Args:
            image_object: image object
            image_key: base image key
        """
        if image_object is not None:
            image_key = image_object.image_key
        elif image_key is not None:
            pass
        else:
            raise ValueError
        self.image_objects.pop(image_key)

    def to_dict(self):
        """Cover scence object to dict, usually required when dump json."""
        keep_key = [
            "is_tp",
            "is_fn",
            "match_gt_obj3d_id",
            "score",
            "drot",
            "dxyp",
        ]
        obj3d = {id: obj.to_dict() for id, obj in self.objects_3d.items()}
        [
            [
                obj3d[id].update({k: v})
                for k, v in info.items()
                if k in keep_key
            ]
            for id, info in self.objects_3d_info.items()
        ]
        img_info = {}
        for key, img_obj in self.image_objects.items():
            img_obj_dict = img_obj.to_dict()
            img_info[key] = img_obj_dict
        results = dict(  # noqa
            scence_key=self.scence_key,
            objects_3d=obj3d,
            image_objects=img_info,
            meta=self.meta,
        )
        return results


class TrackletObj(object):
    """Single object in a tracklet for temporal bev detection.

    Args:
        clip_timestamp: key frame of a clip.
        track_id: track id of the object.
        scence_key: scence key of the frame the object belongs to,
            eg. "LS912_1685955776136".
        pre_scence_key: scence key of the previous frame the object belongs to.
        info_type: type of the input info, eg. "gt", "pred".
        ego_pose: ego pose in the current scence the object belongs to.
        object_3d: the object in a tracklet.
        meta: tracklet meta info.
    """

    def __init__(
        self,
        clip_timestamp: str,
        track_id: str,
        scence_key: str,
        pre_scence_key: str,
        info_type: str,
        ego_pose: List[float] = None,
        object_3d: Instance3dObj = None,
        meta: Any = None,
    ):
        self.clip_timestamp = clip_timestamp
        self.track_id = track_id
        self.scence_key = scence_key
        self.object_3d = object_3d
        self.object_3d_info = {
            "ego_pose": ego_pose,
        }
        self.pre_scence_key = pre_scence_key

        self.delta_time = (
            (
                int(self.scence_key.split("_")[-1])
                - int(self.pre_scence_key.split("_")[-1])
            )
            / 1000
            if self.pre_scence_key
            else None
        )
        self.meta = meta
        self.info_type = info_type
        if self.info_type == "gt":
            self.update_global_loc(ego_pose)

    def update_global_loc(self, ego_pose=None):
        """Update the global location & yaw based on ego pose.

        Args:
            ego_pose: ego pose in the current scence the object belongs to.
        """
        if not ego_pose and not self.object_3d_info["ego_pose"]:
            global_loc = None
        else:
            ego_pose = (
                ego_pose if ego_pose else self.object_3d_info["ego_pose"]
            )
            trans = ego_pose[1:4]
            rot = R.from_quat(ego_pose[4:]).as_matrix()
            T_vcs2global = np.zeros((4, 4), dtype=np.float32)
            T_vcs2global[:3, :3] = rot
            T_vcs2global[:3, 3] = trans
            T_vcs2global[3, 3] = 1.0

            vcs_loc = self.object_3d.location
            vcs_loc = np.array(vcs_loc + [1])
            global_loc = (T_vcs2global @ vcs_loc).tolist()[:-1]

            vcs_yaw_vec = [
                np.cos(self.object_3d.rotation_y),
                np.sin(self.object_3d.rotation_y),
                0,
            ]
            global_yaw_vec = (rot @ vcs_yaw_vec).tolist()
            global_yaw = np.arctan2(global_yaw_vec[1], global_yaw_vec[0])

        self.object_3d_info.update(
            {
                "global_loc": global_loc,
                "global_yaw": global_yaw,
            }
        )

    def convert_to_dict(self):
        """Convert tracklet object to dict, usually required when dump json."""
        results = dict(  # noqa
            clip_timestamp=self.clip_timestamp,
            track_id=self.track_id,
            scence_key=self.scence_key,
            pre_scence_key=self.pre_scence_key,
            objects_3d=self.object_3d.to_dict(),
            objects_3d_info=self.object_3d_info,
            meta=self.meta,
        )
        return results


class ClipObj(object):
    """Single clip (contains multiple frames) for temporal bev detection.

    Args:
        clip_timestamp: scence key of the key frame in the clip.
        info_type: type of the input info, eg. "gt", "pred".
        scence_objects: The scence objects contained in a clip.
        meta: Clip meta info.


    Examples::
        demo_instance1 = Instance3dObj(
            dimensions=[1.8, 2.5, 4.5],
            location=[1,1,2],
            rotation_y=0.5,
        )
        demo_instance2 = Instance3dObj(
            dimensions=[2.0, 2.8, 4.7],
            location=[10,-10,2],
            rotation_y=0.2,
        )

        # demo_scence1 contains 2 3d objects:
        demo_scence1 = ScenceObj(
            scence_key="LS912_1685955776136",
            objects_3d=[demo_instance1, demo_instance2],
        )
        # demo_scence2 contains 0 3d object:
        demo_scence2 = ScenceObj(
            scence_key="LS912_1685955776036",
            objects_3d=[],
        )
        # demo_scence3 contains 0 3d object:
        demo_scence3 = ScenceObj(
            scence_key="LS912_1685955775936",
            objects_3d=[],
        )

        # demo_clip contains 3 frames:
        demo_clip = ClipObj(
            clip_timestamp="LS912_1685955776136",
            info_type="gt",
            scence_objects=[demo_scence1, demo_scence2, demo_scence3],
        )
    """

    def __init__(
        self,
        clip_timestamp: str,
        info_type: str,
        scence_objects: List[ScenceObj] = None,
        meta: Any = None,
    ):
        self.clip_timestamp = clip_timestamp
        self.info_type = info_type
        self.scence_obj = (
            {obj.scence_key: obj for obj in scence_objects}
            if scence_objects
            else {}
        )
        self.scence_obj_info = {}
        self.tracklet_dict = None
        self.meta = meta
        self.sequence_list = []
        self.temporal_info_list = ["velocity", "acceleration", "yaw_rate"]

    def add_scenceObj(self, scence_obj):
        """Add signle scence frame and build scence info.

        Args:
            scence_obj: base scence frame
        """
        assert isinstance(scence_obj, ScenceObj)
        self.scence_obj[scence_obj.scence_key] = scence_obj
        self.scence_obj_info[scence_obj.scence_key] = {}

    def add_scenceObj_list(self, scence_objects):
        """Add multiple scence frame and build scence info.

        Args:
            scence_objects: list of scence frame
        """
        for scence_obj in scence_objects:
            if (
                self.info_type == "gt"
                and scence_obj.meta.get("clip_timestamp", None)
                != self.clip_timestamp
            ):
                continue
            self.add_scenceObj(scence_obj)

    def get_sequence_list(self):
        """Get ascending order of scence key in a clip."""
        sequence_list = []
        timestamp_list = []
        plate = list(self.scence_obj.keys())[0].split("_")[0]
        for scence_key in self.scence_obj.keys():
            # scence_key: plate_timestamp
            timestamp_list.append(int(scence_key.split("_")[-1]))
        timestamp_list = sorted(set(timestamp_list))
        sequence_list = [
            f"{plate}_{timestamp}" for timestamp in timestamp_list
        ]
        return sequence_list

    def get_tracklets(self):
        """Get tracklets in a clip based on the track ids."""
        tracklet_dict = defaultdict(list)
        self.sequence_list = self.get_sequence_list()
        pre_scence_key = None
        if self.info_type == "gt":
            for scence_key in self.sequence_list:
                for idx, instance in self.scence_obj[
                    scence_key
                ].objects_3d.items():
                    if (
                        self.scence_obj[scence_key].objects_3d_info[idx][
                            "is_fn"
                        ]
                        != 0
                    ):  # filter fn & ignored
                        continue
                    tracklet_dict[instance.meta["uid"]].append(
                        TrackletObj(
                            clip_timestamp=self.clip_timestamp,
                            track_id=instance.meta["uid"],
                            scence_key=scence_key,
                            object_3d=instance,
                            pre_scence_key=pre_scence_key,
                            info_type=self.info_type,
                            ego_pose=self.scence_obj[scence_key].meta[
                                "ego_pose"
                            ],
                        )
                    )
                    pre_scence_key = scence_key
        elif self.info_type == "pred":
            for scence_key in self.sequence_list:
                for idx, object_3d in self.scence_obj[
                    scence_key
                ].objects_3d.items():
                    object_3d_info = self.scence_obj[
                        scence_key
                    ].objects_3d_info[idx]
                    if (
                        object_3d_info["is_tp"] != 1.0
                        or object_3d_info.get("match_gt_obj3d_id", None)
                        is None
                    ):
                        continue
                    match_gt_obj3d_id = object_3d_info["match_gt_obj3d_id"]
                    if match_gt_obj3d_id == -1:
                        continue
                    tracklet_dict[match_gt_obj3d_id].append(
                        TrackletObj(
                            clip_timestamp=self.clip_timestamp,
                            track_id=match_gt_obj3d_id,
                            scence_key=scence_key,
                            object_3d=object_3d,
                            pre_scence_key=pre_scence_key,
                            info_type=self.info_type,
                            ego_pose=None,
                        )
                    )
                pre_scence_key = scence_key
        self.tracklet_dict = tracklet_dict
        return tracklet_dict


def parse_to_struct(
    infos_dict: dict,
    eval_class: str,
    info_type: str = "gt",
    disable_prelabel_2d_warning=False,
    do_temporal_aidieval=False,
):

    assert info_type in ["gt", "pred"]

    if info_type == "gt":
        assert (
            "temporal_clip" in infos_dict if do_temporal_aidieval else True
        ), "temporal_clip should exist if do_temporal_aidieval."
        timestamp = infos_dict["timestamp"]
        clip_timestamp = infos_dict.get("temporal_clip", timestamp)
        scence_tag = list(infos_dict["imgs_meta"].values())[0].get(
            "attrs", None
        )
        scence_objs = ScenceObj(
            scence_key=timestamp,
            meta={
                "eval_vis_cfg": infos_dict["eval_vis_cfg"],
                "scence_tag": scence_tag,
                "clip_timestamp": clip_timestamp,
                "ego_pose": infos_dict.get("ego_pose", None),
            },
        )
        assert eval_class in infos_dict, f"{eval_class} not in gt file!"
        imgs_meta = infos_dict["imgs_meta"]
        annos = infos_dict[eval_class]

        for obj_info in annos["objects"]:
            category = None
            if "attrs" in obj_info:
                if "sub_type" in obj_info["attrs"]:
                    category = obj_info["attrs"]["sub_type"]
            obj = Instance3dObj(
                dimensions=obj_info["in_vcs"]["dim"],
                location=obj_info["in_vcs"]["location"],
                depth_idx=0,
                rotation_y=obj_info["in_vcs"]["yaw"],
                category=category,
                score=obj_info.get("score", None),
                ignore=obj_info.get("ignore", True),
                velocity=obj_info["in_vcs"].get("velocity", None),
                acceleration=obj_info["in_vcs"].get("acceleration", None),
                yaw_rate=obj_info["in_vcs"].get("yaw_rate", None),
                meta={"uid": int(obj_info["uid"])},
            )
            scence_objs.add_obj3d(obj)

        for camera, anno in annos["view_anno"].items():
            camera_objs = []
            info_2d_list = []
            meta = imgs_meta[camera]
            for uid, obj in anno["objects"].items():
                uid = int(uid)
                info_2d = dict(  # noqa
                    bbox=obj["bbox"],  # 3d object project to image
                    bbox_2d=obj["bbox_2d"],  # detection 2d prelabel result
                    ignore=obj["ignore"],
                )
                info_2d_list.append(info_2d)
                scence_obj3d = scence_objs.get_obj3d_by_uid(uid)
                # if any 2d ignore is False, obj in scence is False
                scence_obj3d.ignore = (
                    obj["ignore"] if not obj["ignore"] else scence_obj3d.ignore
                )
                camera_objs.append(scence_obj3d)
            if "objects_2d" not in anno or anno["objects_2d"] is None:
                prelabel_obj2d = None
                if not disable_prelabel_2d_warning:
                    logger.warning(
                        f"Anno of image_key:{meta['image_key']} NOT provide prelabel 2d box!"  # noqa
                    )
            else:
                prelabel_obj2d = anno["objects_2d"]
                not_ass = True
                for obj2d in prelabel_obj2d:
                    if "ass_id" not in obj2d:
                        obj2d.update({"ass_id": -1})
                    else:
                        not_ass = False
                if not_ass and not disable_prelabel_2d_warning:
                    logger.warning(
                        f"Prelabel 2d box is NOT match with ANY 3d box in image_key:{meta['image_key']}, maybe info lacking HERE!"  # noqa
                    )
            img_obj = ImgObj(
                image_key=meta["image_key"],
                objects_3d=camera_objs,
                calib=meta["calib"],
                width=meta["shape"][1],
                height=meta["shape"][0],
                meta={
                    "prelabel_objects_2d": prelabel_obj2d,
                    "camera": camera,
                    "image_source": meta.get("image_source", None),
                    "camera_model": meta.get("camera_model", "PinholeCamera"),
                },
            )
            [
                img_obj.save_obj3d_in_img_info(obj, info)
                for obj, info in zip(camera_objs, info_2d_list)
            ]
            scence_objs.add_image_object(img_obj)
    else:
        objs_infos = infos_dict.get(eval_class)
        assert (
            objs_infos is not None
        ), f"{eval_class} not in prediction file!!!"
        scence_objs = ScenceObj(
            scence_key=infos_dict["timestamp"],
            image_objects=None,
            meta={
                "image_keys": infos_dict["image_keys"],
                "clip_timestamp": infos_dict.get(
                    "temporal_clip", infos_dict["timestamp"]
                ),
            },
        )
        for obj_info in objs_infos:
            category = None
            if "attrs" in obj_info:
                if "sub_type" in obj_info["attrs"]:
                    category = obj_info["attrs"]["sub_type"]
            obj = Instance3dObj(
                dimensions=obj_info.get("dimensions"),
                location=obj_info.get("location"),
                depth_idx=0,
                rotation_y=obj_info.get("rotation_y"),
                category=category,
                score=obj_info.get("score", None),
                ignore=obj_info.get("ignore", False),
            )
            scence_objs.add_obj3d(obj)

    return scence_objs


def parse_frame_list_to_clip(frame_list, clip_timestamp, info_type):
    """Parse frame list into a clip, whose key frame is clip_timestamp.

    Args:
        frame_list: frame list of ScenceObj to parse.
        clip_timestamp: key frame of the clip to be generated.
        info_type: type of the input info, eg. "gt", "pred".
    """
    clip_obj = ClipObj(
        clip_timestamp=clip_timestamp,
        info_type=info_type,
    )
    clip_obj.add_scenceObj_list(frame_list)

    return clip_obj


def group_scence_by_clip(scence_list, info_type, refer_clip_group=None):
    """Group scence list into clips.

    Args:
        scence_list: scence list of ScenceObj to be grouped.
        info_type: type of the input info, eg. "gt", "pred".
        refer_clip_group: the reference way to group clips,
            necessary if info_type is "pred",
            since pred will be grouped in the same way as gt does.
    """
    assert (
        refer_clip_group if info_type == "pred" else True
    ), "refer_clip_group should not be empty if info_type is pred."

    all_clips = []
    clip_group = defaultdict(list)
    for scence_item in scence_list:
        if info_type == "gt" and scence_item.meta.get("clip_timestamp", None):
            clip_group[scence_item.meta["clip_timestamp"]].append(scence_item)
        elif info_type == "pred":
            clip_group[
                scence_item.meta.get("clip_timestamp", scence_item.scence_key)
            ].append(scence_item)

    if info_type == "pred" and refer_clip_group:
        new_clip_group = defaultdict(list)
        for clip_timestamp, scence_list in refer_clip_group.items():
            for scence_item in scence_list:
                if clip_group.get(scence_item.scence_key, None):
                    new_clip_group[clip_timestamp].append(
                        clip_group[scence_item.scence_key][0]
                    )

        clip_group = new_clip_group

    for clip_timestamp, frame_list in clip_group.items():
        all_clips.append(
            parse_frame_list_to_clip(frame_list, clip_timestamp, info_type)
        )

    return all_clips, clip_group
