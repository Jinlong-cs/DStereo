import math
import os
from typing import List

from hat.registry import OBJECT_REGISTRY


@OBJECT_REGISTRY.register
class KeyPointAnnoJsonTs(object):
    """
    Transform keypoint annotation.

    To generic keypoint and detection annotation.

    Args:
        num_kps:
            the number of kps in anno
        class_name:
            classname for kps
    """

    def __init__(
        self, num_kps: int, conf_ignore_list: List, class_name: str = "vehicle"
    ):
        self.num_kps = num_kps
        self.num_box = 0
        self.conf_ignore_list = conf_ignore_list
        self.class_name = class_name

    def __call__(self, item: List):
        """
        Transform keypoint annotation.

        Args:
            item:
                Annotation in Horizon image data annotation.

        Returns:
            trav_result:
                Bounding box and keypoint information.
            img_abs_path:
                Image absolute path.

            trav_result:

                When no conf, returns

                .. code-block:: python
                    [
                        bbox_type, bbox_clur, bbox_occ, bbox_ignore, bbox_xmin,
                        bbox_ymin, bbox_xmax, bbox_ymax, kp_back_xmin, kp_back_ymin,
                        kp_front_xmax, kp_front_ymax, occ_back, occ_front
                    ]

                When conf exists, returns

                .. code-block:: python
                    [
                        bbox_type, bbox_clur, bbox_occ, bbox_ignore, bbox_xmin,
                        bbox_ymin, bbox_xmax, bbox_ymax, kp_back_xmin, kp_back_ymin,
                        kp_front_xmax, kp_front_ymax, occ_back, occ_front, conf_back,
                        conf_front
                    ]
        """  # noqa
        (img_dir, anno) = item[0]

        if (
            self.class_name not in anno.keys()
            or "p_WheelKeyPoints_" + str(self.num_kps) not in anno.keys()
        ):
            return None
        if "belong_to" not in anno.keys():
            return None

        img_abs_path = os.path.abspath(img_dir)

        # create bbox dict
        data_bbox = anno[self.class_name]
        bbox_dict = {}
        for bbox_ele in data_bbox:
            if "id" not in bbox_ele.keys():
                continue
            bbox_dict[int(bbox_ele["id"])] = bbox_ele

        # parse matching dict {kps_id: bbox_id}
        matching = anno["belong_to"]
        matching_dict = {}
        for matching_ele in matching:
            matching_list = matching_ele.strip().split(":")

            bbox_id = int(matching_list[0].strip().split("|")[-1])
            kps_id = int(matching_list[1].strip().split("|")[-1])

            matching_dict[kps_id] = bbox_id

        # traverse
        trav_result = []

        data_wheel_kps = anno["p_WheelKeyPoints_" + str(self.num_kps)]
        has_conf = isinstance(data_wheel_kps[0]["point_attrs"][0], dict)
        for kps_ele in data_wheel_kps:
            temp_result = []
            if "id" not in kps_ele.keys():
                continue

            kps_id = kps_ele["id"]
            if kps_id not in matching_dict.keys():
                continue

            if "data" not in kps_ele or len(kps_ele["data"]) < self.num_kps:
                continue

            matched_bbox = bbox_dict[matching_dict[kps_id]]
            matched_bbox_attrs = matched_bbox["attrs"]

            temp_result.append(matched_bbox_attrs["type"])
            if "blur" in matched_bbox_attrs.keys():
                temp_result.append(matched_bbox_attrs["blur"])
            else:
                temp_result.append("Normal")
            temp_result.append(
                matched_bbox_attrs.get("occlusion", "fullvisible")
            )
            temp_result.append(matched_bbox_attrs.get("ignore", "no"))
            temp_result.append(str(matched_bbox["data"][0]))
            temp_result.append(str(matched_bbox["data"][1]))
            temp_result.append(str(matched_bbox["data"][2]))
            temp_result.append(str(matched_bbox["data"][3]))
            for kps_i in range(self.num_kps):
                temp_result.append(str(kps_ele["data"][kps_i][0]))
                temp_result.append(str(kps_ele["data"][kps_i][1]))
            if isinstance(kps_ele["point_attrs"][0], str):
                for kps_i in range(self.num_kps):
                    temp_result.append(kps_ele["point_attrs"][kps_i])
            elif isinstance(kps_ele["point_attrs"][0], dict):
                kps_attr = kps_ele["point_attrs"]
                for kps_i in range(self.num_kps):
                    # occlusion
                    temp_result.append(
                        kps_attr[kps_i]["point_label"]["occlusion"]
                    )
                for kps_i in range(self.num_kps):
                    # confidence
                    temp_conf = kps_attr[kps_i]["point_label"][
                        "Corner_confidence"
                    ]
                    if temp_conf in self.conf_ignore_list:
                        continue
                    temp_result.append(temp_conf)
            else:
                raise TypeError('data "point_attrs" type error')

            # update valid box number
            self.num_box += 1
            trav_result.append(temp_result)

        return (
            img_abs_path,
            trav_result,
            anno["width"],
            anno["height"],
            has_conf,
        )


@OBJECT_REGISTRY.register
class KeyPointAnnoTs(object):
    """
    Key point annotation transformer.

    Args:
        num_kps:
            the number of kps in anno
    """

    def __init__(
        self,
        num_kps,
        valid_types,
        min_width,
        min_height,
        kps_occ_id_dict,
        occ_ignore_list,
        obj_ele_num_2kps,
        obj_ele_num_3kps,
        obj_ele_num_xkps,
        skip_invalid=True,
    ):
        # assert num_kps in [2, 3], 'num_kps should be 2 or 3.'
        self.valid_types = valid_types
        self.min_width = min_width
        self.min_height = min_height
        self.kps_occ_id_dict = kps_occ_id_dict
        self.occ_ignore_list = occ_ignore_list

        self.num_kps = num_kps
        self.obj_ele_num_2kps = obj_ele_num_2kps
        self.obj_ele_num_3kps = obj_ele_num_3kps
        self.obj_ele_num_xkps = obj_ele_num_xkps

        self.ret_num = 0
        self.num_valid_bbox = 0
        self.kps_count_dict = {}
        self.skip_invalid = skip_invalid

    def check_occ(self, *occ):
        for data in occ:
            if data not in self.kps_occ_id_dict:
                return False
        return True

    def __call__(
        self,
        img_abs_path,
        trav_result=None,
        width=None,
        height=None,
        has_conf=False,
    ):
        if img_abs_path is None:
            return None
        else:
            img_abs_path, trav_result, width, height, has_conf = img_abs_path

        elem_per_inst = self.obj_ele_num_xkps
        if has_conf:
            elem_per_inst += self.num_kps

        num_obj = len(trav_result)
        if num_obj == 0:
            return None

        if elem_per_inst != len(trav_result[0]):
            print("[!] Error anno data ")
            if self.skip_invalid:
                return None
            else:
                raise RuntimeError

        # generate label, traverse valid obj
        boxes, gt_classes, keypoints = [], [], []
        for obj_data in trav_result:
            cls_cur, blur, occ, ignore = obj_data[:4]

            if cls_cur not in self.valid_types:
                continue

            if blur != "Normal" and blur != "no":
                continue

            # check object occ
            if occ in self.occ_ignore_list:
                continue

            # check 2d bbox dimension
            x1, y1, x2, y2 = map(float, obj_data[4 : 4 + 4])
            bbox_width, bbox_height = x2 - x1, y2 - y1
            if bbox_width < self.min_width or bbox_height < self.min_height:
                continue

            # assign box, category and keypoints
            boxes.append([x1, y1, x2, y2])
            gt_classes.append(1)

            keypoints_per_obj = []
            for kps_i in range(self.num_kps):
                # 1 + 3 + 4 + num_kps * 2 + num_kps
                occ_kp_xkps = obj_data[8 + self.num_kps * 2 + kps_i]

                if not self.check_occ(occ_kp_xkps):
                    print("[!] Error anno data %s" % img_abs_path)
                    if self.skip_invalid:
                        return None
                    else:
                        raise RuntimeError

                keypoints_per_obj.extend(
                    [
                        float(obj_data[8 + 2 * kps_i]),
                        float(obj_data[8 + 2 * kps_i + 1]),
                        self.kps_occ_id_dict[occ_kp_xkps],
                    ]
                )

                if occ_kp_xkps not in self.kps_count_dict:
                    self.kps_count_dict[occ_kp_xkps] = 0
                self.kps_count_dict[occ_kp_xkps] += 1
            keypoints.append(keypoints_per_obj)
        # no valid obj
        if len(boxes) <= 0 or len(gt_classes) <= 0 or len(keypoints) <= 0:
            return None

        self.num_valid_bbox += len(boxes)

        # assign label
        roi_rec = {
            "image": img_abs_path,
            "height": height,
            "width": width,
            "boxes": boxes,
            "gt_classes": gt_classes,
            "keypoints": keypoints,
        }
        self.ret_num += 1
        return roi_rec


@OBJECT_REGISTRY.register
class KeyPointAnnoNumTs(object):
    def __init__(self, class_name, sub_class_name, ts_sub_class_name):
        self.class_name = class_name
        self.sub_class_name = sub_class_name
        self.ts_sub_class_name = ts_sub_class_name

    def _anno_trans(self, kps_points, kps_attr, matched_bbox_attrs):
        NotImplemented

    def _get_match_boxes(self, anno):
        data_bbox = anno[self.class_name]
        bbox_dict = {}
        for bbox_ele in data_bbox:
            if "id" not in bbox_ele.keys():
                continue
            bbox_dict[int(bbox_ele["id"])] = bbox_ele

        matching = anno["belong_to"]
        matching_dict = {}
        for matching_ele in matching:
            matching_list = matching_ele.strip().split(":")
            bbox_id = int(matching_list[0].strip().split("|")[-1])
            kps_id = int(matching_list[1].strip().split("|")[-1])
            matching_dict[kps_id] = bbox_id

        return bbox_dict, matching_dict

    def __call__(self, item):
        (x, anno) = item
        anno = anno.copy()
        if (
            self.class_name not in anno.keys()
            or self.sub_class_name not in anno.keys()
        ):
            return ((x, anno),)
        if "belong_to" not in anno.keys():
            return ((x, anno),)

        bbox_dict, matching_dict = self._get_match_boxes(anno)

        ts_kps_data = []
        og_kps_data = anno[self.sub_class_name]

        for kps_ele in og_kps_data:
            if "id" not in kps_ele.keys():
                continue
            if "point_attrs" not in kps_ele.keys():
                continue
            if "data" not in kps_ele.keys():
                continue

            kps_id = kps_ele["id"]
            kps_attr = kps_ele["point_attrs"]
            kps_points = kps_ele["data"]

            if kps_id not in matching_dict.keys():
                continue

            matched_bbox = bbox_dict[matching_dict[kps_id]]
            matched_bbox_attrs = matched_bbox["attrs"]
            ts_kps_points, ts_kps_attr = self._anno_trans(
                kps_points, kps_attr, matched_bbox_attrs
            )

            if len(ts_kps_points) == 0:
                continue

            kps_ele["point_attrs"] = ts_kps_attr
            kps_ele["data"] = ts_kps_points
            ts_kps_data.append(kps_ele)
        if len(ts_kps_data) != 0:
            anno[self.ts_sub_class_name] = ts_kps_data
            anno.pop(self.sub_class_name)
            anno["belong_to"] = [
                x.replace(self.sub_class_name, self.ts_sub_class_name)
                for x in anno["belong_to"]
            ]
        return ((x, anno),)


@OBJECT_REGISTRY.register
class VehicleKps8to2Ts(KeyPointAnnoNumTs):
    def __init__(self, class_name, sub_class_name, ts_sub_class_name):
        super(VehicleKps8to2Ts, self).__init__(
            class_name=class_name,
            sub_class_name=sub_class_name,
            ts_sub_class_name=ts_sub_class_name,
        )

    def _anno_trans(self, kps_points, kps_attr, matched_bbox_attrs):
        offset = 10
        kps_occ = list(
            map(lambda point: point["point_label"]["occlusion"], kps_attr)
        )
        kps_ignore = list(
            map(lambda point: point["point_label"]["ignore"], kps_attr)
        )
        truncation = matched_bbox_attrs.get("truncation", "None")

        if len(kps_points) < 4 or len(kps_ignore) < 4:
            return [], []

        if truncation != "None":
            # box with truncation, vehicle partly visible
            left_pt_valid = True
            left_angle_valid = True
            left_attr_valid = kps_ignore[0] != "yes" or kps_ignore[1] != "yes"
            right_pt_valid = True
            right_angle_valid = True
            right_attr_valid = kps_ignore[3] != "yes" or kps_ignore[2] != "yes"
            if left_attr_valid:
                if kps_ignore[0] != "yes":
                    occ_valid = kps_occ[0] != "self_occluded"
                elif kps_ignore[1] != "yes":
                    occ_valid = kps_occ[1] != "self_occluded"
                _left_occ_valid = occ_valid
            else:
                _left_occ_valid = False
            if right_attr_valid:
                if kps_ignore[3] != "yes":
                    occ_valid = kps_occ[3] != "self_occluded"
                elif kps_ignore[2] != "yes":
                    occ_valid = kps_occ[2] != "self_occluded"
                _right_occ_valid = occ_valid
            else:
                _right_occ_valid = False
            # if the whole left or right part is outside the image, skip
            left_occ_valid = _left_occ_valid and right_attr_valid
            right_occ_valid = left_attr_valid and _right_occ_valid
        else:
            # box without truncation, vehicle fully visible
            # point valid
            left_pt_valid = kps_points[1] != kps_points[0]
            right_pt_valid = kps_points[2] != kps_points[3]
            # attribute valid
            left_attr_valid = kps_ignore[0] != "yes" and kps_ignore[1] != "yes"
            right_attr_valid = (
                kps_ignore[3] != "yes" and kps_ignore[2] != "yes"
            )
            # angle valid
            vh_vector_left = (
                kps_points[1][0] - kps_points[0][0],
                kps_points[1][1] - kps_points[0][1],
            )
            angle = math.atan2(vh_vector_left[1], vh_vector_left[0])
            angle_left = angle * 180 / math.pi
            angle_left = (angle_left + 360) % 180
            left_angle_valid = angle_left > (90 + offset) or angle_left < (
                90 - offset
            )
            vh_vector_right = (
                kps_points[2][0] - kps_points[3][0],
                kps_points[2][1] - kps_points[3][1],
            )
            angle = math.atan2(vh_vector_right[1], vh_vector_right[0])
            angle_right = angle * 180 / math.pi
            angle_right = (angle_right + 360) % 180
            right_angle_valid = angle_right > (90 + offset) or angle_right < (
                90 - offset
            )

            # occ valid
            left_num_vis = int(kps_occ[1] != "self_occluded") + int(
                kps_occ[0] != "self_occluded"
            )
            right_num_vis = int(kps_occ[3] != "self_occluded") + int(
                kps_occ[2] != "self_occluded"
            )
            if left_num_vis > right_num_vis:
                left_occ_valid = True
                right_occ_valid = False
            elif left_num_vis < right_num_vis:
                left_occ_valid = False
                right_occ_valid = True
            else:
                if left_num_vis == 0:
                    left_occ_valid = False
                    right_occ_valid = False
                else:
                    # if the same num, decide by vehicle light points
                    left_num_vis = int(kps_occ[5] != "self_occluded") + int(
                        kps_occ[4] != "self_occluded"
                    )
                    right_num_vis = int(kps_occ[7] != "self_occluded") + int(
                        kps_occ[6] != "self_occluded"
                    )
                    if left_num_vis > right_num_vis:
                        left_occ_valid = True
                        right_occ_valid = False
                    elif left_num_vis < right_num_vis:
                        left_occ_valid = False
                        right_occ_valid = True
                    else:
                        left_occ_valid = False
                        right_occ_valid = False
        # overall decision
        left_valid = (
            left_pt_valid
            and left_attr_valid
            and left_angle_valid
            and left_occ_valid
        )
        right_valid = (
            right_pt_valid
            and right_attr_valid
            and right_angle_valid
            and right_occ_valid
        )

        if left_valid:
            return [kps_points[0], kps_points[1]], [kps_attr[0], kps_attr[1]]
        elif right_valid:
            return [kps_points[3], kps_points[2]], [kps_attr[3], kps_attr[2]]
        else:
            return [], []
