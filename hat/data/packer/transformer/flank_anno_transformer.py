import copy
import os
from typing import Dict, List, Tuple

from hat.registry import OBJECT_REGISTRY, build_from_registry


@OBJECT_REGISTRY.register
class VehicleFlankAnnoTs(object):
    """
    Annotation transformer for Vehicle-Flank detection.

    which format the
    ground-line and side-edge to flank-data

    Args:
        parent_classname:
            the classname of vehicle
        child_classname:
            the classname of vehicle flank
        anno_adapter:
            the adapter for different annotated dataset,
            such as key-points 8 dataset
        min_flank_width:
            the min valid flank width for instance to be packed
        empty_value:
            the default value used to fill empty flank
        vis_dir:
            if is not None, save the transform visualize result to this dir
    """

    def __init__(
        self,
        parent_classname: str = "vehicle",
        child_classname: str = "vehicle_flank",
        anno_adapter: "AnnoAdapter" = None,
        min_flank_width: int = 4,
        empty_value: int = -10000,
        vis_dir: str = None,
    ):
        self.parent_classname = parent_classname
        self.child_classname = child_classname
        if anno_adapter is None:
            self.anno_adapter = None
        elif isinstance(anno_adapter, dict):
            anno_adapter = build_from_registry(anno_adapter)  # noqa
            assert isinstance(anno_adapter, AnnoAdapter)
            self.anno_adapter = anno_adapter
        elif isinstance(anno_adapter, AnnoAdapter):
            self.anno_adapter = anno_adapter
        else:
            raise NotImplementedError
        self.min_flank_width = min_flank_width
        self.empty_value = empty_value
        self.vis_dir = vis_dir

    def __call__(self, item: Tuple):
        """
        Annotation transformer for Vehicle-Flank detection.

        Args:
            item:
                which contains <image, anno>

        Returns:
            record:
                The record for one image to be packed
                {
                    'image': image_url
                    'height': image_height
                    'width': image_width
                    'bboxes': each:
                        {
                            'data': <x1,y1,x2,y2>
                            'class_id':
                        }
                    'flanks': each:
                        {
                            'data': <left_bottom_point, right_bottom_point,
                                    right_top_point, left_top_point>
                            'class_id':
                        }
                    'ignore_regions':
                        {
                            'left_top': bbox[0:2],
                            'right_bottom': bbox[2:4],
                            'class_id':
                        }
                }
        """
        if self.anno_adapter is not None:
            item = self.anno_adapter(item)
        img_dir, anno = item

        image_url = os.path.abspath(img_dir)
        if image_url is None:
            return None

        bbox_lst, flank_lst, ignore_lst = [], [], []
        parent_map = get_instance_map(anno, self.parent_classname)
        child_map = get_instance_map(anno, self.child_classname)
        couples = get_couples(
            anno, self.parent_classname, self.child_classname
        )  # noqa
        parent_id_coupled = set()
        # process the coupled vehicle and flank
        for parent_id, child_id in couples:
            parent = parent_map.get(parent_id, None)
            child = child_map.get(child_id, None)
            if parent is None or child is None:
                continue

            parent_id_coupled.add(parent_id)
            x1, y1, x2, y2 = [float(val) for val in parent["data"]]
            if (
                parent["attrs"]["ignore"].lower() == "yes"
                or child["attrs"]["faced_flank"] == "unknown"
            ):
                # ignore
                ignore_lst.append(
                    {
                        "left_top": [x1, y1],
                        "right_bottom": [x2, y2],
                        "class_id": [float(1)],
                    }
                )
            elif child["attrs"]["faced_flank"] in ["rear", "head"]:
                # negative
                bbox_lst.append(
                    {"data": [x1, y1, x2, y2], "class_id": float(1)}
                )
                flank_lst.append({"data": child["data"], "class_id": float(0)})
            elif child["attrs"]["faced_flank"] in ["left", "right"]:
                flank_values = []
                for point in child["data"]:
                    flank_values += point
                if self.empty_value in flank_values:
                    # ignore
                    ignore_lst.append(
                        {
                            "left_top": [x1, y1],
                            "right_bottom": [x2, y2],
                            "class_id": [float(1)],
                        }
                    )
                else:
                    left_bottom, right_bottom = child["data"][0:2]
                    if right_bottom[0] - left_bottom[0] < self.min_flank_width:
                        # ignore
                        ignore_lst.append(
                            {
                                "left_top": [x1, y1],
                                "right_bottom": [x2, y2],
                                "class_id": [float(1)],
                            }
                        )
                    else:
                        # positive
                        bbox_lst.append(
                            {"data": [x1, y1, x2, y2], "class_id": float(1)}
                        )
                        flank_lst.append(
                            {"data": child["data"], "class_id": float(1)}
                        )
            else:
                raise NotImplementedError

        # no valid flank instance, just delete the images
        if len(flank_lst) == 0:
            return None

        # process the uncoupled vehicle to ignore
        for parent_id, parent in parent_map.items():
            if parent_id in parent_id_coupled:
                continue
            x1, y1, x2, y2 = [float(val) for val in parent["data"]]
            ignore_lst.append(
                {
                    "left_top": [x1, y1],
                    "right_bottom": [x2, y2],
                    "class_id": [float(1)],
                }
            )

        # assign label
        record = {
            "image": image_url,
            "height": anno["height"],
            "width": anno["width"],
            "bboxes": bbox_lst,
            "flanks": flank_lst,
            "ignore_regions": ignore_lst,
        }

        return record


@OBJECT_REGISTRY.register
class AnnoAdapter(object):
    def __init__(
        self,
        parent_classname: str,
        child_classname: str,
        target_classname: str,
    ):
        """
        Anno-adapter interface.

        Args:
            parent_classname:
                the parent bbox classname; such as 'vehicle'
            child_classname:
                the child instance classname of the origin anno dataset,
                such as 'vehicle_kps_8'
            target_classname:
                the target instance class name, such as 'vehicle_flank'
        """
        self.parent_classname = parent_classname
        self.child_classname = child_classname
        self.target_classname = target_classname

    def __call__(self, item):
        x, anno = item
        new_anno = copy.deepcopy(anno)

        parent_map = get_instance_map(anno, self.parent_classname)
        couples = get_couples(
            anno, self.child_classname, self.parent_classname
        )  # noqa
        couples_map = {one_id: other_id for one_id, other_id in couples}

        new_child_lst = []
        new_belong_to_lst = []
        for child in anno.get(self.child_classname, []):
            child_id = child["id"]
            parent_id = couples_map[child_id]
            parent = parent_map.get(parent_id, None)
            if parent is None or child is None:
                continue
            try:
                child = self.adapt(parent, child)
            except (KeyError, IndexError):
                # some error data exist on anno dataset
                continue
            new_child_lst.append(child)
            belong_to = f"{self.parent_classname}|{parent_id}:{self.target_classname}|{child_id}"  # noqa
            new_belong_to_lst.append(belong_to)
        new_anno[self.target_classname] = new_child_lst
        new_anno["belong_to"] = new_belong_to_lst

        return x, new_anno

    def adapt(self, parent: Dict, child: Dict) -> Dict:
        """
        Process coupled parent and child to flank instance.

        Args:
            parent:
                the instance dict of vehicle bbox
            child:
                the instance dict of child(such as key-points 8)

        Returns：
            vehicle_flank:
                {
                    'id':
                    'data': [left_bottom_point, right_bottom_point,
                            left_top_point, right_top_point]
                    'attrs': {
                        'faced_flank': value in ['right','left','head','rear','unknown']    # noqa
                    }
                }

        """
        raise NotImplementedError


@OBJECT_REGISTRY.register
class AnnoAdapterKps8(AnnoAdapter):
    def __init__(
        self,
        parent_classname,
        child_classname,
        target_classname,
        empty_value=-10000,
    ):
        super(AnnoAdapterKps8, self).__init__(
            parent_classname, child_classname, target_classname
        )  # noqa
        self.empty_value = empty_value

    def adapt(self, parent, child) -> dict:
        faced_flank = self.get_faced_flank_info(parent, child)
        vehicle_flank = self.get_vehicle_flank(parent, child, faced_flank)

        return vehicle_flank

    def get_vehicle_flank(
        self: Dict, parent: Dict, child: str, faced_flank: Dict
    ) -> Dict:
        """
        Adapt kps 8.

        Args:
            parent:
            child:
            faced_flank:
                value in ['right','left','head','rear','unknown']

        Returns:
            vehicle_flank:
                {
                    'id':
                    'data': [left_bottom_point, right_bottom_point,
                            left_top_point, right_top_point]
                    'attrs': {
                        'faced_flank': value in ['right','left','head','rear','unknown']    # noqa
                    }
                }

        """
        if faced_flank in ["head", "rear", "unknown"]:
            # NOTE: 1. 'head' and 'rear' are negative instance for flank task
            #       2. 'unknown' are ignore instance for flank task
            return {
                "id": child["id"],
                "data": [
                    [self.empty_value, self.empty_value] for _ in range(4)
                ],  # noqa
                "attrs": {"faced_flank": faced_flank},
            }
        elif faced_flank in ["left", "right"]:
            # NOTE: 'left' and 'right' are negative instance for flank task
            # while truncate the wheel out, you need ignore the instance for ground line task   # noqa
            (
                left_bottom_point,
                right_bottom_point,
                left_top_point,
                right_top_point,
            ) = self.get_vehicle_flank_data(parent, child, faced_flank)
            return {
                "id": child["id"],
                "data": [
                    left_bottom_point,
                    right_bottom_point,
                    left_top_point,
                    right_top_point,
                ],
                "attrs": {"faced_flank": faced_flank},
            }
        else:
            raise ValueError("Error faced_flank: {}".format(faced_flank))

    def get_vehicle_flank_data(
        self, parent: Dict, child: Dict, faced_flank: str
    ):
        """
        Get data.

        Args:
            parent:
            child:
            faced_flank:


        """
        assert faced_flank in ["right", "left"]
        x1, y1, x2, y2 = [float(val) for val in parent["data"]]
        points = child["data"]
        points_attr = child["point_attrs"]
        if faced_flank == "left":
            left_wheel_idx, right_wheel_idx = 1, 0
            left_edge_idx, right_edge_idx = 5, 4
        else:
            left_wheel_idx, right_wheel_idx = 3, 2
            left_edge_idx, right_edge_idx = 7, 6

        left_edge_ignore = self.is_ignore_point(points_attr[left_edge_idx])
        if left_edge_ignore:
            left_edge_x = x1
        else:
            left_edge_x = float(points[left_edge_idx][0])
        right_edge_ignore = self.is_ignore_point(points_attr[right_edge_idx])
        if right_edge_ignore:
            right_edge_x = x2
        else:
            right_edge_x = float(points[right_edge_idx][0])
        left_top_point = [float(left_edge_x), float(y1)]
        right_top_point = [float(right_edge_x), float(y1)]

        left_wheel_point = points[left_wheel_idx]
        left_wheel_ignore = self.is_ignore_point(points_attr[left_wheel_idx])
        right_wheel_point = points[right_wheel_idx]
        right_wheel_ignore = self.is_ignore_point(points_attr[right_wheel_idx])
        if left_wheel_ignore or right_wheel_ignore:
            # can`t get the vehicle ground line
            left_bottom_y = self.empty_value
            right_bottom_y = self.empty_value
        else:
            left_bottom_y, right_bottom_y = self.get_intersections_to_vertical(
                left_wheel_point,
                right_wheel_point,
                [left_edge_x, right_edge_x],
            )
        left_bottom_point = [float(left_edge_x), float(left_bottom_y)]
        right_bottom_point = [float(right_edge_x), float(right_bottom_y)]

        return (
            left_bottom_point,
            right_bottom_point,
            left_top_point,
            right_top_point,
        )  # noqa

    def get_intersections_to_vertical(self, one_point, other_point, loc_x_lst):
        one_x, one_y = one_point
        other_x, other_y = other_point
        delta_x = one_x - other_x
        delta_y = one_y - other_y
        if delta_x == 0:
            return [self.empty_value] * len(loc_x_lst)
        else:
            slope = delta_y / delta_x
            return [slope * (x - other_x) + other_y for x in loc_x_lst]

    def get_faced_flank_info(self, parent: Dict, child: Dict) -> str:
        """
        Get info.

        Args:
            parent:
            child:

        Returns:
            faced_flank:
                value in ['right','left','head','rear','unknown']
        """
        child_attrs = child.get("attrs", {})
        if "faced_flank" in child_attrs:
            return child_attrs["faced_flank"]

        points = child["data"]
        points_attrs = child["point_attrs"]
        faced_flank_set = set()
        for start_idx, end_idx in [(0, 1), (3, 2), (4, 5), (7, 6)]:
            start_point, end_point = points[start_idx], points[end_idx]
            start_attr, end_attr = (
                points_attrs[start_idx],
                points_attrs[end_idx],
            )  # noqa
            faced = self.faced_flank_info(
                start_point, end_point, start_attr, end_attr
            )  # noqa
            if faced == "unknown":
                continue
            faced_flank_set.add(faced)
        if len(faced_flank_set) == 0:
            faced_flank = "unknown"
        elif len(faced_flank_set) == 1:
            faced_flank = faced_flank_set.pop()
        else:
            # TODO: diff the rear and head @feng02.li
            faced_flank = "rear"

        return faced_flank

    @staticmethod
    def faced_flank_info(
        start_point: int, end_point: int, start_attr: str, end_attr: str
    ) -> str:
        """
        Convert info.

        Args:
            start_point:
            end_point:
            start_attr:
            end_attr:

        Returns:
            faced_flank:
                value in ['right','left','head','rear','unknown']

        """
        start_ignore = AnnoAdapterKps8.is_ignore_point(start_attr)
        end_ignore = AnnoAdapterKps8.is_ignore_point(end_attr)
        start_x, start_y = start_point
        end_x, end_y = end_point
        if not start_ignore and not end_ignore:
            if start_x < end_x:
                return "right"
            elif start_x > end_x:
                return "left"
            elif start_y > end_y:
                return "rear"
            else:
                return "head"
        else:
            return "unknown"

    @staticmethod
    def is_ignore_point(point_attr: Dict):
        """
        Check if ignored.

        Args:
            point_attr:
                {
                    'occlusion':
                    'Corner_confidence':
                    'ignore':
                    'position':
                }


        """
        ignore = point_attr.get("point_label", {}).get("ignore", "no").lower()

        return ignore == "yes"


def get_instance_map(anno: Dict, classname: Dict, valid_types: List = None):
    """
    Get instance map.

    Args:
        anno:
        classname:
        valid_types:

    Returns:
        mapping:
            each record <instance_id, instance>
    """
    mapping = {}
    for instance in anno.get(classname, []):
        idx = instance.get("id", None)
        if idx is None or idx in mapping:
            continue
        if (
            valid_types is not None
            and instance["attrs"]["type"] not in valid_types
        ):
            continue
        mapping[int(idx)] = copy.deepcopy(instance)

    return mapping


def get_couples(anno, one_classname, other_classname):
    pairs = []
    for belong_to in anno.get("belong_to", []):
        one, other = belong_to.strip().split(":")
        one_name, one_id = one.split("|")
        other_name, other_id = other.split("|")
        tmp_map = {one_name: int(one_id), other_name: int(other_id)}
        parent_id = tmp_map.get(one_classname)
        child_id = tmp_map.get(other_classname)
        pairs.append((parent_id, child_id))

    return pairs
