import copy
import logging
import os
import os.path as osp
from typing import List, Optional, Tuple

import numpy as np

from hat.utils.deprecate import deprecated_warning
from hat.utils.package_helper import require_packages

try:
    from auto_dp import Database, Image, Pack
except ImportError:
    Database = None
    Image = None
    Pack = None
from hat.registry import OBJECT_REGISTRY

try:
    from hatbc.auto_dp import DataBase
    from hatbc.resource_manager import get_resource
except ImportError:
    DataBase = None
    get_resource = None

from .camera import Camera
from .flank_anno_transformer import get_couples, get_instance_map

logger = logging.Logger(__name__)


@OBJECT_REGISTRY.register
class AnnoKPS8ToKps2(object):
    def __init__(
        self,
        bbox_classname: Optional[str] = "vehicle",
        src_classname: Optional[str] = "vehicle_kps_8",
        tgt_classname: Optional[str] = "p_WheelKeyPoints_2",
        padding_value: Optional[float] = 0,
    ):
        self.bbox_classname = bbox_classname
        self.src_classname = src_classname
        self.tgt_classname = tgt_classname
        self.padding_value = padding_value

    def __call__(self, item):
        x, anno = item
        if anno is None:
            return x, anno

        parent_map = get_instance_map(anno, self.bbox_classname)
        child_map = get_instance_map(anno, self.src_classname)
        couples = get_couples(anno, self.bbox_classname, self.src_classname)

        tgt_parent_lst = []
        tgt_child_lst = []
        tgt_couples = []
        for parent_id, child_id in couples:
            parent = parent_map.get(parent_id, None)
            child = child_map.get(child_id, None)
            if parent is None or child is None:
                continue
            if parent.get("attrs", {}).get("ignore", "no") == "yes":
                continue
            try:
                faced_flank = self.get_faced_flank_info(parent, child)
            except TypeError:
                # some keyerror exist on the annotation dataset
                logger.warning("Some Error Data, while get_faced_flank_info")
                continue
            kps, attrs = self.get_target_kps(parent, child, faced_flank)
            if kps is None or attrs is None:
                continue
            tgt_parent_lst.append(parent)
            child_attrs = {"faced_flank": faced_flank}
            child.update(
                {
                    "data": kps,
                    "point_attrs": attrs,
                    "attrs": child_attrs,
                }
            )
            tgt_child_lst.append(child)
            tgt_couples.append(
                f"{self.bbox_classname}|{parent_id}:{self.tgt_classname}|{child_id}"  # noqa
            )

        anno.update(
            {
                self.bbox_classname: tgt_parent_lst,
                self.tgt_classname: tgt_child_lst,
                "belong_to": tgt_couples,
            }
        )

        return x, anno

    def get_target_kps(self, parent, child, faced_flank):
        if faced_flank in ["head", "rear"]:
            return self.get_faced_kps(parent, child, faced_flank)
        elif faced_flank in ["left", "right"]:
            src_kps = child.get("data", None)
            src_attrs = child.get("point_attrs", None)
            if src_kps is None or src_attrs is None:
                return None, None
            tgt_indices = self.get_tartget_kps_indices(
                parent, child, faced_flank
            )
            tgt_kps = [src_kps[idx] for idx in tgt_indices]
            tgt_attrs = [src_attrs[idx] for idx in tgt_indices]
            return tgt_kps, tgt_attrs
        elif faced_flank in ["unknown"]:
            return None, None
        else:
            raise NotImplementedError

    def get_faced_kps(self, parent, child, faced_flank):
        assert faced_flank in ["head", "rear"]
        kps = [[self.padding_value, self.padding_value] for _ in range(2)]
        attrs = child.get("point_attrs", [])[0:2]

        return kps, attrs

    def get_tartget_kps_indices(self, parent, child, faced_flank):
        if faced_flank == "left":
            return 1, 0
        elif faced_flank == "right":
            return 3, 2
        else:
            raise NotImplementedError

    def get_faced_flank_info(self, parent, child):
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
    def faced_flank_info(start_point, end_point, start_attr, end_attr) -> str:
        start_ignore = _is_ignore_point(start_attr)
        end_ignore = _is_ignore_point(end_attr)
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


@OBJECT_REGISTRY.register
class AddCameraParam(object):
    @require_packages("auto_dp")
    def __init__(
        self,
        camera_param_key: Optional[str] = "",
    ):
        # TODO: remove dependency on auto_dp
        deprecated_warning(
            author="zihan.qiu",
            old_name="Camera",
            deprecation_version="v1.2.0",
            removal_version="v1.4.0",
        )
        self.camera_param_key = camera_param_key
        self.db = self.get_database()
        assert Image is not None and Pack is not None

    def __call__(self, item):
        x, anno = item
        if anno is None:
            return x, anno

        if self.camera_param_key in anno:
            return x, anno

        camera_param = self.get_camera_params(anno)
        if camera_param is not None:
            camera_param["height"] = anno["height"]
            camera_param["width"] = anno["width"]
            anno[self.camera_param_key] = camera_param

        return x, anno

    def get_camera_params(self, anno):
        try:
            if self.camera_param_key and self.camera_param_key in anno:
                camera_param = anno[self.camera_param_key]
            else:
                pack_uuid = None
                pack_name = None
                if "image_uuid" in anno:
                    image_uuid = anno["image_uuid"]
                    query = Image().where(f"_id = '{image_uuid}'")
                    results = self.db.execute(query)
                    image = results.images[0]
                    pack_uuid = image.puid
                else:
                    image_key = anno["image_key"]
                    pack_name = image_key.split("/")[-1].split("__")[0]
                camera_param, _ = self.get_pack_camera_param(
                    pack_name, pack_uuid
                )
                return camera_param
        except Exception:
            return None

    def get_pack_camera_param(
        self,
        pack_name: Optional[str] = None,
        pack_uuid: Optional[str] = None,
    ):
        if pack_uuid is not None:
            query_pack = Pack().where(f"_id = '{pack_uuid}'")
        elif pack_name is not None:
            query_pack = Pack().where(
                f'name like "{pack_name}" and frames > 0'
            )  # noqa
        else:
            raise ValueError("pack_name and pack_uuid given at least one")

        results = self.db.execute(query_pack)
        assert len(results.packs) == 1, "[ ERROR] No exist packs: {}".format(
            query_pack
        )  # noqa
        pack = results.packs[0]
        camera_param = pack.camera_params[0]
        camera_loc = pack.camera_loc
        assert len(camera_loc) == 1

        return camera_param, camera_loc[0]

    def get_database(self):
        if DataBase is not None and get_resource is not None:
            db = get_resource(DataBase)
        else:
            config = osp.join(os.environ["HOME"], ".hobot/sdk.yaml")
            assert Database is not None
            db = Database(config)

        return db


@OBJECT_REGISTRY.register
class FlankCornersInterpolationAnnoTs(object):
    def __init__(
        self,
        parent_classname: str,
        child_classname: str,
        interval: Optional[int] = 2,
        interpolation_indices: Optional[List[Tuple[int, int]]] = None,
        camera_param_key: Optional[str] = None,
        valid_types: Optional[List[str]] = None,
    ):
        """
        Annotation transformer for distorted key points.

        Args:
            parent_classname:
                the classname of parent bbox
            child_classname:
                the classname of key points
            num_key_points:
                the number of key points
            num_interpolation_points:
                the number sample dense of the numpoints
            interpolation_points_dist:
                the distance of points sample
            interpolation_indices: Optional[List[Tuple[int, int]]]
                the indices of which lines will be interpolated
            camera_param_key:
                the key of camera param
            db: Database
        """
        self.parent_classname = parent_classname
        self.child_classname = child_classname
        self.interpolation_indices = interpolation_indices
        self.camera_param_key = camera_param_key
        self.valid_types = valid_types

        self.interpolator = PointInterpolator(
            interval=interval,
            camera_param=None,
        )

    def __call__(self, item):
        _, anno = item
        if anno is None:
            return None

        image_url = os.path.abspath(anno["image_url"])
        if image_url is None:
            logger.warning("image url is None")
            return None

        try:
            camera_param = anno.get(self.camera_param_key, None)
            if camera_param is None:
                # skip the image without camera param
                logger.warning("camera param is None")
                return None
            self.interpolator.set_camera(camera_param)
        except TypeError:
            logger.warning(f"Error camera param @ {image_url}")
            return None

        parent_map = get_instance_map(anno, self.parent_classname)
        child_map = get_instance_map(anno, self.child_classname)
        couples = get_couples(
            anno, self.parent_classname, self.child_classname
        )

        bboxes = []
        keypoints = []
        flank_classes = []
        interpolation_points = []
        for parent_id, child_id in couples:
            parent = parent_map.get(parent_id, None)
            child = child_map.get(child_id, None)
            if parent is None or child is None:
                continue
            bbox_type = parent.get("attrs", {}).get("type", None)
            if self.valid_types and (
                bbox_type is None or bbox_type not in self.valid_types
            ):
                continue
            parent_boxes = parent.get("data", None)
            points = child.get("data", None)
            if points is None or parent_boxes is None:
                continue

            flank_class = 0
            # (num_interpolation,num_sample_points,point_dimension)
            all_point_sequence = []
            faced_flank = child.get("attrs", {}).get("faced_flank", "unknown")
            if faced_flank in ["unknown"]:
                continue
            elif faced_flank in ["rear", "head"]:
                # negative sample
                for one_idx, other_idx in self.interpolation_indices:
                    one_point = points[one_idx]
                    other_point = points[other_idx]
                    # (num_sample_points,point_dimension)
                    point_sequence = np.asarray(
                        [one_point, other_point], dtype=np.float32
                    )
                    all_point_sequence.append(point_sequence.tolist())
                flank_class = 0
            elif faced_flank in ["left", "right"]:
                # positive sample
                any_ignore = any(
                    [
                        _is_ignore_point(attr)
                        for attr in child.get("point_attrs", {})
                    ]
                )
                if any_ignore:
                    continue
                all_occluded = all(
                    [
                        _is_occ_point(attr)
                        for attr in child.get("point_attrs", {})
                    ]
                )
                if all_occluded:
                    continue
                # (num_interpolation,num_sample_points,point_dimension)
                for one_idx, other_idx in self.interpolation_indices:
                    one_point = points[one_idx]
                    other_point = points[other_idx]
                    point_sequence = self.interpolator(one_point, other_point)
                    point_sequence = np.asarray(point_sequence)
                    # (num_sample_points,point_dimension)
                    point_sequence = point_sequence.reshape(
                        -1, point_sequence.shape[-1]
                    )
                    all_point_sequence.append(point_sequence.tolist())
                flank_class = 1
            else:
                raise NotImplementedError(
                    f"Unsupport faced flank: {faced_flank}"
                )  # noqa

            bboxes.append(parent_boxes)
            keypoints.append(points)
            flank_classes.append(flank_class)
            interpolation_points.append(all_point_sequence)

        assert len(bboxes) == len(keypoints) == len(interpolation_points)

        if len(bboxes) == 0:
            logger.warning("number of object is zero")
            return None

        roi_rec = {
            "image": image_url,
            "height": anno["height"],
            "width": anno["width"],
            "bboxes": copy.deepcopy(bboxes),
            "keypoints": copy.deepcopy(keypoints),
            "flank_classes": copy.deepcopy(flank_classes),
            "interpolation_points": copy.deepcopy(interpolation_points),
            "interpolation_indices": copy.deepcopy(self.interpolation_indices),
        }

        return roi_rec

    def __debug_vis(self, roi_rec):
        import cv2

        from .viz import VizFlankCornersMatrixGluon

        vis = VizFlankCornersMatrixGluon(
            save_flag=True, save_path="output"  # noqa
        )
        img = cv2.imread(roi_rec["image"])
        vis(img, roi_rec)


class PointInterpolator(object):
    def __init__(
        self,
        interval: Optional[float] = 1,
        camera_param: Optional[dict] = None,
    ):
        if camera_param is None:
            self.camera = None
        else:
            self.camera = self.set_camera(camera_param)
        self.is_fisheye = False
        self.interval = interval

    def set_camera(self, camera_param, undistorted_scale=1):
        if "fov" in camera_param:
            self.is_fisheye = True if camera_param["fov"] > 180 else False
        elif "distort" in camera_param:
            self.is_fisheye = (
                True if len(camera_param["distort"]) == 4 else False
            )
        else:
            raise NotImplementedError(
                "You should specify image type in config file"
                "or add fov in camera param file"
            )
        camera = Camera()
        camera.img_size_ = [camera_param["height"], camera_param["width"]]
        camera.buildTransformMatrixFromCameraParamDict(camera_param)
        camera.setUndistortInformation(self.is_fisheye, undistorted_scale)
        self.camera = camera

    def __call__(self, point_one, point_other):
        input_points = np.array([point_one, point_other], dtype=np.float64)
        pts_undistorted = (
            self.camera.mapPointsFromOriginalImageToUndistortedImage(
                input_points
            )
        )
        interpolated_point_sequence, key_positions = self.interpolate_points(
            pts_undistorted
        )
        point_sequence = (
            self.camera.mapPointsFromUndistortedImageToOriginalImage(
                np.array(interpolated_point_sequence)
            )
        )

        return point_sequence

    def interpolate_points(self, pts_undistorted):
        point_pairs = self.make_point_pairs(pts_undistorted)
        point_groups = []
        for point_pair in point_pairs:
            line_func = self.line_fitting(point_pair)
            start = point_pair[0][0]
            end = point_pair[1][0]
            inverse_flag = False
            if start > end:
                inverse_flag = True
                start, end = end, start
            num_pts = int((end - start) * 1.0 / self.interval)
            if num_pts > 1:
                pts_x = np.linspace(start, end, num=num_pts)
                if inverse_flag is True:
                    pts_x = pts_x[::-1]
                pts_x = np.array(pts_x)
                pts_y = line_func(pts_x)
                interpolated_pts = np.vstack([pts_x, pts_y]).T
            else:
                if inverse_flag is False:
                    interpolated_pts = np.array(point_pair)
                else:
                    interpolated_pts = np.array(point_pair[::-1])
            point_groups.append(interpolated_pts.tolist())
        point_sequence, key_positions = self.make_point_sequence(point_groups)

        return point_sequence, key_positions

    def make_point_pairs(self, points):
        point_pairs = []
        for point_id in range(len(points) - 1):
            point_pairs.append([points[point_id], points[point_id + 1]])
        point_pairs.append([points[-1], points[0]])

        return point_pairs

    def make_point_sequence(self, point_groups):
        point_sequence = []
        key_positions = []
        for point_group in point_groups:
            key_positions.append(len(point_sequence))
            if len(point_group) == 1:
                point_sequence.extend(point_group)
            else:
                point_sequence.extend(point_group[:-1])

        return point_sequence, key_positions

    def line_fitting(self, pts_pair):
        # fit a line between the 2 points
        # return a function: y = a * x + b
        x1, y1 = pts_pair[0]
        x2, y2 = pts_pair[1]
        a = (y1 - y2) * 1.0 / (x1 - x2 + 1e-6)
        b = (x1 * y2 - x2 * y1) * 1.0 / (x1 - x2 + 1e-6)

        def line_func(pts_x):
            pts_y = a * pts_x + b
            return pts_y

        return line_func

    def find_nn_points(self, pts_x_src, pts_x_dst):
        num_src = len(pts_x_src)
        num_dst = len(pts_x_dst)
        assert num_src >= num_dst
        assert num_dst > 2
        pts_x_src = np.reshape(pts_x_src, (num_src, 1))
        pts_x_dst = np.reshape(pts_x_dst, (num_dst, 1))
        A = np.hstack([pts_x_src] * num_dst)
        B = np.hstack([pts_x_dst] * num_src)
        dist2 = A ** 2 + B.T ** 2 - 2 * A * B.T
        ptx_idx = np.argmin(dist2, axis=0)

        return ptx_idx

    def point_sampling(self, point_sequence, key_positions, input_points):
        # split point_sequence into point_groups
        point_groups = []
        num_groups = len(key_positions)
        for idx in range(num_groups):
            key_position = key_positions[idx]
            point_sequence[key_position] = input_points[idx]
            if idx == 0:
                continue
            last_key_position = key_positions[idx - 1]
            point_group = point_sequence[
                last_key_position : (key_position + 1)
            ]
            point_groups.append(point_group)
            if idx == num_groups - 1:
                point_group = np.vstack(
                    [point_sequence[key_position:], point_sequence[0:1, :]]
                )
                point_groups.append(point_group)

        # point sampling
        sampled_point_groups = []
        for point_group in point_groups:
            assert len(point_group) >= 2
            if len(point_group) == 2:
                num_pts = 2
            else:
                point_group = np.array(point_group)
                pts_x_src = point_group[:, 0]
                diff = np.abs(pts_x_src[1:] - pts_x_src[:-1])
                max_diff = np.max(diff)
                max_diff = np.maximum(max_diff, 2)
                start = pts_x_src[0]
                end = pts_x_src[-1]
                inverse_flag = False
                if start > end:
                    inverse_flag = True
                    start, end = end, start
                num_pts = int((end - start) * 1.0 / max_diff)
                if max_diff < 20:
                    pts_y_src = point_group[:, 1]
                    num_pixels = int(np.max(pts_y_src) - np.min(pts_y_src))
                    target_num_pts = np.maximum(num_pixels / 20, 4)
                    num_pts = np.minimum(num_pts, target_num_pts)
            if num_pts > 3:
                num_pts = int(num_pts)
                pts_x = np.linspace(start, end, num=num_pts)
                if inverse_flag is True:
                    pts_x = pts_x[::-1]
                pts_x_dst = np.array(pts_x)
                sampled_pts_idx = self.find_nn_points(pts_x_src, pts_x_dst)
                sampled_point_group = point_group[sampled_pts_idx, :]
            else:
                sampled_point_group = point_group[[0, -1]]
            sampled_point_groups.append(sampled_point_group.tolist())

        point_sequence, _ = self.make_point_sequence(sampled_point_groups)

        return point_sequence


def _is_ignore_point(point_attr):
    ignore = point_attr.get("point_label", {}).get("ignore", "no").lower()

    return ignore == "yes"


def _is_occ_point(point_attr):
    occ = (
        point_attr.get("point_label", {})
        .get("occlusion", "full_visible")
        .lower()
    )

    return occ != "full_visible"


@OBJECT_REGISTRY.register
class AnnoKPS8ToFlankKps4(AnnoKPS8ToKps2):
    def __init__(
        self,
        bbox_classname: Optional[str] = "vehicle",
        src_classname: Optional[str] = "vehicle_kps_8",
        tgt_classname: Optional[str] = "flank_kps_4",
        padding_value: Optional[float] = 0,
    ):
        super().__init__(
            bbox_classname, src_classname, tgt_classname, padding_value
        )

    def get_faced_kps(self, parent, child, faced_flank):
        assert faced_flank in ["head", "rear"]
        kps = [[self.padding_value, self.padding_value] for _ in range(4)]
        attrs = child.get("point_attrs", [])[0:4]

        return kps, attrs

    def get_tartget_kps_indices(self, parent, child, faced_flank):
        if faced_flank == "left":
            return 1, 0, 5, 4
        elif faced_flank == "right":
            return 3, 2, 7, 6
        else:
            raise NotImplementedError


@OBJECT_REGISTRY.register
class AnnoFlankKps2ToFlankCorners2(object):
    def __init__(
        self,
        bbox_classname: Optional[str] = "vehicle",
        src_classname: Optional[str] = "flank_points_4",
        tgt_classname: Optional[str] = "p_WheelKeyPoints_2",
        camera_param_key: Optional[str] = None,
        padding_value: Optional[float] = 0,
    ):
        self.bbox_classname = bbox_classname
        self.src_classname = src_classname
        self.tgt_classname = tgt_classname
        self.camera_param_key = camera_param_key
        self.padding_value = padding_value
        self.interpolator = PointInterpolator()

    def __call__(self, item):
        x, anno = item
        if anno is None:
            return x, anno

        camera_param = anno.get(self.camera_param_key, None)
        if camera_param is None:
            # skip the image without camera param
            logger.warning("camera param is None")
            return x, None

        try:
            self.interpolator.set_camera(camera_param)
        except TypeError:
            logger.warning(
                f"Error camera param @ {anno.get('image_key', None)}"
            )
            return x, None

        parent_map = get_instance_map(anno, self.bbox_classname)
        child_map = get_instance_map(anno, self.src_classname)
        couples = get_couples(anno, self.bbox_classname, self.src_classname)

        tgt_parent_lst = []
        tgt_child_lst = []
        tgt_couples = []
        for parent_id, child_id in couples:
            parent = parent_map.get(parent_id, None)
            child = child_map.get(child_id, None)
            if parent is None or child is None:
                continue
            if parent.get("attrs", {}).get("ignore", "no") == "yes":
                continue
            faced_flank = child.get("attrs", {}).get("faced_flank", "unknown")
            if faced_flank in ["head", "rear"]:
                kps, attrs = self.get_faced_kps(parent, child)
            elif faced_flank in ["left", "right"]:
                kps, attrs = self.flank_to_kps2(parent, child)
            else:
                continue
            if kps is None or attrs is None:
                continue
            tgt_parent_lst.append(parent)
            child.update(
                {
                    "data": kps,
                    "point_attrs": attrs,
                }
            )
            tgt_child_lst.append(child)
            tgt_couples.append(
                f"{self.bbox_classname}|{parent_id}:{self.tgt_classname}|{child_id}"  # noqa
            )

        anno.update(
            {
                self.bbox_classname: tgt_parent_lst,
                self.tgt_classname: tgt_child_lst,
                "belong_to": tgt_couples,
            }
        )

        return x, anno

    def get_faced_kps(self, parent, child):
        faced_flank = child.get("attrs", {}).get("faced_flank", "unknown")
        assert faced_flank in ["rear", "head"]
        points = [[self.padding_value, self.padding_value] for _ in range(2)]
        attrs = child["point_attrs"][0:2]

        return points, attrs

    def flank_to_kps2(
        self,
        parent,
        child,
    ):
        origin_pts = np.asarray(child["data"], dtype=np.float64)
        undistorted_pts = self.interpolator.camera.mapPointsFromOriginalImageToUndistortedImage(  # noqa
            origin_pts
        )
        left_wheel_point, right_wheel_point = undistorted_pts[0:2]
        left_edge_point, right_edge_point = undistorted_pts[2:4]
        points = self.get_intersection(
            left_wheel_point,
            right_wheel_point,
            [left_edge_point[0], right_edge_point[0]],
        )
        if points is None:
            attrs = copy.deepcopy(child["point_attrs"])[0:2]
            attrs[0]["point_label"]["ignore"] = "yes"
            attrs[1]["point_label"]["ignore"] = "yes"
            return origin_pts[0:2], attrs

        points = self.interpolator.camera.mapPointsFromUndistortedImageToOriginalImage(  # noqa
            points
        )
        points = points.reshape([-1, 2]).tolist()

        is_ignore = any(
            [
                val["point_label"]["ignore"] == "yes"
                for val in child["point_attrs"]
            ]
        )
        attrs = copy.deepcopy(child["point_attrs"])[0:2]
        if is_ignore:
            attrs[0]["point_label"]["ignore"] = "yes"
            attrs[1]["point_label"]["ignore"] = "yes"

        return points, attrs

    def get_intersection(self, pt_one, pt_other, coord_x):
        one_x, one_y = pt_one
        other_x, other_y = pt_other
        delta_x = one_x - other_x
        if delta_x == 0:
            return None
        delta_y = one_y - other_y
        slope = delta_y / delta_x

        points = []
        for x in coord_x:
            y = slope * (x - other_x) + other_y
            points.append([x, y])
        return np.asarray(points).reshape([-1, 2])
