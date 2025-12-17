import copy
import logging
from collections import defaultdict
from dataclasses import fields

import cv2
import numpy as np

try:
    from pycocotools import mask as coco_mask
except ImportError:
    coco_mask = None

from hat.core.box_utils import bbox_clamp
from hat.core.data_struct.app_struct import DetObjects
from hat.core.data_struct.base_struct import (
    DetBoxes2D,
    DetBoxes3D,
    Lines2D,
    Mask,
    MultipleBox2D,
    MultipleBoxes2D,
    Points2D_2,
)
from hat.core.virtual_camera import (
    CylindricalCamera,
    FisheyeCamera,
    PinholeCamera,
    SphericalCamera,
)
from hat.core.virtual_camera.utils import get_cam_uuid
from hat.registry import OBJECT_REGISTRY
from hat.utils.package_helper import check_packages_available

logger = logging.getLogger(__name__)
uv_map_record_dict = defaultdict()
bpu_uv_map_record_dict = defaultdict()


def _load_map_from_local_path(map_path, output_shape):
    with open(map_path, "rb") as f:
        raw_data = f.read()
        map_value = np.frombuffer(raw_data, dtype=np.float32)
        map_value = map_value.reshape(output_shape)
        return map_value


def load_uvmap(map_x_file, map_y_file, uvmap_h, uvmap_w):
    mapx = _load_map_from_local_path(map_x_file, (uvmap_h, uvmap_w, 1))
    mapy = _load_map_from_local_path(map_y_file, (uvmap_h, uvmap_w, 1))
    uv_map_xy = np.concatenate([mapx, mapy], axis=2)
    logger.info(f"shapes of uv_map from software: {uv_map_xy.shape}")
    return uv_map_xy


@OBJECT_REGISTRY.register
class CameraStandardization(object):
    CAMERA_MODELS = {
        "cylindrical": CylindricalCamera,
        "fisheye": FisheyeCamera,
        "pinhole": PinholeCamera,
        "spherical": SphericalCamera,
    }

    def __init__(
        self,
        image_width=None,
        image_height=None,
        focal_u=None,
        focal_v=None,
        center_u=None,
        center_v=None,
        distort=None,
        roll=None,
        pitch=None,
        yaw=None,
        camera_model=None,
        meta_key=("anno", "meta"),
        default_calib=None,
        default_camera_model="pinhole",
        warping_on_bpu=False,
        post_resize_scale=1.0,
        crop_region=None,
        enable_inv_trans=True,
        transform_2d=False,
        probability=1.0,
        project_3d_to_vcs=False,
        use_LUT=False,
        ignore_mask_remap_stride=4,
        encode_ignore_mask=False,
        uvmap=None,
        uvmap_invert=None,
    ):
        self.image_width = image_width
        self.image_height = image_height
        self.focal_u = focal_u
        self.focal_v = focal_v
        self.center_u = center_u
        self.center_v = center_v
        self.distort = distort
        self.roll = roll
        self.pitch = pitch
        self.yaw = yaw
        self.camera_model = camera_model
        self.meta_key = meta_key
        self.default_calib = default_calib
        self.default_camera_model = default_camera_model
        self.warping_on_bpu = warping_on_bpu
        self.post_resize_scale = post_resize_scale
        self.crop_region = crop_region
        self.enable_inv_trans = enable_inv_trans
        self.transform_2d = transform_2d
        self.probability = probability
        self.project_3d_to_vcs = project_3d_to_vcs
        self.use_LUT = use_LUT
        self.ignore_mask_remap_stride = int(ignore_mask_remap_stride)
        self.encode_ignore_mask = encode_ignore_mask
        self.uvmap = uvmap
        self.uvmap_invert = uvmap_invert

    def _project_2Dlabels2dstcam(self, data, src_cam, dst_cam):
        for key in ["gt_boxes", "ig_regions"]:
            if key not in data or data[key].shape[0] == 0:
                continue
            data[key][:, :4] = bbox_clamp(
                src_cam.project_bbox2d2dstcam(dst_cam, data[key][:, :4]),
                im_hw=dst_cam.image_size[::-1],
            )
        return data

    def _project_labels2dstcam(self, data, src_cam, dst_cam, pitch=0):
        labels = data["anno"]["objects"]
        bboxes = []
        locations = []
        rotation_ys = []
        for label in labels:
            if "bbox" in label and label["bbox"] is not None:
                bboxes.append(label["bbox"])
            if "bbox_2d" in label and label["bbox_2d"] is not None:
                bboxes.append(label["bbox_2d"])
            if "in_camera" in label:
                locations.append(label["in_camera"]["location"])
                rotation_ys.append(label["in_camera"]["rotation_y"])
        bboxes = np.array(bboxes)
        locations = np.array(locations)
        rotation_ys = np.array(rotation_ys)

        if bboxes.shape[0] == 0:
            return data
        bboxes[..., 2:] += bboxes[..., :2]
        bboxes_output = bbox_clamp(
            src_cam.project_bbox2d2dstcam(dst_cam, bboxes),
            im_hw=dst_cam.image_size[::-1],
        )
        bboxes_output[..., 2:4] -= bboxes_output[..., :2]
        bboxes_output = bboxes_output.tolist()

        rot_vector = np.stack(
            [
                np.cos(pitch) * np.cos(rotation_ys),
                np.sin(pitch) * np.sin(rotation_ys),
                -np.cos(pitch) * np.sin(rotation_ys),
            ],
            axis=-1,
        )
        projected = src_cam.project_cam2dstCam(
            dst_cam, np.concatenate([locations, rot_vector])
        )
        rot_vector = projected[locations.shape[0] :]
        locations = projected[: locations.shape[0]].tolist()
        rotation_ys = np.arctan2(-rot_vector[:, 2], rot_vector[:, 0])

        bbox_index = 0
        bbox3d_index = 0
        for label in labels:
            if "bbox" in label and label["bbox"] is not None:
                label["bbox"] = bboxes_output[bbox_index]
                bbox_index += 1
            if "bbox_2d" in label and label["bbox_2d"] is not None:
                label["bbox_2d"] = bboxes_output[bbox_index]
                bbox_index += 1
            if "in_camera" in label:
                label["in_camera"]["location"] = locations[bbox3d_index]
                label["in_camera"]["depth"] = locations[bbox3d_index][-1]
                label["in_camera"]["rotation_y"] = rotation_ys[bbox3d_index]
                bbox3d_index += 1
        return data

    def _update_camera_parameters(
        self, calib_all, standardized_calib_all=None
    ):
        if isinstance(calib_all, list):
            if standardized_calib_all is None:
                standardized_calib_all = [None] * len(calib_all)
            return [
                self._update_camera_parameters(x, y)
                for x, y in zip(calib_all, standardized_calib_all)
            ]
        if calib_all is None:
            return calib_all
        if standardized_calib_all is not None:
            calib_all.update(standardized_calib_all)
            return calib_all

        for key in [
            "image_width",
            "image_height",
            "focal_u",
            "focal_v",
            "center_u",
            "center_v",
            "roll",
            "pitch",
            "yaw",
            "camera_model",
        ]:
            attr = getattr(self, key)
            if isinstance(attr, list):
                attr = np.random.choice(attr)

            if attr is not None:
                calib_all[key] = attr

        if self.distort is not None:
            if "param" in calib_all["distort"]:
                calib_all["distort"]["param"] = self.distort
            else:
                calib_all["distort"] = self.distort
        return calib_all

    def _get_camera(self, calib_all, camera_model=None):
        if isinstance(calib_all, list):
            if not isinstance(camera_model, list):
                camera_model = [camera_model] * len(calib_all)
            return [
                self._get_camera(x, y) for x, y in zip(calib_all, camera_model)
            ]

        if camera_model is None:
            camera_model = self.default_camera_model
        camera = self.CAMERA_MODELS[camera_model]()
        if calib_all is not None:
            camera = camera.init_cam_param_by_dict(calib_all)
        elif self.image_width is not None and self.image_height is not None:
            camera.image_size = [self.image_width, self.image_height]
        return camera

    def _img2buf(self, img, color_space):
        if isinstance(img, list):
            return [self._img2bug(x, color_space) for x in img]
        if color_space == "rgb":
            convertor_mode = cv2.COLOR_RGB2YUV_I420
        elif color_space == "bgr":
            convertor_mode = cv2.COLOR_BGR2YUV_I420
        else:
            raise AssertionError(f"Unsupport color space: {color_space}.")
        img = cv2.cvtColor(img, convertor_mode)
        return img.tobytes()

    def _get_uv_map(self, src_cam, dst_cam):
        if isinstance(src_cam, list):
            assert len(src_cam) == len(dst_cam)
            return [self._get_uv_map(x, y) for x, y in zip(src_cam, dst_cam)]
        record_key = get_cam_uuid(src_cam, dst_cam)
        if record_key in uv_map_record_dict:
            uv_map = uv_map_record_dict[record_key]
        elif src_cam.uuid == dst_cam.uuid:
            uv_map = np.ascontiguousarray(
                np.indices(dst_cam.image_size, dtype=np.float32).transpose(
                    2, 1, 0
                )
            )
            uv_map = (uv_map[..., 0:1], uv_map[..., 1:2])
        else:
            uv_map = src_cam.generate_mapping(dst_cam)
            uv_map_record_dict[record_key] = uv_map
        return uv_map

    def _get_valid_uv_map(self, uv_map, dst_shape=None):
        if isinstance(uv_map, (tuple, list)):
            assert len(uv_map) == 2 and len(uv_map[0].shape) == 3
            uv_map = np.concatenate(uv_map, axis=-1)

        if self.crop_region is not None:
            uv_map = uv_map[
                self.crop_region[1] : self.crop_region[3],
                self.crop_region[0] : self.crop_region[2],
            ]
            uv_map = uv_map - np.array(self.crop_region[:2])
        if dst_shape is not None:
            h, w = dst_shape
            scale = np.array([uv_map.shape[1] / w, uv_map.shape[0] / h])
        else:
            scale = self.post_resize_scale
            h, w = [int(x / scale) for x in uv_map.shape[:2]]
        uv_map = (
            cv2.resize(uv_map, (w, h), interpolation=cv2.INTER_NEAREST) / scale
        )
        return np.float32(uv_map)

    def _get_bpu_uv_map(self, uv_map, src_cam, dst_cam):
        if isinstance(uv_map, list):
            return [self._get_bpu_uv_map(x, src_cam, dst_cam) for x in uv_map]
        record_key = get_cam_uuid(src_cam, dst_cam)
        if record_key in bpu_uv_map_record_dict:
            bpu_uv_map = bpu_uv_map_record_dict[record_key]
            return bpu_uv_map
        if self.uvmap is not None:
            bpu_uv_map = self.uvmap
        else:
            bpu_uv_map = self._get_valid_uv_map(uv_map)
        h, w = bpu_uv_map.shape[:2]
        pixel = np.indices((w, h), dtype=np.float32).transpose(2, 1, 0)
        bpu_uv_map = bpu_uv_map - pixel
        bpu_uv_map_record_dict[record_key] = bpu_uv_map
        return bpu_uv_map

    def _remap(self, image, uv_map):
        if isinstance(image, list):
            assert len(image) == len(uv_map)
            output = [self._remap(x, y) for x, y in zip(image, uv_map)]
            return output
        image = cv2.remap(
            image,
            uv_map[0],
            uv_map[1],
            cv2.INTER_NEAREST,
        )
        return image

    def __call__(self, data):
        if np.random.uniform() > self.probability:
            return data

        meta = data
        for key in self.meta_key:
            meta = meta[key]

        if "calib_all" not in meta:
            src_calib_all = copy.deepcopy(self.default_calib)
        else:
            src_calib_all = meta["calib_all"]

        src_cam = self._get_camera(src_calib_all)
        if isinstance(src_calib_all, dict) and "pitch" in src_calib_all:
            pitch = src_calib_all["pitch"]
        else:
            pitch = 0
        dst_calib_all = self._update_camera_parameters(
            src_calib_all, meta.get("standardized_calib_all")
        )
        dst_cam = self._get_camera(dst_calib_all)

        uv_map = self._get_uv_map(src_cam, dst_cam)
        if self.warping_on_bpu:
            meta["uv_map"] = self._get_bpu_uv_map(uv_map, src_cam, dst_cam)
        elif "img" in data:
            data["img"] = self._remap(data["img"], uv_map)
        elif "ori_img" in data:
            image = self._remap(data["ori_img"], uv_map)
            data["img_buf"] = self._img2buf(image, data.get("color_space"))

        if "ignore_mask" in meta:
            if coco_mask is None:
                check_packages_available("pycocotools")
            ignore_mask = coco_mask.decode(meta["ignore_mask"])
            h, w = ignore_mask.shape[:2]
            stride = self.ignore_mask_remap_stride
            ignore_mask = self._remap(
                ignore_mask,
                [
                    np.asfortranarray(uv_map[0][::stride, ::stride]),
                    np.asfortranarray(uv_map[1][::stride, ::stride]),
                ],
            )
            ignore_mask = cv2.resize(
                ignore_mask, (w, h), interpolation=cv2.INTER_NEAREST
            )
            ignore_mask = np.asfortranarray(ignore_mask)
            if self.encode_ignore_mask:
                if coco_mask is None:
                    check_packages_available("pycocotools")
                meta["ignore_mask"] = coco_mask.encode(ignore_mask)
            else:
                meta["ignore_mask"] = ignore_mask

        if "anno" in data and "objects" in data["anno"]:
            data = self._project_labels2dstcam(data, src_cam, dst_cam, pitch)

        if self.transform_2d:
            data = self._project_2Dlabels2dstcam(data, src_cam, dst_cam)
        if not isinstance(dst_cam, list) and dst_calib_all is not None:
            meta["calib"] = np.concatenate(
                [dst_cam.camera_matrix, np.array([0, 0, 0])[:, None]], axis=1
            )
            meta["distCoeffs"] = np.array(dst_cam.distcoeffs)

        if self.enable_inv_trans and src_calib_all is not None:
            transform_meta = {"src_cam": src_cam, "dst_cam": dst_cam}
        else:
            transform_meta = {"src_cam": None}
        if "transform_meta" in data:
            data["transform_meta"].append(transform_meta)
        else:
            data["transform_meta"] = [transform_meta]

        return data

    def inverse_transform(self, task_model_outs, transform_meta):
        src_cam = transform_meta.get("src_cam", None)
        dst_cam = transform_meta.get("dst_cam", None)
        if src_cam is None or dst_cam is None:
            return task_model_outs
        if isinstance(task_model_outs, DetBoxes3D):
            task_model_outs = self._handler_box3d(
                task_model_outs, src_cam, dst_cam
            )

        if isinstance(task_model_outs, (DetBoxes2D, MultipleBox2D)):
            task_model_outs = self._handler_box2d(
                task_model_outs, src_cam, dst_cam
            )

        if isinstance(task_model_outs, MultipleBoxes2D):
            task_model_outs = self._handler_multi_box2d(
                task_model_outs, src_cam, dst_cam
            )

        if isinstance(task_model_outs, Mask):
            task_model_outs = self._handler_mask(
                task_model_outs, src_cam, dst_cam
            )

        if isinstance(task_model_outs, (Lines2D, Points2D_2)):
            task_model_outs = self._handler_point(
                task_model_outs, src_cam, dst_cam
            )

        if isinstance(task_model_outs, DetObjects):
            sub_structs = [f.name for f in fields(type(task_model_outs))]
            for struct in sub_structs:
                setattr(
                    task_model_outs,
                    struct,
                    self.inverse_transform(
                        getattr(task_model_outs, struct), transform_meta
                    ),
                )
        return task_model_outs

    def _handler_box3d(self, task_model_outs, src_cam, dst_cam):
        locations = task_model_outs.locations.cpu().numpy()
        if locations.shape[0] > 0:
            yaw = task_model_outs.yaw.cpu().numpy()
            rot_vector = np.stack(
                [np.cos(yaw), np.zeros_like(yaw), -np.sin(yaw)],
                axis=-1,
            )

            if self.project_3d_to_vcs:
                locations = dst_cam.project_cam2vcs(locations)
                poseMat_cam2vcs = np.linalg.inv(dst_cam.poseMat_vcs2cam)
                rot_vector = np.dot(rot_vector, poseMat_cam2vcs[:3, :3].T)
                yaw = np.arctan2(rot_vector[..., 1], rot_vector[..., 0])
            else:
                locations = dst_cam.project_cam2dstCam(src_cam, locations)
                rot_vector = dst_cam.project_cam2dstCam(src_cam, rot_vector)
                yaw = np.arctan2(-rot_vector[..., 2], rot_vector[..., 0])

            locations = task_model_outs.x.new_tensor(locations)
            yaw = task_model_outs.x.new_tensor(yaw)

            task_model_outs.x = locations[..., 0]
            task_model_outs.y = locations[..., 1]
            task_model_outs.z = locations[..., 2]
            task_model_outs.yaw = yaw
        return task_model_outs

    def _handler_box2d(self, task_model_outs, src_cam, dst_cam):
        boxes = task_model_outs.boxes.cpu().numpy()
        if boxes.shape[0] > 0:
            if not self.use_LUT:
                boxes = dst_cam.project_bbox2d2dstcam(src_cam, boxes)
            else:
                boxes = self._project_bboxes2dstcam_byLUT(
                    dst_cam, src_cam, boxes
                )
            boxes = task_model_outs.boxes.new_tensor(boxes)
            task_model_outs.boxes = boxes
        return task_model_outs

    def _handler_multi_box2d(self, task_model_outs, src_cam, dst_cam):
        for i, boxes in enumerate(task_model_outs.boxes_list):
            _boxes = boxes.cpu().numpy()
            if boxes.shape[0] > 0:
                if not self.use_LUT:
                    _boxes = dst_cam.project_bbox2d2dstcam(src_cam, _boxes)
                else:
                    _boxes = self._project_bboxes2dstcam_byLUT(
                        dst_cam, src_cam, _boxes
                    )
                _boxes = boxes.new_tensor(_boxes)
                task_model_outs.boxes_list[i] = _boxes
        return task_model_outs

    def _handler_mask(self, task_model_outs, src_cam, dst_cam):
        record_key = get_cam_uuid(dst_cam, src_cam)
        if record_key in uv_map_record_dict:
            uv_map = uv_map_record_dict[record_key]
        else:
            uv_map = dst_cam.generate_mapping(src_cam)
            uv_map_record_dict[record_key] = uv_map
        output_size = task_model_outs.mask.shape[:2]
        if self.uvmap_invert is not None:
            uv_map = self.uvmap_invert
        else:
            uv_map = self._get_valid_uv_map(uv_map, output_size)
        mask = task_model_outs.mask.cpu().numpy()
        mask = cv2.remap(
            mask,
            uv_map[..., 0],
            uv_map[..., 1],
            cv2.INTER_NEAREST,
            borderValue=255,
        )
        task_model_outs.mask = task_model_outs.mask.new_tensor(mask)
        return task_model_outs

    def _handler_point(self, task_model_outs, src_cam, dst_cam):
        points0 = task_model_outs.points0.points.cpu().numpy()
        if points0.shape[0] > 0:
            points1 = task_model_outs.points1.points.cpu().numpy()

            if not self.use_LUT:
                points0 = dst_cam.project_pixel2dstCam(src_cam, points0)
                points1 = dst_cam.project_pixel2dstCam(src_cam, points1)
            else:
                points0 = self._project_pixel2dstCam_byLUT(
                    dst_cam, src_cam, points0
                )
                points1 = self._project_pixel2dstCam_byLUT(
                    dst_cam, src_cam, points1
                )

            task_model_outs.points0.points = (
                task_model_outs.points0.points.new_tensor(points0)
            )
            task_model_outs.points1.points = (
                task_model_outs.points1.points.new_tensor(points1)
            )
        return task_model_outs

    def _project_pixel2dstCam_byLUT(self, src_cam, dst_cam, points):
        record_key = get_cam_uuid(dst_cam, src_cam)
        if record_key in uv_map_record_dict:
            uv_map = uv_map_record_dict[record_key]
        else:
            uv_map = dst_cam.generate_mapping(src_cam)
            uv_map_record_dict[record_key] = uv_map
        uv_map = self._get_valid_uv_map(uv_map)
        points = cv2.remap(
            uv_map,
            points[:, 0:1],
            points[:, 1:2],
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )
        return points[:, 0]

    def _project_bboxes2dstcam_byLUT(self, src_cam, dst_cam, boxes):
        num_box = boxes.shape[0]
        points = np.stack(
            [
                boxes[..., 0],
                (boxes[..., 1] + boxes[..., 3]) / 2,
                boxes[..., 2],
                (boxes[..., 1] + boxes[..., 3]) / 2,
                (boxes[..., 0] + boxes[..., 2]) / 2,
                boxes[..., 1],
                (boxes[..., 0] + boxes[..., 2]) / 2,
                boxes[..., 3],
            ],
            axis=-1,
        ).reshape(num_box * 4, 2)
        points = self._project_pixel2dstCam_byLUT(src_cam, dst_cam, points)
        points = points.reshape(num_box, 4, 2)
        x1 = np.min(points[..., 0], axis=1)
        y1 = np.min(points[..., 1], axis=1)
        x2 = np.max(points[..., 0], axis=1)
        y2 = np.max(points[..., 1], axis=1)
        boxes = np.stack([x1, y1, x2, y2], axis=-1)
        return boxes
