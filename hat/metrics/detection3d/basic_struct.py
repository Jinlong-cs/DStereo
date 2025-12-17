from typing import Any, Dict, List, Optional, Union

import numpy as np

try:
    from hatbc.message.structure import BBox2D, BBox3D
except ImportError:
    BBox2D, BBox3D = None, None

from hat.utils.package_helper import require_packages


class Instance3dObj(object):
    """Object 3D struct.

    Args:
        bbox: 2d bbox.
        dimensions: dimensions, default in [height, width, length] format.
        location: locations, in [x, y, z] format.
        rotation_y: yaw angle.
        score: confidence score. Defaults to None.
        ignore: whether to ignore this object. Defaults to False.
        velocity: velocity, in [x, y, z] format, m/s, in vcs.
        acceleration: acceleration, in [x, y, z] format, m/s^2, in vcs.
        yaw_rate: yaw_rate, deg/s, in vcs coordinates.
        dim_format: dimensions format. Defaults to "hwl".
        depth_idx: idx of depth in location.
            z-axis refer to depth in camera coord, x-axis in vcs coord.
        meta: other necessary info.
    """

    @require_packages("hatbc")
    def __init__(
        self,
        dimensions: List[float],
        location: List[float],
        rotation_y: float,
        bbox: List[float] = None,
        category: Optional[str] = None,
        score: Optional[float] = None,
        ignore: Optional[bool] = False,
        velocity: List[float] = None,
        acceleration: List[float] = None,
        yaw_rate: Optional[float] = None,
        dim_format: Optional[str] = "hwl",
        depth_idx: Optional[int] = 2,
        meta: Any = None,
    ):
        self.bbox = bbox
        self.dimensions = dimensions
        self.location = location
        self.rotation_y = rotation_y
        self.velocity = velocity
        self.acceleration = acceleration
        self.yaw_rate = yaw_rate

        self.ignore = ignore
        self.score = score
        self.category = category
        self.meta = meta

        assert dim_format in ["hwl", "whl", "lwh"]
        assert depth_idx in [0, 1, 2]
        self.dim_format = dim_format
        self.depth_idx = depth_idx

    @property
    def depth(self):
        return self.location[self.depth_idx]

    @property
    def bbox_2d(self) -> "BBox2D":
        bbox_2d = BBox2D(data=self.bbox, score=self.score)

        return bbox_2d

    @property
    def bbox_3d(self) -> "BBox3D":
        # hwl -> whl
        # hatbc.message.BBox3D used dim in "whl" format.
        # https://gitlab.hobot.cc/ptd/algorithm/ai-platform-algorithm/hatbc/-/blob/master/python/hatbc/message/structure.py  # noqa E501
        dim = self.get_dim_by_format(out_format="whl")
        bbox_3d = BBox3D(
            dim=dim,
            loc=self.location,
            yaw=self.rotation_y,
            score=self.score,
        )  # noqa E501

        return bbox_3d

    def get_dim_by_format(self, out_format: str = "hwl"):
        target_dims = [
            self.dimensions[self.dim_format.index(out_format[0])],
            self.dimensions[self.dim_format.index(out_format[1])],
            self.dimensions[self.dim_format.index(out_format[2])],
        ]
        return target_dims

    def to_dict(self):
        # only keep 4-digit number
        result = dict(  # noqa
            dimensions=[round(d, 4) for d in self.dimensions],
            dim_format=self.dim_format,
            location=[round(loc, 4) for loc in self.location],
            rotation_y=round(self.rotation_y, 4),
            category=self.category,
            ignore=self.ignore,
            score=round(self.score, 4)
            if isinstance(self.score, float)
            else self.score,
            meta=self.meta,
        )
        return result


class ImgObj(object):
    """Single frame object for detection3d.

    Args:
        image_key: Image name.
        instances: Ground truth or prediction objects
        width: Image width. Defaults to None.
        height: Image height. Defaults to None.
        calib: Calibration. Defaults to None.
        dist_coeffs: Distortion coefficient. Defaults to None.
        ignore_mask: Ignore mask in gt label. Defaults to None.
        lidar_to_camera: Args for converting lidar to camera. Defaults to None.
        camera_model: FisheyeCamera,CylindricalCamera,SphericalCamera or None.
        attrs: Attributes e.g. `tags`. Defaults to None.
    """

    def __init__(
        self,
        image_key: str,
        instances: List[Instance3dObj],
        width: int = None,
        height: int = None,
        calib: Union[np.ndarray, List] = None,
        dist_coeffs: Union[np.ndarray, List] = None,
        ignore_mask: dict = None,
        lidar_to_camera: dict = None,
        camera_model: Optional[str] = None,
        attrs: Optional[Dict[str, Any]] = None,
    ):

        self.instances = instances
        self.image_key = image_key
        self.calib = calib
        self.dist_coeffs = dist_coeffs
        self.width = width
        self.height = height
        self.ignore_mask = ignore_mask
        self.lidar_to_camera = lidar_to_camera
        self.camera_model = camera_model
        self.attrs = attrs


def parse_to_struct(infos_dict: dict, eval_class: str, info_type: str = "gt"):

    assert info_type in ["gt", "pred"]

    objs_infos = infos_dict.get(eval_class, None)

    objs = []

    if objs_infos is not None:
        for obj_info in objs_infos:
            obj = Instance3dObj(
                bbox=obj_info["bbox"]
                if info_type == "gt"
                else obj_info["bbox_2d"],  # noqa E501
                dimensions=obj_info.get("dimensions"),
                location=obj_info.get("location"),
                rotation_y=obj_info.get("rotation_y"),
                score=obj_info.get("score", None),
                ignore=obj_info.get("ignore", False),
            )
            objs.append(obj)

    camera_model = infos_dict.get("camera_model", None)
    if camera_model is not None:
        camera_model = (
            camera_model + "Camera"
            if "Camera" not in camera_model
            else camera_model
        )
        assert camera_model in [
            "FisheyeCamera",
            "CylindricalCamera",
            "SphericalCamera",
        ], f"camera_model in json file should be Fisheye, Cylindrical, Spherical or None. But get {camera_model}"  # noqa E501

    return ImgObj(
        image_key=infos_dict["image_key"],
        instances=objs,
        calib=infos_dict.get("calib", None),
        dist_coeffs=infos_dict.get("distCoeffs", None),
        width=infos_dict.get("width", None),
        # Note: image 'height' field in GT annotation file may be misspelled.
        # used 'hight' in GT:
        # https://gitlab.hobot.cc/auto/perception/ad/horizonadas_evalkit/-/blob/dev/examples/detection_3d/example_data/gt.json  # noqa E501
        # https://gitlab.hobot.cc/auto/perception/ad/horizonadas_evalkit/-/blob/dev/adas_eval/detection_3d/auto.py#L242 # noqa E501
        height=infos_dict.get("hight")
        if "hight" in infos_dict
        else infos_dict.get("height", None),
        lidar_to_camera=infos_dict.get("lidar_to_camera", None),
        ignore_mask=infos_dict.get("ignore_mask", None),
        camera_model=camera_model,
        attrs=infos_dict.get("attrs", {}),
    )
