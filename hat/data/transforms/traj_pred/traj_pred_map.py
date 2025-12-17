# Copyright (c) Horizon Robotics. All rights reserved.

import colorsys
import copy
import hashlib
import logging
import os
import random
from glob import glob
from typing import Callable, Dict, List, Optional

import cv2
import numpy as np
from PIL import Image
from scipy.sparse import bmat, lil_matrix
from tqdm import tqdm

from hat.core.traj_pred_typing import PathLike
from hat.core.traj_pred_utils import (
    Affine2D,
    interpolate_trajectory_from_path,
    normalize_yaw,
)
from hat.data.transforms.traj_pred.traj_pred_coords import DetectCurve
from hat.registry import OBJECT_REGISTRY

__all__ = [
    "OccupancyMapRender",
    "GetBEVLocalMapByTimestamp",
    "GetBEVHomography",
    "MapAugmentation",
    "NuScenesMapServer",
    "NuScenesRenderedMap",
    "VectorNetStructuredMapServer",
    "ArgoverseStructredMapServer",
    "ArgoverseStructredVizHelper",
    "GetBevVectorizedMap",
]

logger = logging.getLogger(__name__)


@OBJECT_REGISTRY.register
class OccupancyMapRender:
    """The occupancy map render.

    Here, we render all the obstacle bounding boxes in each context
    frames to a occupancy map. \

    To use, the user should construct a `OccupancyMapRender` instance with the
    required initialization parameters. The user should use this tranform after
    `GenBoundingBox`, `PhyToImage` (or `PhyToBEV`) and `GetSeqDataFrameMask`. \

    After instantiation, the callable function will return a new dict that
    contains the following changes: \
        1. a new key 'rendered_obs' was added.
    """

    def __init__(
        self,
        map_height: int,
        map_width: int,
        render_ego: bool = True,
        normalize: bool = False,
    ):
        """Initialize method.

        Args:
            map_height (int): rendered map height.
            map_width (int): rendered map width.
            render_ego (bool, optional): whether to render ego vehicle.
                Default to True.
            normalize (bool, optional): whether to normalize the road map
                from [0, 255] -> [-1, 1]. Default to False.
        """
        self.map_height = map_height
        self.map_width = map_width
        self.render_ego = render_ego
        self.normalize = normalize

    def __call__(self, sample: Dict) -> Dict:
        """Callable function.

        The following keys are required:
            1. 'seq_df'.
            2. 'all_ctx_frame'

        Please make sure that:
            1. `seq_df` is a dataframe and have the following columns:
                'frame_id', 'img_x0', 'img_y0', 'img_x1', 'img_y1',
                'img_x2', 'img_y2', 'img_x3', 'img_y3', 'img_ego_x0',
                'img_ego_y0', 'img_ego_x1', 'img_ego_y1', 'img_ego_x2',
                'img_ego_y2', 'img_ego_x3', 'img_ego_y3'
            2. Therefore, this tranform method must be used after
                `GenBoundingBox` and `PhyToImage` (or `PhyToBEV`).

        Args:
            sample (Dict): input original sample. The `sample` should be
                obtained by the `__getitem__` of trajectory dataset (base
                class `hat.data.datasets.traj_pred_dataset.BaseTrajDataset`)

        Returns:
            sample (Dict): the updated sample as described above.
        """
        seq_df = sample["seq_df"]
        # Rendered obs map should generate from sampled data by default.
        if "sampled_ctx_frame_id" in sample:
            all_ctx_frame = sample["sampled_ctx_frame_id"]
        else:
            all_ctx_frame = sample["ctx_frame_id"]

        # Render historical frames as polygons.
        past_bboxes = []
        for frame_id in all_ctx_frame:
            frame_df = seq_df.loc[seq_df["frame_id"] == frame_id]
            past_bboxes.append(
                self.render_one_frame_bboxes(
                    frame_df, self.map_height, self.map_width
                )
            )

        rendered_obs = np.concatenate([*past_bboxes], axis=2)
        if self.normalize:
            # Norm the occumancy map. [0, 255] -> [-1, 1].
            rendered_obs = (rendered_obs - 128) / 128

        sample["rendered_obs"] = rendered_obs
        return sample

    def render_one_frame_bboxes(
        self, frame_df, map_height, map_width
    ):  # noqa: D205,D400
        """Render obstacle bounding bboxes (in one frame) and
        generate the occupancy map. The values are 0~255.

        Args:
            frame_df: (pd.DataFrame): a DataFrame with the obstacle
                position information of a certain timestamp.
            map_height (int): the height of the rendered map.
            map_width (int): the width of the rendered map.

        Returns:
            ndarray: Rendered bboxes ndarray for the frame
        """
        if frame_df.empty:
            return np.zeros([map_height, map_width, 1])

        # Pad the base map to allow incomplete polygon rendering.
        base = np.zeros([3 * map_height, 3 * map_width])
        # Get the vehicle bounding boxes.

        # fmt: off
        obs_bbox_cols = [
            "img_x0", "img_y0", "img_x1", "img_y1",
            "img_x2", "img_y2", "img_x3", "img_y3",
        ]
        obs_bbox = frame_df[obs_bbox_cols].values
        if self.render_ego:
            ego_bbox_cols = [
                "img_ego_x0", "img_ego_y0", "img_ego_x1", "img_ego_y1",
                "img_ego_x2", "img_ego_y2", "img_ego_x3", "img_ego_y3",
            ]
            ego_bbox = np.unique(frame_df[ego_bbox_cols], axis=0)
            img_bboxes = np.concatenate([ego_bbox, obs_bbox], axis=0)
        else:
            img_bboxes = obs_bbox

        # fmt: on

        for img_bbox in img_bboxes:
            v_x = img_bbox[::2]
            v_y = img_bbox[1::2]
            if v_x.size == 0 or v_y.size == 0:
                continue
            elif (
                np.min(v_x) < -map_height
                or np.max(v_x) >= 2 * map_height
                or np.min(v_y) < -map_width
                or np.max(v_y) >= 2 * map_width
            ):
                continue
            else:
                corners = np.stack([v_y, v_x], axis=1)[None, :, :].astype(
                    np.int32
                )
                base = cv2.drawContours(
                    base, corners + map_height, -1, (255, 0, 0), -1
                )
        base = base[
            map_height : 2 * map_height, map_width : 2 * map_width, None
        ]
        return base


@OBJECT_REGISTRY.register
class GetBEVLocalMapByTimestamp:
    """Get BEV local map by timestamp.

    The local maps are pre-rendered and indexed by timestamps. Given a certain
    timestamp, we can set a vehicle (in most cases, it is the ego vehicle) as
    the center vehicle, and bind its postion with fixed BEV coordinates. Then,
    we can get the local map surrouding the center vehicle. We save the maps
    as figures with timestamps as file names. \

    Since origin BEV map is named by timestamps, the input dictionary must
    have the key `lcf_timestamp`. This means that this transform must be used
    after `GetLcfTimeStamp`. \

    To use, the user should construct a `GetBEVLocalMapByTimestamp`. After
    instantiation, the callable function will return a new dict that contains
    the following changes: \
        1. the `road_map` was added.
    """

    def __init__(
        self,
        image_dir: PathLike,
        image_suffix: str,
        map_height: int,
        map_width: int,
        data_token2path_mapping: dict,
        is_viz: bool = False,
        render_mapping: Optional[dict] = None,
        map_path_func: Optional[Callable] = None,
        item_key: str = "road_map",
    ):
        """Initialize method.

        Args:
            image_dir (PathLike): the root path to the image directory.
            image_suffix (str): the suffix of the images.
            map_height (int): rendered map height.
            map_width (int): rendered map width.
            data_token2path_mapping (dict): the keys are the date tokens, and
                the values are the path prefix to the origin map file. Usually,
                the bev map files of different dates are stored in different
                paths, and `image_dir` is just their largest common path.
            is_vlz (bool, optional): whether the map is for visualization.
                Default to False.
            render_mapping (dict, optional): the color mapping of different
                road elements. Default to None. If the user wants to get a
                local map that renders different road elements (such as lane
                lines, curbs, sidewalks, etc.) as different colors, he should
                input the mapping between element classifications and color
                configurations here. If this parameter is None, this function
                will directly return the origin map from the file.
            item_key (str, optional): the new key that is added to the batch.
                Default to "road_map".
        """
        self.image_dir = image_dir
        self.image_suffix = image_suffix
        self.map_height = map_height
        self.map_width = map_width
        self.data_token2path_mapping = data_token2path_mapping
        self.is_viz = is_viz
        self.render_mapping = render_mapping
        self.item_key = item_key
        if map_path_func is not None:
            self.map_path_func = map_path_func
        else:
            self.map_path_func = self.default_map_path_func

    def __call__(self, sample: dict):
        """Callable function.

        The following keys are required:
            1. `dataset_prefix`.
            2. `date_token`.
            3. `lcf_timestamp`.

        Args:
            sample (dict): input original sample.

        Returns:
            sample (dict): the sample with added 'road_map'.
        """
        plate = sample["dataset_prefix"]
        date_token = sample["date_token"]
        if "recons_timestamp" in sample:
            timestamp = str(int(sample["recons_timestamp"]))  # 复现软件端badcase
        else:
            timestamp = str(int(sample["lcf_timestamp"]))  # 正常加载dataset
        date = date_token.split("-")[0]
        date_key = plate + date + "_viz" if self.is_viz else plate + date

        bev_bucket_path = self.data_token2path_mapping[date_key]
        bev_map_path = self.map_path_func(
            self.image_dir,
            bev_bucket_path,
            plate,
            date_token,
            timestamp,
            self.image_suffix,
        )

        need_render = True
        if not os.path.exists(bev_map_path):
            # logger.warning(f"Image file {bev_map_path} does not exist!")
            road_map = np.zeros((self.map_height, self.map_width))
        else:
            try:
                with Image.open(bev_map_path) as im:
                    im = np.array(im)
                    road_map = im
                    if len(im.shape) == 3:
                        if np.max(im) < 20:
                            road_map = im[:, :, 0]
                            need_render = True
                        else:
                            need_render = False
            except BaseException:
                logger.warning(f"Image file {bev_map_path} is corrupted!")
                road_map = np.zeros((self.map_height, self.map_width))
        if need_render and self.render_mapping is not None:
            for key, value in self.render_mapping.items():
                road_map[road_map == key] = value
        sample[self.item_key] = road_map
        return sample

    @staticmethod
    def default_map_path_func(
        image_dir, bev_bucket_path, plate, date_token, timestamp, image_suffix
    ):
        date = date_token.split("-")[0]
        bev_map_path = os.path.join(
            image_dir,
            bev_bucket_path,
            plate + date + "_D",
            date_token,
            timestamp + image_suffix,
        )
        return bev_map_path


@OBJECT_REGISTRY.register
class GetBEVHomography:
    """Get the homography matrix.

    This transform method is an important part to simulate Rotated ROIAlign,
    which is not supported now. \

    The output of trajectory prediction backbone is a feature map with the
    shape [src_shape, src_shape, C]. We know the postion and yaw of all
    obstacles in the BEV coordinates, and we want to crop an ROI around this
    postion in the feature map and Align it into the fixed shape [dst_shape,
    dst_shape, C]. Then the homography matrix will be used to get the offset
    and use warpping. \

    To use, the user should construct a `GetBEVHomography`. After
    instantiation, the callable function will return a new dict that contains
    the following changes: \
        1. the `img_homographys` was added. \

    This transform method must be used after `GetTrajPredObjectsInfo` because
    it requires the sample key 'valid_img_coords'.
    """

    def __init__(
        self,
        src_shape,
        dst_shape,
        x_origin_ratio=0.5,
        y_origin_ratio=0.5,
        roi_spatial_scale=32,
        swap_xy=False,
        input_key="valid_img_coords",
        output_key="img_homographys",
    ):
        """Initialize method.

        Args:
            src_shape (int): the length and width of the `ROIAlign` input.
            dst_shape (int): the length and width of the `ROIAlign` output.
            x_origin_ratio (float, optional): the ratio between the origin x
                coordinates and the length of BEV map. Default to 0.5,
                which means the origin is located at the center of BEV map
                (along x direction).
            y_origin_ratio (float, optional): the ratio between the origin y
                coordinates and the width of BEV map. Default to 0.5,
                which means the origin is located at the center of BEV map
                (along y direction).
            roi_spatial_scale (float, optional): the scale between the origin
                input map and `src_shape`.
            swap_xy (bool, optional) whether to swap x and y coordinates in
                the homography matrix. In our definition, we set h=x and w=y,
                but some operations like GridSample set h=y and w=x. If we
                use those operation, we need to set this parameter as True.
            input_key: the required key of this transform, should be the type
                of img_coords.
            output_key: the output key of this transform, should be the type
                of homographys.
        """
        self.roi_scale = roi_spatial_scale
        self.swap_xy = swap_xy
        self.dst_shape = dst_shape
        self.input_key = input_key
        self.output_key = output_key
        x_offset = src_shape * x_origin_ratio
        y_offset = src_shape * y_origin_ratio
        # The four corners of the input feature map (the coordinate is
        # centered at the scaled object origin, and the coordinates are
        # parallel to the object VCS coordinates.)
        src_region = [
            [x_offset, y_offset],
            [x_offset, y_offset - src_shape],
            [x_offset - src_shape, y_offset],
            [x_offset - src_shape, y_offset - src_shape],
        ]
        # The four corners of the (ROIAlign) output feature map.
        dst_region = [
            [0, 0],
            [dst_shape, 0],
            [0, dst_shape],
            [dst_shape, dst_shape],
        ]

        self.trans_dst2obj = cv2.getPerspectiveTransform(
            np.array(dst_region).astype(np.float32),
            np.array(src_region).astype(np.float32),
        )
        self.trans_dst2obj = self.trans_dst2obj.astype(np.float32)
        # Based on the above trans mat, we bind the center obstacle (each
        # valid object) with the same position of the dst map. In the
        # callable function, we get the trans mat from the object scaled
        # VCS to the BEV coordinates.

        meshgrid = np.meshgrid(
            range(dst_shape), range(dst_shape), indexing="xy"
        )
        id_coords = np.stack(meshgrid, axis=0).astype(np.float32)
        ones = np.ones((1, dst_shape, dst_shape), dtype="float32")
        self.pix_coords = np.concatenate([id_coords, ones], axis=0).reshape(
            (1, 3, -1)
        )

    def __call__(self, sample: dict):
        """Callable function.

        The following keys are required in `sample`:
            1. self.input_key (default 'valid_img_coords')

        Args:
            sample (dict): input original sample.

        Returns:
            sample (dict): the sample includes a new value self.output_key
            (default `img_homographys`).
        """
        bev_coords = sample[self.input_key]
        homographys = []
        for obj_img_coords in bev_coords:
            bev_img_x, bev_img_y, bev_img_yaw = obj_img_coords
            src_x_offset = bev_img_x / self.roi_scale
            src_y_offset = bev_img_y / self.roi_scale
            # Trans Matrix from `object coordinates` to `image coordinates`.
            # Here, the so called `object coordinates` (`image coordinates`)
            # is generated by scaling the real obs VCS (BEV) coordinates. The
            # scaling ratio is `self.roi_scale`.
            trans_obj2img = np.array(
                [
                    [np.cos(bev_img_yaw), -np.sin(bev_img_yaw), src_x_offset],
                    [np.sin(bev_img_yaw), np.cos(bev_img_yaw), src_y_offset],
                    [0, 0, 1],
                ],
                dtype=np.float32,
            )
            if self.swap_xy:
                trans_swap = np.array(
                    [[0, 1, 0], [1, 0, 0], [0, 0, 1]], dtype=np.float32
                )
                trans_obj2img = np.matmul(trans_swap, trans_obj2img)
            trans_dst2img = np.matmul(
                trans_obj2img, self.trans_dst2obj
            ).reshape((-1, 3, 3))
            homographys.append(trans_dst2img)

        # get warpoffset
        if not len(homographys):
            sample[self.output_key] = []
        else:
            homographys = np.concatenate(homographys)
            img_points = np.matmul(homographys, self.pix_coords)
            new_pix_coords = (
                img_points[:, :2, :] / img_points[:, 2, :][:, None, :]
            )
            pix_coord_offsets = new_pix_coords - self.pix_coords[:, :2, :]
            pix_coord_offsets = pix_coord_offsets.reshape(
                -1, 2, self.dst_shape, self.dst_shape
            )
            pix_coord_offsets = pix_coord_offsets.transpose(0, 2, 3, 1)

            sample[self.output_key] = pix_coord_offsets
        return sample


@OBJECT_REGISTRY.register
class MapAugmentation:
    """Perform map augmentation.

    This transform method must be used if and only if the user sets the
    parameter `augmentation` in `GenSeqCenter` as True.

    To use, the user should construct a `MapAugmentation`. After
    instantiation, the callable function will return a new dict that contains
    the following changes: \
        1. the `affine_transforms` was added. \
    """

    def __init__(
        self,
        src_resolution,
        dst_resolution,
        map_origin_x,
        map_origin_y,
        swap_xy=False,
    ):
        """Initialize method.

        Args:
            src_resolution (float): the resolution of the source map.
            dst_resolution (float): the resolution of the destination map.
            map_origin_x (float): the x coordinate of the bev orgin [m]
                in the map coordinate.
            map_origin_y (float): the y coordinate of the bev orgin [m]
                in the map coordinate.
            swap_xy (bool, optional) whether to swap x and y coordinates in
                the homography matrix. In our definition, we set h=x and w=y,
                but some operations like GridSample set h=y and w=x. If we
                use those operation, we need to set this parameter as True.
        """
        self.src_resolution = src_resolution
        self.dst_resolution = dst_resolution
        self.map_origin_x = map_origin_x
        self.map_origin_y = map_origin_y
        self.swap_xy = swap_xy

    def __call__(self, sample: dict):
        """Callable function.

        The following keys are required in `sample`:
            1. "seq_df"
            2. "seq_center"
            3. "last_context_frame_id"

        Please make sure that:
            1. `seq_df` is a dataframe and have the following columns:
                'pos_x', 'pos_y', 'yaw', 'frame_id'

        Args:
            sample (Dict): input original sample.

        Returns:
            sample (Dict): the sample includes new keys "affine_transforms"
        """
        # Extract information from sample.
        seq_df = sample["seq_df"]
        seq_center = sample["seq_center"]
        seq_center_x = seq_center.pos_x
        seq_center_y = seq_center.pos_y
        seq_center_yaw = seq_center.yaw
        last_context_frame_id = sample["last_context_frame_id"]

        vals = seq_df.values
        cols = seq_df.columns
        phy_ego_x_col = cols.get_loc("pos_x")
        phy_ego_y_col = cols.get_loc("pos_y")
        ego_yaw_col = cols.get_loc("yaw")
        frame_id_col = cols.get_loc("frame_id")
        lcf_mask = vals[:, frame_id_col] == last_context_frame_id
        ego_x = vals[lcf_mask, phy_ego_x_col].astype("float64")[0]
        ego_y = vals[lcf_mask, phy_ego_y_col].astype("float64")[0]
        ego_yaw = vals[lcf_mask, ego_yaw_col].astype("float64")[0]

        # Calculate the dst-VCS origin coordiates in the src-VCS.
        delta_x, delta_y = Affine2D.coord_translate(
            np.array([seq_center_x]), np.array([seq_center_y]), ego_x, ego_y
        )
        vcs_x, vcs_y = Affine2D.coord_rotate(delta_x, delta_y, ego_yaw)
        vcs_yaw = seq_center_yaw - ego_yaw

        dstimg_to_dstvcs = np.array(
            [
                [-self.dst_resolution, 0, self.map_origin_x],
                [0, -self.dst_resolution, self.map_origin_y],
                [0, 0, 1],
            ],
            dtype=np.float32,
        )
        dstvcs_to_srcvcs = np.array(
            [
                [np.cos(vcs_yaw), -np.sin(vcs_yaw), vcs_x[0]],
                [np.sin(vcs_yaw), np.cos(vcs_yaw), vcs_y[0]],
                [0, 0, 1],
            ],
            dtype=np.float32,
        )
        srcvcs_to_srcimg = np.array(
            [
                [
                    -1 / self.src_resolution,
                    0,
                    self.map_origin_x / self.src_resolution,
                ],
                [
                    0,
                    -1 / self.src_resolution,
                    self.map_origin_y / self.src_resolution,
                ],
                [0, 0, 1],
            ],
            dtype=np.float32,
        )
        trans_mat = np.matmul(dstvcs_to_srcvcs, dstimg_to_dstvcs)
        trans_mat = np.matmul(srcvcs_to_srcimg, trans_mat)
        if self.swap_xy:
            trans_swap = np.array(
                [[0, 1, 0], [1, 0, 0], [0, 0, 1]], dtype=np.float32
            )
            trans_mat = np.matmul(trans_mat, trans_swap)
            trans_mat = np.matmul(trans_swap, trans_mat)

        sample["affine_transforms"] = trans_mat
        return sample


class BaseUnstructuredMapServer:
    """The base class for unstructured map server.

    The map server is designed to work on two modes: online and offline.
    1) The user should implement the `_load_base_map` function and set
        `pre_load_map = True` to enable the offline mode. The base map
        will be preloaded to the memory, the `get` can faster but more
        memory will be used.
    2) The user should implement the `_load_local_map` function and set
        `pre_load_map = False` to enable the online mode. The `get`
        will be slower than the offline mode.

    The base map is a (very large) semantic image for the road plain,
    containing information such as drivable area, road type, etc. But
    in different cases, the users may set different definitions of the
    coordinate system, resulting the different based map chiralities in
    the different sub-class. To avoid this error, we define the
    chiralities in the workflow of this class:

    Suppose the z-axis direction is outward from the map plane, all the
    loaded base map should in left-hand systems. And the output map
    images are in right-hand systems. If viewed in the BEV vision,
    the base map and the output map should in the following coordinates:
            (base)                    (output)
            ---------- x-axis (W)      ---------- y-axis (W)
            |                          |
            |                     ->   |
            |                          |
           y-axis (H)                x-axis (H)

    The public method is `get` for querying a specific locality of the base
    map. The user should provide the center, orientation, and resolution
    of the coordinate by which a local image is cropped. The returned image
    uses the following coordinate convention:
        `h = x`, `w = y`
    """

    def __init__(
        self,
        height: int,
        width: int,
        query_resolution: float,
        crop_expand_ratio: float = 1.5,
        pre_load_map: bool = False,
        load_map_func: Optional[Callable] = None,
        normalize: bool = False,
    ):
        """Initialize method.

        Args:
            height: height of the output image (x-axis).
            width: width of the output image (y-axis).
            query_resolution: resolution for the output map.
            crop_expand_ratio: the ratio of the expanding area
                surrounding the queried area. This expanding area
                is used to avoid loss of information when applying
                rotation.
            pre_load_map: whether to load the entire map in
                initialization.
            load_map_func: the function to load the base map.
                Default to None, and the class will use the internel
                functions.
            normalize: whether to normalize the map value from [0~255]
                to [0, 1].
        """
        assert height > 0, "Output height should be > 0."
        assert width > 0, "Output width should be > 0."
        assert query_resolution > 0, "Output resolution should be > 0."
        self.height = int(height)
        self.width = int(width)
        self.query_resolution = query_resolution
        self.crop_expand_ratio = crop_expand_ratio
        self.pre_load_map = pre_load_map
        self.normalize = normalize
        self.load_map_func = load_map_func
        if pre_load_map:
            if load_map_func is None:
                self.load_map_func = self._load_base_map
        else:
            if load_map_func is None:
                self.load_map_func = self._load_local_map

        # Expand the rectangle by sqrt(2)~=1.5x to avoid loss of information
        # near image boundaries when applying rotation.
        self.crop_area_shape = int(
            max(self.height, self.width) * self.crop_expand_ratio
        )
        self.map_pad_size = int(self.crop_area_shape / 2) + 1

        if pre_load_map:
            (
                self.x_min,
                self.y_min,
                self.x_max,
                self.y_max,
                self.base_map,
            ) = self.load_one_frame_map(
                self.load_map_func, True, self.map_pad_size
            )
        else:
            self.x_min, self.y_min = -np.inf, -np.inf
            self.x_max, self.y_max = np.inf, np.inf

    def _load_base_map(self, **kwargs):  # noqa: D205,D401
        """The map loader for the offline mode.

        It should be implemented by the child classes.
        """
        raise NotImplementedError("`_load_base_map` not implemented.")

    def _load_local_map(self, **kwargs):  # noqa: D205,D401
        """The map loader for the online mode.

        It should be implemented by the child classes.
        """
        raise NotImplementedError("`_load_local_map` not implemented.")

    @staticmethod
    def load_one_frame_map(load_map_func, pad=False, pad_size=0, **kwargs):
        """Load one frame map.

        Args:
            load_map_func (Callable): the function to load map. All the
                parameters of this function should be inputted in kwargs.
            pad (bool): whether to pad zeros around the loaded map.
            pad_size (int): the pad size.

        Return:
            `loaded_map` or a Tuple includes `loaded_map`. If the return
            value is a tuple, the loaded map is the last element.
        """
        loaded_results = load_map_func(**kwargs)
        if type(loaded_results) in [list, tuple]:
            loaded_results = list(loaded_results)
            loaded_map = loaded_results[-1]
        else:
            loaded_map = loaded_results

        if pad:
            if type(loaded_map) is lil_matrix:
                tl = br = lil_matrix((pad_size, pad_size), dtype=np.uint8)
                padded_matrix = [
                    [tl, None, None],
                    [None, loaded_map, None],
                    [None, None, br],
                ]
                loaded_map = bmat(padded_matrix, "lil").toarray()
            elif loaded_map is not None:
                loaded_map = np.pad(
                    loaded_map,
                    ((pad_size, pad_size), (pad_size, pad_size), (0, 0)),
                )

        if type(loaded_results) is list:
            loaded_results[-1] = loaded_map
        else:
            loaded_results = loaded_map
        return loaded_results

    def get(self, x: float, y: float, yaw: float = 0):
        """Get desired images from map according to the parameters input.

        Args:
            x (float): the x coords of the real world.
            y (float): the y coords of the real world.
            yaw (float, optional): the yaw angle in degrees. Defaults to 0.

        Returns:
            crop_img (np.array): image from the map.
        """
        # Check the loaded map and the query center.
        required_params = ["x_min", "y_min", "x_max", "y_max"]
        if self.pre_load_map:
            required_params += ["base_map"]
        for param in required_params:
            assert hasattr(self, param), (
                "The base map is not correctly loaded, the parameter "
                f"{param} is missed."
            )
        if (
            x < self.x_min
            or x > self.x_max
            or y < self.y_min
            or y > self.y_max
        ):
            raise ValueError(
                f"Query center ({x}, {y}) is out of bound. {self.x_min} "
                f"< x < {self.x_max}. {self.y_min} < y < {self.y_max}."
            )

        # Crop an enclosing square map and rotate it to make the x-axis
        # (horizontal right) is along the given `yaw`.
        if self.pre_load_map:
            square_img = self._crop_square_patch(
                x, y, self.x_min, self.y_min, self.map_pad_size
            )
        else:
            local_map_kwargs = {"x": x, "y": y, "yaw": yaw}
            square_img = self.load_one_frame_map(
                load_map_func=self.load_map_func, pad=False, **local_map_kwargs
            )
        rotated_img = self._rotate_image(square_img, yaw)

        # Flip x-y since the output coordinate is h = x, w = y.
        swapped_img = np.swapaxes(rotated_img, 0, 1)
        h, w = swapped_img.shape[0], swapped_img.shape[1]

        # Crop under the rotated coodinate.
        h_min = (h - self.height) // 2
        h_max = h_min + self.height
        w_min = (w - self.width) // 2
        w_max = w_min + self.width
        crop_img = swapped_img[h_min:h_max, w_min:w_max]
        crop_img = np.asarray(crop_img, dtype=np.float32)
        if self.normalize:
            crop_img /= 255.0

        # Check if there is anything in the map, and warn if nothing is there.
        if np.max(crop_img) < 1e-4:
            logger.warning(
                "Max value of the queried map is smaller than 1e-4. "
                "If this msg persist, you are possible using an empty map."
            )
        return crop_img

    def _crop_square_patch(
        self,
        x: float,
        y: float,
        x_min: float,
        y_min: float,
        pad_size: int = 0,
        cur_map: Optional[np.array] = None,
    ):
        """Crop a square image patch from the base map.

        Crop a square image path from the original base map centered at
        `(x, y)`, and has length `self.crop_area_shape`.

        Args:
            x (float): the physical x coordinates.
            y (float): the physical y coordinates.
            x_min (float): the x coords of the left side of the cur_map.
            y_min (float): the y coords of the top side of the cur_map.
            pad_size (int, optional): the padding size. Default to zero.
            cur_map (np.array, optional): the map to crop. Default to None.
                If this parameter is None, the crop function will use the
                `self.base_map`.

        Returns:
            square (np.array): the cropped squre image centered at (x, y).
        """
        if cur_map is None:
            assert hasattr(self, "base_map"), "No map to crop."
            cur_map = self.base_map
        # Generate biased center coord (considering padding).
        w = (x - x_min) / self.query_resolution + pad_size + 0.5
        h = (y - y_min) / self.query_resolution + pad_size + 0.5

        sqr_len = int(self.crop_area_shape)
        w_min = int(w - sqr_len // 2)
        h_min = int(h - sqr_len // 2)
        w_max = int(w_min + sqr_len)
        h_max = int(h_min + sqr_len)
        whole_square = cur_map[h_min - 1 : h_max, w_min - 1 : w_max]
        squ_00 = whole_square[:-1, :-1]
        squ_01 = whole_square[:-1, 1:]
        squ_10 = whole_square[1:, :-1]
        squ_11 = whole_square[1:, 1:]
        weight_00 = (1 + int(w) - w) * (1 + int(h) - h)
        weight_10 = (1 + int(w) - w) * (h - int(h))
        weight_01 = (w - int(w)) * (1 + int(h) - h)
        weight_11 = (w - int(w)) * (h - int(h))
        square = (
            weight_00 * squ_00
            + weight_01 * squ_01
            + weight_10 * squ_10
            + weight_11 * squ_11
        )
        return square

    @staticmethod
    def _rotate_image(image, angle):
        """Rotate the input image.

        Note: make sure the input image is padded to sqrt(2) of its original
        size to avoid unintended cropping near edges.

        Args:
            image (np.array, [H, W, C]): input image.
            angle (float): the rotate angle in degrees.

        Returns:
            new_image (np.array, [H, W, C]): the rotated image.
        """
        h, w = image.shape[:2]
        # Rotate on image center.
        center = (w / 2, h / 2)
        rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
        new_image = cv2.warpAffine(image, rot_mat, (w, h))
        return new_image


@OBJECT_REGISTRY.register
class NuScenesRenderedMap(BaseUnstructuredMapServer):
    """The map server for NuScenes rendered map.

    We rendered all the NuScenes maps into small patches by NuScenes-devkit, \
    because the online calling of NuScenes-devkit is time-comsuming. This \
    class support two modes (online and offline) to get rendered map: \
    1) Offline mode (pre_load_map = True) supports the connection of map \
        patches based on the top-left and bottom-right coordinates recorded \
        in the file names. The name of map patches should be: \
        "{resolution}_map_{left_coor}_{top_coor}_{right_coor}_{bottom_coor}.png"
    2) Online mode (pre_load_map = False) loads local map of a small area \
        when the `get` function is called, it takes more time but reduces the \
        memory requirements.
    """

    def __init__(
        self,
        height: int,
        width: int,
        query_resolution: float,
        map_path: str,
        base_resolution: float,
        crop_expand_ratio: float = 1.5,
        pre_load_map: bool = False,
        load_map_func: Optional[Callable] = None,
        normalize: bool = False,
        basemap_suffix: str = ".png",
        basemap_right_hand_sys: bool = True,
    ):
        """Initialize method.

        Only customized parameters are shown here.

        Args:
            map_path (str): the path of the map.
            base_resolution (float): the resolution of the loaded orignal
                map patches.
            basemap_suffix (str): the suffix of the map files.
            basemap_right_hand_sys (bool): whether the base maps use the
                right-hand coordinate systems. Defaults to True. Since the
                base class defines the inputted map should in left-hand
                coordinate systems. If this parameter is True, the function
                will perform the chiral conversion.
        """
        assert os.path.exists(
            map_path
        ), f"The map path {map_path} does not exist."
        self.map_path = map_path
        self.base_resolution = base_resolution
        self.basemap_suffix = basemap_suffix
        self.basemap_right_hand_sys = basemap_right_hand_sys
        self.map_prefix = str(self.base_resolution) + "_map_"

        all_kwargs = {
            "height": height,
            "width": width,
            "query_resolution": query_resolution,
            "crop_expand_ratio": crop_expand_ratio,
            "pre_load_map": pre_load_map,
            "load_map_func": load_map_func,
            "normalize": normalize,
        }
        super(NuScenesRenderedMap, self).__init__(**all_kwargs)

        if not self.pre_load_map:
            self._get_basemap_info_for_online_getter()

    def _load_base_map(self, **kwargs):  # noqa: D205,D401
        """Load rendered maps for NuScenes dataset.

        This function will only be used when `self.pre_load_map = True`.

        Returns:
            min_x (int): min value of x coordinates.
            max_x (int): max value of x coordinates.
            min_y (int): min value of y coordinates.
            max_y (int): max value of y coordinates.
            city_map (np.array): the loaded base map.
        """
        logger.info("loading rendered map image for NuScenes dataset ...")
        img_files = glob(
            os.path.join(self.map_path, "*" + self.basemap_suffix)
        )
        images = []
        img_shape = None
        for idx, img_file in tqdm(enumerate(img_files), unit="files"):
            img = np.array(Image.open(img_file))
            file_name = img_file.split("/")[-1].split(self.basemap_suffix)[0]
            resolution, _, l, t, r, b = file_name.split("_")
            l, t, r, b = int(l), int(t), int(r), int(b)

            if float(resolution) != self.base_resolution:
                logger.info(f"File {file_name} has wrong resolution.")
                continue
            if img_shape is not None and np.any(img_shape != img.shape):
                logger.info(f"File {file_name} has number of channels.")
                continue

            images.append([img, l, t, r, b])
            if not idx:
                min_x, min_y, max_x, max_y = l, t, r, b
                img_shape = img.shape
            else:
                min_x = min(min_x, l)
                max_x = max(max_x, r)
                min_y = min(min_y, t)
                max_y = max(max_y, b)

        channels = img_shape[-1]
        base_h = int((max_x - min_x) / self.base_resolution)
        base_w = int((max_y - min_y) / self.base_resolution)
        city_map = np.zeros([base_h, base_w, channels], dtype=np.float32)
        for img, l, t, r, b in images:
            h_start = int((l - min_x) / self.base_resolution)
            h_end = int((r - min_x) / self.base_resolution)
            w_start = int((t - min_y) / self.base_resolution)
            w_end = int((b - min_y) / self.base_resolution)
            city_map[h_start:h_end, w_start:w_end, :] = img

        if np.max(city_map) <= 1:
            logger.warning("Converting value range from [0, 1] to [0, 255].")
            city_map = np.array(city_map, dtype=np.float32)
            city_map /= np.max(city_map) + 1e-10
            city_map *= 255.0

        # Resize and save the base map to query resolution.
        dst_h = int(base_h * self.base_resolution / self.query_resolution)
        dst_w = int(base_w * self.base_resolution / self.query_resolution)
        city_map = cv2.resize(np.float32(city_map), (dst_w, dst_h))
        city_map = city_map.astype("uint8")

        # Convert the base map to the left-hand system.
        if self.basemap_right_hand_sys:
            city_map = city_map.transpose((1, 0, 2))

        return min_x, min_y, max_x, max_y, city_map

    def _get_basemap_info_for_online_getter(self):  # noqa: D205,D400,D401
        """Traverse the base map files and get the necessary map
        information.

        This function will only be used when `self.pre_load_map = False`,
        i.e., the `get` operation will be an online process.
        """
        # 1. Parameters of the base map. (in the map coordinates). The
        # coordinate systems has not been converted to meet the chirality
        # requirement of loaded base map). Thus h=x, w=y.
        img_files = glob(
            os.path.join(self.map_path, "*" + self.basemap_suffix)
        )
        img_shape = None
        for idx, img_file in enumerate(img_files):
            file_name = img_file.split("/")[-1].split(self.basemap_suffix)[0]
            resolution, _, l, t, r, b = file_name.split("_")
            l, t, r, b = int(l), int(t), int(r), int(b)

            if float(resolution) != self.base_resolution:
                raise ValueError(f"File {file_name} has wrong resolution.")

            if not idx:
                img = np.array(Image.open(img_file))
                img_shape = img.shape
                min_x, min_y, max_x, max_y = l, t, r, b
            else:
                min_x = min(min_x, l)
                max_x = max(max_x, r)
                min_y = min(min_y, t)
                max_y = max(max_y, b)

        self.x_min, self.x_max = min_x, max_x
        self.y_min, self.y_max = min_y, max_y
        self.base_h, self.base_w, self.num_channels = img_shape
        self.base_x = self.base_h * self.base_resolution
        self.base_y = self.base_w * self.base_resolution
        self.query_x = self.height * self.query_resolution
        self.query_y = self.width * self.query_resolution

        # 2. Parameters for loading the local base map.
        # How to calculate the number in x-axis and y-axis of the loaded map
        # patches surrounding A (query_x, query_y)? The following picture
        # illustrates the method.
        # Suppose A is located at the left boundary of one map patch M. The
        # shape of each map patch is [h, w], the shape of the queried map
        # is [h_q, w_q]. If the query_yaw = pi/4, the queried area occupies
        # the largest space on the x-axis. The number n of map patches on the
        # left of M requires:
        #    n * w >= sqrt(2) / 2 * w_q
        #    i.e., n = np.ceil(sqrt(2) / 2 * w_q / w)
        # The case that A is located at the right boundary is similar, thus
        # the num in w-axis = 2 * n + 1. The num in h-axis is simliar.
        #
        #               ___* ____ ____
        #              |    | M  |    |
        #              |___A|____|*___|
        #            *
        #
        #                   *
        self.num_map_x = int(
            2 * np.ceil(self.query_x * np.sqrt(2) / 2 / self.base_x) + 1
        )
        self.num_map_y = int(
            2 * np.ceil(self.query_y * np.sqrt(2) / 2 / self.base_y) + 1
        )

    def _load_local_map(
        self, x: float, y: float, yaw: float = 0
    ):  # noqa: D205,D401
        """Load rendered maps for NuScenes dataset.

        This function will only be used when `self.pre_load_map = False`.

        Args:
            x (float): the x coords of the real world.
            y (float): the y coords of the real world.
            yaw (float, optional): the yaw angle in degrees. Defaults to 0.

        Returns:
            cur_map (np.array): the loaded base map.
        """
        left_grid = (
            self.x_min + np.floor((x - self.x_min) / self.base_x) * self.base_x
        )
        top_grid = (
            self.y_min + np.floor((y - self.y_min) / self.base_y) * self.base_y
        )
        x_grid_l = int(-(self.num_map_x - 1) / 2)
        x_grid_r = self.num_map_x + x_grid_l + 1
        y_grid_l = int(-(self.num_map_y - 1) / 2)
        y_grid_r = self.num_map_y + y_grid_l + 1
        x_grids = np.arange(x_grid_l, x_grid_r, 1) * self.base_x + left_grid
        y_grids = np.arange(y_grid_l, y_grid_r, 1) * self.base_y + top_grid
        local_x_min = x_grids[0]
        local_y_min = y_grids[0]
        cur_map = np.zeros(
            [
                self.num_map_x * self.base_h,
                self.num_map_y * self.base_w,
                self.num_channels,
            ],
            dtype=np.float32,
        )
        for l, r in zip(x_grids[:-1], x_grids[1:]):  # noqa: E741
            for t, b in zip(y_grids[:-1], y_grids[1:]):
                l, t, r, b = int(l), int(t), int(r), int(b)
                file_name = (
                    self.map_prefix
                    + f"{str(l)}_{str(t)}_{str(r)}_{str(b)}"
                    + self.basemap_suffix
                )
                img_file = os.path.join(self.map_path, file_name)
                if not os.path.exists(img_file):
                    continue
                img = np.array(cv2.imread(img_file))[:, :, ::-1]
                h_start = int((l - local_x_min) / self.base_resolution)
                h_end = int((r - local_x_min) / self.base_resolution)
                w_start = int((t - local_y_min) / self.base_resolution)
                w_end = int((b - local_y_min) / self.base_resolution)
                cur_map[h_start:h_end, w_start:w_end, :] = img

        if np.max(cur_map) <= 1:
            logger.warning("Converting value range from [0, 1] to [0, 255].")
            cur_map = np.array(cur_map, dtype=np.float32)
            cur_map /= np.max(cur_map) + 1e-10
            cur_map *= 255.0

        # Resize and save the base map to query resolution.
        dst_h = int(
            self.num_map_x
            * self.base_h
            * self.base_resolution
            / self.query_resolution
        )
        dst_w = int(
            self.num_map_y
            * self.base_w
            * self.base_resolution
            / self.query_resolution
        )
        cur_map = cv2.resize(np.float32(cur_map), (dst_w, dst_h))

        # Convert the base map to the left-hand system.
        if self.basemap_right_hand_sys:
            cur_map = cur_map.transpose((1, 0, 2))

        cur_map = self._crop_square_patch(
            x, y, x_grids[0], y_grids[0], 0, cur_map
        )

        return cur_map


@OBJECT_REGISTRY.register
class NuScenesMapServer:
    def __init__(
        self,
        height: int,
        width: int,
        query_resolution: float,
        map_path: str,
        base_resolution: float,
        crop_expand_ratio: float = 1.5,
        pre_load_map: bool = False,
        load_map_func: Optional[Callable] = None,
        normalize: bool = False,
        basemap_suffix: str = ".png",
        basemap_right_hand_sys: bool = True,
        item_key: str = "road_map",
    ):
        """Initialize method.

        Args:
            item_key: the new key that is added to the batch.
            The definition of other parameters refer to the class
            `NuScenesRenderedMap`.
        """
        super(NuScenesMapServer, self).__init__()
        self.item_key = item_key
        assert os.path.exists(map_path)
        folders = os.listdir(map_path)
        self.maps = {}
        for fold in folders:
            tmp_map_path = os.path.join(map_path, fold)
            self.maps[str(fold)] = NuScenesRenderedMap(
                height=height,
                width=width,
                query_resolution=query_resolution,
                map_path=tmp_map_path,
                base_resolution=base_resolution,
                crop_expand_ratio=crop_expand_ratio,
                pre_load_map=pre_load_map,
                load_map_func=load_map_func,
                normalize=normalize,
                basemap_suffix=basemap_suffix,
                basemap_right_hand_sys=basemap_right_hand_sys,
            )

    def __call__(self, sample: dict):
        """Callable function.

        The following keys are required:
            1. `seq_center`

        Args:
            sample (dict): input original sample.

        Returns:
            sample (dict): the sample with added `item_key`.
        """
        seq_center = sample["seq_center"]
        seq_index = sample["seq_index"]
        map_name = str(seq_index.traj_group_index.map_id)
        sample[self.item_key] = self.maps[map_name].get(
            seq_center.pos_x, seq_center.pos_y, seq_center.yaw
        )[:, :, 0]
        return sample


@OBJECT_REGISTRY.register
class VectorNetStructuredMapServer:
    """The structured map server for VectorNet.

    If the user wants to use this transform, the dataset must be
    `AutoMultiAgentNaviDataset` and enable the loading of structual road
    information. \

    This transform method requires the sample to have the key
    "valid_img_coords", which means that is must be used after the transform
    "GetTrajPredObjectsInfo". \
    """

    # Supported keys and their order of importance, the preceding elements
    # have higher importance. e.g., "stopline":1, "crosswalk":2,
    # "solid_lane":3, "roadedge":4, "virtuallanelines":5.
    SUPPORTED_KEYS_WITH_ORDERS = [
        "stopline",
        "crosswalk",
        "solid_lane",
        "roadedge",
        "virtuallanelines",
    ]
    SUPPORTED_SAMPLE_MODE = ["uniform"]
    SUPPORTED_OP_FEATS = {
        "turn_dir": 1,
        "pid": 1,
        "pid_pred_succ": 2,
        "pre_pre_point": 2,
        "element_type": 5,  # len of SUPPORTED_KEYS_WITH_ORDERS
    }
    COORDINATE_OP_FEATS = ["pre_pre_point"]
    SELECT_ELE_MODES = [
        "distance_first",
        "ele_type_first",
    ]

    def __init__(
        self,
        map_origin_params: List,
        element_keys: List,
        sample_mode: Dict,
        curve_threshold: float = 1.08,
        polyline_seg_len: int = 10,
        polyline_optional_feats: Optional[List] = None,
        local_ele_seg_thr: int = 100,
        max_num_ele_seg: int = 256,
        image_coordinates: str = "bev",
        shuffle: bool = True,
        scale: Optional[List] = None,
        select_ele_mode: str = "distance_first",
        valid_img_coords_key: str = "valid_img_coords",
    ):
        """Initialize method.

        Args:
            map_origin_params: paramters to calculate the offset of the
                map center in the map coordinates.
            element_keys: elements to consider during extracting map
                features. It should be a subset of `SUPPORTED_KEYS`.
            sample_mode: the sample mode for each element type.
            curve_threshold: the threshold of straight trajectories.
                Defaults to 0.5.
            polyline_seg_len: the length of the polyline segments.
            polyline_optional_feats: list of optional polyline features to
                extract. It should be a subset of the key list of
                `SUPPORTED_OP_FEATS`. Defaults to None.
            local_ele_seg_thr: the distance threshold of valid polyline
                segments (from the obstacle). Defaults to 100 [m].
            max_num_ele_seg: the maximum polyline segments after extracting.
                Defaults to 256.
            image_coordinates: the name of the image coordinates. Defaults
                to "bev". It should be one of ["bev", "img]. The corresponding
                coordinate trans method is ["PhyToBEV", "PhyToImg"].
            shuffle: whether to shuffle road element segments. Defaults to
                True.
            scale: the scale of the extracted features. Default to None,
                which means all the scales are 1.
            select_ele_mode: the mode for selecting road elements. Default to
                "distance_first".
            valid_img_coords_key: the key name of the img coords of valid
                track ids.

        """
        assert (
            len(map_origin_params) == 3
        ), "The length of map origin parameters should be 3."
        assert len(
            element_keys
        ), "The element list for feature extracting is empty."
        for key in element_keys:
            assert (
                key in self.SUPPORTED_KEYS_WITH_ORDERS
            ), f"Unsupported element key {key}."
        if polyline_optional_feats is None:
            polyline_optional_feats = []
        for feat in polyline_optional_feats:
            assert (
                feat in self.SUPPORTED_OP_FEATS.keys()
            ), f"Unsupported extracted polyline feature name {feat}."
        assert (
            select_ele_mode in self.SELECT_ELE_MODES
        ), f"Unsupported select_ele_mode {select_ele_mode}."
        self.map_origin_params = map_origin_params
        self.element_keys = element_keys
        self.sample_mode_of_element = sample_mode
        self.curve_threshold = curve_threshold
        self.polyline_seg_len = polyline_seg_len
        self.polyline_optional_feats = polyline_optional_feats
        self.local_ele_seg_thr = local_ele_seg_thr
        self.max_num_ele_seg = max_num_ele_seg
        self.image_coordinates = image_coordinates
        self.shuffle = shuffle
        self.select_ele_mode = select_ele_mode
        self.valid_img_coords_key = valid_img_coords_key

        if self.image_coordinates == "bev":
            self.reverse = True
            bev_ori_x, bev_ori_y, img_resolu = self.map_origin_params
            self.img_center_offset = [bev_ori_x, bev_ori_y]
            self.img_resolu = img_resolu
        elif self.image_coordinates == "img":
            self.reverse = False
            map_h, map_w, img_resolu = self.map_origin_params
            self.img_center_offset = [map_h / 2, map_w / 2]
            self.img_resolu = img_resolu
        else:
            raise ValueError(
                f"Undefined image coordinates {image_coordinates}"
            )

        # The basic feats is ["start_x", "start_y", "end_x", "end_y"]
        self.coor_feat_col_idx = [[0, 1], [2, 3]]
        col_idx = 4
        for key in polyline_optional_feats:
            if key in self.COORDINATE_OP_FEATS:
                self.coor_feat_col_idx.append([col_idx, col_idx + 1])
            col_idx += self.SUPPORTED_OP_FEATS[key]
        self.road_feat_dim = col_idx
        if scale is None:
            self.road_feat_scale = [1 for _ in range(self.road_feat_dim)]
        else:
            self.road_feat_scale = scale

    def __call__(self, sample: Dict):
        """Callable function.

        The following keys are required in `sample`:
            1. "struct_road"
            2. "valid_img_coords"

        Args:
            sample (Dict): input original sample.

        Returns:
            sample (Dict): the sample includes new keys "struct_road_feats",
                "struct_road_masks", "struct_num_road_elements":
                - The shape of "struct_road_feats" is:
                    [num_obs, max_num_ele_seg, polyline_seg_len-1, num_feats].
                - The shape of "struct_road_masks" is:
                    [num_obs, max_num_ele_seg]
                - The shape of "struct_num_road_elements" is: [num_obs]
        """
        valid_img_coords = sample[self.valid_img_coords_key]
        all_feats = self._extract_ori_feats(sample)
        sample["vcs_vector_map_feats"] = all_feats

        # Sample structural element features for different obstacles.
        idx1, idx2 = self.polyline_seg_len - 2, self.polyline_seg_len - 1
        ele_seg_pts = {}
        for key in self.element_keys:
            key_feats = all_feats[key]
            if len(key_feats):
                ele_seg_pts[key] = np.concatenate(
                    (key_feats[:, 0:1, 0:2], key_feats[:, idx1:idx2, 2:4]),
                    axis=1,
                )
            else:
                ele_seg_pts[key] = []
        all_obs_ele_seg = []
        all_num_valid_polyline = []
        all_valid_polyline_mask = []
        offset_x, offset_y = self.img_center_offset
        img_resolu = self.img_resolu
        for (img_x, img_y, img_yaw) in valid_img_coords:
            if self.reverse:
                vcs_x = -(img_x - offset_x / img_resolu) * img_resolu
                vcs_y = -(img_y - offset_y / img_resolu) * img_resolu
                vcs_yaw = img_yaw - np.pi
            else:
                vcs_x = (img_x - offset_x / img_resolu) * img_resolu
                vcs_y = (img_y - offset_y / img_resolu) * img_resolu
                vcs_yaw = img_yaw

            # Extract the element segments in a circle (the radius is equal to
            # `local_ele_seg_thr` [m]) centered at the obstacle coordinates.
            obs_centered_type_ele = {}
            valid_centered_ele_num = 0
            valid_seg_diff = {}
            for key in self.element_keys:
                if len(all_feats[key]) == 0:
                    obs_centered_type_ele[key] = []
                    valid_centered_ele_num += 0
                    valid_seg_diff[key] = []
                    continue
                obs_centered_ele_seg_pts = (
                    ele_seg_pts[key] - np.array([vcs_x, vcs_y])[None, None, :]
                )
                ele_seg_diff = np.sqrt(
                    np.sum(obs_centered_ele_seg_pts ** 2, axis=-1)
                )
                ele_seg_diff = np.min(ele_seg_diff, axis=1)
                ele_valid = ele_seg_diff < self.local_ele_seg_thr
                valid_seg_diff[key] = ele_seg_diff[np.where(ele_valid)[0]]
                obs_centered_type_ele[key] = all_feats[key][
                    np.where(ele_valid)[0], :, :
                ]
                valid_centered_ele_num += len(obs_centered_type_ele[key])

            # Perform padding or sampling.
            if valid_centered_ele_num == 0:
                obs_centered_valid_ele = np.zeros(
                    [
                        self.max_num_ele_seg,
                        self.polyline_seg_len - 1,
                        self.road_feat_dim,
                    ]
                )
                all_num_valid_polyline.append(valid_centered_ele_num)
                all_valid_polyline_mask.append(
                    np.zeros([self.max_num_ele_seg])
                )
            elif valid_centered_ele_num < self.max_num_ele_seg:
                num_pad = self.max_num_ele_seg - valid_centered_ele_num
                obs_centered_valid_ele = []
                for key in self.element_keys:
                    if len(obs_centered_type_ele[key]):
                        obs_centered_valid_ele.append(
                            obs_centered_type_ele[key]
                        )
                obs_centered_valid_ele = np.concatenate(
                    obs_centered_valid_ele, axis=0
                )
                padding_ele = np.zeros(
                    [num_pad] + list(obs_centered_valid_ele.shape[1:])
                )
                obs_centered_valid_ele = np.concatenate(
                    (obs_centered_valid_ele, padding_ele),
                    axis=0,
                )
                valid_mask = np.concatenate(
                    [np.ones([valid_centered_ele_num]), np.zeros([num_pad])],
                    axis=0,
                )
                all_num_valid_polyline.append(valid_centered_ele_num)
                all_valid_polyline_mask.append(valid_mask)
            else:
                # sampling
                obs_centered_valid_ele = self._sample_road_elements(
                    valid_seg_diff,
                    obs_centered_type_ele,
                    self.select_ele_mode,
                )
                all_num_valid_polyline.append(self.max_num_ele_seg)
                all_valid_polyline_mask.append(np.ones([self.max_num_ele_seg]))

            # Perform coordinate trans.
            num_seg, num_seg_pts, num_feats = obs_centered_valid_ele.shape
            obs_feats = obs_centered_valid_ele.reshape([-1, num_feats])
            for cols in self.coor_feat_col_idx:
                tmp_obs_feats = obs_feats[:, cols]
                trans_x, trans_y = Affine2D.coord_translate(
                    tmp_obs_feats[:, 0], tmp_obs_feats[:, 1], vcs_x, vcs_y
                )
                trans_x, trans_y = Affine2D.coord_rotate(
                    trans_x, trans_y, vcs_yaw
                )
                obs_feats[:, cols[0]] = trans_x
                obs_feats[:, cols[1]] = trans_y
            obs_centered_valid_ele = obs_feats.reshape(
                [num_seg, num_seg_pts, num_feats]
            )

            # Perform shuffle.
            if self.shuffle:
                sf_idx = [i for i in range(self.max_num_ele_seg)]  # noqa: C416
                random.shuffle(sf_idx)
                obs_centered_valid_ele = obs_centered_valid_ele[sf_idx, :, :]
                all_valid_polyline_mask[-1] = all_valid_polyline_mask[-1][
                    sf_idx
                ]

            all_obs_ele_seg.append(obs_centered_valid_ele)

        all_obs_ele_seg = np.stack(all_obs_ele_seg)
        all_valid_polyline_mask = np.stack(all_valid_polyline_mask)
        sample["struct_road_feats"] = all_obs_ele_seg
        sample["struct_road_masks"] = all_valid_polyline_mask
        sample["struct_num_road_elements"] = all_num_valid_polyline
        sample["road_feat_scale"] = self.road_feat_scale
        return sample

    def _sample_road_elements(
        self,
        valid_seg_diff,
        obs_centered_type_ele,
        select_mode,
    ):
        """Extract the index for each element type.

        Args:
            valid_seg_diff (dict): valid seg diff for each element type.
            obs_centered_type_ele (dict): valid feature for each element type.
            select_mode (list): the way to select element.

        Returns:
            index_per_ele (dict): the index for each element type.
        """
        # select elements by distance first.
        if select_mode == "distance_first":
            seg_diff = []
            obs_centered_valid_ele = []
            for key in self.element_keys:
                if len(obs_centered_type_ele[key]) > 0:
                    seg_diff += list(valid_seg_diff[key])
                    obs_centered_valid_ele.append(obs_centered_type_ele[key])
            obs_centered_valid_ele = np.concatenate(
                obs_centered_valid_ele, axis=0
            )
            seg_valid_idx = np.argsort(seg_diff)[: self.max_num_ele_seg]
            obs_centered_valid_ele = obs_centered_valid_ele[seg_valid_idx]

        # select elements by the order in self.SUPPORTED_KEYS_WITH_ORDERS.
        elif select_mode == "ele_type_first":
            acc_ele_num = 0
            obs_centered_valid_ele = []
            for key in self.SUPPORTED_KEYS_WITH_ORDERS:
                if key not in self.element_keys:
                    continue
                eles = obs_centered_type_ele[key]
                seg_diff = valid_seg_diff[key]
                if len(eles) == 0:
                    continue
                if acc_ele_num > self.max_num_ele_seg:
                    raise ValueError(f"Too many road elements: {acc_ele_num}")
                elif acc_ele_num == self.max_num_ele_seg:
                    break
                elif acc_ele_num < self.max_num_ele_seg - len(eles):
                    obs_centered_valid_ele.append(eles)
                    acc_ele_num += len(eles)
                else:
                    num_for_ele_key = self.max_num_ele_seg - acc_ele_num
                    index_per_ele = np.argsort(seg_diff)[:num_for_ele_key]
                    obs_centered_valid_ele.append(eles[index_per_ele])
                    acc_ele_num += num_for_ele_key
            obs_centered_valid_ele = np.concatenate(
                obs_centered_valid_ele, axis=0
            )
        else:
            raise ValueError(f"Undefined selecting mode {select_mode}")
        return obs_centered_valid_ele

    def _extract_ori_feats(self, sample: Dict):
        """Extract the original struct road featues.

        Args:
            sample (Dict): the input original sample.

        Returns:
            all_feats {ele_cls: [num_seg, polyline_seg_len-1, num_feats]}:
                the polylines segment features.
        """
        all_feats = {}
        for key in self.element_keys:
            all_feats[key] = self._extract_ori_feats_by_key(sample, key)
        return all_feats

    def _extract_ori_feats_by_key(self, sample: Dict, key: str):
        """Extract the original struct road featues.

        Args:
            sample (Dict): the input original sample.

        Returns:
            feats ([num_seg, polyline_seg_len-1, num_feats]): the
                polylines segment features.
        """
        if "struct_road" not in sample or key not in sample["struct_road"]:
            return []
        struct_road = sample["struct_road"]
        # Extract structural element features.
        pid_offset = 0
        no_turn_dirs = True if (key == "crosswalk") else False
        try:
            cur_feat, pid_offset = self.extract_line_seg_feats(
                struct_road,
                key,
                pid_offset,
                self.curve_threshold,
                self.polyline_seg_len,
                self.polyline_optional_feats,
                no_turn_dirs,
            )
        except ValueError:
            return []

        if len(cur_feat):
            return np.array(cur_feat)
        else:
            return []

    def sample_polyline_seg(
        self,
        polyline: List,
        step_s: float,
        polyline_seg_len: int,
        sample_mode: str,
        drop_ratio: float = 0.3,
    ):
        """Sample or interpolate the road polyline.

        Args:
            polyline (np.numpy, [n_poly, 3, 2]): the original polyline
                or polygon.
            step_s (float): the sample step.
            polyline_seg_len (int): the fixed number of segments of each
                sampled polyline.
            sample_mode (str): the sample mode of polyline.
            drop_ratio (float): work when sample_mode="original", means
                when the ratio of the len of original polyline to to the
                "polyline_seg_len" is less than "drop_ratio", the
                polyline will be dropped.

        Returns:
            polyline_seg: (np.numpy, [n_polyline_seg, seg_len, 2]): the
                sampled polyline with fixed length.
        """
        if sample_mode == "original":
            polyline = np.concatenate(
                (polyline[0:1, 0:2, 0], polyline[:, 0:2, 1]), axis=0
            )
            polyline = polyline[:: int(step_s)]
            n_seg, mod_seg = (
                len(polyline) // polyline_seg_len,
                len(polyline) % polyline_seg_len,
            )
            if mod_seg < polyline_seg_len * drop_ratio:  # dropping
                polyline = polyline[0 : n_seg * polyline_seg_len]
            else:  # padding, mode = "constant" or "edge"
                polyline = np.pad(
                    polyline,
                    ((0, polyline_seg_len - mod_seg), (0, 0)),
                    "edge",
                )
            polyline_segs = polyline.reshape(-1, polyline_seg_len, 2)

        elif sample_mode == "uniform":
            polyline_diff = polyline[:, 0:2, 1] - polyline[:, 0:2, 0]
            total_length = np.sum(np.sqrt(np.sum(polyline_diff ** 2, axis=1)))
            num_itp_seg = (
                np.ceil((total_length / step_s + 1) / polyline_seg_len)
                * polyline_seg_len
            )
            real_step_s = total_length / num_itp_seg

            # Iterpolate the polyline -> [num_seg, polyline_seg_len, 2]
            itp_s_array = np.arange(0, num_itp_seg, 1) * real_step_s
            itp_polyline, _ = interpolate_trajectory_from_path(
                np.concatenate(
                    (polyline[0:1, 0:2, 0], polyline[:, 0:2, 1]), axis=0
                ),
                itp_s_array,
            )
            polyline_segs = itp_polyline.reshape(-1, polyline_seg_len, 2)
        else:
            raise ValueError(f"Undefined sampling mode {sample_mode}")
        return polyline_segs

    # @staticmethod
    def extract_line_seg_feats(
        self,
        struct_road: Dict,
        element_key: str,
        pid_offset: int = 0,
        curve_threshold: float = 1.08,
        polyline_seg_len: int = 10,
        polyline_optional_feats: Optional[List] = None,
        no_turn_dirs: bool = False,
    ):
        """Split, sample and extract polyline segment features.

        Args:
            struct_road (Dict): the structural polyline information.
            element_key (str): elements to consider during extracting
                map features.
            pid_offset (int, optional): the polyline segment offset. Defaults
                to 0.
            curve_threshold (float, optional): the threshold of straight
                trajectories. Defaults to 0.5.
            polyline_optional_feats (Optional[List], optional): List of
                optional polyline features to extract. It should be a subset
                of the key list of `SUPPORTED_OP_FEATS`.
            no_turn_dirs (bool, optional): whether to skip the calculation
                of turn direction. If the elememt shape is polygon, this
                parameter should be True.

        Returns:
            all_lane_segs (np.numpy, [num_seg, polyline_seg_len-1, num_feats]):
                the polylines segment features.
            pid_offset (int): the updated polyline segment offset.
        """
        assert polyline_optional_feats is not None
        # Original element polylines:
        # list of [num_pts, 3(x,y,z), 2(start,end)].
        elements = struct_road[element_key]
        all_lane_segs = []
        for polyline in elements:
            polyline = np.array(polyline)  # [num_pts, 3(x,y,z), 2(start,end)]
            is_curve = DetectCurve.detect_curve(
                polyline[:, 0, 0], polyline[:, 1, 0], curve_threshold
            )
            # Sample elements.
            sample_mode = self.sample_mode_of_element[element_key]
            if sample_mode == "original":
                step_s = 2
            elif sample_mode == "uniform":
                if is_curve:
                    step_s = 1
                else:
                    step_s = 2
            else:
                raise ValueError(f"Undefined road element type {element_key}")
            polyline_segs = self.sample_polyline_seg(
                polyline,
                step_s,
                polyline_seg_len,
                sample_mode,
            )
            # if not extract complete polyline segments
            if len(polyline_segs) == 0:
                continue

            # Extract features.
            num_seg = len(polyline_segs)
            len_seg = polyline_seg_len - 1
            # 1. Basic lane features: [x_start, y_start, x_end, y_end]
            lane_features = np.concatenate(
                (polyline_segs[:, :-1, :], polyline_segs[:, 1:, :]), axis=-1
            )
            all_seg_feats = [lane_features]
            # pid
            pid = np.arange(0, num_seg, 1) + pid_offset
            pid_offset += num_seg

            # 2. Optional features
            for feat_name in polyline_optional_feats:
                if feat_name == "turn_dir":
                    # int: -1 left, 0 straight, 1 right
                    turn_directions = np.zeros([num_seg])
                    if not no_turn_dirs:
                        for idx, seg in enumerate(polyline_segs):
                            if not DetectCurve.detect_curve(
                                seg[:, 0], seg[:, 1], curve_threshold
                            ):
                                continue
                            yaw = np.arctan2(seg[:, 1], seg[:, 0])
                            delta_yaw = normalize_yaw(yaw[-1] - yaw[0])
                            turn_directions[idx] = 1 if (delta_yaw > 0) else -1
                    turn_directions = turn_directions[:, None, None]
                    turn_directions = turn_directions.repeat(len_seg, axis=1)
                    all_seg_feats.append(turn_directions)
                elif feat_name == "pid":
                    pid_feat = pid[:, None, None].repeat(len_seg, axis=1)
                    all_seg_feats.append(pid_feat)
                elif feat_name == "pid_pred_succ":
                    # the pred and succ pid, -1 means no pred or succ segment.
                    pid_pred = pid - 1
                    pid_pred[np.where(pid_pred < pid_offset - num_seg)[0]] = -1
                    pid_succ = pid + 1
                    pid_succ[np.where(pid_succ >= pid_offset)[0]] = -1
                    pid_pred_succ = np.stack([pid_pred, pid_succ], axis=-1)[
                        :, None, :
                    ]
                    pid_pred_succ = pid_pred_succ.repeat(len_seg, axis=1)
                    all_seg_feats.append(pid_pred_succ)
                elif feat_name == "pre_pre_point":
                    pre_pre_point = np.concatenate(
                        (polyline_segs[:, 0:1, :], polyline_segs[:, :-2, :]),
                        axis=1,
                    )
                    all_seg_feats.append(pre_pre_point)
                elif feat_name == "element_type":
                    num_ele_types = len(self.SUPPORTED_KEYS_WITH_ORDERS)
                    type_onehot = np.zeros((num_seg, len_seg, num_ele_types))
                    ele_type_idx = self.SUPPORTED_KEYS_WITH_ORDERS.index(
                        element_key
                    )
                    type_onehot[:, :, ele_type_idx] = 1
                    all_seg_feats.append(type_onehot)
            basic_feats = np.concatenate(all_seg_feats, axis=-1)
            all_lane_segs.append(basic_feats)
        if len(all_lane_segs):
            all_lane_segs = np.concatenate(all_lane_segs, axis=0)
        return all_lane_segs, pid_offset


@OBJECT_REGISTRY.register
class ArgoverseStructredMapServer(VectorNetStructuredMapServer):
    """The structured map server for VectorNet (use Argoverse dataset).

    If the user wants to use this transform, the dataset must be
    `ArgoverseTdtDataset`.

    This transform method requires the sample to have the key
    "valid_img_coords", which means that is must be used after the transform
    "GetTrajPredObjectsInfo". \
    """

    def __init__(
        self,
        map_origin_params: List,
        element_keys: List,
        sample_mode: Dict,
        curve_threshold: float = 1.08,
        polyline_seg_len: int = 10,
        polyline_optional_feats: Optional[List] = None,
        local_ele_seg_thr: int = 100,
        max_num_ele_seg: int = 256,
        image_coordinates: str = "bev",
        shuffle: bool = True,
        scale: Optional[List] = None,
    ):
        """Initialize method.

        Args:
            the parameter definitions are same as the base class.
        """
        super(ArgoverseStructredMapServer, self).__init__(
            map_origin_params=map_origin_params,
            element_keys=element_keys,
            sample_mode=sample_mode,
            curve_threshold=curve_threshold,
            polyline_seg_len=polyline_seg_len,
            polyline_optional_feats=polyline_optional_feats,
            local_ele_seg_thr=local_ele_seg_thr,
            max_num_ele_seg=max_num_ele_seg,
            image_coordinates=image_coordinates,
            shuffle=shuffle,
            scale=scale,
        )

    def _extract_ori_feats(self, sample: Dict):
        """Extract the original struct road featues.

        Returns:
            used_feats ([num_seg, polyline_seg_len-1, num_feats]): the
                polylines segment features.
        """
        vector_map_feats = sample["vector_map_feats"]
        feat_cols = sample["feat_cols"]
        center_x, center_y, center_yaw = sample["seq_center"]

        poly_len = vector_map_feats.shape[1]
        assert poly_len == self.polyline_seg_len - 1, (
            f"The polylines is error: require {self.polyline_seg_len}-1, "
            f"but {poly_len} is given."
        )
        used_feats = []
        for feat_name in ["basic"] + self.polyline_optional_feats:
            assert (
                feat_name in feat_cols
            ), f"Undefined feature name {feat_name}."
            start_idx, end_idx = feat_cols[feat_name]
            used_feats.append(vector_map_feats[:, :, start_idx:end_idx])
        used_feats = np.concatenate(used_feats, axis=-1)

        # Trans from global coordinates to VCS coordinates.
        num_seg, num_seg_pts, num_feats = used_feats.shape
        obs_feats = used_feats.reshape([-1, num_feats])
        for cols in self.coor_feat_col_idx:
            tmp_obs_feats = obs_feats[:, cols]
            trans_x, trans_y = Affine2D.coord_translate(
                tmp_obs_feats[:, 0], tmp_obs_feats[:, 1], center_x, center_y
            )
            trans_x, trans_y = Affine2D.coord_rotate(
                trans_x, trans_y, center_yaw
            )
            obs_feats[:, cols[0]] = trans_x
            obs_feats[:, cols[1]] = trans_y
        used_feats = obs_feats.reshape([num_seg, num_seg_pts, num_feats])

        return used_feats


@OBJECT_REGISTRY.register
class ArgoverseStructredVizHelper:
    """Generate map for visualization.

    This transform must be used after "ArgoverseStructredMapServer".
    """

    NODE_GRAPH_COLORMAP = {
        "node": [255, 0, 0],
        "adj_2": [0, 0, 255],  # succ
        "adj_3": [0, 0, 255],  # pred
        "adj_4": [0, 255, 0],  # left
        "adj_5": [0, 255, 0],  # right
    }

    def __init__(
        self,
        map_origin_params: List,
        image_coordinates: str = "bev",
        ego_track_id: int = -42,
    ):
        """Initialize method.

        Args:
            map_origin_params: paramters to calculate the offset of the
                map center in the map coordinates.
            image_coordinates: the name of the image coordinates. Defaults
                to "bev". It should be one of ["bev", "img]. The corresponding
                coordinate trans method is ["PhyToBEV", "PhyToImg"].
            ego_track_id: the track id of the ego vehicle.
        """
        self.map_origin_params = map_origin_params
        self.image_coordinates = image_coordinates
        self.ego_track_id = ego_track_id

        if self.image_coordinates == "bev":
            self.reverse = True
            bev_ori_x, bev_ori_y, img_resolu = self.map_origin_params
            self.img_center_offset = [bev_ori_x, bev_ori_y]
            self.img_resolu = img_resolu
        elif self.image_coordinates == "img":
            self.reverse = False
            map_h, map_w, img_resolu = self.map_origin_params
            self.img_center_offset = [map_h / 2, map_w / 2]
            self.img_resolu = img_resolu
        else:
            raise ValueError(
                f"Undefined image coordinates {image_coordinates}"
            )

    def __call__(self, sample: Dict):
        """Callable function.

        The following keys are required in `sample`:
            1. "vcs_vector_map_feats"
            2. "adj_mat"
            3. "valid_track_ids"
            4. "viz_map" (not necessary)

        Args:
            sample (Dict): input original sample.

        Returns:
            sample (Dict): the sample includes new keys "viz_node_graph"
                and "road_map".
        """
        vcs_map_feats = sample["vcs_vector_map_feats"]
        adj_mat = sample["adj_mat"]
        valid_track_ids = sample["valid_track_ids"]

        # Render node graph.
        if self.ego_track_id in valid_track_ids:
            seg_start = vcs_map_feats[:, 0, 0:2]
            seg_end = vcs_map_feats[:, -1, 2:4]
            seg_node_coors = (seg_start + seg_end) / 2

        seg_img_coors = -seg_node_coors if self.reverse else seg_node_coors
        seg_img_coors += np.array(self.img_center_offset)[None, :]
        seg_img_coors /= self.img_resolu
        seg_img_coors = np.round(seg_img_coors).astype(np.int64)

        map_h, map_w = 1024, 1024
        node_map = np.zeros([map_h, map_w, 3])
        for start_idx in range(len(seg_img_coors)):
            for end_idx in range(len(seg_img_coors)):
                tmp_vec_type = int(adj_mat[start_idx, end_idx])
                if tmp_vec_type > 1:
                    tmp_color = self.NODE_GRAPH_COLORMAP[f"adj_{tmp_vec_type}"]
                    tmp_vec = seg_img_coors[[start_idx, end_idx], :]
                    cv2.polylines(
                        node_map,
                        [tmp_vec],
                        isClosed=False,
                        color=tmp_color,
                        thickness=2,
                    )
        for coor in seg_img_coors:
            cv2.circle(
                node_map,
                coor,
                5,
                color=self.NODE_GRAPH_COLORMAP["node"],
                thickness=-1,
            )
        node_map = node_map.transpose(1, 0, 2)
        sample["viz_node_graph"] = node_map

        # Rotate raster graph.
        if "viz_map" in sample:
            yaw = sample["seq_center"].yaw
            viz_rot_map = BaseUnstructuredMapServer._rotate_image(
                sample["viz_map"], -yaw * 180 / np.pi
            )
            sample["road_map"] = viz_rot_map
        return sample


@OBJECT_REGISTRY.register
class GetBevVectorizedMap:
    def __init__(
        self,
        size: int = 512,
        expand_size: int = 3,
        bev_origin_x: float = 72.4,
        bev_origin_y: float = 51.2,
        img_resolution: float = 0.2,
    ):
        # init params
        self.size = size
        self.image_size = (size * expand_size, size * expand_size, 3)

        self.img_resolution = img_resolution
        self.cut_start = size * (expand_size - 1) // 2
        self.cut_end = size * (expand_size - 1) // 2 + size
        self.bev_origin_x = (
            bev_origin_x / img_resolution + self.cut_start
        ) * img_resolution
        self.bev_origin_y = (
            bev_origin_y / img_resolution + self.cut_start
        ) * img_resolution
        self.expand_size = expand_size

        # bgr
        self.color_dict = {
            "solid_lane": (255, 0, 0),
            "roadedge": (0, 0, 255),
            "stopline": (255, 0, 255),
            "crosswalk": (0, 128, 128),
        }

    def __call__(self, sample: dict):

        if "struct_road" in sample:

            struct_road = sample["struct_road"]
            # dst=np.zeros(self.image_size,np.uint8)
            dst = np.ones(self.image_size, np.uint8) * 255

            # plot physical layer elements
            for name, value in self.color_dict.items():
                if name in struct_road:
                    lines = struct_road[name]
                    for line in lines:
                        for tmp_line in line:  # [[x1,x2],[y1,y2],[z1,z2]]
                            arr_line = self.phy2img(
                                (np.array(tmp_line).T)[:, :2]
                            )
                            if len(arr_line) > 0:
                                cv2.line(
                                    img=dst,
                                    pt1=arr_line[0, :],
                                    pt2=arr_line[1, :],
                                    color=value,
                                    thickness=3,
                                    lineType=cv2.LINE_4,
                                )

            # plot locical layer elements
            if "navi" in struct_road:
                logical_lanes = struct_road["navi"]
                if logical_lanes:
                    for str_id in logical_lanes.keys():
                        color = self.hash_to_color(str_id)
                        ori_bbox = np.array(
                            logical_lanes[str_id]["bounding_box"]
                        )
                        # bbox.shape=(n,2)
                        assert (
                            len(ori_bbox.shape) == 2
                        ), "len of ori_bbox's shape should be 2."
                        bbox = self.phy2img(ori_bbox)
                        # bg
                        bg = copy.deepcopy(dst)
                        bg = cv2.fillPoly(img=bg, pts=[bbox], color=color)
                        dst = cv2.addWeighted(
                            src1=dst, alpha=0.3, src2=bg, beta=0.7, gamma=0
                        )

            # clip
            dst = dst[
                self.cut_start : self.cut_end, self.cut_start : self.cut_end, :
            ]

            # flip and rotate
            dst = cv2.rotate(cv2.flip(dst, 0), 0)

            sample["road_map"] = dst
        else:
            sample["road_map"] = np.zeros((self.size, self.size, 3))

        return sample

    def hash_to_color(self, str_id, s=1, v=1):
        md5_hash = hashlib.md5(str(str_id).encode()).hexdigest()
        h = int(md5_hash[:6], 16) / 0xFFFFFF
        r, g, b = colorsys.hsv_to_rgb(h, s, v)
        return (int(b * 255), int(g * 255), int(r * 255))

    def phy2img(self, arr):
        # arr:[[x1,y1],[x2,y2],...]
        img_arr = np.zeros_like(arr)
        img_arr[:, 0] = (self.bev_origin_x - arr[:, 0]) / self.img_resolution
        img_arr[:, 1] = (self.bev_origin_y - arr[:, 1]) / self.img_resolution
        res = np.clip(img_arr, 0, self.size * self.expand_size)
        res = res.astype(np.int32)
        return res
