# Copyright (c) Horizon Robotics. All rights reserved.

import logging
import os
from typing import Callable, Dict, List, Optional, Tuple

import cv2
import msgpack
import msgpack_numpy
import numpy as np
import pandas as pd
from torch.utils.data import Dataset

from hat.utils.pack_type.lmdb import Lmdb
from hat.utils.package_helper import require_packages

try:
    from argoverse.map_representation.map_api import ArgoverseMap
except ImportError:
    ArgoverseMap = None

from hat.core.traj_pred_typing import (
    DEFAULT_COLOR_MAP,
    SeqIndex,
    TrajGroupIndex,
)
from hat.registry import OBJECT_REGISTRY
from .data_packer import Packer

__all__ = [
    "ArgoverseDataset",
    "ArgoverseDatasetFromLMDB",
    "ArgoversePacker",
    "ArgoverseTdtDataset",
]

logger = logging.getLogger(__name__)


@OBJECT_REGISTRY.register
class ArgoverseDataset(Dataset):
    """
    Argoverse trajectory prediction dataset.

    Args:
        data_path: The path of the parent directory of data.
            One data_path could contain many specific datasets,
            such as train, valid and test datasets.
        mode: The name of the dataset directory.
        dataset_size: The sample number to be used.
        transforms: A function transform that takes input
            sample and its target as entry and returns a transformed version.
        meters_map: The extent of input raster_map, in meters.
        resolution: The resolution of raster map and dynamic data.
        center: "AV" or "AGENT", means the center of raster_map is
            autonomous vehicle or the specific agent defined by argoverse.
        drivable_color: The color to fill in the drivable
            area, a tuple with three elements.
        centerline_color: The color to draw the center
            line of lane, a tuple with three elements.
        lane_color: The color to draw in the lane line,
            a tuple with three elements.
        only_agent: Whether to return the valid_flag of agents.
            For test, it must be True.
    """

    FPS = 10
    HISTORY_TIME = 2
    FUTURE_TIME = 3
    CURRENT_FRAME = FPS * HISTORY_TIME - 1
    LAST_FRAME = FPS * (HISTORY_TIME + FUTURE_TIME) - 1
    CATEGORY_MAP = {"AV": 1, "AGENT": 2, "OTHERS": 3}
    CITY_LIST = ["PIT", "MIA"]

    @require_packages("argoverse")
    def __init__(
        self,
        data_path: str,
        mode: str = "train",
        dataset_size: int = -1,
        transforms: Optional[Callable] = None,
        meters_map: float = 100 * 2,
        resolution: float = 0.2,
        center: str = "AV",
        drivable_color: Tuple[float, ...] = (127, 0, 0),
        centerline_color: Tuple[float, ...] = (255, 127, 0),
        lane_color: Tuple[float, ...] = (127, 64, 255),
        only_agent: bool = False,
    ):

        self.data_path = data_path
        self.mode = mode
        self.dataset_size = dataset_size
        self.transforms = transforms
        self.meters_map = meters_map
        self.resolution = resolution
        self.center = center
        self.drivable_color = drivable_color
        self.centerline_color = centerline_color
        self.lane_color = lane_color
        self.only_agent = only_agent

        self._csv_files = None
        self.map_loader = ArgoverseMap(os.path.join(data_path, "map_files"))
        self.map_size = int(round(meters_map / resolution))

    @property
    def csv_files(self):
        if self._csv_files is None:
            file_list = os.path.join(
                self.data_path, self.mode, "file_list.txt"
            )
            if os.path.exists(file_list):
                self._csv_files = eval(open(file_list).read())
            else:
                self._csv_files = os.listdir(self.data_path)
            if self.dataset_size >= 0:
                self._csv_files = self._csv_files[: self.dataset_size]
            sort_fn = lambda x: int(x[:-4])
            self._csv_files.sort(key=sort_fn)
            logger.info(f"dataset size: {len(self._csv_files)}")
        return self._csv_files

    def parse_csv(self, index: int):
        data = ArgoverseDataset.csv_to_numpy(
            os.path.join(
                self.data_path, self.mode, "data", self.csv_files[index]
            )
        )
        return data

    def __getitem__(self, index: int) -> dict:
        coordinate, mask, category, city, time = self.parse_csv(index)

        if self.center == "AV":
            center_x, center_y = coordinate[0, self.CURRENT_FRAME]
        elif self.center == "AGENT":
            center_x, center_y = coordinate[1, self.CURRENT_FRAME]

        start_point = coordinate[:, self.CURRENT_FRAME]
        coordinate = self._meters2pixels(coordinate, center_x, center_y)
        coordinate *= mask[:, :, None]
        mask = mask == 1

        current_mask = mask[:, self.CURRENT_FRAME]
        start_point, coordinate, category, mask = [
            x[current_mask] for x in [start_point, coordinate, category, mask]
        ]

        raster_map = self._get_raster_map(center_x, center_y, city)

        sample = {
            "history": coordinate[:, : self.CURRENT_FRAME + 1][:, ::-1],
            "history_mask": mask[:, : self.CURRENT_FRAME + 1][:, ::-1],
            "future": coordinate[:, self.CURRENT_FRAME + 1 :],
            "future_mask": mask[:, self.CURRENT_FRAME + 1 :],
            "category": category,
            "raster_map": raster_map,
            "index": np.array(int(self.csv_files[index].split(".")[0])),
            "start_point": start_point,  # for submition of test result
        }
        sample["history_state"] = sample["history"] - sample["history"][:, 0:1]
        sample["history_state"] *= sample["history_mask"][..., None]

        if self.only_agent:
            valid_flag = np.zeros(len(sample["history"]))
            valid_flag[1] = 1
            sample["valid_flag"] = valid_flag == 1

        if self.transforms:
            sample = self.transforms(sample)
        return sample

    @staticmethod
    def csv_to_numpy(csv_file: str):
        df = pd.read_csv(csv_file)
        city = df["CITY_NAME"][0]
        tracks = list(df.groupby("TRACK_ID"))
        object_num = len(tracks)

        coordinate = np.zeros([object_num, ArgoverseDataset.LAST_FRAME + 1, 2])
        mask = np.zeros([object_num, ArgoverseDataset.LAST_FRAME + 1])
        category = np.zeros([object_num])

        idx_others = 2
        time = np.sort(np.unique(df["TIMESTAMP"].values))
        time_step_dict = dict(zip(time.tolist(), list(range(50))))
        for _track_id, data in tracks:
            step = np.array(
                [time_step_dict[x] for x in data["TIMESTAMP"].values],
                dtype=np.int32,
            )

            category_tmp = ArgoverseDataset.CATEGORY_MAP[
                data["OBJECT_TYPE"].values[0]
            ]
            xy = data[["X", "Y"]].values

            if category_tmp == 1:
                coordinate[0, step] = xy
                mask[0, step] = 1
                category[0] = category_tmp
            elif category_tmp == 2:
                coordinate[1, step] = xy
                mask[1, step] = 1
                category[1] = category_tmp
            else:
                coordinate[idx_others, step] = xy
                mask[idx_others, step] = 1
                category[idx_others] = category_tmp
                idx_others += 1

        return coordinate, mask, category, city, time

    def __len__(self) -> int:
        return len(self.csv_files)

    def _get_raster_map(self, x: float, y: float, city: str) -> np.ndarray:
        side_length = self.meters_map / 2
        box = [
            x - side_length,
            x + side_length,
            y - side_length,
            y + side_length,
        ]

        lane_polygons = self.map_loader.find_local_lane_polygons(box, city)
        drivable_areas = self.map_loader.find_local_driveable_areas(box, city)

        lane_centerlines = [
            np.round(self._meters2pixels(temp[:, :2], x, y)[:, ::-1]).astype(
                np.int32
            )
            for temp in self._get_lane_centerlines(box, city)
        ]
        lane_polygons = [
            np.round(self._meters2pixels(temp[:, :2], x, y)[:, ::-1]).astype(
                np.int32
            )
            for temp in lane_polygons
        ]
        drivable_areas = [
            np.round(self._meters2pixels(temp[:, :2], x, y)[:, ::-1]).astype(
                np.int32
            )
            for temp in drivable_areas
        ]

        raster_map = np.zeros([self.map_size, self.map_size, 3])
        cv2.fillPoly(raster_map, lane_polygons, color=self.drivable_color)
        cv2.fillPoly(raster_map, drivable_areas, color=self.drivable_color)
        cv2.polylines(
            raster_map,
            lane_polygons,
            isClosed=False,
            color=self.lane_color,
            thickness=int(0.4 / self.resolution),
        )
        cv2.polylines(
            raster_map,
            lane_centerlines,
            isClosed=False,
            color=self.centerline_color,
            thickness=int(0.4 / self.resolution),
        )
        return raster_map

    def _get_lane_centerlines(self, box: List[float], city: str):
        seq_lane_props = self.map_loader.city_lane_centerlines_dict[city]
        lane_centerlines = []

        for lane_props in seq_lane_props.values():
            lane_cl = lane_props.centerline
            if (
                np.min(lane_cl[:, 0]) < box[1]
                and np.min(lane_cl[:, 1]) < box[3]
                and np.max(lane_cl[:, 0]) > box[0]
                and np.max(lane_cl[:, 1]) > box[2]
            ):
                lane_centerlines.append(lane_cl)
        return lane_centerlines

    def _meters2pixels(
        self, coordinate_meters: np.ndarray, center_x: float, center_y: float
    ) -> np.ndarray:
        center = np.array([center_x, center_y])
        coordinate_pixels = (
            coordinate_meters - center + self.meters_map / 2
        ) / self.resolution
        return coordinate_pixels


@OBJECT_REGISTRY.register
class ArgoverseDatasetFromLMDB(ArgoverseDataset):
    """
    Argoverse trajectory prediction dataset in lmdb format.

    Args:
        data_path: The path of the parent directory of data.
            One data_path could contain many specific datasets,
            such as train, valid and test datasets.
        mode: The name of the dataset directory.
        dataset_size: The sample number to be used.
        **kwargs: Kwargs for ArgoverseDataset.
    """

    def __init__(
        self, data_path: str, mode: str, dataset_size: int = -1, **kwargs
    ):
        super(ArgoverseDatasetFromLMDB, self).__init__(
            data_path, mode, dataset_size, **kwargs
        )
        self.argoverse_lmdb = Lmdb(
            os.path.join(self.data_path, self.mode),
            writable=False,
            readahead=False,
        )

    @property
    def csv_files(self):
        if self._csv_files is None:
            self._csv_files = eval(
                open(
                    os.path.join(self.data_path, self.mode, "file_list.txt")
                ).read()
            )
            if self.dataset_size >= 0:
                self._csv_files = self._csv_files[: self.dataset_size]
            sort_fn = lambda x: int(x[:-4])
            self._csv_files.sort(key=sort_fn)
            logger.info(f"dataset size: {len(self._csv_files)}")
        return self._csv_files

    def parse_csv(self, index):
        index = self.csv_files[index]
        raw_data = self.argoverse_lmdb.read(index)
        sample = msgpack.unpackb(
            raw_data, raw=True, object_hook=msgpack_numpy.decode
        )
        city = ArgoverseDataset.CITY_LIST[sample[b"city"]]
        return (
            sample[b"coordinate"],
            sample[b"mask"],
            sample[b"category"],
            city,
            sample[b"time"],
        )


class ArgoversePacker(Packer):
    """
    Packer for converting argoverse dataset from csv format into lmdb format.

    Args:
        src_data_path: The path of the parent directory of data.
        mode: The name of the dataset directory.
        target_data_path: The target path to store lmdb dataset.
        num_workers: Num workers for reading original data.
            while num_workers <= 0 means pack by single process.
            num_workers >= 1 mean pack by num_workers process.
        dataset_size: The sample number to be used.
        **kwargs: Kwargs for Packer.
    """

    def __init__(
        self,
        src_data_path: str,
        mode: str,
        target_data_path: str,
        num_workers: int,
        dataset_size: int = -1,
        **kwargs,
    ):

        if not os.path.exists(target_data_path):
            os.makedirs(target_data_path)
        self.data_path = os.path.join(src_data_path, mode, "data")

        file_list = os.path.join(src_data_path, mode, "file_list.txt")
        if os.path.exists(file_list):
            self.csv_files = eval(open(file_list).read())
        else:
            self.csv_files = os.listdir(self.data_path)
        if dataset_size > 0:
            self.csv_files = self.csv_files[:dataset_size]
        f = open(os.path.join(target_data_path, "file_list.txt"), "w")
        f.write(str(self.csv_files))
        f.close()

        super(ArgoversePacker, self).__init__(
            target_data_path,
            len(self.csv_files),
            "lmdb",
            num_workers,
            **kwargs,
        )
        self.packed = 0

    def pack_data(self, idx):
        f = self.csv_files[idx]
        coordinate, mask, category, city, time = ArgoverseDataset.csv_to_numpy(
            os.path.join(self.data_path, f)
        )
        sample = {
            "coordinate": coordinate,
            "mask": mask,
            "category": category,
            "city": np.array(ArgoverseDataset.CITY_LIST.index(city)),
            "time": time,
        }
        data = msgpack.packb(sample, default=msgpack_numpy.encode)

        if self.packed % 1000 == 0:
            logger.info(f"{idx} {self.packed} / {len(self.csv_files)}")
        self.packed += 1
        return [f, data]

    def _write(self, idx, data):
        self.pack_file.write(data[0], data[1])


@OBJECT_REGISTRY.register
class ArgoverseTdtDataset(ArgoverseDataset):
    """Argoverse trajectory prediction dataset support tdt format.

    Different from `ArgoverseDataset`, this dataset modifies the content
    of the output. This change enables the dataset directly connect to the
    old trajectory prediction transforms (based on tdt data format).
    """

    # fmt:off
    SUPPORTED_MAP_MODE = ["vectorized", "rasterized"]
    # Attributes for vectorized map generation.
    POLYLINE_DIRECTION = {"NONE": 0, "LEFT": -1, "RIGHT": 1}
    POLYLINE_OP_FEATS = [
        "turn_dir", "pre_pre_point", "pid", "pid_pred_succ", "interaction"
    ]
    POLYLINE_FEATS_COLS = {
        "basic": [0, 4],
        "turn_dir": [4, 5],
        "pre_pre_point": [5, 7],
        "pid": [7, 8],
        "pid_pred_succ": [8, 10],
        "interaction": [10, 11],
    }
    # TDT format columns.
    DYN_COLS = [
        "pos_x", "pos_y", "yaw", "x", "y", "obs_yaw", "classification",
        "track_id", "frame_id", "timestamp"
    ]
    CONST_COLS = [
        "map_id", "date", "data_num", "data_version", "center_car_id",
        "height", "width", "length", "obs_height", "obs_width", "obs_length"
    ]
    NAN_COLS = [
        "ego_x0", "ego_x1", "ego_x2", "ego_x3", "ego_y0", "ego_y1", "ego_y2",
        "ego_y3", "ego_z0", "ego_z1", "ego_z2", "ego_z3", "x0", "x1",
        "x2", "x3", "y0", "y1", "y2", "y3", "z0", "z1", "z2", "z3"
    ]
    ZERO_COLS = ["pos_z", "z", "det_score"]
    INT_COLS = [
        "map_id", "date", "data_num", "center_car_id", "data_version",
        "frame_id", "track_id", "classification", "timestamp"
    ]
    ADJACENCY_TYPE = {
        "not_collected": 0,
        "self": 1,
        "succ": 2,
        "pred": 3,
        "left": 4,
        "right": 5
    }
    # fmt:on

    def __init__(
        self,
        data_path: str,
        mode: str = "train",
        dataset_size: int = -1,
        transforms: Optional[Callable] = None,
        center: str = "AV",
        only_agent: bool = False,
        # Parameters for map.
        meters_map: float = 100 * 2,
        resolution: float = 0.2,
        map_mode: str = "vectorized",
        # Parameters for rasterized map.
        color_map: Optional[Dict] = None,
        # Customized parameters
        polyline_seg_len: int = 10,
        for_viz: bool = False,
        ego_track_id: int = -42,
        **kwargs,
    ):
        """Initialize method.

        Args:
            The parameters before `resolution` refer to the doc string
            of the base class `ArgoverseDataset`.
            map_mode: the output map type. It should be one of
                ["vectorized", "rasterized"].
            color_map: the color map of different road element during
                rasterized map renderring. The format refer to:
                    hat.core.traj_pred_typing.DEFAULT_COLOR_MAP
            polyline_seg_len: the length of the polyline segments.
            for_viz: whether the dataset is for visualization. If True,
                a rasterized map will be generated regardless of `map_mode`.
            ego_track_id: the track id of the ego vehicle.
        """
        if color_map is None:
            color_map = DEFAULT_COLOR_MAP
        assert "drivable_areas" in color_map
        assert "solid_lanes" in color_map
        assert "virtuallanes" in color_map
        super(ArgoverseTdtDataset, self).__init__(
            data_path=data_path,
            mode=mode,
            dataset_size=dataset_size,
            transforms=transforms,
            meters_map=meters_map,
            resolution=resolution,
            center=center,
            drivable_color=color_map["drivable_areas"],
            centerline_color=color_map["virtuallanes"],
            lane_color=color_map["solid_lanes"],
            only_agent=only_agent,
        )
        assert (
            map_mode in self.SUPPORTED_MAP_MODE
        ), f"Unsupported map type to generate: {map_mode}."
        self.map_mode = map_mode
        self.for_viz = for_viz
        self.ego_track_id = ego_track_id
        self.polyline_seg_len = polyline_seg_len
        self.lane_radius = meters_map / 2

    def __getitem__(self, index: int) -> dict:
        coordinate, mask, category, city, time = self.parse_csv(index)

        if self.center == "AV":
            center_x, center_y = coordinate[0, self.CURRENT_FRAME]
        elif self.center == "AGENT":
            center_x, center_y = coordinate[1, self.CURRENT_FRAME]

        start_point = coordinate[:, self.CURRENT_FRAME]
        # coordinate = self._meters2pixels(coordinate, center_x, center_y)
        coordinate *= mask[:, :, None]
        mask = mask == 1

        current_mask = mask[:, self.CURRENT_FRAME]
        start_point, coordinate, category, mask = [
            x[current_mask] for x in [start_point, coordinate, category, mask]
        ]
        sample = {
            "history": coordinate[:, : self.CURRENT_FRAME + 1],
            "history_mask": mask[:, : self.CURRENT_FRAME + 1],
            "future": coordinate[:, self.CURRENT_FRAME + 1 :],
            "future_mask": mask[:, self.CURRENT_FRAME + 1 :],
            "category": category,
            "timestamp": time,
            "index": np.array(int(self.csv_files[index].split(".")[0])),
            "start_point": start_point,  # for submition of test result
        }
        sample["history_state"] = sample["history"] - sample["history"][:, 0:1]
        sample["history_state"] *= sample["history_mask"][..., None]

        # TODO (shengzhe.dai): the "rasterized" mode has not finished yet.
        # I will implement a new `_get_raster_map` function to adapt to
        # the rasterized map requirement of trajectory prediction.
        if self.map_mode == "rasterized":
            raster_map = self._get_raster_map(center_x, center_y, city)
            sample["raster_map"] = raster_map
            if self.for_viz:
                sample["viz_map"] = raster_map
        elif self.map_mode == "vectorized":
            poly_features, adj_mat = self._get_vectorized_map(
                center_x, center_y, city
            )
            sample["vector_map_feats"] = poly_features
            sample["feat_cols"] = self.POLYLINE_FEATS_COLS
            sample["adj_mat"] = adj_mat
            sample["adj_type"] = self.ADJACENCY_TYPE
            sample = self._trans_argoverse_raw_data_to_tdt(sample)
            if self.for_viz:
                viz_map = self._get_raster_map(center_x, center_y, city)
                sample["viz_map"] = viz_map

        if self.only_agent:
            valid_flag = np.zeros(len(sample["history"]))
            valid_flag[1] = 1
            sample["valid_flag"] = valid_flag == 1

        if self.transforms:
            sample = self.transforms(sample)
        return sample

    def _get_vectorized_map(self, x: float, y: float, city: str):
        """Get local vectoirized map.

        Args:
            x: the x coordinate of the map center (in global coordinates).
            y: the y coordinate of the map center (in global coordinates).
            city: the name of the city.

        Return:
            poly_features ([num_polylines, polyline_seg_len-1, num_feats]):
                the features of the polylines. The specific definition
                refers `self.POLYLINE_FEATS_COLS`.
            adj_mat ([num_polylines, num_polylines]): the adjacency
                relationship matrix. The type refer to self.ADJACENCY_TYPE
        """
        poly_features = []
        seq_lane_props = self.map_loader.city_lane_centerlines_dict[city]
        nearby_lane_ids = self.map_loader.get_lane_ids_in_xy_bbox(
            x, y, city, self.lane_radius
        )
        pid_mapping = {
            lane_id: idx for idx, lane_id in enumerate(nearby_lane_ids)
        }
        adj_mat = np.zeros([len(nearby_lane_ids), len(nearby_lane_ids)])

        for lane_id in nearby_lane_ids:
            lane_props = seq_lane_props[lane_id]
            centerline = lane_props.centerline

            # Extract adjacency matrix
            adj_mat[
                pid_mapping[lane_id], pid_mapping[lane_id]
            ] = self.ADJACENCY_TYPE["self"]
            left_id = getattr(lane_props, "l_neighbor_id", None)
            right_id = getattr(lane_props, "r_neighbor_id", None)
            succs = getattr(lane_props, "successors", None)
            preds = getattr(lane_props, "predecessors", None)
            if left_id is not None and left_id in pid_mapping:
                adj_mat[
                    pid_mapping[left_id], pid_mapping[lane_id]
                ] = self.ADJACENCY_TYPE["left"]
            if right_id is not None and right_id in pid_mapping:
                adj_mat[
                    pid_mapping[right_id], pid_mapping[lane_id]
                ] = self.ADJACENCY_TYPE["right"]
            if succs is not None:
                for succ in succs:
                    if succ in pid_mapping:
                        adj_mat[
                            pid_mapping[succ], pid_mapping[lane_id]
                        ] = self.ADJACENCY_TYPE["succ"]
            if preds is not None:
                for pred in preds:
                    if pred in pid_mapping:
                        adj_mat[
                            pid_mapping[pred], pid_mapping[lane_id]
                        ] = self.ADJACENCY_TYPE["pred"]

            # Extract vectornet feats
            # 1. Basic lane features [x_start, y_start, x_end, y_end]
            centerline = centerline[:, :2]
            lane_feat = np.hstack((centerline[:-1], centerline[1:]))
            num_poly_pts = lane_feat.shape[0]
            # 2. Optional features:
            op_feats = []
            for feat_name in self.POLYLINE_OP_FEATS:
                if feat_name == "turn_dir":
                    turn_direction = self.POLYLINE_DIRECTION[
                        lane_props.turn_direction
                    ]
                    op_feats.append(
                        np.ones([num_poly_pts, 1]) * turn_direction
                    )
                elif feat_name == "pid":
                    op_feats.append(
                        np.ones([num_poly_pts, 1]) * pid_mapping[lane_id]
                    )
                elif feat_name == "pid_pred_succ":
                    if lane_props.successors is not None and not len(
                        lane_props.successors
                    ):
                        pid_succ = lane_props.successors[0]
                        if pid_succ not in pid_mapping:
                            pid_succ = -1
                        else:
                            pid_succ = pid_mapping[pid_succ]
                    else:
                        pid_succ = -1

                    if lane_props.predecessors is not None and not len(
                        lane_props.predecessors
                    ):
                        pid_pred = lane_props.predecessors[0]
                        if pid_pred not in pid_mapping:
                            pid_pred = -1
                        else:
                            pid_pred = pid_mapping[pid_pred]
                    else:
                        pid_pred = -1

                    pid_pred_succ = np.stack([pid_pred, pid_succ], axis=-1)[
                        None, :
                    ]
                    pid_pred_succ = pid_pred_succ.repeat(num_poly_pts, axis=0)
                    op_feats.append(pid_pred_succ)
                elif feat_name == "pre_pre_point":
                    pre_pre_point = np.concatenate(
                        (centerline[0:1, :], centerline[:-2, :]), axis=0
                    )
                    op_feats.append(pre_pre_point)
                elif feat_name == "interaction":
                    is_intersection = int(lane_props.is_intersection)
                    op_feats.append(
                        np.ones([num_poly_pts, 1]) * is_intersection
                    )
            lane_feat = np.concatenate([lane_feat] + op_feats, axis=-1)
            if len(lane_feat) == self.polyline_seg_len - 1:
                poly_features.append(lane_feat)
        poly_features = np.stack(poly_features)
        return poly_features, adj_mat

    def _extract_traj_feats_for_tdt(
        self,
        coors: np.array,
        masks: np.array,
        category: np.array,
        time: np.array,
        coors_type: str,
        frame_id_offset: int = 0,
    ):
        """Extract trajectory features (dynamic objects) for tdt formatting.

        Args:
            coors ([num_obs, traj_len, 2]): the trajectory coordinates.
            masks ([num_obs, traj_len]): the masks.
            category ([num_obs]): the categories.
            time ([traj_len]): the time stamp array.
            coors_type: one of ["history", "future"]
            frame_id_offset: the first frame_id.

        Returns:
            feats ([num_obs, traj_len, len(self.DYN_COLS)]): the extracted
                dynamic features.
            num_ctx_frames: the number of all frames.
        """
        coors_diff = np.diff(coors, axis=1)
        coors_yaw = np.arctan2(coors_diff[:, :, 1], coors_diff[:, :, 0])
        if coors_type == "history":
            coors_yaw = np.concatenate([coors_yaw[:, 0:1], coors_yaw], axis=-1)
        else:
            coors_yaw = np.concatenate([coors_yaw, coors_yaw[:, -1:]], axis=-1)

        num_obs, num_ctx_frames, _ = coors.shape
        num_feats = len(self.DYN_COLS)
        ego_coors = coors[0:1, :, :]
        ego_masks = masks[0:1, :]
        ego_yaw = coors_yaw[0:1, :]
        if num_obs:
            obs_coors = coors
            obs_masks = masks
            obs_yaw = coors_yaw
            obs_categraphy = category
            ego_coors = np.repeat(ego_coors, num_obs, axis=0)
            ego_yaw = np.repeat(ego_yaw, num_obs, axis=0)
            frame_id = (
                np.repeat(
                    np.arange(0, num_ctx_frames)[None, :, None],
                    num_obs,
                    axis=0,
                )
                + frame_id_offset
            )
            track_id = np.arange(-1, num_obs - 1)
            track_id[0] = self.ego_track_id
            track_id = np.repeat(
                track_id[:, None, None], num_ctx_frames, axis=1
            )
            obs_categraphy = np.repeat(
                obs_categraphy[:, None, None], num_ctx_frames, axis=1
            )
            timestamp = np.repeat(time[None, :, None], num_obs, axis=0)
            feats = np.concatenate(
                [
                    ego_coors,
                    ego_yaw[:, :, None],
                    obs_coors,
                    obs_yaw[:, :, None],
                ],
                axis=-1,
            )
            feats = np.concatenate(
                [feats, obs_categraphy, track_id, frame_id, timestamp], axis=-1
            )
        else:
            obs_coors = np.full_like(ego_coors, np.nan)
            obs_masks = np.full_like(ego_masks, np.nan)
            obs_masks = np.full_like(ego_yaw, np.nan)
            feats = np.full([1, num_ctx_frames, num_feats - 3], np.nan)
            feats = np.concatenate(
                [ego_coors, ego_yaw[:, :, None], feats], axis=-1
            )

        feats = feats.reshape([-1, feats.shape[-1]])
        obs_masks = obs_masks.reshape([-1])
        feats = feats[obs_masks, :]
        return feats, num_ctx_frames

    def _trans_argoverse_raw_data_to_tdt(self, sample: Dict):
        """Trans the argoverse raw data to tdt format."""

        dyn_feats = []
        num_frames = 0
        for coors_type in ["history", "future"]:
            if coors_type == "history":
                time = (
                    sample["timestamp"][: self.CURRENT_FRAME + 1] * 1000
                )  # s -> ms
            else:
                time = (
                    sample["timestamp"][self.CURRENT_FRAME + 1 :] * 1000
                )  # s -> ms
            coors = sample[coors_type]
            masks = sample[f"{coors_type}_mask"]
            tmp_feats, tmp_frames = self._extract_traj_feats_for_tdt(
                coors, masks, sample["category"], time, coors_type, num_frames
            )
            num_frames += tmp_frames
            dyn_feats.append(tmp_feats)
        dyn_feats = np.concatenate(dyn_feats, axis=0)
        df_len = dyn_feats.shape[0]

        # fmt:off
        # There are many parameters below are set as special values. They
        # will not be used actually.
        const_feats = np.array(
            [sample["index"], -123, 0, 0, self.ego_track_id, np.nan,
             1.5, 4, np.nan, 1.5, 4])[None, :]
        # fmt:on
        const_feats = np.repeat(const_feats, df_len, axis=0)

        nan_feats = np.full([df_len, len(self.NAN_COLS)], np.nan)
        zero_feats = np.zeros([df_len, len(self.ZERO_COLS)])
        all_feats = np.concatenate(
            [dyn_feats, const_feats, nan_feats, zero_feats], axis=-1
        )
        all_cols = (
            self.DYN_COLS + self.CONST_COLS + self.NAN_COLS + self.ZERO_COLS
        )

        df = pd.DataFrame(all_feats, columns=all_cols)
        df[self.INT_COLS] = df[self.INT_COLS].round().astype(int)

        sample["seq_df"] = df
        sample["dataset_index"] = int(sample["index"])
        sample["last_context_frame_id"] = sample["history"].shape[1] - 1
        sample["seq_index"] = SeqIndex(
            TrajGroupIndex(sample["index"], -123, 0, 0, self.ego_track_id),
            slice(0, num_frames, 1),
        )
        return sample
