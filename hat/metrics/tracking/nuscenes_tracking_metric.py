# Copyright (c) Horizon Robotics. All rights reserved.

import json
import logging
import math
import os
import pickle as pkl
import time
from collections import defaultdict
from copy import deepcopy
from itertools import count
from multiprocessing import Pool
from typing import Callable, Dict, List, Tuple

import numpy as np
import pandas
import sklearn
import tqdm

try:
    from hatbc.message import (
        Attribute,
        BBox3D,
        CameraFrame,
        CameraParam,
        Frame,
        Image,
        Instance,
        MessageMeta,
        SyncMessages,
        VCSParam,
        Velocity,
    )
except ImportError:
    Attribute, BBox3D, CameraFrame, CameraParam = None, None, None, None
    Frame, Image, Instance, MessageMeta = None, None, None, None
    SyncMessages, VCSParam, Velocity = None, None, None

from hat.registry import OBJECT_REGISTRY
from hat.utils.package_helper import require_packages

try:
    from nuscenes.eval.common.data_classes import EvalBoxes
    from nuscenes.eval.common.utils import Quaternion
    from nuscenes.eval.tracking.algo import TrackingEvaluation
    from nuscenes.eval.tracking.constants import AVG_METRIC_MAP, MOT_METRIC_MAP
    from nuscenes.eval.tracking.data_classes import (
        TrackingBox,
        TrackingConfig,
        TrackingMetricData,
        TrackingMetricDataList,
        TrackingMetrics,
    )
    from nuscenes.eval.tracking.evaluate import TrackingEval
    from nuscenes.eval.tracking.loaders import interpolate_tracks
    from nuscenes.eval.tracking.mot import MOTAccumulatorCustom
    from nuscenes.eval.tracking.utils import (
        create_motmetrics,
        print_threshold_metrics,
    )

    NUSC = True
except ImportError:
    EvalBoxes = None
    Quaternion = None
    TrackingBox = None
    TrackingConfig = None
    TrackingMetrics = None
    TrackingMetricData = None
    TrackingMetricDataList = None
    MOTAccumulatorCustom = None
    TrackingEval = object
    interpolate_tracks = None
    TrackingEvaluation = object
    AVG_METRIC_MAP = None
    MOT_METRIC_MAP = None
    print_threshold_metrics = None
    create_motmetrics = None
    NUSC = False

logger = logging.getLogger(__name__)

__all__ = ["NuscenesTrackingMetric"]

DEFAULT_WORST = {
    "amota": 0.0,
    "amotp": 2.0,
    "recall": 0.0,
    "motar": 0.0,
    "mota": 0.0,
    "motp": 2.0,
    "mt": 0.0,
    "ml": -1.0,
    "faf": 500,
    "gt": -1,
    "tp": 0.0,
    "fp": -1.0,
    "fn": -1.0,
    "ids": -1.0,
    "frag": -1.0,
    "tid": 20,
    "lgd": 20,
}
GT_MAPPING = {"FP": "GT", "MISS": "FN", "SWITCH": "SWITCH", "MATCH": "GT"}
PRED_MAPPING = {"FP": "FP", "MISS": "FN", "SWITCH": "SWITCH", "MATCH": "TP"}
NUSC_CAM = {
    "CAM_FRONT_LEFT": 0,
    "CAM_FRONT": 1,
    "CAM_FRONT_RIGHT": 2,
    "CAM_BACK_LEFT": 3,
    "CAM_BACK": 4,
    "CAM_BACK_RIGHT": 5,
}

NUSC_EXTRA_ARGS = [
    "min_recall",
    "max_boxes_per_sample",
    "num_thresholds",
    "num_workers",
    "scene_parallel",
    "metric_worst",
    "verbose",
]


def track_init():
    return defaultdict(list)


def name_gen(_threshold):
    return "thr_%.4f" % _threshold


@OBJECT_REGISTRY.register
class NuscenesTrackingMetric(TrackingEval):
    """Nuscenes tracking metric Wrapper based on HATBC message struct.

    Args:
        save_dir: Output dir to save file.
        gt: GroundTruth for tracking eval.
        pred: Prediction for tracking eval.
        eval_classes: Names of obejct category to evaluate.
        eval_ranges: Max eval distance for each category, ignore if over.
        dist_fcn: Distance function for pred and gt.
        dist_th_tp: Pred is positive if below this thresholds.
        min_recall: Min recall value in case of too much noise.
        max_boxes_per_sample: Box amount limit in each sample.
        num_thresholds: Number of recall thresholds.
        num_workers: Number of multi-processing evaluation worker.
        scene_parallel: Parallel at scene level if true,
            otherwise at the score threshold level.
        metric_worst: The default metric value if not achieved.
        verbose: Logging if true.
    """

    @require_packages(
        "nuscenes", raise_msg="Please `pip3 install nuscenes-devkit`"
    )
    def __init__(
        self,
        save_dir: str,
        gt: Dict[str, "Frame"],
        pred: Dict[str, "Frame"],
        eval_classes: List[str],
        eval_ranges: Dict[str, float],
        dist_fcn: str = "center_distance",
        dist_th_tp: float = 2.0,
        min_recall: float = 0.1,
        max_boxes_per_sample: int = 500,
        num_thresholds: int = 40,
        num_workers: int = 0,
        scene_parallel: bool = False,
        metric_worst: Dict[str, float] = DEFAULT_WORST,
        verbose=True,
    ):

        # init nusc TrackingConfig
        assert all([cls in eval_ranges for cls in eval_classes])
        assert dist_fcn in ["center_distance"], "%s not supported" % dist_fcn
        assert min_recall > 0 and min_recall < 1
        for k, v in DEFAULT_WORST.items():
            if k not in metric_worst:
                metric_worst[k] = v
        config = {
            "tracking_names": eval_classes,
            "pretty_tracking_names": {c: c for c in eval_classes},
            "tracking_colors": {
                c: "C%d" % i for i, c in enumerate(eval_classes)
            },
            "class_range": eval_ranges,
            "dist_fcn": dist_fcn,
            "dist_th_tp": dist_th_tp,
            "min_recall": min_recall,
            "max_boxes_per_sample": max_boxes_per_sample,
            "metric_worst": metric_worst,
            "num_thresholds": num_thresholds,
        }
        self.dist_fcn = dist_fcn
        self.dist_th_tp = dist_th_tp
        self.num_workers = num_workers
        self.scene_parallel = scene_parallel
        self.cfg = TrackingConfig.deserialize(config)
        self.meta = {}

        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        self.output_dir = save_dir
        self.result_path = save_dir
        self.verbose = verbose
        self.eval_classes = eval_classes
        # assume using viz function in hat
        self.render_classes = None

        # convert gt and pred to hatbc message frame format
        self.scene2samples = self.load_scene_index(gt)
        self.sample2scene = {}
        for scene, samples in self.scene2samples.items():
            for sample in samples:
                self.sample2scene[sample] = scene
        gt_boxes = self.load_boxes(gt)
        pred_boxes = self.load_boxes(pred)

        # filter
        # TODO: add vcs [x, y] filter when range is tuple
        gt_boxes = self.filter_eval_boxes(gt_boxes, config["class_range"])
        pred_boxes = self.filter_eval_boxes(pred_boxes, config["class_range"])

        self.sample_tokens = gt_boxes.sample_tokens

        # convert to tracks
        self.tracks_gt = self.create_tracks(gt_boxes, gt, is_gt=True)
        self.tracks_pred = self.create_tracks(pred_boxes, gt, is_gt=False)

    def load_scene_index(self, frames):
        """Load sorted sample tokens in each scene."""
        scene2samples = {}
        for sample, frame in frames.items():
            scene = frame.topic
            if scene not in scene2samples:
                scene2samples[scene] = []
            scene2samples[scene].append(sample)
        # samples are sorted by timestamp in each scene
        for scene, samples in scene2samples.items():
            scene2samples[scene] = sorted(
                samples, key=lambda x: frames[x].meta.timestamp
            )
        return scene2samples

    def load_boxes(self, frames, classes: List[str] = None):
        """Convert standard format to nusc boxes."""
        # transfer to nusc submission format
        results = {}
        for sample_token, frame in frames.items():
            percs = frame.get_messages("overall")[0].get_instances(classes)
            sample_results = []
            for inst in percs:
                sample_result = {
                    "sample_token": sample_token,
                    "translation": inst.get_attributes(
                        topics=["global_translation"]
                    )[0].value,
                    "size": [
                        inst.bbox3ds[0].dim[0],
                        inst.bbox3ds[0].dim[2],
                        inst.bbox3ds[0].dim[1],
                    ],  # whl to wlh
                    "rotation": inst.get_attributes(
                        topics=["global_rotation"]
                    )[0].value,
                    "tracking_id": inst.track_id,
                    "tracking_name": inst.get_attributes(topics=["tracking"])[
                        0
                    ].value,
                    "tracking_score": inst.get_attributes(topics=["tracking"])[
                        0
                    ].score,
                    "ego_translation": inst.bbox3ds[0].loc,
                    "num_pts": inst.get_attributes(topics=["num_pts"])[
                        0
                    ].value,
                }
                if inst.velocity:
                    sample_result["velocity"] = [
                        inst.velocity.vel_x,
                        inst.velocity.vel_y,
                    ]
                sample_results.append(sample_result)
            assert len(sample_results) <= self.cfg.max_boxes_per_sample
            results[sample_token] = sample_results
        # deserialize
        return EvalBoxes.deserialize(results, TrackingBox)

    def filter_eval_boxes(
        self, eval_boxes: EvalBoxes, max_dist: Dict[str, float]
    ):
        """Apply custom filter function."""
        total, dist_filter = 0, 0
        for sample_token in eval_boxes.sample_tokens:
            total += len(eval_boxes[sample_token])
            eval_boxes.boxes[sample_token] = [
                box
                for box in eval_boxes[sample_token]
                if box.ego_dist < max_dist[box.tracking_name]
            ]
            dist_filter += len(eval_boxes[sample_token])
        if self.verbose:
            logger.info("Original number of boxes: %d" % total)
            logger.info("After distance based filtering: %d" % dist_filter)
        return eval_boxes

    def create_tracks(self, all_boxes, frames, is_gt):
        """Nusc boxes to nusc tracks."""
        # Tracks are stored as dict:
        tracks = defaultdict(track_init)

        # Init all scenes and timestamps to guarantee completeness.
        for scene_token, sample_tokens in self.scene2samples.items():
            for sample_token in sample_tokens:
                timestamp = frames[sample_token].meta.timestamp
                tracks[scene_token][timestamp] = []

        # Group annotations wrt scene and timestamp.
        for sample_token in all_boxes.sample_tokens:
            scene_token = self.sample2scene[sample_token]
            timestamp = frames[sample_token].meta.timestamp
            tracks[scene_token][timestamp] = all_boxes.boxes[sample_token]

        # Notice: Codes below are from nuscenes-devkit.
        # Replace box scores with track score (average box score). This only
        # affects the compute_thresholds method and should be done before
        # interpolation to avoid diluting the original scores with
        # interpolated boxes.
        if not is_gt:
            for _, scene_tracks in tracks.items():
                # For each track_id, collect the scores.
                track_id_scores = defaultdict(list)
                for _, boxes in scene_tracks.items():
                    for box in boxes:
                        track_id_scores[box.tracking_id].append(
                            box.tracking_score
                        )

                # Compute average scores for each track.
                track_id_avg_scores = {}
                for tracking_id, scores in track_id_scores.items():
                    track_id_avg_scores[tracking_id] = np.mean(scores)

                # Apply average score to each box.
                for _, boxes in scene_tracks.items():
                    for box in boxes:
                        box.tracking_score = float(
                            track_id_avg_scores[box.tracking_id]
                        )

        # Interpolate GT and predicted tracks.
        for scene_token in tracks.keys():
            tracks[scene_token] = interpolate_tracks(tracks[scene_token])

            if not is_gt:
                # Make sure predictions are sorted in in time.
                tracks[scene_token] = defaultdict(
                    list,
                    sorted(tracks[scene_token].items(), key=lambda kv: kv[0]),
                )

        return tracks

    def evaluate(self):
        """Codes from nuscenes, except multiprocessing eval."""
        start_time = time.time()
        metrics = TrackingMetrics(self.cfg)

        # -----------------------------------
        # Step 1: Accumulate metric data for all classes
        #         and distance thresholds.
        # -----------------------------------
        if self.verbose:
            logging.info("Accumulating metric data...")
        metric_data_list = TrackingMetricDataList()

        def accumulate_class(curr_class_name):
            curr_ev = ParallelTrackingEvaluation(
                self.tracks_gt,
                self.tracks_pred,
                curr_class_name,
                self.cfg.dist_fcn_callable,
                self.cfg.dist_th_tp,
                self.cfg.min_recall,
                num_thresholds=TrackingMetricData.nelem,
                num_workers=self.num_workers,
                scene_parallel=self.scene_parallel,
                metric_worst=self.cfg.metric_worst,
                verbose=self.verbose,
                output_dir=self.output_dir,
                render_classes=self.render_classes,
            )
            curr_md = curr_ev.accumulate()
            metric_data_list.set(curr_class_name, curr_md)

        for class_name in self.cfg.class_names:
            accumulate_class(class_name)

        # -----------------------------------
        # Step 2: Aggregate metrics from the metric data.
        # -----------------------------------
        if self.verbose:
            logging.info("Calculating metrics...")
        for class_name in self.cfg.class_names:
            # Find best MOTA to determine threshold to pick for traditional
            # metrics. If multiple thresholds have the same value, pick the
            # one with the highest recall.
            md = metric_data_list[class_name]
            if np.all(np.isnan(md.mota)):
                best_thresh_idx = None
            else:
                best_thresh_idx = np.nanargmax(md.mota)

            # Pick best value for traditional metrics.
            if best_thresh_idx is not None:
                for metric_name in MOT_METRIC_MAP.values():
                    if metric_name == "":
                        continue
                    value = md.get_metric(metric_name)[best_thresh_idx]
                    metrics.add_label_metric(metric_name, class_name, value)

            # Compute AMOTA / AMOTP.
            for metric_name in AVG_METRIC_MAP.keys():
                values = np.array(md.get_metric(AVG_METRIC_MAP[metric_name]))
                assert len(values) == TrackingMetricData.nelem

                if np.all(np.isnan(values)):
                    # If no GT exists, set to nan.
                    value = np.nan
                else:
                    # Overwrite any nan value with the worst possible value.
                    np.all(values[np.logical_not(np.isnan(values))] >= 0)
                    values[np.isnan(values)] = self.cfg.metric_worst[
                        metric_name
                    ]
                    value = float(np.nanmean(values))
                metrics.add_label_metric(metric_name, class_name, value)

        # Compute evaluation time.
        metrics.add_runtime(time.time() - start_time)

        return metrics, metric_data_list

    def post_process(self, gt, pred):
        """Convert nusc results to general format."""
        results = {}
        # extract summary
        with open(os.path.join(self.output_dir, "metrics_summary.json")) as f:
            summary = json.load(f)
        results["overall"] = {
            "overview": {k: summary[k] for k in DEFAULT_WORST}
        }
        for cls in self.eval_classes:
            results[cls] = {
                "overview": {
                    k: summary["label_metrics"][k][cls] for k in DEFAULT_WORST
                }
            }

        # extract details
        with open(os.path.join(self.output_dir, "metrics_details.json")) as f:
            details = json.load(f)
        for cls in self.eval_classes:
            details[cls]["precision"] = [
                tp / max(1, tp + fp)
                for tp, fp in zip(details[cls]["tp"], details[cls]["fp"])
            ]
            results[cls]["details"] = details[cls]

        # get MOT association
        n_gt = deepcopy(gt)
        n_pred = deepcopy(pred)
        sort_index = {
            "tp": {},
            "fp": {},
            "switch": {},
            "fn": {},
            "dist": {},
        }
        for cls in self.eval_classes:
            n_gt, n_pred, sort_index = self.compute_asso(
                cls, n_gt, n_pred, sort_index
            )

        return results, {
            "gts": n_gt,
            "preds": n_pred,
            "sort_index": sort_index,
        }

    def compute_asso(self, class_name, gt, pred, sort_index):
        """Get the association for error table."""
        # index gt & pred by scene_id and timestamp
        def reindex(sync_frames):
            indexed = {}
            revert_index = {}
            for sample_id, sync_frame in sync_frames.items():
                scene_id = sync_frame.topic
                if scene_id not in indexed:
                    indexed[scene_id] = {}
                indexed[scene_id][sync_frame.meta.timestamp] = sync_frame
                revert_index[(scene_id, sync_frame.meta.timestamp)] = sample_id
            return indexed, revert_index

        gt_indexed, gt_revert_ind = reindex(gt)
        pred_indexed, pred_revert_ind = reindex(pred)

        if self.num_workers > 0 and self.scene_parallel:
            pool = Pool(self.num_workers)
            multi_args = []
            ret_dict = {}
        res_dict = {}

        for scene_id in self.tracks_gt.keys():
            # Retrieve GT and preds.
            scene_tracks_gt = self.tracks_gt[scene_id]
            scene_tracks_pred = self.tracks_pred[scene_id]
            scene_raw_gt = gt_indexed[scene_id]
            scene_raw_pred = pred_indexed[scene_id]

            if self.num_workers > 0 and self.scene_parallel:
                multi_args.append(
                    [
                        scene_id,
                        scene_tracks_gt,
                        scene_tracks_pred,
                        class_name,
                        None,
                        self.dist_fcn,
                        self.dist_th_tp,
                        scene_raw_gt,
                        scene_raw_pred,
                    ]
                )
            else:
                res_dict[scene_id] = scene_accumulate(
                    scene_tracks_gt,
                    scene_tracks_pred,
                    class_name,
                    None,
                    self.dist_fcn,
                    self.dist_th_tp,
                    scene_raw_gt,
                    scene_raw_pred,
                )

        # gather results
        if (
            self.num_workers > 0
            and self.scene_parallel
            and len(multi_args) > 0
        ):
            for arg in multi_args:
                ret = pool.apply_async(
                    scene_accumulate,
                    arg[1:],
                )
                ret_dict[arg[0]] = ret

            for scene_id, ret in tqdm.tqdm(
                ret_dict.items(), desc="MultiProcessingScenes: "
            ):
                res_dict[scene_id] = ret.get()

        # revert by sample_id
        for scene_id, (gt_res, pred_res, sort_count) in res_dict.items():
            for _, frame in gt_res.items():
                gt[gt_revert_ind[(scene_id, frame.meta.timestamp)]] = frame
            for _, frame in pred_res.items():
                pred[pred_revert_ind[(scene_id, frame.meta.timestamp)]] = frame
            for k, v in sort_count.items():
                if scene_id not in sort_index[k]:
                    sort_index[k][scene_id] = 0
                sort_index[k][scene_id] += v
        return gt, pred, sort_index

    @staticmethod
    def euler_to_quaternion(yaw, pitch=0, roll=0):
        qx = np.sin(roll / 2) * np.cos(pitch / 2) * np.cos(yaw / 2) - np.cos(
            roll / 2
        ) * np.sin(pitch / 2) * np.sin(yaw / 2)
        qy = np.cos(roll / 2) * np.sin(pitch / 2) * np.cos(yaw / 2) + np.sin(
            roll / 2
        ) * np.cos(pitch / 2) * np.sin(yaw / 2)
        qz = np.cos(roll / 2) * np.cos(pitch / 2) * np.sin(yaw / 2) - np.sin(
            roll / 2
        ) * np.sin(pitch / 2) * np.cos(yaw / 2)
        qw = np.cos(roll / 2) * np.cos(pitch / 2) * np.cos(yaw / 2) + np.sin(
            roll / 2
        ) * np.sin(pitch / 2) * np.sin(yaw / 2)
        return [qw, qx, qy, qz]

    @require_packages("hatbc")
    @staticmethod
    def nusc_submission_to_frames(json_path, meta_path, save_path=None):
        """Nusc format to hatbc frame struct."""
        frames = {}
        with open(json_path, "r") as f:
            results = json.load(f)["results"]
        with open(meta_path, "r") as f:
            meta = json.load(f)
        # assume samples are sorted by timestamp
        for sample_token, meta_dic in meta.items():
            scene_token = meta_dic["scene_token"]
            ego_pose = meta_dic["ego_pose_translation"]
            ego_rotation = Quaternion(meta_dic["ego_pose_rotation"]).inverse
            timestamp = meta_dic["timestamp"]
            meta = MessageMeta(
                timestamp=timestamp, device_id="nusc_minival_car"
            )
            sync_frames = SyncMessages(
                topic=scene_token,
                messages=[],
                meta=meta,
            )
            percs = []
            for box in results[sample_token]:
                ego_translation = ego_rotation.rotate(
                    [t1 - t2 for t1, t2 in zip(box["translation"], ego_pose)]
                )
                # convert center z to bottom z
                ego_translation[2] -= box["size"][2] / 2
                percs.append(
                    Instance(
                        topic=box["tracking_name"],
                        track_id=box["tracking_id"],
                        attributes=[
                            Attribute(
                                topic="tracking",
                                value=box["tracking_name"],
                                score=box["tracking_score"],
                            ),
                            Attribute(
                                topic="global_translation",
                                value=box["translation"],
                            ),
                            Attribute(
                                topic="global_rotation",
                                value=box["rotation"],
                            ),
                            Attribute(
                                topic="num_pts",
                                value=box.get("num_pts", -1),
                            ),
                        ],
                        bbox3ds=[
                            BBox3D(
                                dim=[
                                    box["size"][0],
                                    box["size"][2],
                                    box["size"][1],
                                ],  # wlh to whl
                                loc=ego_translation,
                                yaw=Quaternion(box["rotation"]).yaw_pitch_roll[
                                    0
                                ]
                                + ego_rotation.yaw_pitch_roll[0],
                            )
                        ],
                        meta=meta,
                        velocity=Velocity(
                            [box["velocity"][0], box["velocity"][1], 0]
                        ),
                    )
                )
            frame = Frame(topic="overall", meta=meta, perceptions=percs)
            sync_frames.messages.append(frame)
            # camera frame for vis
            for data_path, calib in zip(
                meta_dic["data_paths"], meta_dic["calibs"]
            ):
                yaw_pitch_roll = Quaternion(
                    calib["rotation"]
                ).inverse.yaw_pitch_roll
                camera = CameraFrame(
                    topic="camera",
                    image=Image(
                        url=data_path,
                        name=os.path.basename(data_path),
                    ),
                    camera_param=CameraParam(
                        focal_u=calib["camera_intrinsic"][0][0],
                        focal_v=calib["camera_intrinsic"][1][1],
                        center_u=calib["camera_intrinsic"][0][2],
                        center_v=calib["camera_intrinsic"][1][2],
                        camera_x=0,
                        camera_y=0,
                        camera_z=calib["translation"][2],
                        pitch=0,
                        yaw=0,
                        roll=0,
                        vcs=VCSParam(
                            # roll, pitch, yaw
                            rotation=[
                                yaw_pitch_roll[2] - math.pi / 2,
                                yaw_pitch_roll[1],
                                math.pi / 2 - yaw_pitch_roll[0],
                            ],
                            translation=[
                                calib["translation"][0],
                                calib["translation"][1],
                                0,
                            ],
                        ),
                        distort=[0.0] * 8,
                        image_height=calib["height"],
                        image_width=calib["width"],
                    ),
                    camera_type="PinholeCamera",
                    meta=MessageMeta(
                        timestamp=timestamp,
                        channel=NUSC_CAM[data_path.split("/")[0]],
                        device_id="nusc_minival_car",
                    ),
                )
                sync_frames.messages.append(camera)

            # add sync frames
            new_sample_token = "%s_%s" % (meta.device_id, meta.timestamp)
            frames[new_sample_token] = sync_frames
        if save_path:
            with open(save_path, "wb") as f:
                pkl.dump(frames, f)
        return frames


class ParallelTrackingEvaluation(TrackingEvaluation):
    """Parallel version of tracking eval.

    Args:
        tracks_gt: The ground-truth tracks.
        tracks_pred: The predicted tracks.
        class_name: The current class we are evaluating on.
        dist_fcn: The distance function used for evaluation.
        dist_th_tp: The distance threshold used to determine matches.
        min_recall: The minimum recall value below which we drop
            thresholds due to too much noise.
        num_thresholds: The number of recall thresholds from 0 to 1.
            Note that some of these may be dropped.
        num_workers: The number of parallel eval workers.
        scene_parallel: Parallel at scene level if True,
            otherwise at the score threshold level.
        metric_worst: Mapping from metric name to the fallback value
            assigned if a recall threshold is not achieved.
        verbose: Whether to print to stdout.
        output_dir: Output directory to save renders.
        render_classes: Classes to render to disk or None.
    """

    @require_packages(
        "nuscenes", raise_msg="Please `pip3 install nuscenes-devkit`"
    )
    def __init__(
        self,
        tracks_gt: Dict[str, Dict[int, List[TrackingBox]]],
        tracks_pred: Dict[str, Dict[int, List[TrackingBox]]],
        class_name: str,
        dist_fcn: Callable,
        dist_th_tp: float,
        min_recall: float,
        num_thresholds: int,
        num_workers: int,
        scene_parallel: bool,
        metric_worst: Dict[str, float],
        verbose: bool = True,
        output_dir: str = None,
        render_classes: List[str] = None,
    ):
        super().__init__(
            tracks_gt,
            tracks_pred,
            class_name,
            dist_fcn,
            dist_th_tp,
            min_recall,
            num_thresholds,
            metric_worst,
            verbose,
            output_dir,
            render_classes,
        )
        self.num_workers = num_workers
        self.scene_parallel = scene_parallel
        self.name_gen = name_gen

    def accumulate(self) -> TrackingMetricData:
        """Compute metrics for all recall thresholds of the current class."""
        # Init.
        if self.verbose:
            logging.info(
                "Computing metrics for class %s...\n" % self.class_name
            )
        accumulators = []
        thresh_metrics = []
        if self.num_workers > 0 and not self.scene_parallel:
            pool = Pool(self.num_workers)
            multi_args = []
            ret_list = []
        md = TrackingMetricData()

        # Skip missing classes.
        gt_box_count = 0
        gt_track_ids = set()
        for scene_tracks_gt in self.tracks_gt.values():
            for frame_gt in scene_tracks_gt.values():
                for box in frame_gt:
                    if box.tracking_name == self.class_name:
                        gt_box_count += 1
                        gt_track_ids.add(box.tracking_id)
        if gt_box_count == 0:
            # Do not add any metric. The average metrics will then be nan.
            return md

        # Register mot metrics.
        mh = create_motmetrics()

        # Get thresholds.
        # Note: The recall values are the hypothetical recall (10%, 20%, ..).
        # The actual recall may vary as there is no way to compute it
        # without trying all thresholds.
        thresholds, recalls = self.compute_thresholds(gt_box_count)
        md.confidence = thresholds
        md.recall_hypo = recalls
        if self.verbose:
            logging.info("Computed thresholds\n")

        for t, threshold in enumerate(thresholds):
            # If recall threshold is not achieved, we assign the
            # worst possible value in AMOTA and AMOTP.
            if np.isnan(threshold):
                continue

            # Do not compute the same threshold twice.
            # This becomes relevant when a user submits many boxes
            # with the exact same score.
            if threshold in thresholds[:t]:
                continue

            if self.num_workers > 0 and not self.scene_parallel:
                multi_args.append(threshold)
            else:
                acc, thresh_summary = self.compute_metric(mh, [threshold])
                accumulators.extend(acc)
                thresh_metrics.extend(thresh_summary)

        # gather results
        if (
            self.num_workers > 0
            and not self.scene_parallel
            and len(multi_args) > 0
        ):
            valid_thr_len = len(multi_args)
            step = math.ceil(valid_thr_len / self.num_workers)
            start_idx = 0
            while start_idx < valid_thr_len:
                ret = pool.apply_async(
                    self.compute_metric,
                    (mh, multi_args[start_idx : start_idx + step]),
                )
                ret_list.append(ret)
                start_idx += step

            for ret in tqdm.tqdm(ret_list, desc="MultiProcessingThresholds: "):
                acc, thresh_summary = ret.get()
                accumulators.extend(acc)
                thresh_metrics.extend(thresh_summary)

        # Concatenate all metrics. We only do this for more convenient access.
        if len(thresh_metrics) == 0:
            summary = []
        else:
            summary = pandas.concat(thresh_metrics)

        # Get the number of thresholds which were not achieved (i.e. nan).
        unachieved_thresholds = np.array(
            [t for t in thresholds if np.isnan(t)]
        )
        num_unachieved_thresholds = len(unachieved_thresholds)

        # Get the number of thresholds which were achieved (i.e. not nan).
        valid_thresholds = [t for t in thresholds if not np.isnan(t)]
        assert valid_thresholds == sorted(valid_thresholds)
        num_duplicate_thresholds = len(valid_thresholds) - len(
            np.unique(valid_thresholds)
        )

        # Sanity check.
        assert (
            num_unachieved_thresholds
            + num_duplicate_thresholds
            + len(thresh_metrics)
            == self.num_thresholds
        )

        # Figure out how many times each threshold should be repeated.
        rep_counts = [
            np.sum(thresholds == t) for t in np.unique(valid_thresholds)
        ]

        # Store all traditional metrics.
        for (mot_name, metric_name) in MOT_METRIC_MAP.items():
            # Skip metrics which we don't output.
            if metric_name == "":
                continue

            # Retrieve and store values for current metric.
            if len(thresh_metrics) == 0:
                # Set all the worst possible value if no
                # recall threshold is achieved.
                worst = self.metric_worst[metric_name]
                if worst == -1:
                    if metric_name == "ml":
                        worst = len(gt_track_ids)
                    elif metric_name in ["gt", "fn"]:
                        worst = gt_box_count
                    elif metric_name in ["fp", "ids", "frag"]:
                        worst = np.nan
                    else:
                        raise NotImplementedError

                all_values = [worst] * TrackingMetricData.nelem
            else:
                values = summary.get(mot_name).values
                assert np.all(values[np.logical_not(np.isnan(values))] >= 0)

                # If a threshold occurred more than once,
                # duplicate the metric values.
                assert len(rep_counts) == len(values)
                values = np.concatenate(
                    [([v] * r) for (v, r) in zip(values, rep_counts)]
                )

                # Pad values with nans for unachieved recall thresholds.
                all_values = [np.nan] * num_unachieved_thresholds
                all_values.extend(values)

            assert len(all_values) == TrackingMetricData.nelem
            md.set_metric(metric_name, all_values)

        return md

    def compute_metric(self, mh, thresholds: List[float]):
        accs, sums = [], []
        for threshold in thresholds:
            # Accumulate track data.
            acc, _ = self.accumulate_threshold(threshold)

            # Compute metrics for current threshold.
            thresh_name = self.name_gen(threshold)
            thresh_summary = mh.compute(
                acc, metrics=MOT_METRIC_MAP.keys(), name=thresh_name
            )
            if self.verbose:
                print_threshold_metrics(thresh_summary.to_dict())
            accs.append(acc)
            sums.append(thresh_summary)
        return accs, sums

    def accumulate_threshold(
        self, threshold: float = None
    ) -> Tuple[pandas.DataFrame, List[float]]:
        logging.info("in accumulate threshold")
        accs = []
        # The scores of the TPs. These are used to determine
        # the recall thresholds initially.
        scores = []
        if self.num_workers > 0 and self.scene_parallel:
            pool = Pool(self.num_workers)
            multi_args = []
            ret_list = []

        # Go through all frames and associate ground truth and tracker
        # results. Groundtruth and tracker contain lists for every single
        # frame containing lists detections.
        for scene_id in tqdm.tqdm(
            self.tracks_gt.keys(), disable=not self.verbose, leave=False
        ):
            # Retrieve GT and preds.
            scene_tracks_gt = self.tracks_gt[scene_id]
            scene_tracks_pred = self.tracks_pred[scene_id]

            if self.num_workers > 0 and self.scene_parallel:
                multi_args.append(
                    [
                        scene_tracks_gt,
                        scene_tracks_pred,
                        self.class_name,
                        threshold,
                        self.dist_fcn,
                        self.dist_th_tp,
                    ]
                )
            else:
                acc, score = scene_accumulate(
                    scene_tracks_gt,
                    scene_tracks_pred,
                    self.class_name,
                    threshold,
                    self.dist_fcn,
                    self.dist_th_tp,
                )
                accs.append(acc)
                scores.extend(score)
        # gather results
        if (
            self.num_workers > 0
            and self.scene_parallel
            and len(multi_args) > 0
        ):
            for arg in multi_args:
                ret = pool.apply_async(
                    scene_accumulate,
                    arg,
                )
                ret_list.append(ret)

            for ret in tqdm.tqdm(ret_list, desc="MultiProcessingScenes: "):
                acc, score = ret.get()
                accs.append(acc)
                scores.extend(score)

        # Merge accumulators
        acc_merged = MOTAccumulatorCustom.merge_event_dataframes(accs)

        return acc_merged, scores


@require_packages("hatbc")
def scene_accumulate(
    scene_tracks_gt,
    scene_tracks_pred,
    class_name,
    threshold,
    dist_fcn,
    dist_th_tp,
    raw_gt=None,
    raw_pred=None,
):
    """Accumulate mot result for each scene."""
    # Initialize accumulator and frame_id for this scene
    acc = MOTAccumulatorCustom()
    frame_id = 0  # Frame ids must be unique across all scenes
    scene_scores = []

    if raw_gt is not None:
        new_gt_ids = count()
        scene_gt_ids = {}
        sort_index = {
            "tp": 0,
            "fp": 0,
            "switch": 0,
            "fn": 0,
            "dist": 0,
        }

    for timestamp in scene_tracks_gt.keys():
        # Select only the current class.
        frame_gt = scene_tracks_gt[timestamp]
        frame_pred = scene_tracks_pred[timestamp]
        frame_gt = [f for f in frame_gt if f.tracking_name == class_name]
        frame_pred = [f for f in frame_pred if f.tracking_name == class_name]

        # Threshold boxes by score. Note that the scores were previously
        # averaged over the whole track.
        if threshold is not None:
            frame_pred = [
                f for f in frame_pred if f.tracking_score >= threshold
            ]

        # Abort if there are neither GT nor pred boxes.
        gt_ids = [gg.tracking_id for gg in frame_gt]
        pred_ids = [tt.tracking_id for tt in frame_pred]
        if len(gt_ids) == 0 and len(pred_ids) == 0:
            continue

        # Calculate distances.
        # Note that the distance function is hard-coded in Nuscenes to achieve
        # significant speedups via vectorization.
        # TODO: add relative distance.
        assert (
            dist_fcn == "center_distance"
            or dist_fcn.__name__ == "center_distance"
        )
        if len(frame_gt) == 0 or len(frame_pred) == 0:
            distances = np.ones((0, 0))
        else:
            gt_boxes = np.array([b.translation[:2] for b in frame_gt])
            pred_boxes = np.array([b.translation[:2] for b in frame_pred])
            distances = sklearn.metrics.pairwise.euclidean_distances(
                gt_boxes, pred_boxes
            )

        # Distances that are larger than the threshold won't be associated.
        assert len(distances) == 0 or not np.all(np.isnan(distances))
        distances[distances >= dist_th_tp] = np.nan

        # Accumulate results.
        # Note that we cannot use timestamp as frameid as motmetrics
        # assumes it's an integer.
        acc.update(gt_ids, pred_ids, distances, frameid=frame_id)

        # Store scores of matches, which are used to determine thresholds.
        if threshold is None:
            events = acc.events.loc[frame_id]
            matches = events[events.Type == "MATCH"]
            match_ids = matches.HId.values
            match_scores = [
                tt.tracking_score
                for tt in frame_pred
                if tt.tracking_id in match_ids
            ]
            scene_scores.extend(match_scores)
        else:
            events = None

        if raw_gt is not None:
            # Store scores for scene sorting.
            events = acc.events.loc[frame_id]
            sort_index["tp"] += events[events.Type == "MATCH"].shape[0]
            sort_index["fp"] += events[events.Type == "FP"].shape[0]
            sort_index["switch"] += events[events.Type == "SWITCH"].shape[0]
            sort_index["fn"] += events[events.Type == "MISS"].shape[0]
            sort_index["dist"] += sum(events[events.Type == "MATCH"].D.values)

            # Fillback matching results to raw gt and pred
            frame_raw_gt = raw_gt[timestamp]
            frame_raw_pred = raw_pred[timestamp]
            assert frame_raw_gt.meta.timestamp == timestamp, "%s, %s" % (
                frame_raw_gt.meta.timestamp,
                timestamp,
            )
            assert frame_raw_pred.meta.timestamp == timestamp
            inst_gt = {
                inst.track_id: inst
                for inst in frame_raw_gt.get_messages("overall")[
                    0
                ].get_instances(class_name)
            }
            inst_pred = {
                inst.track_id: inst
                for inst in frame_raw_pred.get_messages("overall")[
                    0
                ].get_instances(class_name)
            }

            matches = events[events.Type != "RAW"]
            pred_ids = [
                None if i == float("NaN") else i for i in matches.HId.values
            ]
            gt_ids = [None if i == -1 else i for i in matches.OId.values]
            vis_gt_ids = []
            for i in gt_ids:
                if i is None:
                    v_id = "-1"
                elif i in scene_gt_ids:
                    v_id = scene_gt_ids[i]
                else:
                    v_id = str(next(new_gt_ids))
                    scene_gt_ids[i] = v_id
                vis_gt_ids.append(v_id)

            match_types = matches.Type.values
            dists = matches.D.values
            # TODO: deal with interpolated boxes
            for g_id, p_id, v_id, mtype, dist in zip(
                gt_ids, pred_ids, vis_gt_ids, match_types, dists
            ):
                if g_id is not None and g_id in inst_gt:
                    inst_gt[g_id].attributes.append(
                        Attribute(
                            topic="asso",
                            value={
                                "match_id": p_id,
                                "match_type": GT_MAPPING[mtype],
                                "match_dist": dist,
                                "vis_id": v_id,
                            },
                        )
                    )
                if p_id is not None and p_id in inst_pred:
                    inst_pred[p_id].attributes.append(
                        Attribute(
                            topic="asso",
                            value={
                                "match_id": g_id,
                                "match_type": PRED_MAPPING[mtype],
                                "match_dist": dist,
                                "vis_id": v_id,
                            },
                        )
                    )

        # Increment the frame_id, unless there are no boxes
        # (equivalent to what motmetrics does).
        frame_id += 1

    if raw_gt is not None:
        return raw_gt, raw_pred, sort_index
    else:
        return acc, scene_scores
