import copy
import logging
import math
import multiprocessing as mp
import os
import os.path as osp
import time
from collections import defaultdict

import numpy as np
from scipy.optimize import linear_sum_assignment

from hat.core.box3d_utils import rotate_iou
from hat.core.rotate_box_utils import let_iou_2d

logger = logging.getLogger(__name__)
num_sample_pts = 41  # num of sample points in trk evaluation.
MAX_SUBPROCESS_NUM = 10  # max num of thread in evaluation.
EPS = 1e-9


class MetaData:
    """Utility class to load data."""

    def __init__(self, load_groundtruth):
        """Initialize the object given the parameters."""
        self.load_groundtruth = load_groundtruth

    def add_targets(self, fields):
        self.frame = int(float(fields[0]))
        self.track_id = int(float(fields[1]))
        self.obj_type = fields[2]
        self.truncation = int(float(fields[3]))
        self.occlusion = int(float(fields[4]))
        self.obs_angle = float(fields[5])
        self.x1 = float(fields[6])
        self.y1 = float(fields[7])
        self.x2 = float(fields[8])
        self.y2 = float(fields[9])
        self.h = float(fields[10])
        self.w = float(fields[11])
        self.len = float(fields[12])
        self.X = float(fields[13])
        self.Y = float(fields[14])
        self.Z = float(fields[15])
        self.yaw = float(fields[16])
        self.valid = False
        if not self.load_groundtruth:
            self.score = float(fields[17]) if len(fields) > 17 else -1

    def get_box(self):
        return np.array(
            [
                (self.x1 + self.x2) / 2.0,
                (self.y1 + self.y2) / 2.0,
                (self.x2 - self.x1),
                (self.y2 - self.y1),
                -self.yaw,
            ]
        )


class MetricVars:
    """Utility class to save metrics."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.tp = self.itp = self.fn = self.ifn = 0
        self.fp = self.igt = self.itr = self.overlap = 0
        self.igt_itr = 0  # matched but both ignored
        self.n_gt = self.n_trk = 0
        self.scores = []
        self.ignored_fn = self.ignored_tp = 0
        self.ignored_trackers_num = 0
        self.ignored_pairs = 0
        self.trajectories = None
        self.ignored_flag_on_traj = None

        self.MODP_t = []
        self.MT = self.ML = self.PT = 0
        self.IDs = self.Frag = 0
        self.ids_list = []
        self.frag_list = []

    def summary(self, metric_val=None):
        if metric_val is None:
            self.tp -= self.ignored_tp
            self.fn -= self.ignored_fn
            self.itp += self.ignored_tp
            self.ifn += self.ignored_fn
            self.n_gt -= self.ignored_fn + self.ignored_tp
            self.igt += self.ignored_fn + self.ignored_tp
            self.itr += self.ignored_trackers_num
            return (
                self.tp >= 0
                and self.fn >= 0
                and self.fp >= 0
                and (
                    self.n_gt
                    == self.tp + self.fn + self.ignored_fn + self.ignored_tp
                )
                and (
                    self.n_trk
                    == self.tp
                    + self.fp
                    + self.ignored_trackers_num
                    + self.ignored_tp
                    - self.ignored_pairs
                )
            )
        else:
            self.tp += metric_val.tp
            self.fn += metric_val.fn
            self.itp += metric_val.itp
            self.ifn += metric_val.ifn
            self.n_gt += metric_val.n_gt
            self.n_trk += metric_val.n_trk
            self.igt += metric_val.igt
            self.itr += metric_val.itr
            self.igt_itr += metric_val.igt_itr
            self.fp += metric_val.fp
            self.overlap += metric_val.overlap


class TrackingEvaluation(object):
    """Tracking statistics.

    (CLEAR MOT, id-switches, fragments, ML/PT/MT, precision/recall)
    MOTA   - Multi-object tracking accuracy in [0,100]
    MOTP   - Multi-object tracking precision in [0,100] (3D) / [td,100] (2D)
    MOTAL  - Multi-object tracking accuracy in [0,100] with
            log10(id-switches)

    id-switches - number of id switches
    fragments   - number of fragmentations

    MT, PT, ML - number of mostly tracked, partially tracked and mostly lost
                trajectories

    recall         - recall = percentage of detected targets
    precision      - precision = percentage of correctly detected targets
    FAR            - number of false alarms per frame
    falsepositives - number of false positives (FP)
    missed         - number of missed targets (FN)

    Args:
        gt_dir: path of gt.
        result_dir: path of predicion result.
        class_names: class names.
        eval_2diou: whether eval 2d task.
        max_truncation: maximum truncation level of an object.
        min_area: minimum area of a valid prediction box.
        max_occlusion: maximum occlusion level of an object.
        min_overlap_2d: minimum overlap ratio for 2d evaluation.
        min_overlap_3d: minimum overlap ratio for 3d evaluation.
        eval_mode: the way used to count the metric, in ["bev_iou",
            "let_iou"]
        let_iou_param: the parameters used to calculate the let_iou metric.
    """

    def __init__(
        self,
        gt_dir: str,
        result_dir: str,
        class_names: str,
        eval_2diou: bool = False,
        max_truncation: int = 0,
        min_area: float = 0.2,
        max_occlusion: int = 2,
        min_overlap_2d: float = 0.5,
        min_overlap_3d: float = 0.25,
        eval_mode: str = "bev_iou",
        let_iou_param: dict = None,
    ):
        self.gt_dir = gt_dir
        self.result_dir = result_dir
        self.class_names = class_names
        self.eval_mode = eval_mode
        self.let_iou_param = let_iou_param

        # Get the info of the packs need to be evaluated.
        filename_test_mapping = osp.join(gt_dir, "evaluate_tracking.seqmap")
        assert osp.exists(
            filename_test_mapping
        ), "The seqmap file should be provided first."
        with open(filename_test_mapping, "r") as f:
            lines = f.readlines()
            self.sequence_names = [
                line.split(" ")[0].strip() for line in lines
            ]
            self.sequence_frames = [
                int(line.split(" ")[-1].strip())
                - int(line.split(" ")[-2].strip())
                + 1
                for line in lines
            ]
            self.sequence_start_frame_idx = [
                int(line.split(" ")[-2].strip()) - 1 for line in lines
            ]
            self.sequence_num = len(self.sequence_names)
        self.result_dir = result_dir
        # Now the subdir is constant, as the output dir in model is constant.
        self.pred_txt_dir = osp.join(result_dir, "kf_dt")
        assert osp.exists(
            self.pred_txt_dir
        ), f"The dir {self.pred_txt_dir} is not existed"
        self.gt_txt_dir = osp.join(result_dir, "kf_gt")
        assert osp.exists(
            self.gt_txt_dir
        ), f"The dir {self.gt_txt_dir} is not existed"
        self.ids_save_file = osp.join(
            self.result_dir, f"metric/ids_{self.class_names}.json"
        )  # used to save id switch
        self.frg_save_file = osp.join(
            self.result_dir, f"metric/frg_{self.class_names}.json"
        )  # used to save fragment

        self.min_overlap = min_overlap_2d if eval_2diou else min_overlap_3d
        self.eval_2diou = eval_2diou
        self.max_truncation = max_truncation
        self.max_occlusion = max_occlusion
        self.min_area = min_area
        self.reset()

    def load_trajectory(self, loading_groundtruth: bool):
        """Load ground truth.

        Args:
        loading_groudtruth: load the groundtruth or track result.

        """
        try:
            load_state = self._load_data(
                self.gt_txt_dir
                if (loading_groundtruth)
                else self.pred_txt_dir,
                cls=self.class_names,
                loading_groundtruth=loading_groundtruth,
            )
            return load_state
        except IOError:
            return False

    @staticmethod
    def parallel_wrapper_load_data(
        seq,
        s_name,
        loading_groundtruth,
        root_dir,
        cls,
        seq_frames,
        seq_frames_start_idx,
        return_dict,
    ):
        """Func to load data parallel.

        Args:
            seq: seq index.
            s_name: the name of sequence.
            loading_groudtruth: load the groundtruth or track result.
            root_dir: data root.
            cls: class name.
            seq_frames: the number of frames in the sequence.
            seq_frames_start_idx: the start index of the sequence.
            return_dict: the dict to save the result.
        """
        logger.info(f"Load data from {s_name} using thread: {seq}...")
        t_data = MetaData(loading_groundtruth)
        filename = osp.join(root_dir, f"{s_name}.txt")
        frame_data = [[] for x in range(seq_frames)]
        with open(filename, "r") as f:
            lines = f.readlines()

        classes = [cls.lower(), "dontcare"]
        id_frame_cache = []
        track_ids = []
        trajectories_num = 0
        for line in lines:
            # KITTI tracking benchmark data format:  中间间隔符是空格，
            # 3D的话x1,y1,x2,y2 均为-1
            # (frame,tracklet_id,objType,truncation,occlusion,alpha,x1,
            # y1,x2,y2,h,w,length,X,Y,Z,ry)
            fields = line.strip().split()
            if fields[2].lower() not in classes or (int(fields[1]) == -1):
                continue
            t_data.add_targets(fields)

            id_frame = (t_data.frame, t_data.track_id)
            assert id_frame not in id_frame_cache, (
                "Error! The track ids should be unique. "
                + f"seq_name: {s_name},  is_gt: {loading_groundtruth}"
                + f"frame: {t_data.frame}, track_id: {t_data.track_id}"
            )
            id_frame_cache.append(id_frame)
            try:
                # NOTE: some sequences may not start from frame_idx=0
                frame_data[t_data.frame - seq_frames_start_idx].append(
                    copy.copy(t_data)
                )
            except Exception:
                logger.error("The frame idx is larger than the frame num.")

            if (
                t_data.track_id not in track_ids
                and t_data.obj_type != "dontcare"
            ):
                track_ids.append(t_data.track_id)
                trajectories_num += 1
        return_dict[seq] = [frame_data, trajectories_num]

    def _load_data(self, root_dir, cls, loading_groundtruth=False):
        """Load detections in KITTI format from textfiles."""

        seq_data = []
        # the trajectories num PER SEQUENCE
        seq_n_trajectories = []

        thread_list = []
        manager = mp.Manager()
        return_dict = manager.dict()
        for i in range(self.sequence_num):
            thread = mp.Process(
                target=self.parallel_wrapper_load_data,
                args=(
                    i,
                    self.sequence_names[i],
                    loading_groundtruth,
                    root_dir,
                    cls,
                    self.sequence_frames[i],
                    self.sequence_start_frame_idx[i],
                    return_dict,
                ),
            )
            thread_list.append(thread)
            thread.start()
        for sub_thread in thread_list:
            sub_thread.join()

        for i in range(self.sequence_num):
            seq_data.append(return_dict[i][0])
            seq_n_trajectories.append(return_dict[i][1])

        if not loading_groundtruth:
            self.tracker = seq_data
            self.trajectories_num_all = sum(seq_n_trajectories)
            self.trajectories_num_per_seq = seq_n_trajectories
            if self.trajectories_num_all <= 0:
                logger.info(f"There is no valid trajectories in {root_dir}")
                return False
        else:
            # split ground truth and DontCare areas
            self.groundtruth, self.dcares = [], []
            for seq_gt in seq_data:
                s_g, s_dc = [], []
                for seq_frame_gt in seq_gt:
                    f_g, f_dc = [], []
                    for gg in seq_frame_gt:
                        if gg.obj_type == "dontcare":
                            f_dc.append(gg)
                        else:
                            f_g.append(gg)
                    s_g.append(f_g)
                    s_dc.append(f_dc)
                self.groundtruth.append(s_g)
                self.dcares.append(s_dc)
            self.gt_trajectories_num_all = sum(seq_n_trajectories)
            self.gt_trajectories_num_per_seq = seq_n_trajectories
        return True

    def get_threshold(self, scores, num_gt, num_sample_pts=41):
        """
        Discretize the recall to get the corresponding threshold.

        Based on score of true positive to discretize the recall.

        Args:
            scores: the list of the scores of the true positives.
            num_gt: the all num of gt

        """
        scores = np.array(scores)
        scores.sort()
        scores = scores[::-1]
        current_recall = 0
        num_gt = float(num_gt)
        thresholds, recalls = [], []
        num_pos = len(scores)
        for i in range(num_pos):
            # 将recall 划分41 分
            l_recall = (i + 1) / num_gt
            r_recall = min(i + 2, num_pos) / num_gt
            if (r_recall - current_recall) < (current_recall - l_recall) and (
                i < num_pos - 1
            ):
                continue
            thresholds.append(scores[i])
            recalls.append(current_recall)
            current_recall += 1 / (num_sample_pts - 1.0)
        return thresholds[1:], recalls[1:]

    def reset(self):
        self.n_ignored_tr_total = 0
        self.fp = self.fn = self.n_gt = self.tp = 0
        self.MOTA = self.MOTP = self.MOTAL = 0
        self.sMOTA = self.MODA = self.MODP = 0
        self.recall = self.precision = 0
        self.F1 = self.FAR = 0
        self.total_overlap = 0
        self.fragments = self.id_switches = 0
        self.MT = self.PT = self.ML = 0
        self.scores = []
        self.itp = self.ifn = self.n_igt = 0
        self.n_tr = self.n_itr = 0

    @staticmethod
    def compute_seq_metrics(
        seq_idx: int = 0,
        thresh: float = -10000,
        input_dict: dict = None,
        output_dict: dict = None,
        eval_mode: str = "bev_iou",
        let_iou_param: dict = None,
    ):
        """
        Compute the tracking metrics on each sequence.

        Args:
            seq_idx: The index of sequence
            thresh: Used to filter the trajectories.
                Only keep the predict trajectories whose avg score larger
                than the thresh.

        """
        logger.info(
            f"compute seq metric on {seq_idx} process @thresh:{thresh}..."
        )
        # initialize the metrics in seq
        seq_metrics = MetricVars()
        last_ids = [[], []]
        ids_list, frg_list = [], []
        # index the trajectory by track_id, and anno whether the prediction
        # on each frame is ignored.
        seq_metrics.trajectories = defaultdict(list)
        seq_metrics.ignored_flag_on_traj = defaultdict(list)

        # filter out the tracklet with average score large than the
        # threshold, and replacing confidence with average score.
        tracker_id_score = {}
        # seq_gt (List(List(MetaData))): gt objects in each frame.
        # seq_dc (List(List(MetaData))): dont care objects in each frame.
        # seq_tracker_before (List(List(MetaData))): naive output from model.
        seq_gt = input_dict["seq_gt"]
        seq_dc = input_dict["seq_dc"]
        seq_tracker_before = input_dict["seq_tracker_before"]

        for tracks_tmp in seq_tracker_before:
            for trk_tmp in tracks_tmp:
                id_tmp = trk_tmp.track_id
                score_tmp = trk_tmp.score
                if id_tmp not in tracker_id_score:
                    tracker_id_score[id_tmp] = []
                tracker_id_score[id_tmp].append(score_tmp)

        id_avg_score = {}
        to_delete_id = []
        for trk_id, score_list in tracker_id_score.items():
            avg_score = sum(score_list) / len(score_list) * 1.0
            id_avg_score[trk_id] = avg_score
            if avg_score < thresh:
                to_delete_id.append(trk_id)
        seq_tracker = []
        for tracks_tmp in seq_tracker_before:
            seq_tracker.append([])
            for trk_tmp in tracks_tmp:
                id_tmp = trk_tmp.track_id
                if id_tmp not in to_delete_id:
                    trk_tmp.score = id_avg_score[id_tmp]
                    seq_tracker[-1].append(trk_tmp)

        def _get_overlap(gt_boxes, pred_boxes, eval_mode, **let_iou_param):
            if eval_mode == "let_iou":
                if gt_boxes.size > 0 and pred_boxes.size > 0:
                    let_rotate_ious, al = let_iou_2d(
                        gt_boxes, pred_boxes, **let_iou_param
                    )
                    return let_rotate_ious * al
                else:
                    return np.array([[]])
            elif eval_mode == "bev_iou":
                return rotate_iou(gt_boxes, pred_boxes)
            else:
                raise NotImplementedError

        for frame_idx in range(len(seq_gt)):
            # initialize the metrics in frame, used to check data
            frame_metrics = MetricVars()
            gt = seq_gt[frame_idx]
            dc = seq_dc[frame_idx]
            pred = seq_tracker[frame_idx]
            frame_metrics.n_gt = len(gt)
            frame_metrics.n_trk = len(pred)

            # build the cost matrix and get the match index with
            # hungarian method. row is gt, column is det
            gt_boxes = np.array([gt_.get_box() for gt_ in gt])
            pred_boxes = np.array([trk.get_box() for trk in pred])
            cost_matrix = 1 - _get_overlap(
                gt_boxes, pred_boxes, eval_mode, **let_iou_param
            )
            asso_indices = linear_sum_assignment(cost_matrix)
            # init, each gt traj corresponding match index is -1
            matched_track_ids = [
                [g.track_id for g in gt],
                [-1] * len(gt),
            ]  # gt's trackid and its correspondint pred track id
            for gg in gt:
                # init for update
                gg.tracker = -1
                gg.id_switch = 0
                gg.fragmentation = 0
                seq_metrics.ignored_flag_on_traj[gg.track_id].append(False)
                seq_metrics.trajectories[gg.track_id].append(-1)

            for row, col in zip(*asso_indices):
                c = cost_matrix[row][col]
                if c < 1 - input_dict["min_overlap"]:
                    gt[row].tracker = pred[col].track_id
                    pred[col].valid = True
                    matched_track_ids[1][row] = pred[col].track_id
                    gt[row].distance = c
                    frame_metrics.overlap += 1 - c
                    gt[row].overlap = 1 - c
                    seq_metrics.trajectories[gt[row].track_id][-1] = pred[
                        col
                    ].track_id

                    frame_metrics.tp += 1
                    seq_metrics.scores.append(pred[col].score)
                else:
                    gt[row].tracker = -1
                    frame_metrics.fn += 1

            # associate tracker with DontCare areas
            ignored_trackers = defaultdict(list)
            dc_boxes = np.array([dc_b.get_box() for dc_b in dc])
            for tt in pred:
                if tt.valid:
                    continue
                # area approxmation
                tt_area = abs(tt.x1 - tt.x2) * abs(tt.y1 - tt.y2)
                if tt_area < input_dict["min_area"]:
                    ignored_trackers[tt.track_id] = 1
                    frame_metrics.ignored_trackers_num += 1
                    tt.ignored = True
                    continue
                else:
                    ignored_trackers[tt.track_id] = -1
                overlap = _get_overlap(
                    np.array([tt.get_box()]),
                    dc_boxes,
                    eval_mode,
                    **let_iou_param,
                )
                if overlap.size > 0 and (
                    overlap.min() > input_dict["min_overlap"] and not tt.valid
                ):
                    # not asso to gt, but asso to dont care area, then ignore
                    tt.ignored = True
                    frame_metrics.ignored_trackers_num += 1
                    ignored_trackers[tt.track_id] = 1

            # check for ignored FN/TP
            for gg in gt:
                if (
                    gg.occlusion > input_dict["max_occlusion"]
                    or gg.truncation > input_dict["max_truncation"]
                ):
                    seq_metrics.ignored_flag_on_traj[gg.track_id][-1] = True
                    gg.ignored = True
                    if gg.tracker < 0:
                        frame_metrics.ignored_fn += 1
                    else:
                        frame_metrics.ignored_tp += 1
                        frame_metrics.overlap -= gg.overlap
                    if gg.tracker >= 0 and (ignored_trackers[gg.tracker] > 0):
                        frame_metrics.ignored_pairs += 1
            # ignored_tp: already associated, but should be ignored
            # ignored_fn: already missed, but should be ignored

            frame_metrics.fp = (
                len(pred)
                - frame_metrics.tp
                - frame_metrics.ignored_trackers_num
                - frame_metrics.ignored_tp
                + frame_metrics.igt_itr
            )
            frame_metrics.fn += len(gt) - len(asso_indices[0])
            assert frame_metrics.summary(), (
                "Some Error in the data."
                + f"seq_name: {input_dict['seq_name']}, "
                + f"frame_idx: {frame_idx}"
            )

            seq_metrics.summary(frame_metrics)

            # check for id switches or Fragmentations
            # frag will be more than id switch, switch happens only
            # when id is different but detection exists
            # frag happens when id switch or detection is missing
            for i, tt in enumerate(matched_track_ids[0]):
                # tt is the gt traj id
                if tt in last_ids[0]:
                    idx = last_ids[0].index(tt)
                    lid = last_ids[1][idx]
                    tid = matched_track_ids[1][i]

                    if tid != lid and lid != -1 and tid != -1:
                        gt[i].id_switch = 1
                        if thresh < 0:
                            ids_list.append([seq_idx, frame_idx, lid, tid])
                    elif tid != lid and lid != -1:
                        gt[i].fragmentation = 1
                        if thresh < 0:
                            frg_list.append([seq_idx, frame_idx - 1, lid])

            # save current index
            last_ids = matched_track_ids
            # compute MODP_t
            MODP_t = (
                frame_metrics.overlap / frame_metrics.tp
                if (frame_metrics.tp > 0)
                else 1
            )
            seq_metrics.MODP_t.append(MODP_t)

        seq_metrics.ids_list = ids_list
        seq_metrics.frag_list = frg_list
        # compute MT/PT/ML, fragments, idswitches
        if len(seq_metrics.trajectories) == 0:
            output_dict[seq_idx] = seq_metrics
            return
        for g, ign_g in zip(
            seq_metrics.trajectories.values(),
            seq_metrics.ignored_flag_on_traj.values(),
        ):
            if all(ign_g):
                seq_metrics.ignored_trackers_num += 1
                continue
            if all([this == -1 for this in g]):
                seq_metrics.ML += 1
                continue
            # compute tracked frames in trajectory
            last_id = g[0]
            # the number of matched obj in cur traj
            tracked = 1 if g[0] >= 0 else 0
            # the valid number of objs in cur_traj
            len_gt = 0 if ign_g[0] else 1
            for f in range(1, len(g)):
                if ign_g[f]:
                    last_id = -1
                    continue
                len_gt += 1
                if (
                    last_id != g[f]
                    and last_id != -1
                    and g[f] != -1
                    and g[f - 1] != -1
                ):
                    # matched success in both frames, but id change.
                    seq_metrics.IDs += 1
                if (
                    f < len(g) - 1
                    and g[f - 1] != g[f]
                    and last_id != -1
                    and g[f] != -1
                    and g[f + 1] != -1
                ):
                    seq_metrics.Frag += 1
                if g[f] != -1:
                    tracked += 1
                    last_id = g[f]
            # the last frame of each traj
            if (
                len(g) > 1
                and g[f - 1] != g[f]
                and last_id != -1
                and g[f] != -1
                and not ign_g[f]
            ):
                seq_metrics.Frag += 1

            # compute MT/PT/ML
            tracking_ratio = tracked / float(len(g) - sum(ign_g))
            if tracking_ratio > 0.8:
                seq_metrics.MT += 1
            elif tracking_ratio < 0.2:
                seq_metrics.ML += 1
            else:  # 0.2 <= tracking_ratio <= 0.8
                seq_metrics.PT += 1
        output_dict[seq_idx] = seq_metrics

    def compute_3rd_party_metrics(self, thresh=-100, recall_thresh=1.0):
        """
        Compute the tracking metrics.

            - Stiefelhagen 2008: Evaluating Multiple Object Tracking
                Performance: The CLEAR MOT Metrics
              MOTA, MOTAL, MOTP
            - Nevatia 2008: Global Data Association for Multi-Object
                Tracking Using Network Flows
              MT/PT/ML

        """

        ids_list, frag_list, MODP_t = [], [], []

        input_dict = [
            {
                "seq_gt": self.groundtruth[i],
                "seq_dc": self.dcares[i],
                "seq_tracker_before": self.tracker[i],
                "min_overlap": self.min_overlap,
                "min_area": self.min_area,
                "max_occlusion": self.max_occlusion,
                "max_truncation": self.max_truncation,
                "seq_name": self.sequence_names[i],
            }
            for i in range(self.sequence_num)
        ]

        manager = mp.Manager()
        return_dict = manager.dict()
        max_subprocess_num = min(MAX_SUBPROCESS_NUM, self.sequence_num)
        group = self.sequence_num // max_subprocess_num
        for group_idx in range(group + 1):
            process_indices = range(
                group_idx * max_subprocess_num,
                min((group_idx + 1) * max_subprocess_num, self.sequence_num),
            )
            pool_list = []
            for seq_idx in process_indices:
                thread = mp.Process(
                    target=self.compute_seq_metrics,
                    args=(
                        seq_idx,
                        thresh,
                        copy.deepcopy(input_dict[seq_idx]),
                        return_dict,
                        self.eval_mode,
                        self.let_iou_param,
                    ),
                )
                pool_list.append(thread)
                thread.start()
            for thread in pool_list:
                thread.join()

        for seq_idx in range(self.sequence_num):
            seq_metrics = return_dict[seq_idx]
            self.n_ignored_tr_total += seq_metrics.ignored_trackers_num
            self.MT += seq_metrics.MT
            self.ML += seq_metrics.ML
            self.PT += seq_metrics.PT
            self.fp += seq_metrics.fp
            self.tp += seq_metrics.tp
            self.fn += seq_metrics.fn
            self.itp += seq_metrics.itp
            self.ifn += seq_metrics.ifn
            self.n_igt += seq_metrics.igt
            self.n_tr += seq_metrics.n_trk
            self.n_itr += seq_metrics.itr
            self.id_switches += seq_metrics.IDs
            self.fragments += seq_metrics.Frag
            self.n_gt += seq_metrics.n_gt
            self.total_overlap += seq_metrics.overlap
            ids_list.extend(seq_metrics.ids_list)
            frag_list.extend(seq_metrics.frag_list)
            MODP_t.extend(seq_metrics.MODP_t)
            self.scores.extend(seq_metrics.scores)

        valid_gt_num = self.gt_trajectories_num_all - self.n_ignored_tr_total
        self.MT = self.MT / valid_gt_num if valid_gt_num > 0 else 0
        self.PT = self.PT / valid_gt_num if valid_gt_num > 0 else 0
        self.ML = self.ML / valid_gt_num if valid_gt_num > 0 else 0
        self.recall = self.tp / (self.tp + self.fn + 1e-9)
        self.precision = self.tp / (self.tp + self.fp + 1e-9)
        self.F1 = (
            2.0
            * self.precision
            * self.recall
            / (self.precision + self.recall + 1e-9)
        )
        self.FAR = self.fp / sum(self.sequence_frames)
        self.MOTA = 1 - (self.fn + self.fp + self.id_switches) / float(
            self.n_gt + 1e-9
        )
        self.MODA = 1 - (self.fn + self.fp) / float(self.n_gt + 1e-9)
        self.sMOTA = min(
            1,
            max(
                0,
                1
                - (
                    self.fn
                    + self.fp
                    + self.id_switches
                    - (1 - recall_thresh) * self.n_gt
                )
                / float(recall_thresh * self.n_gt + 1e-9),
            ),
        )
        self.MOTP = self.total_overlap / float(self.tp + 1e-9)
        LOG_IDs = math.log10(self.id_switches) if self.id_switches > 0 else 0
        self.MOTAL = (
            1 - (self.fn + self.fp + LOG_IDs) / float(self.n_gt)
            if self.n_gt > 0
            else "inf"
        )
        self.MODP = sum(MODP_t) / sum(self.sequence_frames)
        self.num_gt = self.tp + self.fn
        if thresh < 0:
            import json

            with open(self.ids_save_file, "w") as f:
                json.dump(ids_list, f)
            with open(self.frg_save_file, "w") as f:
                json.dump(frag_list, f)
        return True

    # format output string
    def print_entry(self, key, val, width=(70, 10)):
        """Pretty print an entry in a table fashion."""

        s_out = key.ljust(width[0])
        if type(val) == int:
            s = "%%%dd" % width[1]
            s_out += s % val
        elif type(val) == float:
            s = "%%%d.4f" % (width[1])
            s_out += s % val
        else:
            s_out += ("%s" % val).rjust(width[1])
        return s_out

    def create_summary_details(self):
        """
        Generate and mail a summary of the results.

        If mailpy.py is present, the summary is instead printed.
        """

        summary = "\n"

        summary += "evaluation: single summary".center(80, "=") + "\n"
        summary += (
            self.print_entry(
                "Multiple Object Tracking Accuracy (MOTA)", self.MOTA
            )
            + "\n"
        )
        summary += (
            self.print_entry(
                "Multiple Object Tracking Precision (MOTP)", float(self.MOTP)
            )
            + "\n"
        )
        summary += (
            self.print_entry(
                "Multiple Object Tracking Accuracy (MOTAL)", self.MOTAL
            )
            + "\n"
        )
        summary += (
            self.print_entry(
                "Multiple Object Detection Accuracy (MODA)", self.MODA
            )
            + "\n"
        )
        summary += (
            self.print_entry(
                "Multiple Object Detection Precision (MODP)", float(self.MODP)
            )
            + "\n"
        )
        summary += "\n"
        summary += self.print_entry("Recall", self.recall) + "\n"
        summary += self.print_entry("Precision", self.precision) + "\n"
        summary += self.print_entry("F1", self.F1) + "\n"
        summary += self.print_entry("False Alarm Rate", self.FAR) + "\n"
        summary += "\n"
        summary += self.print_entry("Mostly Tracked", self.MT) + "\n"
        summary += self.print_entry("Partly Tracked", self.PT) + "\n"
        summary += self.print_entry("Mostly Lost", self.ML) + "\n"
        summary += "\n"
        summary += self.print_entry("True Positives", self.tp) + "\n"
        summary += self.print_entry("Ignored True Positives", self.itp) + "\n"
        summary += self.print_entry("False Positives", self.fp) + "\n"
        summary += self.print_entry("False Negatives", self.fn) + "\n"
        summary += self.print_entry("Ignored False Negatives", self.ifn) + "\n"
        summary += self.print_entry("ID-switches", self.id_switches) + "\n"
        summary += self.print_entry("Fragmentations", self.fragments) + "\n"
        summary += "\n"
        summary += (
            self.print_entry(
                "Ground Truth Objects (Total)", self.n_gt + self.n_igt
            )
            + "\n"
        )
        summary += (
            self.print_entry("Ignored Ground Truth Objects", self.n_igt) + "\n"
        )
        summary += (
            self.print_entry(
                "Ground Truth Trajectories", self.gt_trajectories_num_all
            )
            + "\n"
        )
        summary += "\n"
        summary += (
            self.print_entry("Tracker Objects (Total)", self.n_tr) + "\n"
        )
        summary += (
            self.print_entry("Ignored Tracker Objects", self.n_itr) + "\n"
        )
        summary += (
            self.print_entry("Tracker Trajectories", self.trajectories_num_all)
            + "\n"
        )
        summary += "=" * 80
        summary += "\n"

        return summary

    def create_summary_simple(self, threshold, recall, metrics_dict=None):
        """
        Generate and mail a summary of the results.

        If mailpy.py is present, the summary is instead printed.
        """

        summary = ""

        summary += (
            "\n"
            + (
                "evaluation with score: %f, expected recall: %f"
                % (threshold, recall)
            ).center(80, "=")
            + "\n"
        )

        summary += "{0: ^7}{1: ^7}{2: ^7}{3: ^7}{4: ^7}{5: ^7}{6: ^7}{7: ^7}{8: ^7}{9: ^7}{10: ^7}{11: ^7}{12: ^7}{13: ^7}\n".format(  # noqa
            "sMOTA",
            "MOTA",
            "MOTP",
            "MT",
            "ML",
            "IDS",
            "FRAG",
            "F1",
            "Prec",
            "Recall",
            "FAR",
            "TP",
            "FP",
            "FN",
        )

        if metrics_dict is None:
            summary += "{:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:5d} {:5d} {:.4f} {:.4f} {:.4f} {:.4f} {:5d} {:5d} {:5d}\n".format(  # noqa
                self.sMOTA,
                self.MOTA,
                self.MOTP,
                self.MT,
                self.ML,
                self.id_switches,
                self.fragments,
                self.F1,
                self.precision,
                self.recall,
                self.FAR,
                self.tp,
                self.fp,
                self.fn,
            )
        else:
            summary += "{:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:5d} {:5d} {:.4f} {:.4f} {:.4f} {:.4f} {:5d} {:5d} {:5d}\n".format(  # noqa
                metrics_dict["sMOTA"],
                metrics_dict["mota"],
                metrics_dict["motp"],
                metrics_dict["MT"],
                metrics_dict["ML"],
                metrics_dict["id_switches"],
                metrics_dict["fragments"],
                metrics_dict["F1"],
                metrics_dict["precision"],
                metrics_dict["recall"],
                metrics_dict["FAR"],
                metrics_dict["tp"],
                metrics_dict["fp"],
                metrics_dict["fn"],
            )
        summary += "=" * 80

        return summary

    def get_stats_string(self, threshold=None, recall=None, metrics_dict=None):
        """Save the statistics in a whitespace separate file."""

        if threshold is None:
            summary = self.create_summary_details()
        else:
            summary = self.create_summary_simple(
                threshold, recall, metrics_dict
            )
        return summary


def calculate_mot_metric_per_thresh(i, e, thresh, recall, return_dict):
    logger.info(f"computing metric @ {i}th thresh: {thresh}...")
    pe = copy.deepcopy(e)
    pe.reset()
    pe.compute_3rd_party_metrics(thresh, recall)
    data_tmp = {
        "mota": pe.MOTA,
        "motp": pe.MOTP,
        "sMOTA": pe.sMOTA,
        "MT": pe.MT,
        "ML": pe.ML,
        "id_switches": pe.id_switches,
        "fragments": pe.fragments,
        "F1": pe.F1,
        "precision": pe.precision,
        "recall": pe.recall,
        "FAR": pe.FAR,
        "tp": pe.tp,
        "fp": pe.fp,
        "fn": pe.fn,
        "moda": pe.MODA,
        "summary": pe.get_stats_string(),
    }
    return_dict[i] = data_tmp


def e2e_dynamic_bevtracking_eval(
    gt_dir="",
    result_dir="",
    predefined_classes=("car",),
    eval_mode="bev_iou",
    let_iou_param=None,
    iou_threshold=0.2,
):
    """
    Entry point for tracking evaluation in nuscenes format.

    It will load the data and start evaluation for the classes
    predefined in predefined_classes.
    """
    logger.info("Processing Result for Tracking Benchmark in Nuscenes format.")
    os.makedirs(os.path.join(result_dir, "metric"), exist_ok=True)
    trk_metric_all = {}
    for c in predefined_classes:
        # Evaluate each class.
        # Init the tracker evaluator
        trk_metric_all.update(
            {
                c: {
                    "sAMOTA": 0,
                    "AMOTA": 0,
                    "AMOTP": 0,
                }
            }
        )
        start_time = time.perf_counter()
        e = TrackingEvaluation(
            gt_dir,
            result_dir,
            class_names=c,
            eval_mode=eval_mode,
            let_iou_param=let_iou_param,
            min_overlap_3d=iou_threshold,
        )
        if not e.load_trajectory(loading_groundtruth=False):
            continue
        if not e.load_trajectory(loading_groundtruth=True):
            continue
        assert len(e.groundtruth) == len(
            e.tracker
        ), "The number of gt and pred sequences should be equal."
        e.compute_3rd_party_metrics()
        best_mota = 0
        thresh_list, recall_list = e.get_threshold(e.scores, e.num_gt)

        thread_num = len(thresh_list)
        if thread_num == 0:
            continue
        pool_list = []
        manager = mp.Manager()
        return_dict = manager.dict()

        for i in range(thread_num):
            thread = mp.Process(
                target=calculate_mot_metric_per_thresh,
                args=(i, e, thresh_list[i], recall_list[i], return_dict),
            )
            pool_list.append(thread)
            thread.start()
        for thread in pool_list:
            thread.join()

        mota, motp, sMOTA = 0, 0, 0
        summary = ""
        best_thresh_str = ""
        for idx, (threshold_tmp, recall_tmp) in enumerate(
            zip(thresh_list, recall_list)
        ):
            data_tmp = {}
            data_tmp = return_dict[idx]
            mota_tmp = data_tmp["mota"]
            mota += data_tmp["mota"]
            motp += data_tmp["motp"]
            sMOTA += data_tmp["sMOTA"]
            if mota_tmp > best_mota:
                best_mota = mota_tmp
                best_thresh_str = return_dict[idx]["summary"]
            summary += e.get_stats_string(threshold_tmp, recall_tmp, data_tmp)

        summary += best_thresh_str
        summary += ("\n evaluation: average over recall").center(
            80, "="
        ) + "\n"
        summary += "sAMOTA  AMOTA  AMOTP \n"

        summary += "{:.4f} {:.4f} {:.4f}\n".format(
            sMOTA / (num_sample_pts - 1.0),
            mota / (num_sample_pts - 1.0),
            motp / (num_sample_pts - 1.0),
        )
        summary += "=" * 80
        time7 = time.perf_counter()
        logger.info(f"time on class: [{c}]: {time7 - start_time}")
        logger.info(summary)

        with open(
            os.path.join(result_dir, "trk_results_{}.txt".format(c)),
            "w",
        ) as fwr:
            fwr.write(summary)
        trk_metric_all.update(
            {
                c: {
                    "sAMOTA": sMOTA / (num_sample_pts - 1.0),
                    "AMOTA": mota / (num_sample_pts - 1.0),
                    "AMOTP": motp / (num_sample_pts - 1.0),
                }
            }
        )
    return trk_metric_all
