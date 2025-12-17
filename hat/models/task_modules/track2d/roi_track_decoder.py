from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn

from hat.core.box_utils import bbox_overlaps
from hat.registry import OBJECT_REGISTRY

__all__ = ["TrackDecoder", "QuasiDenseEmbedTracker", "TrackPredict"]


eps = 1e-6


@OBJECT_REGISTRY.register
class TrackDecoder(nn.Module):
    """Decoder for Track."""

    def __init__(self):
        super().__init__()

    @torch.no_grad()
    def forward(
        self,
        batch_rois: List[torch.Tensor],
        head_out: Dict[str, torch.Tensor],
    ):
        batch_track_feat = head_out["track_feat"]

        bs = len(batch_rois)
        batch_track_feat = batch_track_feat.reshape(
            bs,
            -1,
            batch_track_feat.shape[1],
        )
        # batch tensor to List[tensor]
        pred_track_feats = [feat for feat in batch_track_feat]  # noqa: C416
        return {"pred_track_feats": pred_track_feats}


def np_softmax(x, dim=-1):
    x = np.exp(x)
    x_sum = x.sum(axis=dim)
    x_sum = np.expand_dims(x_sum, axis=dim)
    return x / x_sum


@OBJECT_REGISTRY.register
class QuasiDenseEmbedTracker(object):
    """QDTracker post-process.

    Args:
        init_score_thr: New bbox score bigger than init_score_thr would init a
            new tracklet.
        obj_score_thr: New bbox similarity bigger than match_tracklet_thr but
            score smaller or equal than obj_score_thr would be removal.
        match_similarity_thr: New bbox with tracklet similarity bigger than
            this value would been matched.
        memo_tracklet_frames: Tracklet with no new bbox matched for more this
            num frames would be removal from tracklet memory.
        memo_backdrop_frames: Backdrop tracklet keep number of this value
            frames.
        memo_momentum: Tracklet embedding momentum update.
        nms_similarity_thr: Min score and similarity with tracklet bigger than
            this value would be removal.
            (NMS, maybe some regress not well bbox overlap with the object).
        nms_backdrop_iou_thr: Do NMS on backdrop tracklet using this threshold.
        nms_class_iou_thr: Do NMS on high score bbox using this threshold.
        with_category: Class agnostic match or not.
        match_metric: Similarity metric, only support 'bisoftmax', 'softmax'
            and 'cosine'.
        with_bbox_dist: Whether to use bbox center distance to calculate
            similarity.
        match_bbox_dist_thr: Bbox center distance threshold, distance bigger
            than this value the similarity would be set to 0.
        match_appearance_ratio: Whether to use distance, final similarity mat
            would be  match_appearance_ratio * feat_sim_mat +
            (1.0 - match_appearance_ratio) * dist_sim_mat
    """

    def __init__(
        self,
        init_score_thr: float = 0.8,
        obj_score_thr: float = 0.5,
        match_similarity_thr: float = 0.5,
        memo_tracklet_frames: int = 10,
        memo_backdrop_frames: int = 1,
        memo_momentum: float = 0.8,
        nms_similarity_thr: float = 0.5,
        nms_backdrop_iou_thr: float = 0.3,
        nms_class_iou_thr: float = 0.5,
        with_category: bool = True,
        match_metric: str = "bisoftmax",
        with_bbox_dist: bool = False,
        match_bbox_dist_thr: float = 640.0,
        match_appearance_ratio: float = 0.8,
    ):
        assert 0 <= memo_momentum <= 1.0
        assert memo_tracklet_frames >= 0
        assert memo_backdrop_frames >= 0
        self.init_score_thr = init_score_thr
        self.obj_score_thr = obj_score_thr
        self.match_similarity_thr = match_similarity_thr
        self.memo_tracklet_frames = memo_tracklet_frames
        self.memo_backdrop_frames = memo_backdrop_frames
        self.memo_momentum = memo_momentum
        self.nms_similarity_thr = nms_similarity_thr
        self.nms_backdrop_iou_thr = nms_backdrop_iou_thr
        self.nms_class_iou_thr = nms_class_iou_thr
        self.with_category = with_category
        assert match_metric in ["bisoftmax", "softmax", "cosine"]
        self.match_metric = match_metric
        self.with_bbox_dist = with_bbox_dist
        self.match_bbox_dist_thr = match_bbox_dist_thr
        self.match_appearance_ratio = match_appearance_ratio

        self.num_tracklets = 0
        self.tracklets = {}
        self.backdrops = []

    def reset(self):
        """Reset the buffer of the tracker."""
        self.num_tracks = 0
        self.tracks = {}
        self.backdrops = []

    @property
    def empty(self):
        return False if self.tracklets else True

    def update_memo(self, ids, bboxes, embeds, labels, frame_id):
        tracklet_inds = ids > -1

        # update memo
        for id, bbox, embed, label in zip(
            ids[tracklet_inds],
            bboxes[tracklet_inds],
            embeds[tracklet_inds],
            labels[tracklet_inds],
        ):
            id = int(id)
            if id in self.tracklets.keys():
                # (matched) update tracklet
                velocity = (bbox - self.tracklets[id]["bbox"]) / (
                    frame_id - self.tracklets[id]["last_frame"]
                )
                self.tracklets[id]["bbox"] = bbox
                self.tracklets[id]["embed"] = (
                    1 - self.memo_momentum
                ) * self.tracklets[id]["embed"] + self.memo_momentum * embed
                self.tracklets[id]["last_frame"] = frame_id
                self.tracklets[id]["label"] = label
                self.tracklets[id]["velocity"] = (
                    self.tracklets[id]["velocity"]
                    * self.tracklets[id]["acc_frame"]
                    + velocity
                ) / (self.tracklets[id]["acc_frame"] + 1)
                self.tracklets[id]["acc_frame"] += 1
            else:
                # (no matched and no removal) init a new tracklet
                self.tracklets[id] = {
                    "bbox": bbox,
                    "embed": embed,
                    "label": label,
                    "last_frame": frame_id,
                    "velocity": np.zeros_like(bbox),
                    "acc_frame": 0,
                }

        backdrop_inds = np.nonzero(ids == -1)[0]
        ious = bbox_overlaps(bboxes[backdrop_inds, :-1], bboxes[:, :-1])
        # backdrop bbox NMS
        for i, ind in enumerate(backdrop_inds):
            if (ious[i, :ind] > self.nms_backdrop_iou_thr).any():
                backdrop_inds[i] = -1
        backdrop_inds = backdrop_inds[backdrop_inds > -1]

        self.backdrops.insert(
            0,
            {
                "bboxes": bboxes[backdrop_inds],
                "embeds": embeds[backdrop_inds],
                "labels": labels[backdrop_inds],
            },
        )

        # pop memo
        invalid_ids = []
        for k, v in self.tracklets.items():
            # tracklet disappeared > self.memo_tracklet_frames would be removal
            if frame_id - v["last_frame"] >= self.memo_tracklet_frames:
                invalid_ids.append(k)
        for invalid_id in invalid_ids:
            self.tracklets.pop(invalid_id)

        # old backdrops would be removal
        if len(self.backdrops) > self.memo_backdrop_frames:
            self.backdrops.pop()

    @property
    def memo(self):
        memo_embeds = []
        memo_ids = []
        memo_bboxes = []
        memo_labels = []
        memo_vs = []
        for k, v in self.tracklets.items():
            memo_bboxes.append(v["bbox"][None, :])
            memo_embeds.append(v["embed"][None, :])
            memo_ids.append(k)
            memo_labels.append(v["label"].reshape(1, 1))
            memo_vs.append(v["velocity"][None, :])
        memo_ids = np.array(memo_ids, dtype="int64").reshape(1, -1)

        for backdrop in self.backdrops:
            backdrop_ids = np.full(
                (1, backdrop["embeds"].shape[0]), -1, dtype="int64"
            )
            backdrop_vs = np.zeros_like(backdrop["bboxes"])
            memo_bboxes.append(backdrop["bboxes"])
            memo_embeds.append(backdrop["embeds"])
            memo_ids = np.concatenate([memo_ids, backdrop_ids], axis=1)
            memo_labels.append(backdrop["labels"][:, None])
            memo_vs.append(backdrop_vs)

        memo_bboxes = np.concatenate(memo_bboxes, axis=0)
        memo_embeds = np.concatenate(memo_embeds, axis=0)
        memo_labels = np.concatenate(memo_labels, axis=0).squeeze(1)
        memo_vs = np.concatenate(memo_vs, axis=0)
        return (
            memo_bboxes,
            memo_labels,
            memo_embeds,
            memo_ids.squeeze(0),
            memo_vs,
        )

    def match(
        self,
        bboxes: np.ndarray,
        labels: np.ndarray,
        embeds: np.ndarray,
        frame_id: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Track match.

        New objects do match with temporary tracklet.

        Args:
            bboxes: Pred boxes, shape (num_bbox, 5), each bbox
                is (x1, y1, x2, y2, score).
            labels: Pred boxes classes, shape (num_bbox, ).
            pred_scores: Pred boxes scores, shape (num_bbox, ).
            embeds: Pred track feat embeds, shape (num_bbox, emb_len).
            frame_id: Frame index.

        Returns::

            bboxes: Ndarray shape (num_bbox, 5), each bbox
                is (x1, y1, x2, y2, score).
            labels: Ndarray shape (num_bbox, ).
            ids: Ndarray shape (num_bbox, ).
            sim_ret: Ndarray shape (num_bbox, ), similarity with
                previous frame objects, no matched boxes similarity
                is 0.0 (include the first frame objects).

        """
        inds = np.argsort(bboxes[:, -1])[::-1]
        assert (
            inds - np.array(range(bboxes.shape[0]), dtype=inds.dtype)
        ).sum() == 0, "bboxes should be sorted before matched!"
        bboxes = bboxes[inds, :]
        labels = labels[inds]
        embeds = embeds[inds, :]

        # duplicate removal for potential backdrops and cross classes
        # removal behavior is same with NMS,
        # difference is use dynamic IOU threshold accordding to the bbox score
        valids = np.ones((bboxes.shape[0]))
        ious = bbox_overlaps(bboxes[:, :-1], bboxes[:, :-1])
        for i in range(1, bboxes.shape[0]):
            # if bbox score < self.obj_score_thr (FP / background),
            # low NMS IOU threshhold,
            # otherwise, (cross foreground / TP) high NMS IOU threshold.
            thr = (
                self.nms_backdrop_iou_thr
                if bboxes[i, -1] < self.obj_score_thr
                else self.nms_class_iou_thr
            )
            # NMS removal (set some bbox valid to 0)
            if (ious[i, :i] > thr).any():
                valids[i] = 0
        valids = valids == 1
        bboxes = bboxes[valids, :]
        labels = labels[valids]
        embeds = embeds[valids, :]

        # init ids container
        ids = np.full((bboxes.shape[0],), -1, dtype="int64")
        sim_ret = np.full((bboxes.shape[0],), 0.0, dtype="float")

        # match if buffer is not empty
        if bboxes.shape[0] > 0 and not self.empty:
            (
                memo_bboxes,
                memo_labels,
                memo_embeds,
                memo_ids,
                memo_vs,
            ) = self.memo

            if self.match_metric == "bisoftmax":
                feats = np.dot(embeds, memo_embeds.T)
                d2t_sim_mat = np_softmax(feats, dim=1)
                t2d_sim_mat = np_softmax(feats, dim=0)
                sim_mat = (d2t_sim_mat + t2d_sim_mat) / 2
            elif self.match_metric == "softmax":
                feats = np.dot(embeds, memo_embeds.T)
                sim_mat = np_softmax(feats, dim=1)
            elif self.match_metric == "cosine":
                sim_mat = np.dot(
                    embeds
                    / (
                        np.linalg.norm(embeds, ord=2, axis=1, keepdims=True)
                        + eps
                    ),
                    (
                        memo_embeds
                        / (
                            np.linalg.norm(
                                memo_embeds, ord=2, axis=1, keepdims=True
                            )
                            + eps
                        )
                    ).T,
                )
            else:
                raise NotImplementedError

            # if bbox class is different with tracklet class, set this bbox similarity to zero  # noqa
            if self.with_category:
                cat_same = labels.reshape(-1, 1) == memo_labels.reshape(1, -1)
                sim_mat *= cat_same

            # bbox distance
            if self.with_bbox_dist:
                bboxes_cx, bboxes_cy = (bboxes[:, 2] - bboxes[:, 0]) / 2.0, (
                    bboxes[:, 3] - bboxes[:, 1]
                ) / 2.0
                memo_bboxes_cx, memo_bboxes_cy = (
                    memo_bboxes[:, 2] - memo_bboxes[:, 0]
                ) / 2.0, (memo_bboxes[:, 3] - memo_bboxes[:, 1]) / 2.0
                dist = np.sqrt(
                    (bboxes_cx[:, None] - memo_bboxes_cx[None, :]) ** 2
                    + (bboxes_cy[:, None] - memo_bboxes_cy[None, :]) ** 2
                )
                dist_sim_mat = 1.0 - dist / self.match_bbox_dist_thr
                # dist_sim_mat[dist_sim_mat < 0.0] = 0.0
                sim_mat = (
                    self.match_appearance_ratio * sim_mat
                    + (1.0 - self.match_appearance_ratio) * dist_sim_mat
                )
                sim_mat[sim_mat < 0.0] = 0.0

            for i in range(bboxes.shape[0]):
                memo_ind = np.argmax(sim_mat[i, :], axis=0)
                similarity = sim_mat[i, memo_ind]
                id = memo_ids[memo_ind]
                sim_ret[i] = similarity
                if similarity > self.match_similarity_thr:
                    # bbox matched with tracklet
                    if id > -1:
                        # if bboxe score > obj_score_thr, really matched,
                        # otherwise, remove. (background)
                        if bboxes[i, -1] > self.obj_score_thr:
                            ids[i] = id
                            sim_mat[:i, memo_ind] = 0
                            sim_mat[i + 1 :, memo_ind] = 0
                        else:
                            if similarity > self.nms_similarity_thr:
                                ids[i] = -2
        # no removal & no matched bbox & bbox score>init_score_thr
        # would be set to a new tracklet id
        new_inds = (ids == -1) & (bboxes[:, 4] > self.init_score_thr)
        num_news = new_inds.sum()
        ids[new_inds] = np.arange(
            self.num_tracklets, self.num_tracklets + num_news, dtype="int64"
        )
        self.num_tracklets += num_news

        # matched bbox would be used to update the tracklet status (embedding & velocity)  # noqa
        # no matched bbox and new tracklet id bbox (score>init_score_thr) would be used init a new tracklet # noqa
        # no NMS removal & no matched & score<init_score_thr, would be set to backdrop.  # noqa
        self.update_memo(ids, bboxes, embeds, labels, frame_id)

        return bboxes, labels, ids, sim_ret


@OBJECT_REGISTRY.register
class TrackPredict(object):
    """Track2d predictor.

    Wrapper for track2d, given a tracker and per frame track2d feature,
    get the track result (track_id).

    Args:
        tracker: tracker to used to do track.
    """

    def __init__(self, tracker: Any):
        self.tracker = tracker

    def __call__(
        self,
        frame_index: int,
        pred_boxes: np.ndarray,
        pred_classes: np.ndarray,
        pred_scores: np.ndarray,
        pred_embeds: np.ndarray,
    ) -> Dict:
        """Track predictor.

        Args:
            frame_index: Frame index, shold start from 1,
                which if follwing the rules of MOT benchmark.
            pred_boxes: Pred boxes, shape (num_bbox, 4).
            pred_classes: Pred boxes classes, shape (num_bbox, ).
            pred_scores: Pred boxes scores, shape (num_bbox, ).
            pred_embeds: Pred track feat embeds, shape (num_bbox, emb_len).

        Returns:
            Track result dict include the following keys:
                pred_track_ids: Ndarray shape (num_bbox, ).
                pred_track_classes: Ndarray shape (num_bbox, ).
                pred_track_boxes: Ndarray shape (num_bbox, 5),
                    each bbox is (x1, y1, x2, y2, score).
                pred_track_similarity: Ndarray shape (num_bbox, ).
                    similarity with previous frame objects, no matched
                    boxes similarity is 0.0 (include the first frame objects).
        """

        # first frame index is 1 in sequence
        assert frame_index > 0, (
            "frame index should start from 1, which is following"
            "the rules of MOT benchmark"
        )
        if frame_index == 1:
            self.tracker.reset()
        result = {}
        if pred_boxes.shape[0] == 0:
            result.update(
                pred_track_ids=np.empty(0),
                pred_track_classes=np.empty(0),
                pred_track_boxes=np.empty((0, 5)),  # (x1,y1,x2,y2,score)
                pred_track_similarity=np.empty(0),
            )
            return result

        # match tracklet
        bboxes, labels, ids, similarity = self.tracker.match(
            bboxes=np.hstack((pred_boxes, pred_scores.reshape(-1, 1))),
            labels=pred_classes,
            embeds=pred_embeds,
            frame_id=frame_index,
        )
        result.update(
            pred_track_ids=ids,
            pred_track_classes=labels,
            pred_track_boxes=bboxes,  # (x1,y1,x2,y2,score)
            pred_track_similarity=similarity,
        )
        return result
