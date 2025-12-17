import numpy as np
import pytest

from hat.models.task_modules.track2d.roi_track_decoder import (
    QuasiDenseEmbedTracker,
    TrackPredict,
)


def test_track_predict():
    track_output_list = [
        dict(
            frame_index=1,
            pred_boxes=np.array(
                [
                    [200.0, 200.0, 500.0, 500.0],
                ]
            ),
            pred_classes=np.array([1.0]),
            pred_scores=np.array([0.99]),
            pred_embeds=np.array(
                [
                    [1.0] * 64,
                ]
            ),
        ),
        dict(
            frame_index=2,
            pred_boxes=np.array(
                [
                    [200.0, 200.0, 500.0, 500.0],
                ]
            ),
            pred_classes=np.array([1.0]),
            pred_scores=np.array([0.99]),
            pred_embeds=np.array(
                [
                    [1.0] * 64,
                ]
            ),
        ),
        dict(
            frame_index=3,
            pred_boxes=np.array(
                [
                    [2000.0, 2000.0, 5000.0, 5000.0],
                ]
            ),
            pred_classes=np.array([1.0]),
            pred_scores=np.array([0.99]),
            pred_embeds=np.array(
                [
                    [0.0] * 64,
                ]
            ),
        ),
        dict(
            frame_index=4,
            pred_boxes=np.array(
                [
                    [200.0, 200.0, 500.0, 500.0],
                ]
            ),
            pred_classes=np.array([1.0]),
            pred_scores=np.array([0.1]),
            pred_embeds=np.array(
                [
                    [1.0] * 64,
                ]
            ),
        ),
        dict(
            frame_index=5,
            pred_boxes=np.array(
                [
                    [200.0, 200.0, 500.0, 500.0],
                ]
            ),
            pred_classes=np.array([1.0]),
            pred_scores=np.array([0.99]),
            pred_embeds=np.array(
                [
                    [1.0] * 64,
                ]
            ),
        ),
    ]

    qdtracker = QuasiDenseEmbedTracker(
        init_score_thr=0.9,
        obj_score_thr=0.7,
        match_similarity_thr=0.25,
        memo_tracklet_frames=20,
        memo_backdrop_frames=1,
        memo_momentum=0.8,
        nms_similarity_thr=0.25,
        nms_backdrop_iou_thr=0.1,
        nms_class_iou_thr=0.35,
        with_category=True,
        # match_metric='bisoftmax')
        match_metric="cosine",
        with_bbox_dist=True,
        match_bbox_dist_thr=640,
        match_appearance_ratio=0.8,
    )

    track_predict = TrackPredict(qdtracker)

    for frame_net_out in track_output_list:
        frame_index = frame_net_out["frame_index"]
        pred_boxes = frame_net_out["pred_boxes"]
        pred_classes = frame_net_out["pred_classes"]
        pred_scores = frame_net_out["pred_scores"]
        pred_embeds = frame_net_out["pred_embeds"]
        track_result = track_predict(
            frame_index=frame_index,
            pred_boxes=pred_boxes,
            pred_classes=pred_classes,
            pred_scores=pred_scores,
            pred_embeds=pred_embeds,
        )
        print(track_result)
        if frame_index == 3:
            # new object
            assert track_result["pred_track_ids"][0] == 1
        elif frame_index == 4:
            # object score is too low to match with previous object.
            assert track_result["pred_track_ids"][0] < 0
        else:
            # predict track id init from 0
            assert track_result["pred_track_ids"][0] == 0


if __name__ == "__main__":
    pytest.main(["-s", __file__])
