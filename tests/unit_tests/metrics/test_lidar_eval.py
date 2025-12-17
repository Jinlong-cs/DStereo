import os

import pytest

from hat.registry import build_from_registry
from tests import HAT_BUCKET_EXISTS, HAT_BUCKET_PATH


@pytest.mark.skipif(not HAT_BUCKET_EXISTS, reason="need HAT bucket")
def testLidarDetEval():

    data_root = os.path.join(
        HAT_BUCKET_PATH, "unit_test_data/lidar/Datasets_m1"
    )
    work_dir = os.path.join(data_root, "HAT_SAVE")
    val_anno = "info_training_all_2022-06-17.pkl"
    classes = ["Vehicle", "Cyclist", "Pedestrian"]
    label_to_name = dict()
    for i in range(len(classes)):
        label_to_name[i] = classes[i]

    matching = dict(
        dir_name="IoUMatching",
        class_names=classes,
        iou_thresholds=[0.7, 0.5, 0.5],
    )
    id2name = label_to_name
    gt_classes_map = dict(
        Car="Vehicle",
        Cyclist="Cyclist",
        Tricycle="Vehicle",
        Pedestrian="Pedestrian",
        Construction="Vehicle",
        Truck="Vehicle",
        Bus="Vehicle",
    )
    workflow = [
        dict(
            type="DetEvaluation",
            af_work_dir=work_dir,
            range_limit=[
                [0, 1000],
                [0, 20],
                [20, 50],
                [50, 80],
                [80, 120],
                [120, 150],
            ],
            bev_metric=False,
            vis_pr_curve=False,
            failure_case=False,
            filter_by_x=False,
            result_name="aph_result",
        )
    ]
    gt_pkl = os.path.join(data_root, val_anno)
    test_orientation = ("rear", "front")
    test_rear_range = [[0, 1000], [0, 10], [10, 20], [20, 30]]
    save_pkl_path = work_dir
    lidar_eval = dict(
        type="LidarDetEval",
        id2name=id2name,
        gt_classes_map=gt_classes_map,
        gt_pkl=gt_pkl,
        work_dir=data_root,
        matching=matching,
        workflow=workflow,
        test_orientation=test_orientation,
        save_pkl_path=save_pkl_path,
        test_rear_range=test_rear_range,
    )
    lidar_eval = build_from_registry(lidar_eval)

    lidar_eval._init_states()
    # TODO test on_batch_end by shijie.sun
    # The train config has not been determined yet,
    # so we have not model outputs to test on_batch_end
    # and aph result is right or not.
