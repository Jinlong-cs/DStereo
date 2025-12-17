# Copyright (c) Horizon Robotics. All rights reserved.

import os

import pytest
import torch
from tqdm import tqdm

from hat.callbacks.traj_pred_viz import TrajPredViz
from hat.data.collates.traj_pred_collates import collate_multipath_viz
from hat.data.datasets.traj_pred_dataset import BaseTrajDataset
from hat.models.task_modules.traj_pred.post_process import (
    MultipathPostProcessor,
)
from projects.prediction.configs.processed_dataset import HAT_UNITTEST_PREFIX
from tests import SD_AlGORITHM_BUCKET_PATH  # , SD_AlGORITHM_BUCKET_EXISTS
from tests.unit_tests.core.test_traj_pred_utils import (  # noqa: E501
    gen_autourban_example_data_loader,
)
from tests.unit_tests.models.task_modules.traj_pred.structures.test_multipath import (  # noqa: E501
    gen_example_multipath_model_v2,
    gen_example_roadmap_color_table,
)


@pytest.mark.skipif(True, reason="requiring SD_Algorithm bucket")
def test_traj_pred_viz_callback(tmpdir):
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    tdt_dir = os.path.join(
        SD_AlGORITHM_BUCKET_PATH,
        HAT_UNITTEST_PREFIX,
        "datasets/auto_urban_8days/hdfs_save/csv_trans/concat_tdt/",  # noqa: E501
    )
    tdt_yaml_dir = os.path.join(
        SD_AlGORITHM_BUCKET_PATH,
        HAT_UNITTEST_PREFIX,
        "datasets/auto_urban_8days/hdfs_save/csv_trans/concat_tdt/tdt_meta.yaml",  # noqa: E501
    )
    img_dir = os.path.join(SD_AlGORITHM_BUCKET_PATH, "mengyuan")
    viz_dir = tmpdir

    # Build dataloader.
    viz_dataloader = gen_autourban_example_data_loader(
        tdt_dir, tdt_yaml_dir, img_dir, enable_high_freq=True
    )
    viz_dataloader.collate_fn = collate_multipath_viz
    # Build model.
    anchor_file = os.path.join(
        SD_AlGORITHM_BUCKET_PATH,
        HAT_UNITTEST_PREFIX,
        "anchors/ZGC_anchors_kmeans_dirnorm_add_lc48_remove_straight70_fixbug.pkl",  # noqa: E501
    )
    anchor_num = 106
    viz_head_name_dict = {
        "traj_head": "traj_head",
        "track_valid_head": "track_valid_head",
    }
    post_process_dict = {
        "traj_head": MultipathPostProcessor(
            filter_reverse=True,
            use_traj_nms=False,
            nms_params=dict(
                nms_threshold=0.05,
                nms_prob_threshold=0.01,
                nms_num_trajs=10,
                change_probs=False,
            ),
        ),
    }
    color_table = gen_example_roadmap_color_table()
    model = gen_example_multipath_model_v2(
        anchor_file,
        anchor_num,
        color_table,
        post_process_dict=post_process_dict,
    )
    model = model.cuda()
    model.fuse_model()
    model.set_qconfig()

    bev_origin_x, bev_origin_y, img_resolution = 72.4, 51.2, 0.2
    callback = TrajPredViz(
        head_names=viz_head_name_dict,
        height=512,
        width=512,
        resolution=img_resolution,
        map_origin=[bev_origin_x, bev_origin_y],
        video_save_path=viz_dir,
        video_name="event_test",
        video_fps=10,
        viz_on_every_epoch=True,
        ego_track_id=-BaseTrajDataset.ANSWER,
        num_traj_to_viz=5,
        num_workers=4,
    )

    model.eval()
    model.cuda()
    with torch.no_grad():
        callback.clear()
        for batch in tqdm(viz_dataloader):
            output = model(batch)
            callback.collect(batch, output)
        callback.render()
    assert os.path.exists(os.path.join(viz_dir, "event_test.mp4"))
