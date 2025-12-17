import numpy as np
import pytest
import torch

from hat.models.task_modules.motr import MotrPostProcess
from tests.unit_tests.models.task_modules.motr.test_criterion import (
    generate_empty_tracks,
)

input_size = (800, 1422)


def test_motr_post_process():

    motr_post_process = MotrPostProcess()

    outputs_classes_head = []
    outputs_coords_head = []
    out_hs = torch.randn((1, 256, 4, 128))
    for _ in range(6):
        outputs_classes_head.append(torch.randn((1, 1, 4, 128)))
        outputs_coords_head.append(torch.randn((1, 4, 4, 128)))

    track_instance = generate_empty_tracks(256, 256)
    empty_track_instance = generate_empty_tracks(256, 256)
    fake_track_instance = generate_empty_tracks(256, 256)

    seq_data = dict(
        img=[torch.randn((1, 3, input_size[0], input_size[1]))],
        query_pos=torch.randn((1, 256, 2, 128), dtype=torch.float),
        mask_query=torch.ones((1, 1, 1, 256), dtype=torch.float),
        ref_pts=torch.randn((1, 4, 2, 128), dtype=torch.float),
        gt_bboxes=[
            torch.tensor(
                [[10, 10, 20, 20], [20, 20, 40, 40]], dtype=torch.float
            ),
        ],
        gt_classes=[torch.tensor([0, 0], dtype=torch.float)],
        gt_ids=[torch.tensor([0, 1], dtype=torch.float)],
        img_name=["test_img.jpg"],
        seq_name=["test_seq"],
        img_shape=[
            np.array([3, input_size[0], input_size[1]]),
        ],
        scale_factor=torch.tensor(
            [1.4253, 1.4264, 1.4253, 1.4264], dtype=torch.float
        ),
    )

    frame_id = 0
    seq_frame_id = 0
    seq_name = seq_data["seq_name"][0]
    motr_post_process.eval()

    outputs = motr_post_process(
        track_instance,
        empty_track_instance,
        fake_track_instance,
        out_hs,
        outputs_classes_head,
        outputs_coords_head,
        seq_data=seq_data,
        frame_id=frame_id,
        seq_frame_id=seq_frame_id,
        seq_name=seq_name,
    )

    assert len(outputs) == 3


if __name__ == "__main__":
    pytest.main(["-s", __file__])
