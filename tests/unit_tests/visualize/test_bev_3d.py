import numpy as np
import torch

from hat.visualize.bev_3d import Bev3DVisualize


def get_fake_output():
    output = {
        "bev3d_dim": torch.rand((1, 100, 3)),
        "bev3d_rot": torch.randn((1, 100)),
        "bev3d_loc_z": torch.rand((1, 100)),
        "bev3d_cls_id": torch.randint(0, 2, (1, 100)),
        "bev3d_score": torch.rand((1, 100)),
        "bev3d_ct": torch.randn((1, 100, 2)).to(bool),
    }

    return output


def test_vis_bev_3d():
    torch.manual_seed(0)
    output = get_fake_output()
    output_np = {}
    for k, v in output.items():
        output_np[k] = v.cpu().numpy()
    bs = 0
    pred_bboxes = np.concatenate(
        [
            output_np["bev3d_ct"][bs],
            output_np["bev3d_loc_z"][bs][:, None],
            output_np["bev3d_dim"][bs],
            output_np["bev3d_rot"][bs][:, None],
        ],
        axis=1,
    )
    class_id = output_np["bev3d_cls_id"][bs][:, None]
    score = output_np["bev3d_score"][bs][:, None]
    bev_img = np.zeros((512, 512, 3))

    bev_boxes_img = Bev3DVisualize.draw_bev_boxes(
        bev_img=bev_img,
        pred_bboxes=pred_bboxes,
        class_id=class_id,
        score=score,
        bev_size=(512, 512),
        bev_range=(-30.0, -51.2, 72.4, 51.2),
        score_threshold=2,
        thickness=2,
        color=(0, 0, 255),  # red
    )

    assert bev_boxes_img.shape == (512, 512, 3)
