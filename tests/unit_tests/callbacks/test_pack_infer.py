import os

import cv2
import numpy as np

from hat.callbacks.pack_infer import ANCSaveVisualize


class FackVis:
    def __init__(
        self,
        range_mode,
    ):
        self.range_mode = range_mode

    def __call__(self, batch, model_outs, task, save_path):
        cv2.imwrite(os.path.join(save_path, "test.png"), model_outs)


def test_save_visualize(tmpdir):

    save_dir = os.path.join(tmpdir, "example-save-vis/")

    vis = ANCSaveVisualize(
        task_list=["test_task"],
        visualize_list=[
            [FackVis(range_mode="wide"), FackVis(range_mode="small")]
        ],
        save_dir=save_dir,
    )
    batch = ({}, "test_task")
    model_outs = np.random.randint(0, 255, (512, 960, 3), dtype=np.uint8)
    vis.on_batch_end(batch, model_outs)

    assert os.path.exists(
        os.path.join(tmpdir, "example-save-vis/test_task_wide/test.png")
    )
    assert os.path.exists(
        os.path.join(tmpdir, "example-save-vis/test_task_small/test.png")
    )
