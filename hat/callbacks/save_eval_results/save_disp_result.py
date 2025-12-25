import os
import pickle
import numpy as np
import cv2
from hat.callbacks.callbacks import CallbackMixin
from hat.registry import OBJECT_REGISTRY
from DStereo.common import disp2rgb, uncert2rgb, view_infer_result


@OBJECT_REGISTRY.register
class SaveDisp(CallbackMixin):
    def __init__(
        self,
        task_name,
        maxdisp=192,
        output_dir=None,
        color=(252, 247, 192),
    ):
        self.task_name = task_name
        self.maxdisp = maxdisp
        self.color = color
        self.output_dir = output_dir
        if output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)

    def unpad(self, l, r, d, pred, uncert, input_size):
        if list(l.shape[:2]) != input_size:
            height, width = input_size
            new_height = (height + 31) // 32 * 32
            new_width = (width + 31) // 32 * 32
            top_pad = (new_height - height) // 2
            bottom_pad = new_height - height - top_pad
            left_pad = (new_width - width) // 2
            right_pad = new_width - width - left_pad

            l = l[top_pad : top_pad + height, left_pad : left_pad + width, :]
            r = r[top_pad : top_pad + height, left_pad : left_pad + width, :]
            d = d[top_pad : top_pad + height, left_pad : left_pad + width]
            pred = pred[top_pad : top_pad + height, left_pad : left_pad + width]
            uncert = uncert[top_pad : top_pad + height, left_pad : left_pad + width]

        return l, r, d, pred, uncert

    def on_batch_end(self, batch, model_outs, train_metrics, **kwargs):
        for sample_idx in range(len(batch["dataset_name"])):
            left = batch["left_img"][sample_idx]
            right = batch["right_img"][sample_idx]
            disp_gt = batch["gt_disp"][sample_idx].detach().cpu().numpy()
            pred = model_outs[0][sample_idx, ...].detach().cpu().numpy()
            initdisp = model_outs[1][sample_idx, ...].detach().cpu().numpy()
            left, right, disp_gt, pred, initdisp = self.unpad(
                left, right, disp_gt, pred, initdisp, batch["origin_shape"][sample_idx]
            )

            view_gt = disp2rgb(disp_gt, self.maxdisp, 1)
            view_pred = disp2rgb(pred, self.maxdisp, 1)
            cv2.imwrite(
                os.path.join(
                    self.output_dir,
                    os.path.splitext(batch["left_img_name"][sample_idx])[0].split("/")[
                        -1
                    ]
                    + "_left.png",
                ),
                left,
            )
            cv2.imwrite(
                os.path.join(
                    self.output_dir,
                    os.path.splitext(batch["left_img_name"][sample_idx])[0].split("/")[
                        -1
                    ]
                    + "_right.png",
                ),
                right,
            )
            cv2.imwrite(
                os.path.join(
                    self.output_dir,
                    os.path.splitext(batch["left_img_name"][sample_idx])[0].split("/")[
                        -1
                    ]
                    + "_gt.png",
                ),
                view_gt,
            )
            cv2.imwrite(
                os.path.join(
                    self.output_dir,
                    os.path.splitext(batch["left_img_name"][sample_idx])[0].split("/")[
                        -1
                    ]
                    + "_pred.png",
                ),
                view_pred,
            )
          

@OBJECT_REGISTRY.register
class SaveCalibdata(CallbackMixin):
    """
    Save OccFlow result for visualization.

    Args:
        output_dir: Output dir for saving results.
    """

    def __init__(
        self,
        output_dir: str,
    ):
        self.output_dir = output_dir
        os.makedirs(os.path.join(output_dir, "infra1"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "infra2"), exist_ok=True)

    def on_batch_begin(self, batch, global_step_id, **kwargs):
        assert (
            len(batch["dataset_name"]) == 1
        ), "test_batch_size_per_gpu in DStereo/DStereoPlus.py must be 1"
        left_cropped = batch["left_img_yuv"][0].transpose(2, 0, 1)
        left_cropped = np.ascontiguousarray(left_cropped)
        right_cropped = batch["right_img_yuv"][0].transpose(2, 0, 1)
        right_cropped = np.ascontiguousarray(right_cropped)
        left_cropped.tofile(
            os.path.join(self.output_dir, "infra1", "%d.npy" % global_step_id)
        )
        right_cropped.tofile(
            os.path.join(self.output_dir, "infra2", "%d.npy" % global_step_id)
        )
