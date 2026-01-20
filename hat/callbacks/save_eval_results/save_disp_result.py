import os
import pickle
import numpy as np
import cv2
from PIL import Image
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
            init_source = None
            if isinstance(model_outs, (list, tuple)) and len(model_outs) > 1:
                init_source = model_outs[1]
            if init_source is None:
                init_source = model_outs[0]
            initdisp = init_source[ sample_idx, ...].detach().cpu().numpy()
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

    @staticmethod
    def _infer_crop_hw(batch_img):
        if batch_img is None:
            return 352, 640
        h = int(batch_img.shape[-2])
        w = int(batch_img.shape[-1])
        return h, w

    @staticmethod
    def _bgr2nv12(image):
        image = image.astype(np.uint8)
        height, width = image.shape[0], image.shape[1]
        frame_size = width * height
        nv12 = np.zeros(frame_size + frame_size // 2, dtype=np.int32)

        bgr24_reshaped = image.reshape((height, width, 3)).astype(np.int32)
        b, g, r = bgr24_reshaped[..., 0], bgr24_reshaped[..., 1], bgr24_reshaped[..., 2]

        y_plane = ((66 * r + 129 * g + 25 * b + 128) >> 8) + 16
        y_plane = np.clip(y_plane, 0, 255).astype(np.uint8)
        nv12[:frame_size] = y_plane.flatten()

        uv_b = b[::2, ::2]
        uv_g = g[::2, ::2]
        uv_r = r[::2, ::2]
        uv_plane = np.zeros((height * width // 2), dtype=np.int32)

        u = ((-38 * uv_r - 74 * uv_g + 112 * uv_b + 128) >> 8) + 128
        v = ((112 * uv_r - 94 * uv_g - 18 * uv_b + 128) >> 8) + 128

        u = np.clip(u, 0, 255).astype(np.uint8).flatten()
        v = np.clip(v, 0, 255).astype(np.uint8).flatten()

        uv_plane[0::2] = u
        uv_plane[1::2] = v
        nv12[frame_size:] = uv_plane.flatten()

        return np.ascontiguousarray(nv12.astype(np.uint8))

    @staticmethod
    def _nv12Toyuv444(data, height, width):
        nv12_data = data.flatten()
        yuv444 = np.empty([height, width, 3], dtype=np.uint8)
        yuv444[:, :, 0] = nv12_data[: width * height].reshape(height, width)
        u = nv12_data[width * height :: 2].reshape(height // 2, width // 2)
        yuv444[:, :, 1] = Image.fromarray(u).resize((width, height), resample=0)
        v = nv12_data[width * height + 1 :: 2].reshape(height // 2, width // 2)
        yuv444[:, :, 2] = Image.fromarray(v).resize((width, height), resample=0)
        return np.ascontiguousarray(yuv444.astype(np.uint8))

    def preprocess(self, p, crop_width=640, crop_height=352):
        img = cv2.imread(p)
        if img is None:
            raise ValueError(f"Failed to read image: {p}")
        img = img[:, :, :3]
        h, w, _ = img.shape

        ratio = w / h
        nh = int(crop_width / ratio)
        img_resized = cv2.resize(img, (crop_width, nh))
        if nh < crop_height:
            top_pad = (crop_height - nh) // 2
            bottom_pad = crop_height - nh - top_pad
            if top_pad > 0 or bottom_pad > 0:
                img_resized = cv2.copyMakeBorder(
                    img_resized,
                    top_pad,
                    bottom_pad,
                    0,
                    0,
                    cv2.BORDER_CONSTANT,
                    value=0,
                )
        elif nh > crop_height:
            top = int((nh - crop_height + 1) * 0.5)
            bottom = top + crop_height
            img_resized = img_resized[top:bottom, ...]

        if img_resized.shape != (crop_height, crop_width, 3):
            raise ValueError(f"Unexpected resized shape {img_resized.shape} for {p}")

        img_nv12 = self._bgr2nv12(img_resized)
        img_yuv444 = self._nv12Toyuv444(img_nv12, crop_height, crop_width)

        input = img_yuv444.astype(np.float32)
        input = input / 128.0 - 1.0
        input = np.transpose(input, (2, 0, 1))
        input = np.ascontiguousarray(input[None, ...])

        return input

    def on_batch_begin(self, batch, global_step_id, **kwargs):
        assert (
            len(batch["dataset_name"]) == 1
        ), "test_batch_size_per_gpu in DStereo/DStereoPlus.py must be 1"
        img = batch.get("img", None)
        crop_height, crop_width = self._infer_crop_hw(img)

        left_path = batch["left_img_name"][0]
        right_path = batch["right_img_name"][0]

        left_input = self.preprocess(
            left_path, crop_width=crop_width, crop_height=crop_height
        )
        right_input = self.preprocess(
            right_path, crop_width=crop_width, crop_height=crop_height
        )

        left_save = (left_input + 1.0) * 128.0
        right_save = (right_input + 1.0) * 128.0

        file_name = left_path
        if "left" in file_name:
            file_name = file_name.replace("_left", "")
        file_name = os.path.basename(file_name).split(".")[0]
        left_save.tofile(os.path.join(self.output_dir, "infra1", file_name + ".npy"))
        right_save.tofile(os.path.join(self.output_dir, "infra2", file_name + ".npy"))
