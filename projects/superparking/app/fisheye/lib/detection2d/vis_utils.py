import os

import cv2
import numpy as np

from hat.callbacks.callbacks import CallbackMixin
from ..utils import reorganize_sod_batch_results

FOCAL_V = 442


def cal_pixel_height_cylindrically(height, sense_dist):
    return height / sense_dist * FOCAL_V


def visualize_bboxes_batch(
    batch,
    pred,
    task_name,
    color_map=None,
    show_score=False,
    show_label=False,
    show_gt=False,
    save_dir=None,
    class_names=None,
):
    """
    visual det val result
    """
    if len(pred) == 0:
        return

    pred_color = (0, 0, 255)
    gt_color = (0, 255, 0)
    for img_i, per_img_pred in enumerate(pred):
        img = batch[0]["img"][img_i]
        img = (
            img.detach().cpu().numpy() * 128.0 + 128.0
        )  # TODO: change according to
        img = img.astype(np.uint8).transpose((1, 2, 0))
        # yuv to bgr
        img = cv2.cvtColor(img, cv2.COLOR_YUV2BGR)

        if show_gt:
            gt_bboxes = batch[0]["gt_bboxes"][img_i]
            gt_labels = batch[0]["gt_classes"][img_i]
            gt_difficults = batch[0]["gt_difficult"][img_i]

            for igt, (gt_bbox, gt_label) in enumerate(
                zip(gt_bboxes, gt_labels)
            ):
                if gt_difficults is not None and gt_difficults[igt] == 1:
                    continue
                x1, y1, x2, y2, label = (
                    *list(map(int, gt_bbox[:4])),
                    int(gt_label),
                )
                img = vis_bbox(
                    img,
                    (x1, y1),
                    (x2, y2),
                    gt_color,
                    label=label,
                    show_label=show_label,
                )

        per_img_pred = per_img_pred.detach().cpu().numpy()
        gt_real_label = batch[0]["gt_labels"][img_i]
        for i in range(per_img_pred.shape[0]):
            bbox = per_img_pred[i]
            x1, y1, x2, y2, score, label = (
                *list(map(int, bbox[:4])),
                round(bbox[4], 2),
                int(bbox[5]),
            )
            pred_color = color_map[label] if color_map else pred_color
            if label not in gt_real_label:
                continue
            img = vis_bbox(
                img,
                (x1, y1),
                (x2, y2),
                pred_color,
                score,
                label,
                show_score,
                show_label,
            )

        if save_dir is None:
            save_dir = f"./tmp_output/{task_name}_pred_result/"
        class_name = ""

        if class_names is not None and show_gt:
            uni_clss = []
            for cls in gt_labels.unique():
                if cls >= 0:
                    uni_clss.append(cls)
            if len(uni_clss) == 0:
                continue
            # assert (
            #     len(uni_clss) == 1
            # ), "only support evaluation on dataset with 1 class labeled"
            class_name = class_names[uni_clss[0]]
        image_save_dir = os.path.join(save_dir, class_name)
        if not os.path.exists(image_save_dir):
            os.makedirs(image_save_dir)
        image_name = batch[0]["img_name"][img_i].replace("/", "_")
        cv2.imwrite(os.path.join(image_save_dir, image_name), img)


def vis_bbox(
    image,
    p1,
    p2,
    color,
    score=None,
    label=None,
    show_score=False,
    show_label=False,
):
    cv2.rectangle(image, p1, p2, color, 1)
    bbox_info = ""
    if show_score and score is not None:
        bbox_info += "s-{:.2f} ".format(score)
    if show_label and label is not None:
        bbox_info += "c-{:d} ".format(label)
    cv2.putText(image, bbox_info, p1, cv2.FONT_HERSHEY_COMPLEX, 0.5, color)
    return image


class VisualizationCallBack(CallbackMixin):
    def __init__(
        self,
        task_name="",
        save_dir=None,
        draw_results=False,
        color_map=None,
        difficulties=None,
        filter_condition=None,
        show_gt=True,
        class_names=None,
    ):
        self.save_dir = save_dir
        self.do_draw = draw_results
        self.task_name = task_name
        self.color_map = color_map
        self.filter_contition = filter_condition
        self.difficulties = difficulties
        self.show_gt = show_gt
        self.class_names = class_names

    def on_batch_end(self, batch, model_outs, **kwargs):
        if self.filter_contition is not None and (
            not self.filter_contition(batch)
        ):
            return

        targets, pred_bboxes_batch = reorganize_sod_batch_results(
            batch,
            model_outs,
            self.difficulties,
            filter_pred=False,
            filter_gt=self.show_gt,
        )
        batch[0]["gt_difficult"] = targets["gt_difficult"]

        if self.do_draw:
            visualize_bboxes_batch(
                batch,
                pred_bboxes_batch,
                self.task_name,
                show_score=False,
                show_label=False,
                show_gt=self.show_gt,
                save_dir=self.save_dir,
                color_map=self.color_map,
                class_names=self.class_names,
            )
