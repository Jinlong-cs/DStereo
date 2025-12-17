import logging

import numpy as np
import torch.distributed as dist
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import classification_report

from hat.metrics.metric import EvalMetric
from hat.registry import OBJECT_REGISTRY

__all__ = ["PersonPostionMetric"]

logger = logging.getLogger(__name__)


@OBJECT_REGISTRY.register
class PersonPostionMetric(EvalMetric):
    """Evaluation person postion.

    This class prints metrics such as confusion matrix, precision,
    recall, f1-score for the person position task.

    Args:
        iou_thresh: Iou threshold to match predict box and gt box.
        num_class: Class number of person position.
    """

    def __init__(
        self,
        iou_thresh: float,
        num_class: int,
    ):
        name = ["person_position"]
        super(PersonPostionMetric, self).__init__(name)
        self.iou_thresh = iou_thresh
        self.num_class = num_class

    def _init_states(self):
        self.add_state(
            "_results",
            default=[],
        )

    def _remove_redundance_result(self, results):
        img_name_list = []
        new_result = []
        for ret in results:
            img_name = ret["image_name"]
            if img_name not in img_name_list:
                img_name_list.append(img_name)
                new_result.append(ret)
            else:
                logging.warn(
                    f"find redundance img {img_name} result, remove it now"
                )
        return new_result

    def _iou_score(self, rec1, rec2):
        left_column_max = max(rec1[0], rec2[0])
        right_column_min = min(rec1[2], rec2[2])
        up_row_max = max(rec1[1], rec2[1])
        down_row_min = min(rec1[3], rec2[3])
        if left_column_max >= right_column_min or down_row_min <= up_row_max:
            return 0.0
        else:
            area1 = (rec1[2] - rec1[0]) * (rec1[3] - rec1[1])
            area2 = (rec2[2] - rec2[0]) * (rec2[3] - rec2[1])
            area_cross = (down_row_min - up_row_max) * (
                right_column_min - left_column_max
            )
            return area_cross / (area1 + area2 - area_cross + 1e-9)

    def _update_confusion_matrix(self, results):
        results = self._remove_redundance_result(results)
        logging.info("pred results num: {}".format(len(results)))
        eval_matrix = np.zeros((self.num_class, self.num_class))
        pred_result = []
        gt_result = []
        for result in results:
            pred_boxes = result["pred_boxes"]
            pred_position = result["pred_position"]
            gt_boxes = result["gt_boxes"]
            gt_position = result["gt_position"]
            score_matrix = np.zeros((len(gt_boxes), len(pred_boxes)))
            for i in range(len(pred_boxes)):
                rec2 = pred_boxes[i]
                for j in range(len(gt_boxes)):
                    rec1 = gt_boxes[j]
                    score_matrix[j, i] = self._iou_score(rec1, rec2)
            row_ind, col_ind = linear_sum_assignment(-score_matrix)
            sub_eval_matrix = np.zeros((self.num_class, self.num_class))
            sub_pred_result = []
            sub_gt_result = []
            for k in range(len(col_ind)):
                if score_matrix[row_ind[k], col_ind[k]] > self.iou_thresh:
                    pred_p = int(pred_position[col_ind[k]])
                    gt_p = int(gt_position[row_ind[k]])
                    sub_eval_matrix[gt_p, pred_p] += 1
                    sub_pred_result.append(pred_p)
                    sub_gt_result.append(gt_p)
                else:
                    pass
            eval_matrix += sub_eval_matrix
            pred_result.extend(sub_pred_result)
            gt_result.extend(sub_gt_result)

        self._eval_output(eval_matrix, pred_result, gt_result)

    def _eval_output(self, matrix, pred, gt):
        log_str = ""
        if 4 == self.num_class:
            log_str = "\n===================== confusion_matrix\n"
            log_str += "++++++++++   unknown: %6d %6d %6d %6d\n" % (
                matrix[0, 0],
                matrix[0, 1],
                matrix[0, 2],
                matrix[0, 3],
            )
            log_str += "++++++++++    driver: %6d %6d %6d %6d\n" % (
                matrix[1, 0],
                matrix[1, 1],
                matrix[1, 2],
                matrix[1, 3],
            )
            log_str += "++++++++++   copilot: %6d %6d %6d %6d\n" % (
                matrix[2, 0],
                matrix[2, 1],
                matrix[2, 2],
                matrix[2, 3],
            )
            log_str += "++++++++++ passenger: %6d %6d %6d %6d\n" % (
                matrix[3, 0],
                matrix[3, 1],
                matrix[3, 2],
                matrix[3, 3],
            )
            log_str += (
                "===================== precision/recall/f1-score/support\n"
            )
            t = classification_report(
                np.asarray(gt + [0, 1, 2, 3]),
                np.asarray(pred + [0, 1, 2, 3]),
                labels=[0, 1, 2, 3],
                target_names=["0", "1", "2", "3"],
                output_dict=True,
            )
            log_str += "++++++++++   unknown: %.4f %.4f %.4f %6d\n" % (
                t["0"]["precision"],
                t["0"]["recall"],
                t["0"]["f1-score"],
                t["0"]["support"],
            )
            log_str += "++++++++++    driver: %.4f %.4f %.4f %6d\n" % (
                t["1"]["precision"],
                t["1"]["recall"],
                t["1"]["f1-score"],
                t["1"]["support"],
            )
            log_str += "++++++++++   copilot: %.4f %.4f %.4f %6d\n" % (
                t["2"]["precision"],
                t["2"]["recall"],
                t["2"]["f1-score"],
                t["2"]["support"],
            )
            log_str += "++++++++++ passenger: %.4f %.4f %.4f %6d\n" % (
                t["3"]["precision"],
                t["3"]["recall"],
                t["3"]["f1-score"],
                t["3"]["support"],
            )
        logging.info(log_str)

    def compute(self):
        """Get evaluation metrics."""
        results = []
        # if distributed is used, gather data from all process.
        if dist.is_initialized():
            dist.barrier()
            world_size = dist.get_world_size()
            gather_data = [None for _ in range(world_size)]
            # gather results from all process.
            dist.all_gather_object(gather_data, self._results)
            for data in gather_data:
                for item in data:
                    results.append(item)

            # do val only on rank 0
            if dist.get_rank() == 0:
                self._update_confusion_matrix(results)
            self.reset()
            return -1

        else:
            results = self._results
            self._update_confusion_matrix(results)
            self.reset()

            return -1

    def update(self, batch, preds):
        batch_data = batch

        img_name = batch_data["image_name"]
        batch_size = len(img_name)
        gt_position = batch_data["gt_position"]
        gt_boxes = batch_data["gt_boxes"]
        pred_position = preds["pred_position"]
        pred_boxes = preds["pred_boxes"]

        for idx in range(batch_size):
            result = {}
            result["pred_position"] = (
                pred_position[idx].detach().cpu().numpy().tolist()
            )
            result["pred_boxes"] = (
                pred_boxes[idx].detach().cpu().numpy().tolist()
            )
            result["gt_position"] = (
                gt_position[idx].detach().cpu().numpy().tolist()
            )
            result["gt_boxes"] = gt_boxes[idx].detach().cpu().numpy().tolist()
            result["image_name"] = img_name[idx]
            self._results.append(result)
