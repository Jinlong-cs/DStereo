from typing import Tuple

import numpy as np
import torch
from sklearn.linear_model import LinearRegression

from hat.registry import OBJECT_REGISTRY
from hat.visualize.hpp import HPPViz
from .metric import EvalMetric

__all__ = ["HPPMetric"]


@OBJECT_REGISTRY.register
class HPPMetric(EvalMetric):
    def __init__(
        self,
        name: str = "HPPMetric",
        pixel_thresh: float = 20,
        match_thresh: float = 0.85,
        img_shape: Tuple[int, int] = (512, 256),  # W x H
        poly_fit: bool = False,
        dis_threshold: int = 5,
        img_saved_path: str = None,
        num_lanes: float = 4,
        visualized: bool = True,
    ):
        """HPP Metric.

        Args:
            name (str, optional): metric name. Defaults to "HPPMetric".
            pixel_thresh (float, optional): maximum horizontal distance
                threshold. Defaults to 20.
            match_thresh (float, optional): match threshold. Defaults to 0.85.
            img_shape (Tuple[int, int], optional): image shape.
                Defaults to (512, 256).
            dis_threshold (int, optional): distance threshold for directly
                sample points. Defaults to 5.
            img_saved_path (str, optional): visualized image saved path.
                Defaults to None.
            num_lanes (float, optional): the max number of lanes in
                tusimple metric. Defaults to 4.
            visualized (bool, optional): whether visualized odometry on
                images or not. Defaults to True.
        """
        super(HPPMetric, self).__init__(name)
        self.pixel_thresh = pixel_thresh
        self.match_thresh = match_thresh
        self.img_shape = img_shape
        self.poly_fit = poly_fit
        self.dis_threshold = dis_threshold
        self.img_saved_path = img_saved_path
        self.num_lanes = num_lanes
        self.visualized = visualized
        self.hpp_viz = HPPViz(img_saved_path)

    def _init_states(self):

        self.add_state(
            "accuracy",
            default=torch.tensor(0.0),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "fp",
            default=torch.tensor(0.0),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "fn",
            default=torch.tensor(0.0),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "num",
            default=torch.tensor(0.0),
            dist_reduce_fx="sum",
        )

    def get_angle(self, xs, y_samples):
        xs, ys = xs[xs >= 0], y_samples[xs >= 0]
        lr = LinearRegression()
        if len(xs) > 1:
            lr.fit(ys[:, None], xs)
            k = lr.coef_[0]
            theta = np.arctan(k)
        else:
            theta = 0
        return theta

    def line_accuracy(self, pred, gt, thresh):
        # x value ​​less than 0 will be specified as -100,
        # in case of affecting the calculation of positive points.
        pred = np.array([p if p >= 0 else -100 for p in pred])
        gt = np.array([g if g >= 0 else -100 for g in gt])
        return np.sum(np.where(np.abs(pred - gt) < thresh, 1.0, 0.0)) / len(gt)

    def bench(self, pred, gt, y_samples):
        if any(len(p) != len(y_samples) for p in pred):
            raise Exception("Format of odometry error.")
        angles = [
            self.get_angle(np.array(x_gts), np.array(y_samples))
            for x_gts in gt
        ]
        threshs = [self.pixel_thresh / np.cos(angle) for angle in angles]
        line_accs = []
        fp, fn = 0.0, 0.0
        matched = 0.0
        for x_gts, thresh in zip(gt, threshs):
            accs = [
                self.line_accuracy(np.array(x_preds), np.array(x_gts), thresh)
                for x_preds in pred
            ]
            max_acc = np.max(accs) if len(accs) > 0 else 0.0
            if max_acc < self.match_thresh:
                fn += 1
            else:
                matched += 1
            line_accs.append(max_acc)
        fp = len(pred) - matched
        # perhaps in tusimple metric, they only consider four-lane scene,
        # thus they set num_lanes to 4.
        if len(gt) > self.num_lanes and fn > 0:
            fn -= 1
        acc = sum(line_accs)
        if len(gt) > self.num_lanes:
            acc -= min(line_accs)
        return (
            acc / max(min(self.num_lanes, len(gt)), 1.0),
            fp / len(pred) if len(pred) > 0 else 0.0,
            fn / max(min(len(gt), self.num_lanes), 1.0),
        )

    def resample_x_along_y(self, h_samples, pred_points, poly_fit):
        # 2d points need to convert to X-axis coordinate
        # that sampled by fixed y coordinate
        if poly_fit:
            # apply polynomial curve to fit predicting points,
            # then resample x values according to fixed y.
            # for efficiency and accuracy, we use 3-order polynomial curve.
            if pred_points.shape[0] == 1:
                return -1
            coeff = np.polyfit(pred_points[:, 1], pred_points[:, 0], 3)
            pred_x = list(np.polyval(coeff, h_samples))
            return [pred_x]
        else:
            # First find the y coordinate of the point closest to
            # the fixed h_samples, and then select the x coordinate
            # of the point
            pred_x = []
            for h_sample in h_samples:
                dis = np.abs(h_sample - pred_points[:, 1])
                idx = np.argmin(dis)
                # If the minimum distance is greater than the threshold,
                # the point is discarded directly, and the -2 mark is used
                if dis[idx] > self.dis_threshold:
                    pred_x.append(-2)
                else:
                    pred_x.append(round(pred_points[:, 0][idx], 3))
            return [pred_x]

    def filter_neg_value(self, h_samples, gt_x):
        if isinstance(h_samples, torch.Tensor):
            h_samples = h_samples.cpu().numpy()
        if isinstance(gt_x, torch.Tensor):
            gt_x = gt_x.cpu().numpy()
        h_samples = list(h_samples[h_samples >= 0])
        temp_lanes = []
        gt_x = gt_x[
            None,
        ]
        for gt_lane in gt_x:
            # np_gt_lane = gt_lane.cpu().numpy()
            gt_lane = gt_lane[gt_lane > 0]
            temp_lanes.append(list(gt_lane))
        return h_samples, temp_lanes

    def update(self, batch, preds):
        gt_points = batch["points"]
        batch_size = gt_points.shape[0]
        for idx in range(batch_size):
            gt_point = gt_points[idx]
            pred_odom = preds["coordinate"][idx]
            if self.visualized:
                assert self.img_saved_path is not None
                self.hpp_viz(
                    batch["img_path"][idx],
                    preds["img_name"][idx],
                    gt_point,
                    pred_odom,
                )
            gt_h_samples, gt_x = self.filter_neg_value(
                gt_point[:, 1], gt_point[:, 0]
            )
            # sort predicted points along y-axis
            pred_odom = pred_odom[pred_odom[:, 1].argsort()]
            pred_odom = self.resample_x_along_y(
                gt_h_samples, pred_odom, poly_fit=self.poly_fit
            )
            # the number of points is 0
            if pred_odom == -1:
                acc, p, n = 0.0, 1.0, 1.0
            else:
                acc, p, n = self.bench(pred_odom, gt_x, gt_h_samples)
            self.num += 1
            self.accuracy += acc
            self.fp += p
            self.fn += n

    def compute(self):
        fp = self.fp / self.num
        fn = self.fn / self.num
        acc = self.accuracy / self.num

        return {"Accuracy": acc.item(), "FP": fp.item(), "FN": fn.item()}
