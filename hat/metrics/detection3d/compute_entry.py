from collections import defaultdict
from typing import Dict, List

import numpy as np

from hat.metrics.detection2d.utils import calap, calar
from .basic_struct import Instance3dObj


# ------------------ Auto ------------------
class AutoEvalCalculator(object):
    @staticmethod
    def dx(gt_x, pred_x):
        return np.abs(gt_x - pred_x)

    @staticmethod
    def sdx(gt_x, pred_x):
        return gt_x - pred_x

    @staticmethod
    def dvx(gt_vx, pred_vx):
        return np.abs(gt_vx - pred_vx)

    @staticmethod
    def sdvx(gt_vx, pred_vx):
        return gt_vx - pred_vx

    @staticmethod
    def dax(gt_ax, pred_ax):
        return np.abs(gt_ax - pred_ax)

    @staticmethod
    def day(gt_ay, pred_ay):
        return np.abs(gt_ay - pred_ay)

    @staticmethod
    def dy(gt_y, pred_y):
        return np.abs(gt_y - pred_y)

    @staticmethod
    def sdy(gt_y, pred_y):
        return gt_y - pred_y

    @staticmethod
    def dvy(gt_vy, pred_vy):
        return np.abs(gt_vy - pred_vy)

    @staticmethod
    def sdvy(gt_vy, pred_vy):
        return gt_vy - pred_vy

    @staticmethod
    def dxy(dx, dy):
        return (dx ** 2 + dy ** 2) ** 0.5

    @staticmethod
    def dxp(dx, gt_x):
        return dx / (np.abs(gt_x) + 1e-6)

    @staticmethod
    def sdxp(sdx, gt_x):
        return sdx / (np.abs(gt_x) + 1e-6)

    @staticmethod
    def dyp(dy, gt_y):
        return dy / (np.abs(gt_y) + 1e-6)

    @staticmethod
    def sdyp(sdy, gt_y):
        return sdy / (np.abs(gt_y) + 1e-6)

    @staticmethod
    def dxyp(dxy, gt_x, gt_y):
        return dxy / (np.abs(gt_x ** 2 + gt_y ** 2) ** 0.5 + 1e-6)

    @staticmethod
    def dh(gt_h, pred_h):
        return np.abs(pred_h - gt_h)

    @staticmethod
    def sdh(gt_h, pred_h):
        return gt_h - pred_h

    @staticmethod
    def dw(gt_w, pred_w):
        return np.abs(pred_w - gt_w)

    @staticmethod
    def sdw(gt_w, pred_w):
        return gt_w - pred_w

    @staticmethod
    def dl(gt_l, pred_l):
        return np.abs(gt_l - pred_l)

    @staticmethod
    def sdl(gt_l, pred_l):
        return gt_l - pred_l

    @staticmethod
    def dhp(dh, gt_h):
        return dh / (np.abs(gt_h) + 1e-6)

    @staticmethod
    def sdhp(sdh, gt_h):
        return sdh / (np.abs(gt_h) + 1e-6)

    @staticmethod
    def dwp(dw, gt_w):
        return dw / (np.abs(gt_w) + 1e-6)

    @staticmethod
    def sdwp(sdw, gt_w):
        return sdw / (np.abs(gt_w) + 1e-6)

    @staticmethod
    def dlp(dl, gt_l):
        return dl / (np.abs(gt_l) + 1e-6)

    @staticmethod
    def sdlp(sdl, gt_l):
        return sdl / (np.abs(gt_l) + 1e-6)

    @staticmethod
    def abs_rot(gt_yaw, pred_yaw):
        gt_yaw = np.rad2deg(gt_yaw) % 360
        pred_yaw = np.rad2deg(pred_yaw) % 360
        return np.abs(gt_yaw - pred_yaw)

    @staticmethod
    def dyaw(gt_yaw, pred_yaw):
        return gt_yaw - pred_yaw

    @staticmethod
    def drot(abs_rot):
        return np.minimum(abs_rot, 360.0 - abs_rot)

    @staticmethod
    def sdrot(dyaw):
        sdrot = dyaw - np.round(dyaw / (np.pi / 2.0)) * (np.pi / 2.0)
        return sdrot * 180.0 / np.pi

    @staticmethod
    def dxy_10p_error(dxyp, threshold=0.1):
        return dxyp <= threshold

    @staticmethod
    def rot_cls_error_p(dyaw):
        return dyaw > np.pi / 2

    @staticmethod
    def dyaw_rate(gt, pred):
        return np.abs(gt - pred)


def cal_tp_error_auto(
    image_key,
    all_dets,
    all_gts,
    matched_dict,
    inds,
    det_assigns,
    mask,
    lidar_error,
    dep_thresh,
    y_thresh,
    gt_matched,
    y_gt_matched,
    metrics,
    metrics_y,
):
    diou_error = 1.0 - matched_dict["overlaps"][mask]
    matched_scores = np.array(
        [all_dets[image_key][assign]["score"] for assign in det_assigns]
    )
    pred_dim = np.array(
        [all_dets[image_key][assign]["dimensions"] for assign in det_assigns]
    )
    if lidar_error:
        pred_loc = np.array(
            [
                all_dets[image_key][assign]["location_lidar"]
                for assign in det_assigns
            ]
        )
        pred_yaw = np.array(
            [
                all_dets[image_key][assign]["yaw_lidar"]
                for assign in det_assigns
            ]
        )
        gt_dim = np.array(
            [all_gts[image_key][ind]["dimensions"] for ind in inds]
        )
        gt_loc = np.array(
            [all_gts[image_key][ind]["location_lidar"] for ind in inds]
        )
        gt_yaw = np.array(
            [all_gts[image_key][ind]["yaw_lidar"] for ind in inds]
        )
        # here gt_depth means x coordinate in lidar coordinate system.
        gt_depth = np.array(
            [all_gts[image_key][ind]["location_lidar"][0] for ind in inds]
        )  # noqa
    else:
        # camera error
        pred_loc = np.array(
            [all_dets[image_key][assign]["location"] for assign in det_assigns]
        )
        pred_yaw = np.array(
            [
                all_dets[image_key][assign]["rotation_y"]
                for assign in det_assigns
            ]
        )
        gt_dim = np.array(
            [all_gts[image_key][ind]["dimensions"] for ind in inds]
        )
        gt_loc = np.array(
            [all_gts[image_key][ind]["location"] for ind in inds]
        )
        gt_yaw = np.array(
            [all_gts[image_key][ind]["rotation_y"] for ind in inds]
        )
        gt_depth = np.array(
            [all_gts[image_key][ind]["depth"] for ind in inds]
        )  # noqa
    if lidar_error:
        # from lidar to camera.
        pred_loc_copy = pred_loc.copy()
        gt_loc_copy = gt_loc.copy()
        pred_loc[:, [2]] = pred_loc_copy[:, [0]]
        pred_loc[:, [0]] = pred_loc_copy[:, [1]]
        gt_loc[:, [2]] = gt_loc_copy[:, [0]]
        gt_loc[:, [0]] = gt_loc_copy[:, [1]]

    dx = AutoEvalCalculator.dx(gt_loc[:, 2], pred_loc[:, 2])
    sdx = AutoEvalCalculator.sdx(gt_loc[:, 2], pred_loc[:, 2])
    dy = AutoEvalCalculator.dy(gt_loc[:, 0], pred_loc[:, 0])
    sdy = AutoEvalCalculator.sdy(gt_loc[:, 0], pred_loc[:, 0])
    dxy = AutoEvalCalculator.dxy(dx, dy)
    dxp = AutoEvalCalculator.dxp(dx, gt_loc[:, 2])

    sdxp = AutoEvalCalculator.sdxp(sdx, gt_loc[:, 2])
    dyp = AutoEvalCalculator.dyp(dy, gt_loc[:, 0])
    sdyp = AutoEvalCalculator.sdyp(sdy, gt_loc[:, 0])
    dxyp = AutoEvalCalculator.dxyp(dxy, gt_loc[:, 2], gt_loc[:, 0])
    dxy_10p_error = AutoEvalCalculator.dxy_10p_error(dxyp)

    dh = AutoEvalCalculator.dh(gt_dim[:, 0], pred_dim[:, 0])
    sdh = AutoEvalCalculator.sdh(gt_dim[:, 0], pred_dim[:, 0])

    dw = AutoEvalCalculator.dw(gt_dim[:, 1], pred_dim[:, 1])
    sdw = AutoEvalCalculator.sdw(gt_dim[:, 1], pred_dim[:, 1])
    dl = AutoEvalCalculator.dl(gt_dim[:, 2], pred_dim[:, 2])
    sdl = AutoEvalCalculator.sdl(gt_dim[:, 2], pred_dim[:, 2])

    dhp = AutoEvalCalculator.dhp(dh, gt_dim[:, 0])
    sdhp = AutoEvalCalculator.sdhp(sdh, gt_dim[:, 0])
    dwp = AutoEvalCalculator.dwp(dw, gt_dim[:, 1])
    sdwp = AutoEvalCalculator.sdwp(sdw, gt_dim[:, 1])

    dlp = AutoEvalCalculator.dlp(dl, gt_dim[:, 2])
    sdlp = AutoEvalCalculator.sdlp(sdl, gt_dim[:, 2])

    abs_rot = AutoEvalCalculator.abs_rot(gt_yaw, pred_yaw)
    dyaw = AutoEvalCalculator.dyaw(gt_yaw, pred_yaw)
    drot = AutoEvalCalculator.drot(abs_rot)

    sdrot = AutoEvalCalculator.sdrot(dyaw)
    rot_cls_error_p = AutoEvalCalculator.rot_cls_error_p(dyaw)

    res = {
        "dx": dx,
        "sdx": sdx,
        "dxp": dxp,
        "sdxp": sdxp,
        "dy": dy,
        "sdy": sdy,
        "dyp": dyp,
        "sdyp": sdyp,
        "dxy": dxy,
        "dxyp": dxyp,
        "dh": dh,
        "sdh": sdh,
        "dhp": dhp,
        "sdhp": sdhp,
        "dw": dw,
        "sdw": sdw,
        "dwp": dwp,
        "sdwp": sdwp,
        "dl": dl,
        "sdl": sdl,
        "dlp": dlp,
        "sdlp": sdlp,
        "drot": drot,
        "sdrot": sdrot,
        "diou_error": diou_error,
        "dxy_10p_error": dxy_10p_error,
        "rot_cls_error_p": rot_cls_error_p,
        "scores": matched_scores,
    }

    # split gt_bojs to different dep_thresh.
    dep_thresh_inds = np.sum(
        gt_depth[:, np.newaxis] > dep_thresh, axis=-1
    )  # noqa
    # unique dep_thresh_inds to dep_inds
    dep_inds, cnts = np.unique(dep_thresh_inds, return_counts=True)
    gt_matched[dep_inds.tolist()] += cnts
    # put eval data to metrics.
    for key, val in metrics.items():
        for dep_ind in dep_inds:
            mask = dep_thresh_inds == dep_ind
            val[dep_ind] += res[key][mask].tolist()
    if y_thresh:
        # split gt_bojs to different y_thresh.
        y_thresh_inds = np.sum(gt_loc[:, [0]] > y_thresh, axis=-1)  # noqa
        # unique dep_thresh_inds to y_inds
        y_inds, cnts = np.unique(y_thresh_inds, return_counts=True)
        y_gt_matched[y_inds.tolist()] += cnts
        for key, val in metrics_y.items():
            for y_ind in y_inds:
                mask = y_thresh_inds == y_ind
                val[y_ind] += res[key][mask].tolist()
    return res


# ----------------- Nuscense -----------------


def get_tp_fp_nuscense(
    gt_list: List[Instance3dObj],
    pred_list: List[Instance3dObj],
    iou_threshold: float,
    enable_ignore: bool = False,
) -> Dict:

    preds = sorted(pred_list, key=lambda x: x.score, reverse=True)
    gts = gt_list

    num_pred = len(preds)
    tp = np.zeros(num_pred)
    fp = np.zeros(num_pred)
    conf = np.zeros(num_pred)
    gt_checked = np.zeros(len(gts))
    ignored_mask = np.zeros(num_pred)

    for pidx, det in enumerate(preds):
        conf[pidx] = det.score
        max_overlap = -np.inf
        jmax = -1

        if len(gts) > 0:
            overlaps = [
                det.bbox_3d.iou(gt.bbox_3d, axis=1) for gt in gts
            ]  # axis = 1 use height intersection
            max_overlap = np.max(overlaps)
            jmax = np.argmax(overlaps)

        if max_overlap > iou_threshold:
            if enable_ignore and gts[jmax].ignore:
                ignored_mask[pidx] = 1
                continue
            if gt_checked[jmax] == 0:
                tp[pidx] = 1.0
                gt_checked[jmax] = 1
            else:
                fp[pidx] = 1.0
        else:
            fp[pidx] = 1.0

    if enable_ignore:
        valid_mask = (1 - ignored_mask).astype(bool)
        tp = tp[valid_mask]
        fp = fp[valid_mask]
        conf = conf[valid_mask]
    ret = {"tp": tp, "fp": fp, "conf": conf}

    return ret


def summarize_nuscenes(
    res_by_img: List[dict],
    gt_count: int,
    target_recalls: List[float],
    target_precisions: List[float],
) -> Dict:
    """Get nuscenes metric results.

    Args:
        res_by_img: List of tp & fp info. Format:
            [
                {
                    tp: np.ndarray with shape (N),
                    fp: np.ndarray with shape (N),
                    conf: np.ndarray with shape (N),
                }
                ...
            ]
        gt_count: Number of ground truth box.
        target_recalls: Target recalls.
        target_precisions: Target precisions.

    Returns:
        Dict: summary.
    """
    res_all = defaultdict(lambda: [])
    num_image = len(res_by_img)
    for res in res_by_img:
        res_all["tp"] += [res["tp"]]
        res_all["fp"] += [res["fp"]]
        res_all["conf"] += [res["conf"]]

    tp = np.concatenate(res_all["tp"])
    fp = np.concatenate(res_all["fp"])
    conf = np.concatenate(res_all["conf"])

    argsort = np.argsort(-conf)
    conf = conf[argsort]
    tp = tp[argsort]
    fp = fp[argsort]

    fp = np.cumsum(fp, axis=0)
    tp = np.cumsum(tp, axis=0)

    if gt_count == 0:
        recalls = np.zeros(tp.shape)
    else:
        recalls = tp / float(gt_count)
    assert np.all(0 <= recalls) & np.all(recalls <= 1)

    precisions = tp / (tp + fp + 1e-6)
    assert np.all(0 <= precisions) & np.all(precisions <= 1)
    ap, recall, precision = calap(recalls, precisions)
    recall = np.array(recall)
    precision = np.array(precision)

    fppi = fp / float(num_image)
    fppi += 1e-6
    detection_rate = (tp - fp) / (gt_count + 1e-6)
    ar = calar(fppi, recalls)

    results = {}
    results["aps"] = {
        "ap": ap,
        "rec": recall[-1] if len(recall) > 0 else np.inf,
        "detection_rate": detection_rate[-1]
        if len(detection_rate) > 0
        else np.inf,
        "precision": precision[-1] if len(precision) > 0 else np.inf,
        "ar": ar,
        "num_image": num_image,
        "num_gt": gt_count,
        "num_tp": tp[-1] if len(tp) else 0,
        "num_fp": fp[-1] if len(fp) else 0,
    }
    results["recall"] = recall.tolist()
    results["precision"] = precision.tolist()
    results["fppi"] = fppi.tolist()
    results["conf"] = conf.tolist()
    results["detection_rate"] = detection_rate.tolist()
    results["fp"] = fp.tolist()

    results["scores_lut"] = {}
    thres_rec_pre_tp_fp_gt = []
    for thres in np.unique(conf):
        select = conf >= thres
        sub_tp = np.max(tp[select])
        sub_fp = np.max(fp[select])
        rec = sub_tp / (gt_count + 1e-6)
        pre = sub_tp / (sub_tp + sub_fp + 1e-6)
        thres_rec_pre_tp_fp_gt.append(
            [thres, rec, pre, sub_tp, sub_fp, gt_count]
        )
        results["scores_lut"][float(thres)] = {
            "threshold": thres,
            "recall": rec,
            "precision": pre,
            "num_tp": sub_tp,
            "num_fp": sub_fp,
            "num_gt": gt_count,
        }
    thres_rec_pre_tp_fp_gt = np.array(thres_rec_pre_tp_fp_gt)

    results["target_recalls"] = {}
    if len(thres_rec_pre_tp_fp_gt) == 0:
        pass
    else:
        for target_recall in target_recalls:
            valid_idx = np.where(thres_rec_pre_tp_fp_gt[:, 1] > target_recall)[
                0
            ]
            if valid_idx.shape[0] > 0:
                idx = valid_idx[thres_rec_pre_tp_fp_gt[valid_idx, 1].argmin()]
            else:
                idx = np.argmin(target_recall - thres_rec_pre_tp_fp_gt[:, 1])
            results["target_recalls"][target_recall] = {
                "threshold": thres_rec_pre_tp_fp_gt[idx, 0],
                "recall": thres_rec_pre_tp_fp_gt[idx, 1],
                "precision": thres_rec_pre_tp_fp_gt[idx, 2],
                "num_tp": thres_rec_pre_tp_fp_gt[idx, 3],
                "num_fp": thres_rec_pre_tp_fp_gt[idx, 4],
                "num_gt": thres_rec_pre_tp_fp_gt[idx, 5],
            }
        results["target_precisions"] = {}
        for target_precision in target_precisions:
            valid_idx = np.where(
                thres_rec_pre_tp_fp_gt[:, 2] > target_precision
            )[0]
            if valid_idx.shape[0] > 0:
                idx = valid_idx[thres_rec_pre_tp_fp_gt[valid_idx, 2].argmin()]
            else:
                idx = np.argmin(
                    target_precision - thres_rec_pre_tp_fp_gt[:, 2]
                )
            results["target_precisions"][target_precision] = {
                "threshold": thres_rec_pre_tp_fp_gt[idx, 0],
                "recall": thres_rec_pre_tp_fp_gt[idx, 1],
                "precision": thres_rec_pre_tp_fp_gt[idx, 2],
                "num_tp": thres_rec_pre_tp_fp_gt[idx, 3],
                "num_fp": thres_rec_pre_tp_fp_gt[idx, 4],
                "num_gt": thres_rec_pre_tp_fp_gt[idx, 5],
            }

    summary = {"AP": ap, "AR": ar, "all_results": results}
    return summary
