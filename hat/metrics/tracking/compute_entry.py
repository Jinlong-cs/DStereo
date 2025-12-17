from typing import Dict

import numpy as np

from hat.metrics.detection3d.compute_entry import AutoEvalCalculator

DEFAULT_ERROR = [
    "dxp",
    "dx",
    "sdxp",
    "sdy",
    "dy",
    "dvx",
    "sdvx",
    "dvy",
    "sdvy",
    "dax",
    "day",
    "dl",
    "sdl",
    "dw",
    "sdw",
    "dh",
    "drot",
    "drot90",
    "sdrot",
    "dyaw_rate",
]
FILTER_FIELD = ["class", "x", "y", "depth", "cipv"]


def cal_tp_error(
    pred,
    gt,
) -> Dict:
    tps = [
        perc
        for perc in pred.get_messages("overall")[0].perceptions
        if len(perc.get_attributes(topics=["asso"])) > 0
        and perc.get_attributes(topics=["asso"])[0].value.get(
            "match_type", "FP"
        )
        == "TP"
    ]
    tp_match_gt_id_list = [
        tp.get_attributes(topics=["asso"])[0].value["match_id"] for tp in tps
    ]
    gt_d = {
        perc.track_id: perc
        for perc in gt.get_messages("overall")[0].perceptions
    }
    gt_list, tp_list = [], []
    for gt_id, tp in zip(tp_match_gt_id_list, tps):
        if gt_id in gt_d:
            gt_list.append(gt_d[gt_id])
            tp_list.append(tp)

    if len(gt_list) > 0:
        gt_dim = np.array([perc.bbox3ds[0].dim for perc in gt_list])
        gt_yaw = np.array([perc.bbox3ds[0].yaw for perc in gt_list])
        gt_loc = np.array([perc.bbox3ds[0].loc for perc in gt_list])
        gt_depth = np.linalg.norm(gt_loc[:, :2], axis=1)
        gt_velocity = np.array([perc.velocity.velocity for perc in gt_list])
        gt_accelerate = []
        for perc in gt_list:
            _acceleration = perc.get_attributes(topics=["accelerate"])
            if len(_acceleration) == 0:
                _acceleration = perc.get_attributes(topics=["acceleration"])
            gt_accelerate.append(_acceleration)
        gt_accelerate = np.array(
            [
                [0] * 3 if len(acce) == 0 else acce[0].value
                for acce in gt_accelerate
            ]
        )
        gt_yawrate = [perc.get_attributes("yaw_rate") for perc in gt_list]
        gt_yawrate = np.array(
            [
                0 if len(yawrate) == 0 else yawrate[0].value
                for yawrate in gt_yawrate
            ]
        )

        gt_class = np.array([perc.topic for perc in tp_list])

        tp_cipv = [perc.get_attributes(topics=["cipv"]) for perc in tp_list]
        tp_cipv = np.array(
            [0 if len(cipv) == 0 else cipv[0].value for cipv in tp_cipv]
        )

        tp_dim = np.array([perc.bbox3ds[0].dim for perc in tp_list])
        tp_yaw = np.array([perc.bbox3ds[0].yaw for perc in tp_list])
        tp_loc = np.array([perc.bbox3ds[0].loc for perc in tp_list])
        tp_velocity = np.array([perc.velocity.velocity for perc in tp_list])
        tp_accelerate = []
        for perc in tp_list:
            _acceleration = perc.get_attributes(topics=["accelerate"])
            if len(_acceleration) == 0:
                _acceleration = perc.get_attributes(topics=["acceleration"])
            tp_accelerate.append(_acceleration)
        tp_accelerate = np.array(
            [
                [0] * 3 if len(acce) == 0 else acce[0].value
                for acce in tp_accelerate
            ]
        )
        tp_yawrate = [perc.get_attributes("yaw_rate") for perc in tp_list]
        tp_yawrate = np.array(
            [
                0 if len(yawrate) == 0 else yawrate[0].value
                for yawrate in tp_yawrate
            ]
        )

        tp_pred_score = np.array(
            [
                perc.get_attributes(topics=["tracking"])[0].score
                for perc in tp_list
            ]
        )

        dx = AutoEvalCalculator.dx(gt_loc[:, 0], tp_loc[:, 0])
        sdx = AutoEvalCalculator.sdx(gt_loc[:, 0], tp_loc[:, 0])
        dy = AutoEvalCalculator.dy(gt_loc[:, 1], tp_loc[:, 1])
        sdy = AutoEvalCalculator.sdy(gt_loc[:, 1], tp_loc[:, 1])
        dxy = AutoEvalCalculator.dxy(dx, dy)
        dxp = AutoEvalCalculator.dxp(dx, gt_loc[:, 0])
        sdxp = AutoEvalCalculator.sdxp(sdx, gt_loc[:, 0])
        dvx = AutoEvalCalculator.dvx(gt_velocity[:, 0], tp_velocity[:, 0])
        sdvx = AutoEvalCalculator.sdvx(gt_velocity[:, 0], tp_velocity[:, 0])
        dax = AutoEvalCalculator.dax(gt_accelerate[:, 0], tp_accelerate[:, 0])
        day = AutoEvalCalculator.day(gt_accelerate[:, 1], tp_accelerate[:, 1])
        dyp = AutoEvalCalculator.dyp(dy, gt_loc[:, 1])
        dvy = AutoEvalCalculator.dvy(gt_velocity[:, 1], tp_velocity[:, 1])
        sdvy = AutoEvalCalculator.sdvy(gt_velocity[:, 1], tp_velocity[:, 1])
        sdyp = AutoEvalCalculator.sdyp(sdy, gt_loc[:, 1])
        dxyp = AutoEvalCalculator.dxyp(dxy, gt_loc[:, 0], gt_loc[:, 1])
        dxy_10p_error = AutoEvalCalculator.dxy_10p_error(dxyp)
        dh = AutoEvalCalculator.dh(gt_dim[:, 1], tp_dim[:, 1])
        sdh = AutoEvalCalculator.sdh(gt_dim[:, 1], tp_dim[:, 1])
        dw = AutoEvalCalculator.dw(gt_dim[:, 0], tp_dim[:, 0])
        sdw = AutoEvalCalculator.sdw(gt_dim[:, 0], tp_dim[:, 0])
        dl = AutoEvalCalculator.dl(gt_dim[:, 2], tp_dim[:, 2])
        sdl = AutoEvalCalculator.sdl(gt_dim[:, 2], tp_dim[:, 2])
        dhp = AutoEvalCalculator.dhp(dh, gt_dim[:, 1])
        sdhp = AutoEvalCalculator.sdhp(sdh, gt_dim[:, 1])
        dwp = AutoEvalCalculator.dwp(dw, gt_dim[:, 0])
        sdwp = AutoEvalCalculator.sdwp(sdw, gt_dim[:, 0])
        dlp = AutoEvalCalculator.dlp(dl, gt_dim[:, 2])
        sdlp = AutoEvalCalculator.sdlp(sdl, gt_dim[:, 2])
        abs_rot = AutoEvalCalculator.abs_rot(gt_yaw, tp_yaw)
        dyaw = AutoEvalCalculator.dyaw(gt_yaw, tp_yaw)
        drot = AutoEvalCalculator.drot(abs_rot)
        sdrot = AutoEvalCalculator.sdrot(dyaw)
        drot90 = np.abs(sdrot)
        rot_cls_error_p = AutoEvalCalculator.rot_cls_error_p(dyaw)
        dyaw_rate = AutoEvalCalculator.dyaw_rate(gt_yawrate, tp_yawrate)

        res_metric = {
            "dx": dx,
            "sdx": sdx,
            "dxp": dxp,
            "sdxp": sdxp,
            "dvx": dvx,
            "sdvx": sdvx,
            "dax": dax,
            "day": day,
            "dy": dy,
            "sdy": sdy,
            "dyp": dyp,
            "sdyp": sdyp,
            "dvy": dvy,
            "sdvy": sdvy,
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
            "drot90": drot90,
            "sdrot": sdrot,
            "dxy_10p_error": dxy_10p_error,
            "rot_cls_error_p": rot_cls_error_p,
            "score": tp_pred_score,
            "depth": gt_depth,
            "x": gt_loc[:, 0],
            "y": gt_loc[:, 1],
            "class": gt_class,
            "cipv": tp_cipv,
            "dyaw_rate": dyaw_rate,
        }
    else:
        res_metric = {}

    return res_metric


def obs_filter(value, query):
    if type(query) is list:
        if type(query[0]) is list:
            index = obs_filter(value, query[0])
            if len(query) == 1:
                return index
            else:
                return np.logical_or(index, obs_filter(value, query[1:]))
        else:
            minq, maxq = query[:2]
            index_min = value >= minq
            index_max = value < maxq
            index = np.logical_and(index_min, index_max)
    else:
        index = value == query
    return index


def statistic_error_by_filter(
    error_raw,
    filters,
    eval_error=None,
):
    if "dx" not in error_raw or error_raw["dx"].shape[0] == 0:
        return []

    if eval_error is None:
        eval_error = DEFAULT_ERROR

    error_table = []
    for filter_kq in filters:
        error_dict = {}
        index = np.array([True] * len(error_raw["dx"]))
        for key, query in filter_kq.items():
            # TODO: support more custom filter
            assert key in FILTER_FIELD
            error_dict[key] = query
            index = np.logical_and(index, obs_filter(error_raw[key], query))
        error_dict = {f: error_dict.get(f, "-") for f in FILTER_FIELD}
        error_dict["obj_num"] = index.sum()
        for error_key in eval_error:
            error_match = error_raw[error_key][index]
            error_mean = error_match[~np.isnan(error_match)].mean()
            error_dict[error_key] = error_mean
        error_table.append(error_dict)
    return error_table
