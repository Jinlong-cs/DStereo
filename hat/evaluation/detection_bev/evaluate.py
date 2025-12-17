import json
import logging
from multiprocessing import Pool
from typing import List

import tqdm

from hat.metrics.detection_bev.basic_struct import parse_to_struct
from hat.metrics.detection_bev.compute_entry import match_by_key
from hat.metrics.detection_bev.metric import HorizonBEVDetMetric
from hat.utils.config import Config

logger = logging.getLogger(__name__)


def evaluate(
    gt_file: str, pred_file: str, config_file: str, output_dir: str = "./"
) -> List[dict]:
    logger.info("exec bev detection evaluation")
    logger.info("gt_file: %s" % gt_file)
    logger.info("det_file: %s" % pred_file)
    logger.info("config_file: %s" % config_file)
    logger.info("output_dir: %s" % output_dir)

    cfg = Config.fromfile(config_file)
    eval_cfg = cfg.get("eval_type")[0]
    do_temporal_aidieval = eval_cfg.get("do_temporal_aidieval", False)
    eval_class = eval_cfg["eval_class"]
    num_worker = eval_cfg.get("num_worker", 0)
    disable_prelabel_2d_warning = eval_cfg.get(
        "disable_prelabel_2d_warning", False
    )
    skip_single_frame_aidieval = eval_cfg.get(
        "skip_single_frame_aidieval", False
    )
    gts = []
    eval_vis_cfg = dict(  # noqa
        vis_image_layout=eval_cfg["vis_camera_layout"],
        vis_camera=eval_cfg["eval_cameras"],
        bird_eye_size=eval_cfg["vis_bird_eye_size"],
        vcs_range=eval_cfg["vis_vcs_range"],
        vis_score_threshold=eval_cfg.get("vis_score_threshold", None),
        is_multi_category_eval=True
        if eval_cfg.get("eval_multi_category", None) is not None
        else False,
    )
    for _, gt in enumerate(open(gt_file)):
        gt = json.loads(gt)
        gt.update({"eval_vis_cfg": eval_vis_cfg})
        gt = parse_to_struct(
            gt,
            eval_class,
            info_type="gt",
            disable_prelabel_2d_warning=disable_prelabel_2d_warning,
            do_temporal_aidieval=do_temporal_aidieval,
        )
        gts.append(gt)

    preds = []
    for _, pred in enumerate(open(pred_file)):
        pred = json.loads(pred)
        pred.update({"eval_vis_cfg": eval_vis_cfg})
        pred = parse_to_struct(
            pred,
            eval_class,
            info_type="pred",
            disable_prelabel_2d_warning=disable_prelabel_2d_warning,
            do_temporal_aidieval=do_temporal_aidieval,
        )
        preds.append(pred)

    logger.info(f"Load GT: {len(gts)} samples.")
    logger.info(f"Load Prediction: {len(preds)} samples.")
    metric_setting = cfg["eval_type"][0]
    metric = HorizonBEVDetMetric(save_dir=output_dir, **metric_setting)
    eval_count = 0
    if num_worker == 0:
        for pred, gt in tqdm.tqdm(
            match_by_key(preds, gts, "scence_key"), desc="Process: "
        ):
            if pred is not None:
                metric.update([pred], [gt])
                eval_count += 1
    else:
        pool = Pool(num_worker)
        ret_list = []
        for pred, gt in match_by_key(preds, gts, "scence_key"):
            if pred is not None:
                ret = pool.apply_async(metric.update, ([pred], [gt]))
                eval_count += 1
                ret_list.append(ret)
        pool.close()
        all_gts = []
        all_preds = []
        for ret in tqdm.tqdm(ret_list, desc="Process: "):
            gts = ret.get()[0]
            preds = ret.get()[1]
            [gt.update_id() for gt in gts]
            [pred.update_id() for pred in preds]
            all_gts += gts
            all_preds += preds
        metric.all_gts = all_gts
        metric.all_preds = all_preds

    logger.info(f"Eval Prediction: {eval_count} samples.")

    result = {}
    if not skip_single_frame_aidieval:
        result.update(metric.get())
    if do_temporal_aidieval:
        result.update(metric.temporal_get())
    return result
