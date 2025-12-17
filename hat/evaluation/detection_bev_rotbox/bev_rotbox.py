import json
import logging
from multiprocessing import Pool
from typing import List

import tqdm

from hat.metrics.detection_bev_rotbox.basic_struct import parse_to_struct
from hat.metrics.detection_bev_rotbox.compute_entry import match_by_key
from hat.metrics.detection_bev_rotbox.metric import HorizonRotBoxDetMetric
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
    eval_class = eval_cfg["eval_class"]
    num_worker = eval_cfg.get("num_worker", 0)
    gts = []
    eval_vis_cfg = dict(  # noqa
        vis_image_layout=eval_cfg["vis_camera_layout"],
        vis_bird_eye_size=eval_cfg["vis_bird_eye_size"],
        vcs_range=eval_cfg["vis_vcs_range"],
        bev_size=eval_cfg["bev_size"],
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
        )
        preds.append(pred)

    logger.info(f"Load GT: {len(gts)} samples.")
    logger.info(f"Load Prediction: {len(preds)} samples.")
    metric_setting = cfg["eval_type"][0]
    metric = HorizonRotBoxDetMetric(save_dir=output_dir, **metric_setting)
    eval_count = 0

    if num_worker == 0:
        for pred, gt in tqdm.tqdm(
            match_by_key(preds, gts, "timestamp"), desc="Process: "
        ):
            if pred is None:
                metric.update([], [gt])
            else:
                metric.update([pred], [gt])
                eval_count += 1
    else:
        pool = Pool(num_worker)
        ret_list = []
        for pred, gt in match_by_key(preds, gts, "timestamp"):
            if pred is None:
                res = pool.apply_async(metric.update, ([], [gt]))
            else:
                res = pool.apply_async(metric.update, ([pred], [gt]))
                eval_count += 1
                ret_list.append(res)
        pool.close()
        for res in tqdm.tqdm(ret_list, desc="Process: "):
            for k, v in res.get().items():
                if isinstance(v, list):
                    metric.all_res[k].extend(v)
                else:
                    metric.all_res[k].extend([v])
    logger.info(f"Eval Prediction: {eval_count} samples.")
    result = metric.get()
    return result
