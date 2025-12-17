import json
import logging
import time
from collections import defaultdict
from multiprocessing import Pool
from typing import List

import tqdm

from hat.metrics.detection3d.auto import HorizonAutoMetric
from hat.metrics.detection3d.basic_struct import parse_to_struct
from hat.metrics.detection3d.nuscense_3d import HorizonNuscense3DMetric
from hat.utils.config import Config

logger = logging.getLogger(__name__)


def evaluate(
    gt_file: str, pred_file: str, config_file: str, output_dir: str
) -> List[dict]:

    logger.info("exec detection 3d evaluation")
    logger.info("gt_file: %s" % gt_file)
    logger.info("config_file: %s" % config_file)
    logger.info("output_dir: %s" % output_dir)

    cfg = Config.fromfile(config_file)
    eval_class = cfg.get("eval_type")[0]["eval_class"]

    gts = {}
    with open(gt_file, "r", encoding="utf-8", errors="replace") as f:
        for line in f.readlines():
            gt = json.loads(line)
            gt = parse_to_struct(gt, eval_class=eval_class, info_type="gt")
            gts[gt.image_key] = gt

    preds = {}
    with open(pred_file, "r", encoding="utf-8", errors="replace") as f:
        for line in f.readlines():
            pred = json.loads(line)
            pred = parse_to_struct(
                pred, eval_class=eval_class, info_type="pred"
            )
            preds[pred.image_key] = pred

    logger.info(f"Load GT {len(gts.keys())} images.")
    logger.info(f"Load Prediction {len(preds.keys())} images.")

    image_keys = list(set(gts.keys()).intersection(set(preds.keys())))
    logger.info(f"Evaluate {len(image_keys)} images")

    results = []
    for metric_setting in cfg.get("eval_type"):
        if "Auto" in metric_setting:
            metric_type = "Auto"
            metric_setting.pop("Auto")
            metric_cls = HorizonAutoMetric(
                save_dir=output_dir, **metric_setting
            )
        elif "NuScenes" in metric_setting:
            metric_type = "NuScenes"
            metric_setting.pop("NuScenes")
            metric_cls = HorizonNuscense3DMetric(
                save_dir=output_dir, **metric_setting
            )
        else:
            metric_type = None
            metric_cls = None
            continue
        logger.info(f"Start {metric_type} eval")
        start_time = time.time()

        num_worker = metric_cls.num_worker

        if num_worker <= 1:
            for image_key in tqdm.tqdm(image_keys, desc="Process: "):
                metric_cls.update([gts[image_key]], [preds[image_key]])
        else:
            pool = Pool(num_worker)
            ret_list = []
            for image_key in image_keys:
                ret = pool.apply_async(
                    metric_cls.update, ([gts[image_key]], [preds[image_key]])
                )
                ret_list.append(ret)
            pool.close()

            all_gts = defaultdict(list)
            all_dets = defaultdict(list)
            gt_count = 0
            img_info_list = []

            for ret in tqdm.tqdm(ret_list, desc="Process: "):

                ret_gts = ret.get()[0]
                ret_dets = ret.get()[1]

                all_gts.update(ret_gts)
                all_dets.update(ret_dets)

                if isinstance(metric_cls, HorizonNuscense3DMetric):
                    gt_count += ret.get()[2]
                elif isinstance(metric_cls, HorizonAutoMetric):
                    img_info_list += ret.get()[2]
                else:
                    continue

            metric_cls.all_gts = all_gts
            metric_cls.all_dets = all_dets
            if isinstance(metric_cls, HorizonNuscense3DMetric):
                metric_cls.gt_count = gt_count
            elif isinstance(metric_cls, HorizonAutoMetric):
                metric_cls.img_info_list = img_info_list
            else:
                continue

        result = metric_cls.get()
        if result is not None:
            results.append(result)
        logger.info(
            f"Finish {metric_type} eval, cost time: {time.time() - start_time}s"  # noqa E501
        )

    logger.info("finished detection 3d eval.")
    return results
