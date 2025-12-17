# Copyright (c) Horizon Robotics. All rights reserved.

import json
import logging

from hat.core.eval_setting.parameter_def import DetectionSettingConfig
from hat.metrics.detection2d import Detection2dSingle
from hat.utils.config import Config

logger = logging.getLogger(__name__)


def evaluate(gt_file: str, det_file: str, config_file: str, output_dir: str):
    """Evaluate for detection 2d single.

    Args:
        gt_file: Path for ground truth json file.
        det_file: Path for prediction result json file.
        config_file: Path for setting file.
        output_dir: Path for output directory.
    """
    logger.info("exec detection 2d single evaluation")
    logger.info("gt_file: %s" % gt_file)
    logger.info("det_file: %s" % det_file)
    logger.info("config_file: %s" % config_file)
    logger.info("output_dir: %s" % output_dir)

    cfg = Config.fromfile(config_file)
    cfg = DetectionSettingConfig(**cfg._cfg_dict)

    metric = Detection2dSingle(cfg, output_dir)

    gts = {}
    preds = {}

    for _, gt in enumerate(open(gt_file)):
        gt = json.loads(gt)
        gts[gt["image_key"]] = gt
    for _, pred in enumerate(open(det_file)):
        pred = json.loads(pred)
        preds[pred["image_key"]] = pred

    logger.info(f"Load GT {len(gts.keys())} images.")
    logger.info(f"Load Prediction {len(preds.keys())} images.")

    # The order of image keys will slightly affect the ap value.
    # horizonadas_evalkit keep the order of image keys in gt.json,
    # while hat will sort the image keys to keep the calculation of
    # ap value stable, and avoid being affected by the order of
    # image keys in gt.json.
    image_keys = list(set(gts.keys()).intersection(set(preds.keys())))
    image_keys.sort(reverse=False)
    logger.info(f"Evaluate {len(image_keys)} images")

    # About the specific format of the gt and pred, please refer to
    # https://horizonrobotics.feishu.cn/wiki/wikcnleVkKSnsRGr7Ct8UiNXN6c#oTfkFr
    for img_key in image_keys:
        metric.update(gts[img_key], preds[img_key])

    summary = metric.get()
    return summary
