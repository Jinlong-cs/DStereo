import json
import logging
from typing import Optional

from hat.core.eval_setting.parameter_def import ClassificationSettingConfig
from hat.metrics.classification import ClassificationMetric
from hat.utils.config import Config

logger = logging.getLogger(__name__)


def load_json(json_path):
    data = {}
    try:
        with open(json_path, "r") as fin:
            for line in fin:
                record = json.loads(line)
                data[record["image_key"]] = record
    except IOError as e:
        logger.exception(e)
        logger.error(f"can't load {json_path}")

    return data


def evaluate(
    gt_path: str,
    pred_path: str,
    config_path: str,
    output_dir: str,
    image_dir: Optional[str] = None,
):
    """Evaluate for classification.

    Args:
        gt_path: Path for ground truth json file.
        pred_path: Path for prediction result json file.
        config_path: Path for setting file.
        output_dir: Path for output directory.
        image_dir: Path for image dir.
    """
    logger.info("exec detection 2d single evaluation")
    logger.info("gt_path: %s" % gt_path)
    logger.info("pred_path: %s" % pred_path)
    logger.info("config_path: %s" % config_path)
    logger.info("output_dir: %s" % output_dir)
    logger.info("image_dir: %s" % image_dir)

    gts = load_json(gt_path)
    preds = load_json(pred_path)

    logger.info(f"Load GT {len(gts)} images.")
    logger.info(f"Load Prediction {len(preds.keys())} images.")

    sort_image_key = False
    if sort_image_key:
        image_keys = list(set(gts.keys()).intersection(set(preds.keys())))
        image_keys.sort(reverse=False)
    else:
        image_keys = gts.keys()
    logger.info(f"Evaluate {len(image_keys)} images")

    config = Config.fromfile(config_path)
    config = ClassificationSettingConfig(**config._cfg_dict)

    metric = ClassificationMetric(config, output_dir, image_dir)

    for img_key in image_keys:
        metric.update(gts[img_key], preds[img_key])

    summary = metric.get()
    return summary
