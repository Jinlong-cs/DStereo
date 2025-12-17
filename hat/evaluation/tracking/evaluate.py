import logging
import pickle as pkl
from typing import List

from hat.metrics.tracking.metric import TrackingMetric
from hat.utils.config import Config

logger = logging.getLogger(__name__)


def evaluate(
    gt_file: str, pred_file: str, config_file: str, output_dir: str = "./"
) -> List[dict]:
    logger.info("exec tracking evaluation")
    logger.info("gt_file: %s" % gt_file)
    logger.info("det_file: %s" % pred_file)
    logger.info("config_file: %s" % config_file)
    logger.info("output_dir: %s" % output_dir)

    cfg = Config.fromfile(config_file)

    # TODO(jiaxi.wu): load from json rather than pickle when hatbc supports
    with open(gt_file, "rb") as f:
        gts = pkl.load(f)
    logger.info(f"Load GT: {len(gts)} samples.")

    with open(pred_file, "rb") as f:
        preds = pkl.load(f)
    logger.info(f"Load Prediction: {len(preds)} samples.")

    metric_setting = cfg["eval_type"][0]
    eval_metric_type = metric_setting.pop("eval_metric_type")

    metric = TrackingMetric(output_dir, eval_metric_type, metric_setting)
    metric.update(preds, gts)
    result = metric.get()
    return result
