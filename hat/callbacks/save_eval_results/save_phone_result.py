# Copyright (c) Horizon Robotics. All rights reserved.
import json
import logging
import os

import torch
import torch.distributed as dist
import torch.nn.functional as F

from hat.registry import OBJECT_REGISTRY
from hat.utils.apply_func import convert_numpy
from hat.utils.distributed import all_gather_object
from .save_eval_result import SaveEvalResult

__all__ = ["SavePhoneResult"]

logger = logging.getLogger(__name__)


@OBJECT_REGISTRY.register
class SavePhoneResult(SaveEvalResult):
    """
    Save OccFlow result for visualization.

    Args:
        output_dir: Output dir for saving results.
    """

    def __init__(
        self,
        output_dir: str,
    ):
        super().__init__(output_dir)
        self.output_dir = output_dir
        if (
            dist.is_initialized()
            and self.rank == 0
            or not dist.is_initialized()
        ):
            os.makedirs(
                self.output_dir,
                exist_ok=True,
            )

    def on_batch_end(self, batch, model_outs, train_metrics, **kwargs):
        epoch_id = kwargs["epoch_id"]

        result = {}
        mode = convert_numpy(batch["mode"])[0]
        labels = convert_numpy(batch["labels"])
        img_locs = convert_numpy(batch["img_loc"])
        logits, losses = model_outs
        logits = logits.detach().cpu()
        softmax_scores = F.softmax(logits, 1)
        preds_score, preds_label = torch.max(softmax_scores, dim=1)

        batch_size = len(preds_score)
        result["labels"] = labels
        result["img_locs"] = img_locs
        result["preds_score"] = preds_score
        result["preds_label"] = preds_label
        txt_path = f"{self.output_dir}/{mode}_{epoch_id:0>2}.txt"
        if dist.is_initialized():
            global_result = [None for _ in range(self.world_size)]
            all_gather_object(global_result, result)
        else:
            global_result = [result]

        if (
            dist.is_initialized()
            and self.rank == 0
            or not dist.is_initialized()
        ):
            for res in global_result:
                for i in range(batch_size):
                    img_loc = res["img_locs"][i]
                    gt = res["labels"][i]
                    pred = res["preds_label"][i]
                    score = res["preds_score"][i]
                    line = {
                        "img_loc": img_loc,
                        "gt": int(gt),
                        "pred": pred.item(),
                        "score": score.item(),
                    }
                    line = json.dumps(line)
                    with open(txt_path, "a") as f:
                        f.write(line)
                        f.write("\n")
