import logging
import os
import struct
from typing import List

import numpy as np
import torch
import torch.distributed as dist

from hat.registry import OBJECT_REGISTRY
from hat.utils.distributed import get_dist_info, rank_zero_only
from .faceid.faceid_eval import eval_gar, load_features
from .metric import EvalMetric

logger = logging.getLogger(__name__)


@OBJECT_REGISTRY.register
class FaceIDGARMetrics(EvalMetric):
    """Evaluation faceid status classification results.

    For more details, refer to wiki.
        http://wiki.hobot.cc/pages/viewpage.action?pageId=226377114

    Args:
        total_num_list: Total eval num list.
        save_dir: Path to save feature and label.
        name_list: Name of this metric instance for display.

    """

    def __init__(
        self,
        total_num_list: List[int],
        save_dir: str,
        name_list: List[str],
    ):
        self.eval_num = len(name_list)
        self.total_num_list = total_num_list

        assert self.eval_num > 0
        assert self.eval_num == len(total_num_list)
        self.save_feat_paths = []
        self.save_label_paths = []
        self.save_result_paths = []
        for name in name_list:
            save_feat_path = f"{name}_fea.bin"
            save_label_path = f"{name}_label.bin"

            self.save_feat_paths.append(os.path.join(save_dir, save_feat_path))
            self.save_label_paths.append(
                os.path.join(save_dir, save_label_path)
            )
            self.save_result_paths.append(
                os.path.join(save_dir, f"{name}-res.txt")
            )

        self.batch_ind = 0
        self.eval_ind = 0

        self.emb_size = None
        self.world_size = 1
        self.save_dir = save_dir
        try:
            os.makedirs(save_dir, exist_ok=True)
        except Exception:
            pass

        super(FaceIDGARMetrics, self).__init__(name_list)

    def reset(self) -> None:
        super().reset()
        self.batch_ind = 0

    def update(
        self,
        label: torch.Tensor,
        feats: torch.Tensor,
    ):
        """
        Update internal buffer with latest predictions.

        Note that the statistics are not available until
        you call self.get() to return the metrics.

        Args:
            preds: model output.
            label: gt.
        """
        batch_size, emb_size = feats.size()
        rank, world_size = get_dist_info()

        cur_save_feat_path = f"{self.save_feat_paths[self.eval_ind]}-{rank}"
        cur_save_label_path = f"{self.save_label_paths[self.eval_ind]}-{rank}"
        local_feats = feats.cpu().numpy().astype(np.float32)
        local_feats = local_feats.reshape(-1)
        local_label = label.cpu().numpy().astype(np.int32)
        local_label = local_label.reshape(-1)

        raw_data = struct.pack("f" * batch_size * emb_size, *local_feats)
        raw_label = struct.pack("i" * batch_size, *local_label)
        if self.batch_ind == 0:
            fp_fea = open(cur_save_feat_path, "wb")
            fp_label = open(cur_save_label_path, "wb")
        else:
            fp_fea = open(cur_save_feat_path, "ab")
            fp_label = open(cur_save_label_path, "ab")
        fp_fea.seek(self.batch_ind * batch_size * emb_size * 4)
        fp_fea.write(raw_data)
        fp_label.seek(self.batch_ind * batch_size * 4)
        fp_label.write(raw_label)
        fp_fea.close()
        fp_label.close()

        self.num_inst += batch_size
        self.batch_ind += 1
        self.emb_size = emb_size
        self.world_size = world_size

    @rank_zero_only
    def _load_feature_and_compute_gar(self):
        feats, labels = load_features(
            self.save_feat_paths[self.eval_ind],
            self.save_label_paths[self.eval_ind],
            self.world_size,
            self.emb_size,
            self.total_num_list[self.eval_ind],
        )
        result_dict = eval_gar(
            feats,
            labels,
            output_path=self.save_result_paths[self.eval_ind],
        )
        return result_dict

    def compute(self):
        """Get evaluation metrics."""
        assert self.emb_size is not None
        if self.world_size > 1:
            dist.barrier()
            self._load_feature_and_compute_gar()
            dist.barrier()
        else:
            self._load_feature_and_compute_gar()

        self.eval_ind += 1

        return self.save_result_paths
