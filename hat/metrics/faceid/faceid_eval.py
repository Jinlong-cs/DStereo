# Copyright (c) Horizon Robotics. All rights reserved.

import logging

import numpy as np

from .faceid_benchmark import FeatSet

logger = logging.getLogger(__name__)

__all__ = [
    "load_features",
    "eval_gar",
]


def load_features(
    save_feat_path: str,
    save_label_path: str,
    world_size: int,
    embedding_size: int,
    total_num: int,
):
    """Load faceid saved features and labels.

    Args:
        save_dir: features dir.
        world_size: num_replicas of feature saving.
        total_num: total features num.
    """

    fea_list = []
    label_list = []
    for rank_i in range(world_size):
        fea_path = f"{save_feat_path}-{rank_i}"
        label_path = f"{save_label_path}-{rank_i}"

        fea = np.fromfile(fea_path, dtype=np.float32)
        fea = fea.reshape(-1, embedding_size)
        fea_list.append(fea)

        label = np.fromfile(label_path, dtype=np.int32)
        label = label.reshape(-1, 1)
        label_list.append(label)
    features = np.concatenate(fea_list, -1)
    features = features.reshape(-1, embedding_size)
    features = features[:total_num, :]

    labels = np.concatenate(label_list, -1)
    labels = labels.reshape(-1, 1)
    labels = labels[:total_num, :]
    labels = labels.reshape(total_num)
    return features, labels


def eval_gar(
    features: np.ndarray, labels: np.ndarray, output_path: str = "./res.txt"
):
    """Do faceid GAR eval.

    Args:
        features: faceid features, shape: [N, emb_size].
        labels: faceid labels, shape: [N].
        output_path: saving results path.
    """

    val_FeatSet = FeatSet(labels, features)
    result_dict = val_FeatSet.GAR(output_path=output_path)
    return result_dict
