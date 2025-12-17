#!/usr/bin/env python3.6
# metric.py

import logging
from decimal import Decimal
from typing import Callable

import numpy as np

logger = logging.getLogger(__name__)

__all__ = [
    "find_metric_threshold",
    "compute_leuclidean_distance",
]


def find_metric_threshold(
    qry_feat: np.ndarray,
    qry_label: np.ndarray,
    ref_feat: np.ndarray,
    ref_label: np.ndarray,
    dist: np.ndarray = None,
    L: Callable = None,
    dump_dist: bool = False,
    dump_intra_path: str = "",
    dump_inter_path: str = "",
    output_path: str = "",
):
    """Find faceid metrice threshold.

    Args:
        qry_feat: query feature, shape: [N, emb_size]
        qry_label: query label, shape: [N]
        ref_feat: ref feat, shape: [M, emb_size]
        ref_label: ref label, shape: [M]
        dist: feature distance
        L: func to compute feature distance
        dump_intra_path: save feature intra dist path
        dump_inter_path: save feature inter dist path
        output_path: output_path
    """
    if dist is None:
        if L is None:
            dist = compute_leuclidean_distance(qry_feat, ref_feat)  # squared
        else:
            dist = L(qry_feat, ref_feat)
    else:
        assert dist.shape == (qry_feat.shape[0], ref_feat.shape[0])

    qry_num = qry_feat.shape[0]

    intra_num = intra_sum = intra_sum2 = 0
    intra_min = 1e20
    intra_max = 0

    inter_num = inter_sum = inter_sum2 = 0
    inter_min = 1e20
    inter_max = 0

    intra_v = []
    inter_v = []
    for lqi in range(qry_num):
        tmp_label = qry_label[lqi]
        ref_label[lqi] = 20000000
        intra_indices = np.where(ref_label == tmp_label)[0]
        ref_label[lqi] = tmp_label

        if intra_indices.shape[0] >= 1:
            intra_dist = dist[lqi][intra_indices]
            intra_num += intra_indices.size
            intra_sum += intra_dist.sum()
            intra_sum2 += (intra_dist ** 2).sum()
            intra_min = min(intra_dist.min(), intra_min)
            intra_max = max(intra_dist.max(), intra_max)
            intra_v.append(intra_dist)

        inter_indices = np.where(ref_label != qry_label[lqi])[0]
        inter_dist = dist[lqi][inter_indices]
        inter_num += inter_indices.size
        inter_sum += inter_dist.sum()
        inter_sum2 += (inter_dist ** 2).sum()
        inter_min = min(inter_dist.min(), inter_min)
        inter_max = max(inter_dist.max(), inter_max)
        inter_v.append(inter_dist)

    intra_avg = intra_sum / intra_num
    intra_std = np.sqrt(intra_sum2 / intra_num - intra_avg ** 2)

    inter_avg = inter_sum / inter_num
    inter_std = np.sqrt(inter_sum2 / inter_num - inter_avg ** 2)

    intra_info = f"Intra Distance: {intra_num}, "
    intra_info += f"{intra_avg:.4f}+-{intra_std:.4f}, "
    intra_info += f"min {intra_min:.4f}, max {intra_max:.4f}"
    logger.info(intra_info)

    inter_info = f"Inter Distance: {inter_num}, "
    inter_info += f"{inter_avg:.4f}+-{inter_std:.4f}, "
    inter_info += f"min {inter_min:.4f}, max {inter_max:.4f}"
    logger.info(inter_info)

    if intra_avg >= inter_avg:
        logger.info("The Metric Feature Is Too Bad!")

    intra_v = np.hstack(intra_v)
    tmp_inter_v = inter_v
    total_len = [0]
    for k in range(len(tmp_inter_v)):
        total_len.append(total_len[-1] + tmp_inter_v[k].size)

    dst_inter_v = np.zeros((total_len[-1]), dtype=np.float32)
    cur_index = 0
    for k in range(len(tmp_inter_v)):
        cur_len = tmp_inter_v[k].size
        dst_inter_v[cur_index : (cur_index + cur_len)] = tmp_inter_v[k][:]
        cur_index += cur_len
    inter_v = dst_inter_v

    FAR = [0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001]
    sort_place = [int(FAR[k] * inter_num) for k in range(len(FAR))]
    inter_v.partition(sort_place)

    result_info = ""
    result = {}
    for k in range(len(FAR)):
        num = int(FAR[k] * inter_num)
        thr = inter_v[num]
        cnt = len(intra_v[intra_v < thr])
        GAR = float(cnt) / intra_num
        metric_key = "GAR_AT_FAR_{:.0E}".format(Decimal(FAR[k]))
        result[metric_key] = round(GAR, 4)
        result_info_ = f"{thr:.4f}  {FAR[k]:.5f}({num}/{inter_num}) "
        result_info_ += f"GAR:{GAR:.5f}({cnt}/{intra_num})"
        logger.info(result_info_)
        result_info += result_info_ + "\n"

    if dump_dist:
        np.save(dump_intra_path, intra_v)
        logger.info(f"dump intra distance to {dump_intra_path}")
        np.save(dump_inter_path, inter_v)
        logger.info(f"dump inter distance to {dump_inter_path}")

    if output_path is not None:
        output_info = intra_info + "\n"
        output_info += inter_info + "\n"
        output_info += result_info
        with open(output_path, "w") as fw:
            fw.write(output_info)

    return result


def ldot(feat: np.ndarray, proj: np.ndarray):
    """Do mtx product.

    Args:
        feat: input mtx 1, shape: [N1, M]
        proj: input mtx 2, shape: [M, N2]
    """
    coord = np.empty((feat.shape[0], proj.shape[1]), dtype=proj.dtype)
    item_size = max(feat.dtype.itemsize, proj.itemsize)
    chunk_sz = int((1 << 30) / item_size / feat.shape[1])  # 1G
    for si in range(0, feat.shape[0], chunk_sz):
        ei = min(feat.shape[0], si + chunk_sz)
        for sj in range(0, proj.shape[1], chunk_sz):
            ej = min(proj.shape[1], sj + chunk_sz)
            coord[si:ei, sj:ej] = np.dot(feat[si:ei], proj[:, sj:ej])
    return coord


def compute_leuclidean_distance(qry: np.ndarray, ref: np.ndarray):
    """Do LEuclidean.

    For every possible pair.

    Args:
        qry: query feature, shape: [N, emb_size]
        ref: ref feature, shape: [M, emb_size]
    """
    qry_sq = (qry ** 2).sum(axis=1).reshape(qry.shape[0], 1)
    ref_sq = (ref ** 2).sum(axis=1).reshape(1, ref.shape[0])
    dist = -2 * ldot(qry, ref.T)
    dist += qry_sq
    dist += ref_sq
    return dist
