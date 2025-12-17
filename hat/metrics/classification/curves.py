from io import BytesIO
from typing import List

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa
from matplotlib.ticker import MultipleLocator  # noqa


def draw_precision_recall_curve(
    precisions: np.array,
    recalls: np.array,
    thresholds: np.array,
    average_precision: np.array,
    category: str,
):
    fig = plt.figure(figsize=(15, 7))
    fig.suptitle(category, fontsize=16)
    subplt = fig.add_subplot(121)
    subplt.plot(recalls, precisions)
    subplt.set_title("Precision Recall, AP: %.4f" % (average_precision))
    subplt.set_ylabel("precision")
    subplt.set_xlabel("recall")
    subplt.set_xlim([0, 1])
    subplt.set_ylim([0, 1])
    subplt.set_xticks(np.arange(0.0, 1, 0.1))
    subplt.set_yticks(np.arange(0.0, 1.01, 0.05))
    subplt.grid(True)
    subplt = fig.add_subplot(122)
    subplt.plot(thresholds, precisions, label="precision")
    subplt.plot(thresholds, recalls, label="recall")
    subplt.set_ylabel("precision & recall")
    subplt.set_xlabel("threshold")
    subplt.set_xlim([min(thresholds), max(thresholds)])
    subplt.set_ylim([0, 1])
    subplt.set_xticks(np.arange(min(thresholds), max(thresholds), 0.1))
    subplt.set_yticks(np.arange(0.0, 1.01, 0.05))
    subplt.set_title("Precision & Recall - Threshold")
    subplt.legend(loc="lower left")
    subplt.grid(True)
    b_io = BytesIO()
    fig.savefig(b_io)
    b_io.seek(0)
    plt.close(fig)
    return b_io


def draw_all_precision_recall_curves(
    categorys: List[str],
    precisions_list: List[np.array],
    recalls_list: List[np.array],
    average_precision_list: List[np.array],
):
    fig = plt.figure(figsize=(7, 7))
    subplt = fig.add_subplot(111)
    for category, precisions, recalls, average_precision in zip(
        categorys, precisions_list, recalls_list, average_precision_list
    ):
        subplt.plot(
            recalls,
            precisions,
            label="%s, AP: %.4f" % (category, average_precision),
        )
    subplt.set_ylabel("precision")
    subplt.set_xlabel("recall")
    subplt.set_xlim([0, 1])
    subplt.set_ylim([0, 1])
    subplt.set_xticks(np.arange(0.0, 1, 0.1))
    subplt.set_yticks(np.arange(0.0, 1.01, 0.05))
    subplt.legend(loc="lower left", fontsize="x-small")
    subplt.set_title(
        "Precision Recall, mAP: %.4f" % (np.mean(average_precision_list)),
        pad=25,
    )
    subplt.grid(True)
    b_io = BytesIO()
    fig.savefig(b_io)
    b_io.seek(0)
    plt.close(fig)
    return b_io


def draw_confusion_matrix(
    classes: List[str], matrix: np.array, title: str, cm_x_label_rot: int = 45
):
    """classes: a list of class names."""
    matrix = matrix.astype(np.float64)
    num_gts = matrix.sum(axis=1)
    num_prs = matrix.sum(axis=0)
    # Normalize by num of gts
    normed_matrix = matrix / num_gts.reshape(-1, 1)
    fig = plt.figure(figsize=(14, 14))
    ax = fig.add_subplot(111)
    cax = ax.matshow(normed_matrix, cmap=plt.cm.viridis)
    fig.colorbar(cax)
    ax.xaxis.set_major_locator(MultipleLocator(1))
    ax.yaxis.set_major_locator(MultipleLocator(1))
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            if np.isnan(normed_matrix[row, col]):
                ax.text(col, row, "NaN", va="center", ha="center", fontsize=12)
            else:
                ax.text(
                    col,
                    row,
                    "%.1f%%" % (normed_matrix[row, col] * 100),
                    va="center",
                    ha="center",
                    fontsize=12,
                )
    classes_with_num_gts = list(
        map(lambda c_n: "%s(%d)" % (c_n[0], c_n[1]), zip(classes, num_gts))
    )
    classes_with_num_prs = list(
        map(lambda c_n: "%s(%d)" % (c_n[0], c_n[1]), zip(classes, num_prs))
    )
    ax.set_xticklabels([None] + classes_with_num_prs, rotation=cm_x_label_rot)
    ax.set_yticklabels([None] + classes_with_num_gts)
    ax.set_xlabel("Predict", fontdict={"fontsize": 20})
    ax.set_ylabel("Groundtruth", fontdict={"fontsize": 20})
    ax.set_title(title, pad=50)

    b_io = BytesIO()
    plt.savefig(b_io, bbox_inches="tight")
    b_io.seek(0)
    plt.close(fig)
    return b_io
