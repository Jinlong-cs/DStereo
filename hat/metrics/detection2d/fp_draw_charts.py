from io import BytesIO
from typing import List, Tuple

import matplotlib

matplotlib.use("AGG")
import matplotlib.pyplot as plt  # noqa


def draw_compare_histogram(
    fp_counts: List[List[dict]], prediction_names: List[str]
):
    height = 20
    num_preds = len(fp_counts)

    fp_type_infos = sorted(fp_counts[0], key=lambda f: f["fp_gt"])
    column_names = list(map(lambda f: f["fp_type"], fp_type_infos))
    row_names = ["gt"] + prediction_names
    gt_row = list(map(lambda fp_info: fp_info["fp_gt"], fp_type_infos))
    pred_rows = list(
        map(
            lambda fp_count: list(
                map(
                    lambda fp_type: list(
                        filter(
                            lambda fp_info: fp_info["fp_type"] == fp_type,
                            fp_count,
                        )
                    )[0]["fp_det"],
                    column_names,
                )
            ),
            fp_counts,
        )
    )
    rows = [gt_row] + pred_rows

    gt_row_text = list(map(str, gt_row))
    pred_row_texts = list(
        map(
            lambda pred_row: list(
                map(
                    lambda pred_gt: "%d (%.1f%%)"
                    % (pred_gt[0], pred_gt[0] * 100.0 / pred_gt[1]),
                    zip(pred_row, gt_row),
                )
            ),
            pred_rows,
        )
    )
    row_texts = [gt_row_text] + pred_row_texts

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111)

    list(
        map(
            lambda row_id_row_row_name: ax.barh(
                y=list(
                    map(
                        lambda column_id: (
                            (num_preds + 2) * column_id
                            + num_preds
                            - row_id_row_row_name[0]
                        )
                        * height,
                        range(len(column_names)),
                    )
                ),
                width=row_id_row_row_name[1][0],
                height=height,
                label=row_id_row_row_name[1][1],
            ),
            enumerate(zip(rows, row_names)),
        )
    )

    ax.set_yticks(
        list(
            map(
                lambda column_id_column_name: (
                    (num_preds + 2) * column_id_column_name[0]
                    + num_preds / 2.0
                )
                * height,
                enumerate(column_names),
            )
        )
    )
    ax.set_yticklabels(column_names)

    list(
        map(
            lambda column_id: list(
                map(
                    lambda row_id_row_row_text: ax.text(
                        row_id_row_row_text[1][0][column_id],
                        (
                            (num_preds + 2) * column_id
                            + num_preds
                            - row_id_row_row_text[0]
                            - 0.5
                        )
                        * height,
                        row_id_row_row_text[1][1][column_id],
                        ha="left",
                        va="bottom",
                    ),
                    enumerate(zip(rows, row_texts)),
                )
            ),
            range(len(column_names)),
        )
    )

    ax.set_title("False Positive Number")
    ax.set_xlabel("number")
    ax.set_ylabel("type")
    ax.legend(loc="lower right")

    b_io = BytesIO()
    fig.savefig(b_io)
    b_io.seek(0)
    plt.close(fig)
    return b_io


def draw_histogram(fp_count: List[dict]):
    return draw_compare_histogram([fp_count], ["pred"])


def _draw_pie(names: Tuple[str], nums: Tuple[int], title: str):
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111)
    paches, texts, autotexts = ax.pie(nums, labels=names, autopct="%1.1f%%")
    ax.set_title(title)

    b_io = BytesIO()
    fig.savefig(b_io)
    b_io.seek(0)
    plt.close(fig)
    return b_io


def draw_gt_pie(fp_count: List[dict]):
    fp_types, gt_nums = zip(
        *list(map(lambda f: (f["fp_type"], f["fp_gt"]), fp_count))
    )
    title = "FP Gt Ratio"
    return _draw_pie(fp_types, gt_nums, title)


def draw_det_pie(fp_count: List[dict]):
    fp_types, gt_nums = zip(
        *list(map(lambda f: (f["fp_type"], f["fp_det"]), fp_count))
    )
    title = "FP Det Ratio"
    return _draw_pie(fp_types, gt_nums, title)
