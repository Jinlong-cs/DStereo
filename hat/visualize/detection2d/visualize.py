from io import BytesIO

import matplotlib
import numpy as np
from prettytable import PrettyTable

matplotlib.use("AGG")
from matplotlib import pyplot as plt  # noqa

DEFAULT_ERROR_TYPES = [
    "iou_err",
    "iou_horizontal_err",
    "iou_vertical_err",
    "width_err",
    "height_err",
]


def plot_compare_errors_curves(
    results_errors, results_name, target_error_types=DEFAULT_ERROR_TYPES
):
    num_ax = len(target_error_types)
    num_row = int(num_ax ** 0.5)
    num_col = num_row
    if num_row * num_col < num_ax:
        num_col += 1
    if num_row * num_col < num_ax:
        num_row += 1
    assert num_row * num_col >= num_ax
    fig = plt.figure(figsize=(6 * num_col, 6 * num_row))
    fig.subplots_adjust(hspace=0.3)
    ax_dict = {}

    for result_errors, result_name in zip(results_errors, results_name):
        for result_error in result_errors:
            error_name = result_error["name"].split()[0]
            if error_name in target_error_types:
                if error_name not in ax_dict:
                    ax = fig.add_subplot(
                        num_row,
                        num_col,
                        target_error_types.index(error_name) + 1,
                    )
                    ax_dict[error_name] = ax
                ax = ax_dict[error_name]
                data = sorted(result_error["data"], reverse=True)
                p_data = list(
                    map(
                        lambda idx_v: (
                            (1 - float(idx_v[0]) / len(data)) * 100,
                            idx_v[1],
                        ),
                        enumerate(data),
                    )
                )
                ax.plot(*zip(*p_data), label=result_name)
    for error_name in ax_dict:
        ax = ax_dict[error_name]
        ax.set_xlabel("percentage (%)", fontsize=16)
        ax.set_ylabel("percentile", fontsize=16)
        ax.set_xlim([0, 1 * 100])
        ax.set_title("%s" % error_name, fontsize=18)
        ax.set_xticks(np.arange(0, 100 * (1 + 0.00001), 0.05 * 100))
        for tick in ax.get_xticklabels():
            tick.set_rotation(45)
        ax.grid()
        ax.legend(loc="upper left", fontsize=14)
    b_io = BytesIO()
    fig.savefig(b_io, bbox_inches="tight")
    b_io.seek(0)
    plt.close(fig)
    return b_io


def generate_compare_errors_table(
    results_tables, results_name, target_error_types=DEFAULT_ERROR_TYPES
):
    tb = PrettyTable()
    header = set(["name", "TP"])  # noqa: C405
    subheader = {}
    sub_col_names = ["mean", "std", "p0.9"]
    rows = []
    rows.append(subheader)
    for result_tables, result_name in zip(results_tables, results_name):
        row = {"name": result_name}
        rows.append(row)
        for table_dict in result_tables:
            if table_dict["name"] == "errors":
                for error_info in table_dict["data"]:
                    error_name = error_info["name"].split()[0]
                    if error_name in target_error_types:
                        if error_name not in header:
                            header.add(error_name)
                            subheader[error_name] = " / ".join(sub_col_names)
                        row[error_name] = "%s / %s / %s" % tuple(  # noqa: C414
                            list(
                                map(
                                    lambda k: str(error_info[k]), sub_col_names
                                )
                            )
                        )
            if table_dict["name"] == "overview":
                if "num_tp" in table_dict["data"][0]:
                    num_tp = int(table_dict["data"][0]["num_tp"])
                else:
                    num_gt = table_dict["data"][0]["num_gt"]
                    recall = table_dict["data"][0]["rec"]
                    num_tp = int(round(num_gt * recall))
                row["TP"] = num_tp

    header = ["name", "TP"] + [
        error_name for error_name in target_error_types if error_name in header
    ]
    tb.field_names = header
    list(
        map(
            lambda d: tb.add_row(
                list(
                    map(
                        lambda field_name: d.get(field_name, ""),
                        tb.field_names,
                    )
                )
            ),
            rows,
        )
    )
    tb.title = "compare errors"
    return tb


def generate_compare_mean_errors_by_track_id_table(
    results_tables, results_names, target_error_types=DEFAULT_ERROR_TYPES
):
    tb = PrettyTable()
    header = set(["name", "TP"])  # noqa: C405
    subheader = {}
    sub_col_names = ["mean", "std", "p0.9"]
    rows = []
    rows.append(subheader)
    for result_tables, result_name in zip(results_tables, results_names):
        row = {"name": result_name}
        rows.append(row)
        for table_dict in result_tables:
            if table_dict["name"] == "mean_errors_by_track_id":
                for error_info in table_dict["data"]:
                    error_name = error_info["name"].split()[0]
                    if error_name in target_error_types:
                        if error_name not in header:
                            header.add(error_name)
                            subheader[error_name] = " / ".join(sub_col_names)
                        row[error_name] = "%s / %s / %s" % tuple(  # noqa: C414
                            list(
                                map(
                                    lambda k: str(error_info[k]), sub_col_names
                                )
                            )
                        )
            if table_dict["name"] == "overview":
                if "num_tp" in table_dict["data"][0]:
                    num_tp = int(table_dict["data"][0]["num_tp"])
                else:
                    num_gt = table_dict["data"][0]["num_gt"]
                    recall = table_dict["data"][0]["rec"]
                    num_tp = int(round(num_gt * recall))
                row["TP"] = num_tp

    header = ["name", "TP"] + [
        error_name for error_name in target_error_types if error_name in header
    ]
    tb.field_names = header
    list(
        map(
            lambda d: tb.add_row(
                list(
                    map(
                        lambda field_name: d.get(field_name, ""),
                        tb.field_names,
                    )
                )
            ),
            rows,
        )
    )
    tb.title = "compare mean_errors by track_id"
    return tb


def generate_compare_errors_by_track_id_table(
    results_tables, results_names, target_error_types=DEFAULT_ERROR_TYPES
):

    if "width_err_rel_to_fit" not in target_error_types:
        target_error_types.append("width_err_rel_to_fit")

    if "height_err_rel_to_fit" not in target_error_types:
        target_error_types.append("height_err_rel_to_fit")

    tb = PrettyTable()
    header = set(["slice_name", "track_id", "name", "TP"])  # noqa: C405
    subheader = {}
    sub_col_names = ["mean", "std", "p0.9"]
    rows = []
    rows.append(subheader)

    data_by_slice_name_and_track_id = {}
    for result_tables, result_name in zip(results_tables, results_names):
        for table_dict in result_tables:
            if table_dict["name"] == "errors_by_track_id":
                for error_info in table_dict["data"]:
                    slice_name = error_info["slice_name"]
                    track_id = error_info["track_id"]

                    if slice_name not in data_by_slice_name_and_track_id:
                        data_by_slice_name_and_track_id[slice_name] = {}

                    if (
                        track_id
                        not in data_by_slice_name_and_track_id[slice_name]
                    ):
                        data_by_slice_name_and_track_id[slice_name][
                            track_id
                        ] = []

                    new_line = True
                    row = {"name": result_name}
                    for tmp_result in data_by_slice_name_and_track_id[
                        slice_name
                    ][track_id]:
                        if result_name == tmp_result["name"]:
                            row = tmp_result
                            new_line = False

                    if new_line:
                        data_by_slice_name_and_track_id[slice_name][
                            track_id
                        ].append(row)
                    error_name = error_info["name"].split()[0]
                    num_tp = error_info["num_tp"]
                    row["TP"] = num_tp

                    if error_name in target_error_types:
                        if error_name not in header:
                            header.add(error_name)
                            subheader[error_name] = " / ".join(sub_col_names)
                        row[error_name] = "%s / %s / %s" % tuple(  # noqa: C414
                            list(
                                map(
                                    lambda k: str(error_info[k]), sub_col_names
                                )
                            )
                        )

    header = ["name", "TP"] + [
        error_name for error_name in target_error_types if error_name in header
    ]
    tb.field_names = ["slice_name", "track_id"] + header
    for (
        slice_name,
        data_from_slice_name,
    ) in data_by_slice_name_and_track_id.items():
        for track_id, data in data_from_slice_name.items():
            list(
                map(
                    lambda d: tb.add_row(
                        [slice_name, track_id]
                        + list(
                            map(
                                lambda field_name: d.get(field_name, ""),
                                header,
                            )
                        )
                    ),
                    data,
                )
            )
    tb.title = "compare errors by track_id"
    return tb
