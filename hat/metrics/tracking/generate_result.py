import json
import os
import pickle
from io import BytesIO
from typing import Dict, List, Optional

import matplotlib
import numpy as np

from hat.metrics.detection2d.generate_result import NpEncoder

matplotlib.use("Agg")


def draw_compare_curves(eval_files, eval_names):
    """Draw compare curves of multi result file.

    Args:
        eval_files: array-like, a list of file objects
        eval_names: array-like, a list of eval names
    """
    import pylab

    fig, (ax1, ax2, ax3) = pylab.subplots(1, 3, figsize=(25, 7))
    min_threshold = 0
    max_threshold = 1
    max_motp = 0
    for eval_file, fn in zip(eval_files, eval_names):
        results = json.load(eval_file)
        recall = results["details"]["recall_hypo"]
        motar = results["details"]["motar"]
        motp = results["details"]["motp"]
        mota = results["details"]["mota"]
        conf = results["details"]["confidence"]
        fil_nan_mota = [(v, c) for v, c in zip(mota, conf) if not np.isnan(v)]
        fil_nan_mota = sorted(fil_nan_mota, key=lambda x: x[0])
        if len(fil_nan_mota) > 0:
            max_mota, max_mota_conf = fil_nan_mota[-1]
        else:
            max_mota, max_mota_conf = float("NaN"), float("NaN")

        # motar-recall curve, indicated to AMOTA
        ax1.plot(
            recall,
            motar,
            label="{} amota:{:0.4f}".format(  # noqa: E501
                fn, results["overview"]["amota"]
            ),
        )
        ax2.plot(
            recall,
            motp,
            label="{} amotp:{:0.4f}".format(fn, results["overview"]["amotp"]),
        )
        ax3.plot(
            conf,
            mota,
            label="{}: max mota {:0.4f} at thr {:0.4f}".format(
                fn,
                max_mota,
                max_mota_conf,
            ),
        )
        fil_nan = lambda l: [i for i in l if not np.isnan(i)]
        min_threshold = min(fil_nan(conf) + [min_threshold])
        max_threshold = max(fil_nan(conf) + [max_threshold])
        max_motp = max(fil_nan(motp) + [max_motp])

    ax1.grid(True)
    ax1.legend(loc="lower left", borderaxespad=0.0, fontsize="xx-small")
    ax1.set_xlabel("recall")
    ax1.set_ylabel("motar")
    ax1.set_xlim([0, 1])
    ax1.set_ylim([0, 1])
    ax1.set_xticks(np.arange(0.0, 1, 0.05))
    ax1.set_yticks(np.arange(0.0, 1, 0.05))
    ax1.set_title("recall vs motar")

    ax2.grid(True)
    ax2.legend(loc="lower right", borderaxespad=0.0, fontsize="xx-small")
    ax2.set_xlabel("recall")
    ax2.set_ylabel("motp")
    ax2.set_xlim([0, 1])
    ax2.set_ylim([0, max_motp])
    ax2.set_xticks(np.arange(0.0, 1.0, 0.05))
    ax2.set_yticks(np.arange(0.0, max_motp + 0.00001, max_motp / 20.0))
    ax2.set_title("recall vs motp")

    ax3.grid(True)
    ax3.legend(loc="upper right", borderaxespad=0.0, fontsize="xx-small")
    ax3.set_xlabel("threshold")
    ax3.set_ylabel("mota")
    ax3.set_xlim([min_threshold, max_threshold])
    ax3.set_ylim([0, 1])
    ax3.set_xticks(
        np.arange(
            min_threshold,
            max_threshold + 0.00001,
            (max_threshold - min_threshold) / 20.0,
        )
    )
    ax3.set_yticks(np.arange(0.0, 1.0, 0.05))
    ax3.set_title("thr vs mota")

    for ax in fig.axes:
        matplotlib.pyplot.sca(ax)
        pylab.xticks(rotation=45)

    b_io = BytesIO()
    fig.savefig(b_io)
    b_io.seek(0)
    pylab.close(fig)
    return b_io


def draw_curves(eval_files, eval_names, output_file):
    b_io = draw_compare_curves(
        [open(eval_file, "r") for eval_file in eval_files], eval_names
    )
    with open(output_file, "wb") as f:
        f.write(b_io.read())
    return b_io


def generate_tables(results, prefix="", suffix=""):
    tables = []
    if len(prefix):
        prefix = prefix + " "
    if len(suffix):
        suffix = " " + suffix
    overall_table = {
        "name": prefix + "overview" + suffix,
        "header": list(results["overview"].keys()),
        "data": [results["overview"]],
    }
    tables.append(overall_table)
    # TODO: add more detailed tables
    # e.g., 70/80/90% recall and relative stats

    # error tables
    if len(results["errors"]) > 0:
        error_table = {
            "name": prefix + "error" + suffix,
            "header": list(results["errors"][0].keys()),
            "data": results["errors"],
        }
        tables.append(error_table)
    return tables


def generate_tracking_result(
    results: Dict,
    results_raw: Dict,
    output_dir: str,
    result_json: Optional[str] = "result.json",
    result_png: Optional[str] = "result.png",
    table_json: Optional[str] = "tables.json",
    eval_classes: Optional[List[str]] = None,
):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # write results json
    result_json_file = os.path.join(output_dir, result_json)
    with open(result_json_file, "w") as f:
        json.dump(results, f, cls=NpEncoder)

    # dump raw results
    # TODO: dump json later
    result_raw_file = os.path.join(output_dir, "result.pkl")
    with open(result_raw_file, "wb") as f:
        pickle.dump(results_raw, f)

    # draw curves
    cls_json_list = []
    for cls in eval_classes:
        cls_results = results[cls]
        result_json_one_cls_file = os.path.join(
            output_dir, f"{cls}_" + result_json
        )
        cls_json_list.append(result_json_one_cls_file)
        with open(result_json_one_cls_file, "w") as f:
            json.dump(cls_results, f, cls=NpEncoder)
    draw_curves(
        cls_json_list,
        eval_classes,
        os.path.join(output_dir, result_png),
    )

    # write table
    tables_json_file = os.path.join(output_dir, table_json)
    old_tables = (
        list(json.load(open(tables_json_file)))
        if os.path.exists(tables_json_file)
        else []
    )
    tables = old_tables + generate_tables(
        results["overall"], prefix="overall", suffix=""
    )  # noqa
    for cls in eval_classes:
        cls_results = results[cls]
        tables += generate_tables(
            cls_results,
            prefix=f"sub_type:{cls}",
            suffix="",
        )
    with open(tables_json_file, "w") as f:
        json.dump(tables, f, indent=2, cls=NpEncoder)
