# Copyright (c) Horizon Robotics. All rights reserved.
from prettytable import PrettyTable


def compare(eval_name, eval_prediction, diff_name, diff_prediction):
    """Compare bev 3d eval table.

    Args:
        eval_prediction: The prediction being compared.
        diff_prediction: The prediction of comparison.
    """

    eval_tb = PrettyTable()
    eval_tb.title = eval_prediction["name"] + f" eval: {eval_name}"
    eval_tb.field_names = eval_prediction["header"]
    eval_data = eval_prediction["data"]
    for data_dict in eval_data:
        eval_tb.add_row(
            list(map(lambda key: data_dict[key], eval_tb.field_names))
        )

    diff_tb = PrettyTable()
    diff_tb.title = diff_prediction["name"] + f" diff: {diff_name}"
    diff_tb.field_names = diff_prediction["header"]
    diff_data = diff_prediction["data"]
    for data_dict in diff_data:
        diff_tb.add_row(
            list(map(lambda key: data_dict[key], diff_tb.field_names))
        )
    res = str(eval_tb) + "\n" + str(diff_tb)
    return res
