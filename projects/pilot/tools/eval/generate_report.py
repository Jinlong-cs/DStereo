import argparse
import logging
import os

from hat.utils import Config
from hat.utils.package_helper import raise_error_if_import_failed

logging.getLogger("").setLevel(logging.DEBUG)

try:
    from report_client import ReportClient
except ImportError:
    ReportClient = None


def generate_report(
    pred_name,
    modules,
    prediction_tags=None,
    diff_pred_name=None,
):

    raise_error_if_import_failed(ReportClient, "report_client")
    assert (
        ReportClient is not None
    ), "cannot import evaluation_client, have you install it?"
    config_path = os.path.join(os.getenv("HOME"), ".eval/config.yaml")
    assert os.path.exists(config_path), (
        "cannot find configure file %s for evaluation system" % config_path
    )

    if diff_pred_name is None:
        report_name = pred_name
    else:
        report_name = f"{pred_name}-vs-{diff_pred_name}"

    report_config = {
        "eval_model_tags": prediction_tags,
        "eval_model_name": pred_name,
        "diff_model_name": diff_pred_name,
        "modules": modules,
    }

    client = ReportClient()
    client.auth.login_by_ldap()
    response = client.report.create_report(
        report_type_id=2,
        report_name=report_name,
        description="",
        config=report_config,
    )

    return response


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        required=True,
        help="eval config file path",
    )
    parser.add_argument(
        "--model-setting",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--model-version",
        type=str,
        required=False,
        default="test",
    )
    parser.add_argument(
        "--diff-pred-name",
        type=str,
        required=False,
        default=None,
    )

    args = parser.parse_args()

    os.environ["HAT_PILOT_MODEL_SETTING"] = args.model_setting
    os.environ["HAT_PILOT_MODEL_VERSION"] = args.model_version

    config = Config.fromfile(args.config)

    datasets = []
    # map detection_3d -> detection
    type_map = {"detection_3d": "detection"}
    for task_type, task_ids in config.dataset_ids.items():
        for task, ids in task_ids.items():
            task = task[0] if isinstance(task, tuple) else task
            datasets.append(
                {
                    "datasets": [int(_id) for _id in ids],
                    "type": type_map[task_type]
                    if task_type in type_map
                    else task_type,
                    "name": task,
                }
            )

    generate_report(
        config.model_name,
        datasets,
        prediction_tags=config.eval_tags,
        diff_pred_name=args.diff_pred_name,
    )
