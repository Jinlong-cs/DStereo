import argparse

from hat.evaluation.detection2d.single import evaluate


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--task-type", type=str, required=True, help="cyclist or person"
    )

    parser.add_argument(
        "--predict-file", type=str, required=True, help="predict json file"
    )

    parser.add_argument(
        "--save-dir", type=str, required=True, help="output dir"
    )

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()
    task_type = args.task_type
    if task_type == "person":
        gt_file = "/horizon-bucket/auto_eval/adas_eval/eval_platform/fs/6033372/datasets/x8b_20220412_cp/data.json"  # noqa
        config_file = "/horizon-bucket/auto_eval/adas_eval/eval_platform/fs/6033372/settings/7366440/Merge_Ped_filter_Day_0-90m.yaml"  # noqa
    elif task_type == "cyclist":
        gt_file = "/horizon-bucket/auto_eval/adas_eval/eval_platform/fs/6033428/datasets/x8b_20220412_cp/data.json"  # noqa
        config_file = "/horizon-bucket/auto_eval/adas_eval/eval_platform/fs/6033428/settings/7366558/Merge_Cyc_filter_Day_0-90m.yaml"  # noqa
    det_file = args.predict_file
    output_dir = args.save_dir

    evaluate(gt_file, det_file, config_file, output_dir)
