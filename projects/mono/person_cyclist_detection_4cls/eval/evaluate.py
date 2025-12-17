import argparse

from hat.evaluation.detection2d.single import evaluate


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--predict-file", type=str, required=True, help="predict json file"
    )

    parser.add_argument(
        "--gt-file", type=str, required=True, help="gt json file"
    )

    parser.add_argument(
        "--setting-file", type=str, required=True, help="setting json file"
    )

    parser.add_argument(
        "--save-dir", type=str, required=True, help="output dir"
    )

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()
    evaluate(args.gt_file, args.predict_file, args.setting_file, args.save_dir)
