import logging

from hat.evaluation.classification.evaluate import evaluate

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


if __name__ == "__main__":
    import argparse
    import os

    parser = argparse.ArgumentParser()
    parser.add_argument("--gt_path", type=str, required=True)
    parser.add_argument("--pred_path", type=str, required=True)
    parser.add_argument("--config_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--image_dir", type=str)
    args = parser.parse_args()

    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir)
    evaluate(
        gt_path=args.gt_path,
        pred_path=args.pred_path,
        config_path=args.config_path,
        output_dir=args.output_dir,
        image_dir=args.image_dir,
    )
