import json
import logging

import cv2

from hat.evaluation.detection2d.fp import evaluate
from hat.visualize.detection2d.draw_fp_samples import (
    draw_sample,
    sort_samples_by_fp,
)

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
        gt_file=args.gt_path,
        det_file=args.pred_path,
        config_file=args.config_path,
        output_dir=args.output_dir,
    )

    with open(f"{args.output_dir}/samples.json") as fin:
        result = json.load(fin)

    samples = sort_samples_by_fp(result, fp_type="other")

    for sample in samples:
        image_key = sample["image_key"]
        image = cv2.imread(os.path.join(args.image_dir, image_key))
        if image is None:
            print(os.path.join(args.image_dir, image_key))
        draw_sample(image, sample)
        output_image_path = os.path.join(
            args.output_dir, image_key.replace("/", "__")
        )
        cv2.imwrite(output_image_path, image)
