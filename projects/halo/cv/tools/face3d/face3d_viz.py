"""viz face3d dataset."""
import argparse
import os

import cv2
import numpy as np

from hat.data.datasets.face3d_dataset import Face3dDataset
from hat.visualize.bbox2d import draw_bbox

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--image-path",
        required=True,
    )
    parser.add_argument(
        "--mask-path",
        required=True,
    )
    parser.add_argument(
        "--anno-path",
        required=True,
    )
    parser.add_argument(
        "--viz-save-dir",
        default=".tmp",
    )
    parser.add_argument(
        "--viz-num",
        default=50,
    )

    args = parser.parse_args()

    dataset = Face3dDataset(
        image_path_list=[args.image_path],
        mask_path_list=[args.mask_path],
        anno_path_list=[args.anno_path],
        stage="finetune",
    )

    os.makedirs(args.viz_save_dir, exist_ok=True)
    for i, data in enumerate(dataset):
        img = data["img"]
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        gt_seg = data["gt_mask"]
        ldmk = data["gt_ldmk"]
        bbox = data["gt_bboxes"]
        img = draw_bbox(img, bbox, color=(250, 0, 0), thickness=1)
        for k in range(len(ldmk)):
            img = cv2.circle(
                img,
                (int(ldmk[k, 0]), int(ldmk[k, 1])),
                radius=2,
                thickness=-1,
                color=(128, 0, 128),
            )
        img = img + gt_seg[:, :, np.newaxis] * 0.2
        cv2.imwrite(os.path.join(args.viz_save_dir, f"{i}.jpg"), img)

        if i > int(args.viz_num):
            break
