# Copyright (c) Horizon Robotics. All rights reserved.
import logging
import os
from multiprocessing import Pool
from typing import Optional

import cv2
from tqdm import tqdm

from hat.metrics.semantic_segmentation.get_files import (
    check_validation,
    dmp_path_to_local,
    get_image_info_dict,
    get_list_json,
    readin_config,
)
from hat.metrics.semantic_segmentation.segmentation import SegMetric

logger = logging.getLogger(__name__)


def eval_one(
    image_name,
    images_dir,
    pred_dir,
    gt_dir,
    up_scale,
    label_remap,
    images_info_dict,
    attr_dict,
    metric,
):
    if not os.path.isdir(images_dir):
        _, image_source = get_list_json(images_dir)
        dmp_image_path = image_source[image_name]
        image_path = dmp_path_to_local(dmp_image_path)
    else:
        image_path = os.path.join(images_dir, image_name)
    pred_path = os.path.join(
        pred_dir, os.path.splitext(image_name)[0] + ".png"
    )
    gt_path = os.path.join(
        gt_dir, os.path.splitext(image_name)[0] + "_label.png"
    )
    try:
        image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        pred_label = cv2.imread(pred_path, -1)
    except Exception as e:
        logger.error(
            "Please check %s or %s is exists." % (image_path, pred_path)
        )
        raise e

    if up_scale > 1:
        pred_label = cv2.resize(
            pred_label,
            (
                pred_label.shape[1] * up_scale,
                pred_label.shape[0] * up_scale,
            ),
            interpolation=cv2.INTER_NEAREST,
        )
    else:
        pass
    try:
        gt_label = cv2.imread(gt_path, -1)[
            : pred_label.shape[0], : pred_label.shape[1]
        ]
        gt_label[pred_label == 255] = 255
    except Exception as e:
        logger.error(
            "Please check %s and related pred is exists. "
            "And you should ensure gt size is same to pred size "
            "and gt_label channel = 1." % (gt_path)
        )
        raise e
    if label_remap is not None:
        gt_label = label_remap[gt_label]

    if attr_dict is None:
        image_tag = None
    else:
        image_tag = attr_dict[image_name]["attrs"]["tags"]

    if images_info_dict is not None:
        image_info = images_info_dict[image_name]
        image_info.update(
            {
                "img_name": image_name,
                "image": image,
                "gt_label": gt_label,
                "pred_label": pred_label,
                "image_tag": image_tag,
            }
        )
    else:
        image_info = {
            "img_name": image_name,
            "image": image,
            "gt_label": gt_label,
            "pred_label": pred_label,
            "image_tag": None,
        }
    return metric.update(image_info)


def evaluate(
    images_dir: str,
    gt_dir: str,
    pred_dir: str,
    config_file: str,
    outfile_dir: str,
    images_json: Optional[str] = None,
    attr_path: Optional[str] = None,
):
    """Semantic segmentation evaluation tool.

    Args:
        images_dir: path to images.
        gt_dir: path to groundtruth labelmap and image info json file.
        pred_dir: path to prediction json file.
        config_file: path to config yaml file.
        outfile_dir: path to output directory.
        images_json: path to image info json file.
    """
    logger.info("exec semantic segmentation evaluation")
    logger.info("images_dir: %s" % images_dir)
    logger.info("gt_dir: %s" % gt_dir)
    logger.info("pred_dir: %s" % pred_dir)
    logger.info("config_file: %s" % config_file)
    logger.info("outfile_dir: %s" % outfile_dir)
    logger.info("images_json: %s" % images_json)
    logger.info("attr_path: %s" % attr_path)
    logger.info("start check validation")
    if not os.path.exists(outfile_dir):
        os.makedirs(outfile_dir)
    validation = check_validation(
        images_dir, gt_dir, pred_dir, images_json=images_json
    )
    if validation is True:
        (
            labels_dict,
            up_scale,
            freespace_dict,
            roi_dict,
            min_gt_num,
            label_remap,
            image_tags,
        ) = readin_config(config_file)

        if images_json is not None:
            images_info_dict = get_image_info_dict(images_json)
        else:
            images_info_dict = None

        if attr_path is not None:
            attr_dict = get_image_info_dict(attr_path)
        else:
            attr_dict = None

        metric = SegMetric(
            outfile_dir=outfile_dir,
            labels_dict=labels_dict,
            freespace_dict=freespace_dict,
            roi_dict=roi_dict,
            min_gt_num=min_gt_num,
            image_tags=image_tags,  # image tag in config
        )
        if os.path.isdir(images_dir):
            images_names = os.listdir(images_dir)
        else:
            _, image_source = get_list_json(images_dir)
            images_names = list(image_source.keys())
        logger.info("all %s images" % len(images_names))
        if not images_names:
            logger.warning("images dir is null")
            images_names = os.listdir(gt_dir)
            images_names = list(
                map(
                    lambda images_name: images_name[:-10] + ".jpg",
                    images_names,
                )
            )
            images_names = list(
                filter(
                    lambda images_name: os.path.exists(
                        os.path.join(
                            pred_dir, os.path.splitext(images_name)[0] + ".png"
                        )
                    ),
                    images_names,
                )
            )
            logger.info(
                "all %s images in gt after filter prediction"
                % len(images_names)
            )

        results = []
        pbar = tqdm(total=len(images_names))

        for idx in range(0, len(images_names), 32):
            p = Pool(8)
            reslist = []
            for image_name in images_names[idx : idx + 32]:
                res = p.apply_async(
                    eval_one,
                    args=(
                        image_name,
                        images_dir,
                        pred_dir,
                        gt_dir,
                        up_scale,
                        label_remap,
                        images_info_dict,
                        attr_dict,
                        metric,
                    ),
                )
                reslist.append(res)
            p.close()
            p.join()
            for i in range(len(reslist)):
                results.append(reslist[i].get())
            pbar.update(32)
        metric.reslist = results
        content = metric.get()
    else:
        logger.error(
            "The data is illegal. \n"
            "Please check whether the dataset or prediction "
            "file or directory structure is legal. \n"
            "If other prediction in the dataset are evaluated successfully, "
            "you only need to check whether the dataset format is correct."
        )
        raise Exception(
            "The data is illegal. \n"
            "Please check whether the dataset or prediction "
            "file or directory structure is legal. \n"
            "If other prediction in the dataset are evaluated successfully, "
            "you only need to check whether the dataset format is correct."
        )
    return content
