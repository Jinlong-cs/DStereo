"""tools of infer and viz"""
import argparse
import logging
import os

import cv2
import horizon_plugin_pytorch as horizon
import torch
from dataset_mappings import dataset_mappings

from hat.data.collates.collates import collate_2d
from hat.data.collates.nusc_collates import (
    collate_nuscenes,
    collate_nuscenes_sequence,
)
from hat.registry import build_from_registry
from hat.utils import qconfig_manager
from hat.utils.apply_func import to_cuda
from hat.utils.config import Config
from hat.utils.logger import init_logger

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        required=True,
        help="model config file path",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="imagenet",
        required=True,
        help="dataset, releated to vis_update func",
    )
    parser.add_argument(
        "--pretrained-ckpt", type=str, default=None, help="pre-trained model"
    )
    parser.add_argument(
        "--is-plot",
        action="store_true",
    )
    parser.add_argument(
        "--use-eager",
        action="store_true",
    )
    parser.add_argument("--device-id", type=int, default=0)
    return parser.parse_args()


def build_quantize_model(model, config, pretrained_ckpt=None, use_eager=False):
    if use_eager:
        model.fuse_model()
    qconfig_manager.set_qconfig_mode(qconfig_manager.QconfigMode.QAT)
    model.set_qconfig()
    if not use_eager:
        qat_model = horizon.quantization.prepare_qat_fx(model)
    else:
        qat_model = horizon.quantization.prepare_qat(model, inplace=False)

    if pretrained_ckpt is None:
        pretrained_ckpt = os.path.join(
            config.ckpt_dir, "qat-checkpoint-best.pth.tar"
        )
        if not os.path.exists(pretrained_ckpt):
            pretrained_ckpt = os.path.join(
                config.ckpt_dir, "calibration-checkpoint-best.pth.tar"
            )
    checkpoint = torch.load(pretrained_ckpt, map_location="cpu")
    logger.info(f"load pretrained model from {pretrained_ckpt}")
    qat_model.load_state_dict(checkpoint["state_dict"], strict=False)

    quantized_model = horizon.quantization.convert(qat_model, inplace=False)
    quantized_model.eval()
    return quantized_model


if __name__ == "__main__":
    args = parse_args()
    config = Config.fromfile(args.config)
    config.is_plot = args.is_plot
    init_logger(f"work_dirs/hat_logss/{config.task_name}_infer_viz")

    torch.cuda.set_device(int(args.device_id))
    horizon.march.set_march(config.get("march", horizon.march.March.BAYES))

    model = build_from_registry(config.model)
    quantized_model = build_quantize_model(
        model, config, args.pretrained_ckpt, args.use_eager
    )
    quantized_model.cuda()
    dataset, _, _, viz_update = dataset_mappings(args.dataset, config)

    if args.dataset == "kitti3d":
        # collate_fn_func = collate_kitti3d
        collate_fn_func = None
    elif args.dataset == "nuscenes" or args.dataset == "detr3d_nuscenes":
        collate_fn_func = collate_nuscenes
    elif args.dataset == "nuscenes_sequence":
        collate_fn_func = collate_nuscenes_sequence
    else:
        collate_fn_func = collate_2d
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        collate_fn=collate_fn_func,
    )

    for ind, data in enumerate(dataloader):

        data = to_cuda(data, device=args.device_id, non_blocking=True)
        results = quantized_model(data)

        if args.dataset == "imagenet":
            img = data["ori_img"][0]
            category = viz_update(img, results[0])
            logger.info(f"The result is: {category}")

        if args.dataset == "voc":
            ori_img = data["ori_img"][0]
            size = data["img"].size()[-2:]
            ori_img = cv2.resize(ori_img, size)
            viz_update(ori_img, results)

        if args.dataset == "coco":
            img = data["resized_ori_img"][0]
            viz_update(img, results)

        if args.dataset == "fcos_coco":
            img = data["ori_img"][0]
            viz_update(img, results)

        if args.dataset == "yolo_coco":
            img = data["resized_ori_img"][0]
            viz_update(img, results)

        if args.dataset == "detr_coco":
            img = data["ori_img"][0]
            viz_update(img, results)

        if args.dataset == "cityscapes":
            img = data["ori_img"][0]
            viz_update(img, results)

        if args.dataset == "flyingchairs":
            img = data["ori_img"][0]
            viz_update(img, results)

        if args.dataset == "culane":
            img = data["ori_img"][0]
            viz_update(img, results)

        if args.dataset == "nuscenes_mono":
            data["img"] = data["padded_img"][0]
            data["filename"] = data["filename"][0]
            viz_update(data, results, "./demo", 0.3)

        if args.dataset == "sceneflow":
            viz_update(data["ori_img"][0], results)

        if args.dataset == "carfusion":
            size = data["img"].size()[-2:]
            viz_update(data["ori_img"][0], size, results)

        if ind == 0:
            break
