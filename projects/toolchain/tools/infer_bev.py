"""tools of infer and viz"""
import argparse
import os

import cv2
import horizon_plugin_pytorch as horizon
import numpy as np
import torch
from dataset_mappings import dataset_mappings
from PIL import Image
from torchvision.transforms.functional import pil_to_tensor
from torchvision.transforms.functional_tensor import resize

from hat.registry import build_from_registry
from hat.utils import qconfig_manager
from hat.utils.config import Config
from hat.utils.logger import init_logger


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        required=True,
        help="train config file path",
    )
    parser.add_argument(
        "--inputs",
        required=True,
        type=str,
        help="Input imgs for current frame.",
    )

    parser.add_argument(
        "--homo", type=str, required=True, help="Input homography."
    )
    parser.add_argument(
        "--temporal-homo",
        type=str,
        required=False,
        help="Input homography for temporal.",
    )
    parser.add_argument(
        "--pretrained-ckpt", type=str, default=None, help="pre-trained model"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=False,
        default="nuscenes",
    )
    parser.add_argument(
        "--num-views",
        type=int,
        default=6,
        required=False,
        help="number of views.",
    )
    parser.add_argument(
        "--input-format",
        type=str,
        default="yuv",
        required=False,
        choices=["yuv", "rgb"],
        help="input image format, yuv or rgb",
    )
    parser.add_argument(
        "--use-eager",
        action="store_true",
    )
    parser.add_argument(
        "--is-plot",
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
    print(f"load pretrained model from {pretrained_ckpt}")
    qat_model.load_state_dict(checkpoint["state_dict"], strict=False)

    quantized_model = horizon.quantization.convert(qat_model, inplace=False)
    quantized_model.eval()
    return quantized_model


def resize_homo(homo, scale):
    view = np.eye(4)
    view[0, 0] = scale[1]
    view[1, 1] = scale[0]
    homo = view @ homo
    return homo


def crop_homo(homo, offset):
    view = np.eye(4)
    view[0, 2] = -offset[0]
    view[1, 2] = -offset[1]
    homo = view @ homo
    return homo


def process_img(img_path, resize_size, crop_size):
    orig_img = cv2.imread(img_path)
    cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB, orig_img)
    orig_img = Image.fromarray(orig_img)
    orig_img = pil_to_tensor(orig_img)
    resize_hw = (
        int(resize_size[0]),
        int(resize_size[1]),
    )

    orig_shape = (orig_img.shape[1], orig_img.shape[2])
    resized_img = resize(orig_img, resize_hw).unsqueeze(0)
    top = int(resize_hw[0] - crop_size[0])
    left = int((resize_hw[1] - crop_size[1]) / 2)
    resized_img = resized_img[:, :, top:, left:]

    return resized_img, orig_shape


if __name__ == "__main__":
    args = parse_args()
    config = Config.fromfile(args.config)
    init_logger(f"work_dirs/hat_logss/{config.task_name}_infer_viz")

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device_id)
    horizon.march.set_march(config.get("march", horizon.march.March.BAYES))

    model = build_from_registry(config.model)

    model = build_quantize_model(
        model, config, args.pretrained_ckpt, args.use_eager
    )

    inputs = args.inputs

    num_views = args.num_views
    config.is_plot = args.is_plot
    _, _, _, viz_update = dataset_mappings(args.dataset, config)

    resize_size = config.resize_shape[1:]
    input_size = config.val_data_shape[1:]

    orig_imgs = []
    for i, img in enumerate(os.listdir(inputs)):
        img = os.path.join(inputs, img)
        img, orig_shape = process_img(img, resize_size, input_size)
        orig_imgs.append({"name": i, "img": img})

    input_imgs = []
    for orig_img in orig_imgs:
        if args.input_format == "yuv":
            input_img = horizon.nn.functional.bgr_to_yuv444(
                orig_img["img"], True
            )
        input_imgs.append(input_img)

    input_imgs = torch.cat(input_imgs)
    input_imgs = (input_imgs - 128.0) / 128.0

    homo = np.load(args.homo)

    top = int(resize_size[0] - input_size[0])
    left = int((resize_size[1] - input_size[1]) / 2)

    scale = (resize_size[0] / orig_shape[0], resize_size[1] / orig_shape[1])
    homo = resize_homo(homo, scale)
    homo = crop_homo(homo, (left, top))
    assert (
        homo.shape[0] == input_imgs.shape[0]
    ), "homography must be same as input imgs"
    model.cuda()
    model.eval()
    input_imgs = input_imgs.cuda()
    homo_cuda = torch.tensor(homo).cuda()
    data = {"img": input_imgs, "ego2img": homo_cuda}
    if args.temporal_homo:
        temporal_homo = np.load(args.temporal_homo)
        data["ego2global"] = temporal_homo
    with torch.no_grad():

        result = model(data)

    meta = {}
    orig_imgs = orig_imgs[:num_views]
    meta["ego2img"] = homo[:num_views]
    viz_update(orig_imgs, result, meta)
