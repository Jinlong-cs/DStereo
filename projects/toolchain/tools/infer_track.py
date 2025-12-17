"""tools of infer and viz"""
import argparse
import glob
import os

import cv2
import horizon_plugin_pytorch as horizon
import torch
from dataset_mappings import dataset_mappings
from PIL import Image
from torchvision.transforms.functional import pil_to_tensor
from torchvision.transforms.functional_tensor import resize

from hat.registry import build_from_registry
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
        "--input-imagedir", type=str, required=True, help="Input file dir."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--pretrained-ckpt", type=str, default=None, help="pre-trained model"
    )
    parser.add_argument(
        "--input-size",
        type=str,
        required=True,
        help="input image size, HxWxC",
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
        "--num",
        type=int,
        default=None,
        required=False,
        help="The num of infer images",
    )
    parser.add_argument(
        "--is-plot",
        action="store_true",
    )
    parser.add_argument("--device-id", type=int, default=0)
    return parser.parse_args()


def build_quantize_model(model, config, pretrained_ckpt=None):
    model.fuse_model()
    model.set_qconfig()
    qat_model = horizon.quantization.prepare_qat(model, inplace=False)

    if pretrained_ckpt is None:
        pretrained_ckpt = os.path.join(
            config.ckpt_dir, "qat-checkpoint-best.pth.tar"
        )
    checkpoint = torch.load(pretrained_ckpt, map_location="cpu")
    print(f"load pretrained model from {pretrained_ckpt}")
    qat_model.load_state_dict(checkpoint["state_dict"], strict=False)

    quantized_model = horizon.quantization.convert(qat_model, inplace=False)
    quantized_model.eval()
    return quantized_model


if __name__ == "__main__":
    args = parse_args()
    config = Config.fromfile(args.config)
    init_logger(f"work_dirs/hat_logss/{config.task_name}_infer_viz")

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device_id)
    torch.cuda.set_device(int(args.device_id))
    horizon.march.set_march(config.get("march", horizon.march.March.BAYES))

    model = build_from_registry(config.test_model)
    quantized_model = build_quantize_model(model, config, args.pretrained_ckpt)

    inputs_imgs = sorted(glob.glob(os.path.join(args.input_imagedir, "*.jpg")))

    input_size = list(map(int, args.input_size.split("x")))
    input_format = args.input_format

    config.is_plot = args.is_plot
    _, _, _, viz_update = dataset_mappings(args.dataset, config)

    num = args.num

    for idx, input_image in enumerate(inputs_imgs):
        if num is not None and idx >= num:
            break
        orig_img = cv2.imread(input_image)
        cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB, orig_img)
        ori_img = orig_img.copy()
        orig_img = Image.fromarray(orig_img)
        orig_img = pil_to_tensor(orig_img)

        resized_img = resize(orig_img, input_size[:2]).unsqueeze(0)

        if input_format == "yuv":
            input_img = horizon.nn.functional.bgr_to_yuv444(resized_img, True)
        else:
            input_img = resized_img
        input_img = (input_img - 128.0) / 128.0

        quantized_model.cuda()
        input_img = input_img.cuda()

        h_scale = input_img.shape[-2] / orig_img.shape[-2]
        w_scale = input_img.shape[-1] / orig_img.shape[-1]
        scale_factor = [
            torch.tensor(
                [w_scale, h_scale, w_scale, h_scale], dtype=torch.float32
            ).cuda()
        ]

        result = quantized_model(
            {
                "frame_data_list": [
                    {
                        "img": [input_img],
                        "scale_factor": scale_factor,
                        "img_name": ["/".join(input_image.split("/")[-3:])],
                        "seq_name": [input_image.split("/")[-3]],
                    },
                ]
            }
        )
        viz_update(ori_img, result)
