"""tools of infer and viz"""
import argparse
import os

import horizon_plugin_pytorch as horizon
import numpy as np
import torch
from dataset_mappings import dataset_mappings

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
        "--input-points", type=str, required=True, help="Input file."
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
    if use_eager:
        qat_model = horizon.quantization.prepare_qat(model, inplace=False)
    else:
        qat_model = horizon.quantization.prepare_qat_fx(model)

    if pretrained_ckpt is None:
        pretrained_ckpt = config.int_infer_predictor["model_convert_pipeline"][
            "converters"
        ][1]["checkpoint_path"]
    checkpoint = torch.load(pretrained_ckpt)
    print(f"load pretrained model from {pretrained_ckpt}")
    qat_model.load_state_dict(checkpoint["state_dict"], strict=False)

    quantized_model = horizon.quantization.convert(qat_model, inplace=False)
    quantized_model.eval()
    return quantized_model


Mappings = {
    "kitti3d": 4,
    "nuscenes_lidar": 5,
    "nuscenes_lidar_multi": 5,
}


if __name__ == "__main__":
    args = parse_args()
    config = Config.fromfile(args.config)

    config.is_plot = args.is_plot
    points = np.fromfile(args.input_points, dtype=np.float32).reshape(
        (-1, Mappings[args.dataset])
    )

    init_logger(f"work_dirs/hat_logss/{config.task_name}_infer_viz")

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device_id)
    horizon.march.set_march(config.get("march", horizon.march.March.BAYES))

    model = build_from_registry(config.model)
    quantized_model = build_quantize_model(
        model, config, args.pretrained_ckpt, args.use_eager
    )

    points = torch.from_numpy(points).cuda()
    quantized_model.cuda()

    data = {
        "points": [points],
    }

    result = quantized_model(data)

    _, _, _, viz_update = dataset_mappings(args.dataset, config)

    viz_update(points, result)
