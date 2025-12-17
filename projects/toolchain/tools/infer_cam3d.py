import argparse
import logging
import os

import horizon_plugin_pytorch as horizon
import torch
from dataset_mappings import dataset_mappings

from hat.data.collates.collates import collate_2d
from hat.registry import build_from_registry
from hat.utils import qconfig_manager
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
        help="train config file path",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="nuscenes_mono",
    )
    parser.add_argument(
        "--pretrained-ckpt", type=str, default=None, help="pre-trained model"
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="./demo01",
        help="save directory",
    )
    parser.add_argument(
        "--score-thr",
        type=float,
        default=0.3,
        help="score threshold",
    )
    parser.add_argument("--device-id", type=int, default=0)
    return parser.parse_args()


def build_quantize_model(model, config, pretrained_ckpt=None):

    model.set_qconfig()
    qconfig_manager.set_qconfig_mode(qconfig_manager.QconfigMode.QAT)
    qat_model = horizon.quantization.prepare_qat_fx(model)
    if pretrained_ckpt is None:
        pretrained_ckpt = os.path.join(
            config.ckpt_dir, "qat-checkpoint-best.pth.tar"
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
    init_logger(f"work_dirs/hat_logss/{config.task_name}_infer_viz")

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device_id)
    torch.cuda.set_device(int(args.device_id))
    horizon.march.set_march(config.get("march", horizon.march.March.BAYES))

    model = build_from_registry(config.model)
    quantized_model = build_quantize_model(model, config, args.pretrained_ckpt)
    quantized_model.cuda()
    dataset, _, _, viz_update = dataset_mappings(args.dataset, config)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        collate_fn=collate_2d,
    )

    for ind, data in enumerate(dataloader):
        data["img"] = data["img"].cuda()
        results = quantized_model(data)
        data["img"] = data["padded_img"][0]
        data["filename"] = data["filename"][0]
        viz_update(data, results, args.save_dir, args.score_thr)
        if ind == 10:
            break
