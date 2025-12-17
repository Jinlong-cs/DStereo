"""align bpu validation tools, Only support int-infer"""
import argparse

import horizon_plugin_pytorch as horizon
import torch
import torchvision

from hat.registry import build_from_registry
from hat.utils import qconfig_manager
from hat.utils.apply_func import to_cuda
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
        "--device-id",
        type=int,
        required=False,
        default=0,
        help="Reference device, only support single device.",
    )
    parser.add_argument(
        "--pretrained-ckpt", type=str, default=None, help="pre-trained model"
    )
    known_args, unknown_args = parser.parse_known_args()
    return known_args, unknown_args


def build_fx_model(model):

    qconfig_manager.set_qconfig_mode(qconfig_manager.QconfigMode.QAT)
    model.set_qconfig()
    qat_model = horizon.quantization.prepare_qat_fx(model)

    return qat_model


def build_eager_model(model):
    model.fuse_model()
    qconfig_manager.set_qconfig_mode(qconfig_manager.QconfigMode.QAT)
    model.set_qconfig()
    qat_model = horizon.quantization.prepare_qat(model, inplace=False)

    return qat_model


def build_quantize_model(model, pretrained_ckpt=None, convert_mode="fx"):
    if convert_mode == "fx":
        qat_model = build_fx_model(model)
    elif convert_mode == "eager":
        qat_model = build_eager_model(model)
    else:
        raise ValueError(f"Does not support convert_mode {convert_mode}")

    checkpoint = torch.load(pretrained_ckpt)
    print(f"load pretrained model from {pretrained_ckpt}")
    qat_model.load_state_dict(checkpoint["state_dict"], strict=False)

    quantized_model = horizon.quantization.convert(qat_model, inplace=False)
    quantized_model.eval()
    return quantized_model


def defualt_prepare_inputs(infer_inputs):

    return [infer_inputs]


def setup_modelinputs(unknown_args):
    modelinputs = {}
    i = 0
    while i < len(unknown_args):
        if "=" in unknown_args[i]:
            modelinputs_info = unknown_args[i].split("=")
            modelinputs_info[0] = modelinputs_info[0].strip("-")
            modelinputs_info[0] = modelinputs_info[0].replace("-", "_")
            modelinputs.update({modelinputs_info[0]: modelinputs_info[1]})
            i += 1
        else:
            modelinputs_info = unknown_args[i].strip("-").replace("-", "_")
            modelinputs.update({modelinputs_info: unknown_args[i + 1]})
            i += 2

    return modelinputs


if __name__ == "__main__":
    args, unknown_args = parse_args()
    torch.cuda.set_device(int(args.device_id))
    config = Config.fromfile(args.config)
    init_logger(f"work_dirs/hat_logss/{config.task_name}_infer_viz")
    horizon.march.set_march(config.get("march"))

    infer_cfg = config.get("infer_cfg")

    model = build_from_registry(infer_cfg.get("model"))

    convert_mode = infer_cfg.get("convert_mode")
    if args.pretrained_ckpt is not None:
        checkpoint_path = args.pretrained_ckpt
    else:
        checkpoint_path = infer_cfg.get("checkpoint_path")

    viz_func = build_from_registry(infer_cfg.get("viz_func"))
    prepare_inputs = infer_cfg.get("prepare_inputs", defualt_prepare_inputs)
    process_inputs = infer_cfg.get("process_inputs")
    process_outputs = infer_cfg.get("process_outputs")

    if len(unknown_args):
        infer_inputs = setup_modelinputs(unknown_args)
    else:
        infer_inputs = infer_cfg.get("infer_inputs")
    transforms = infer_cfg.get("transforms", None)
    if transforms is not None:
        transforms = build_from_registry(transforms)
        transforms = torchvision.transforms.Compose(transforms)

    quantized_model = build_quantize_model(
        model, checkpoint_path, convert_mode
    )

    quantized_model.cuda(args.device_id)

    prepared_inputs = prepare_inputs(infer_inputs)

    for prepared_input in prepared_inputs:
        model_input, vis_inputs = process_inputs(prepared_input, transforms)
        model_input = to_cuda(
            model_input, device=args.device_id, non_blocking=True
        )
        with torch.no_grad():
            model_outputs = quantized_model(model_input)

        outputs = process_outputs(model_outputs, viz_func, vis_inputs)
        if outputs is not None:
            print(outputs)
