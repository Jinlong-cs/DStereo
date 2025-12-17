"""align bpu validation tools, Only support int-infer"""
import argparse

import horizon_plugin_pytorch as horizon
import torch

from hat.callbacks.metric_updater import MetricUpdater
from hat.callbacks.validation import Validation
from hat.engine import Predictor
from hat.engine.processors import BasicBatchProcessor
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
        "--device-id",
        type=int,
        required=False,
        default=0,
        help="Reference device, only support single device.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        required=False,
        default=1,
        help="The number of workers to load image.",
    )
    return parser.parse_args()


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

    checkpoint = torch.load(pretrained_ckpt, map_location="cpu")
    print(f"load pretrained model from {pretrained_ckpt}")
    qat_model.load_state_dict(checkpoint["state_dict"], strict=False)

    quantized_model = horizon.quantization.convert(qat_model, inplace=False)
    quantized_model.eval()
    return quantized_model


def build_predictor(
    model, data_loader, val_metrics, update_metric, config, args
):
    val_batch_processor = BasicBatchProcessor(need_grad_update=False)
    metric_updater = MetricUpdater(
        metric_update_func=update_metric,
        step_log_freq=100000,
        epoch_log_freq=1,
        log_prefix="Validation " + config.task_name,
    )
    val_callback = Validation(
        data_loader=data_loader,
        batch_processor=val_batch_processor,
        callbacks=[metric_updater],
        log_interval=1,
    )
    predictor = Predictor(
        model=model,
        data_loader=data_loader,
        batch_processor=val_batch_processor,
        device=args.device_id,
        callbacks=[val_callback],
        num_epochs=0,
        log_interval=1,
    )
    predictor.val_metrics = val_metrics
    return predictor


if __name__ == "__main__":
    args = parse_args()
    torch.cuda.set_device(int(args.device_id))
    config = Config.fromfile(args.config)
    init_logger(f"work_dirs/hat_logss/{config.task_name}_align_bpu_validation")
    horizon.march.set_march(config.get("march"))

    align_bpu_cfg = config.get("align_bpu_cfg")
    dataset = build_from_registry(align_bpu_cfg.get("dataset"))
    val_metrics = build_from_registry(align_bpu_cfg.get("metrics"))
    update_metric = build_from_registry(align_bpu_cfg.get("metric_updater"))
    model = build_from_registry(align_bpu_cfg.get("model"))
    convert_mode = align_bpu_cfg.get("convert_mode")
    checkpoint_path = align_bpu_cfg.get("checkpoint_path")
    collate_fn_func = align_bpu_cfg.get("collate_fn_func")

    quantized_model = build_quantize_model(
        model, checkpoint_path, convert_mode
    )

    data_loader = torch.utils.data.DataLoader(
        dataset=dataset,
        collate_fn=collate_fn_func,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    predictor = build_predictor(
        model=quantized_model,
        data_loader=data_loader,
        val_metrics=val_metrics,
        update_metric=update_metric,
        config=config,
        args=args,
    )
    predictor.fit()
