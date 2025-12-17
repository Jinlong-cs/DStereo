"""align bpu validation tools, Only support int-infer"""
import argparse

import horizon_plugin_pytorch as horizon
import torch
from dataset_mappings import dataset_mappings

from hat.callbacks.metric_updater import MetricUpdater
from hat.callbacks.validation import Validation
from hat.data.collates.collates import (
    collate_2d,
    collate_lidar3d,
    collate_mot_seq,
)
from hat.data.collates.nusc_collates import (
    collate_nuscenes,
    collate_nuscenes_sequence,
)
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
        "--dataset",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--pretrained-ckpt",
        type=str,
        default=None,
        help="QAT checkpoint for int-infer.",
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
    parser.add_argument(
        "--use-testmodel",
        action="store_true",
        help="Whether use test_model in config",
    )
    parser.add_argument(
        "--use-eager",
        action="store_true",
    )
    parser.add_argument(
        "--march",
        type=str,
        default="bernoulli2",
        help="Use which march for config",
    )
    return parser.parse_args()


march_mapping = {
    "bernoulli2": horizon.march.March.BERNOULLI2,
    "bayes": horizon.march.March.BAYES,
}


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
        pretrained_ckpt = config.int_infer_trainer["model_convert_pipeline"][
            "converters"
        ][1]["checkpoint_path"]

    checkpoint = torch.load(pretrained_ckpt)
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

    march = march_mapping[args.march]
    horizon.march.set_march(config.get("march", march))

    dataset, val_metrics, update_metric, _ = dataset_mappings(
        args.dataset, config
    )
    if args.use_testmodel:
        model = build_from_registry(config.test_model)
    else:
        model = build_from_registry(config.model)
    quantized_model = build_quantize_model(
        model, config, args.pretrained_ckpt, args.use_eager
    )
    if args.dataset == "kitti3d" or ("nuscenes_lidar" in args.dataset):
        collate_fn_func = collate_lidar3d
    elif args.dataset == "bev_nuscenes" or args.dataset == "detr3d_nuscenes":
        collate_fn_func = collate_nuscenes
    elif args.dataset == "bev_nuscenes_sequence":
        collate_fn_func = collate_nuscenes_sequence
    elif args.dataset == "mot17":
        collate_fn_func = collate_mot_seq
    else:
        collate_fn_func = collate_2d

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
