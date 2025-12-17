import argparse
import logging
import os

import horizon_plugin_pytorch as horizon
import torch
import torch_tensorrt

from hat.registry import RegistryContext, build_from_registry
from hat.utils.config import Config
from hat.utils.distributed import get_dist_info
from hat.utils.logger import LOG_DIR, MSGColor, format_msg, init_rank_logger
from hat.utils.setup_env import setup_args_env

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)-15s %(levelname)s %(message)s",
    level=logging.INFO,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--device-ids",
        "-ids",
        type=str,
        required=False,
        default=None,
    )
    known_args, unknown_args = parser.parse_known_args()
    return known_args, unknown_args


if __name__ == "__main__":
    args, args_env = parse_args()
    if args_env:
        setup_args_env(args_env)
    cfg = Config.fromfile(args.config)

    rank, world_size = get_dist_info()
    logger = init_rank_logger(
        rank,
        save_dir=cfg.get("log_dir", LOG_DIR),
        cfg_file=args.config,
        step="export-trt",
        prefix="export-tensorrt-",
    )

    if args.device_ids is not None:
        device = list(map(int, args.device_ids.split(",")))
    else:
        device = cfg.device_ids

    if isinstance(device, (list, tuple)):
        device = device[0]
        logger.warning("Compile tensorrt uses first device by default.")

    logger.info("=" * 50 + "BEGIN EXPORT TENSORRT" + "=" * 50)

    if "march" not in cfg:
        logger.warning(
            format_msg(
                f"Please make sure the march is provided in configs. "
                f"Defaultly use {horizon.march.March.BAYES}",
                MSGColor.RED,
            )
        )
    horizon.march.set_march(cfg.get("march", horizon.march.March.BAYES))

    with RegistryContext():
        trt_solver = build_from_registry(cfg.tensorrt_cfg)
        model = trt_solver["model"].eval()
        pipeline = trt_solver.get("model_convert_pipeline", None)
        if pipeline is not None:
            model = pipeline(model)

    out_dir = trt_solver.get("out_dir", "./tmp_models/")
    os.makedirs(out_dir, exist_ok=True)

    example_input = trt_solver.get("example_inputs")

    if (device is not None) and (device != -1):
        example_input = example_input.cuda(device)
        model.cuda(device)

    traced_model = torch.jit.trace(model, [example_input])
    pt_filename = os.path.join(out_dir, "traced_model.pt")
    torch.jit.save(traced_model, pt_filename)
    logger.info("=" * 50 + "TRACE SUCCESSFULL" + "=" * 50)

    inputs = [torch_tensorrt.Input(**trt_solver["input_cfg"])]
    trt_ts_module = torch_tensorrt.ts.compile(
        traced_model, inputs=inputs, **trt_solver["compile_cfg"]
    )
    ts_filename = os.path.join(out_dir, "trt_ts_module.ts")
    torch.jit.save(trt_ts_module, ts_filename)
    logger.info("=" * 50 + "COMPILE SUCCESSFULL" + "=" * 50)

    logger.info("=" * 50 + "END TENSORRT" + "=" * 50)
