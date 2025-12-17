import argparse
import copy
import logging
import os

import horizon_plugin_pytorch as horizon
import torch

from hat.core.compose_transform import Compose
from hat.registry import RegistryContext, build_from_registry
from hat.utils.apply_func import _as_list
from hat.utils.config import Config
from hat.utils.logger import MSGColor, format_msg
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
    known_args, unknown_args = parser.parse_known_args()
    return known_args, unknown_args


if __name__ == "__main__":
    args, args_env = parse_args()
    if args_env:
        setup_args_env(args_env)
    cfg = Config.fromfile(args.config)

    logger.info("=" * 50 + "BEGIN PROFILER STAGE" + "=" * 50)

    if "march" not in cfg:
        logger.warning(
            format_msg(
                f"Please make sure the march is provided in configs. "
                f"Defaultly use {horizon.march.March.BAYES}",
                MSGColor.RED,
            )
        )
    horizon.march.set_march(cfg.get("march", horizon.march.March.BAYES))

    tool = cfg.model_profiler_solver["tool"]["type"]
    if tool not in ("CheckShared", "CheckFused"):
        # process out_dir
        if cfg.model_profiler_solver["tool"].get("out_dir", None) is None:
            cfg.model_profiler_solver["tool"]["out_dir"] = os.path.join(
                cfg.get("ckpt_dir", "."), "profiler"
            )
        out_dir = cfg.model_profiler_solver["tool"]["out_dir"]
        if not os.path.exists(out_dir):
            os.mkdir(out_dir)

    with RegistryContext():
        profiler = build_from_registry(cfg.model_profiler_solver)
        model = profiler["model"].eval()
        pipelines = _as_list(profiler["model_convert_pipeline"])
        assert (
            len(pipelines) == 2
            if tool
            in ("FeaturemapSimilarity", "CompareWeights", "ModelProfiler")
            else 1
        ), (
            "Two model_convert_pipelines are needed when using featuremap "
            + "similarity tool. Otherwise, only one model_convert_pipeline is "
            + "needed."
        )

        # build models
        models = []
        for pipeline in pipelines:
            m = copy.deepcopy(model)
            models.append(pipeline(m))

        # process inputs
        example_inputs = profiler.get("inputs", None)
        if example_inputs is None:
            dataloader = profiler.get("dataloader", None)
            data_index = profiler.get("data_index", None)
            assert dataloader is not None and data_index is not None, (
                "Please set `inputs` or `dataloader & data_index` in "
                + "model_profiler_solver."
            )
            for i, _data in enumerate(dataloader):
                if i == data_index:
                    break
            logger.info(f"Use index {i} data in dataloader.")
            example_inputs = _data
            batch_transforms = profiler.get("batch_transforms", None)
            if batch_transforms:
                if isinstance(batch_transforms, (list, tuple)):
                    batch_transforms = Compose(batch_transforms)
                example_inputs = batch_transforms(example_inputs)
        elif isinstance(example_inputs, str):
            logger.info(f"Load example inputs from {example_inputs}...")
            if not os.path.exists(example_inputs):
                raise ValueError(f"File path {example_inputs} does not exist.")
            example_inputs = torch.load(example_inputs)

        logger.info(
            "using profiler tool: {}".format(
                cfg.model_profiler_solver["tool"]["type"]
            )
        )

        if tool in ("CompareWeights", "CheckDeployDevice"):
            profiler["tool"](*models)
        else:
            profiler["tool"](*models, example_inputs)
        logger.info("=" * 50 + "END PROFILER" + "=" * 50)
