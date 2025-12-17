import argparse
import logging
import os
import sys
import time
from importlib import import_module
from types import ModuleType
from typing import List

from aidisdk import AIDIClient
from aidisdk.infra import get_docker_image
from aidisdk.model import ModelFileSpec, ModelTags
from models.example_torch_model import HatModel

from hat.utils.config import Config

client = AIDIClient(endpoint="http://aidi.hobot.cc")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def trace_config_deps(cfg: str, exist_py_deps: List[ModuleType]):
    # TODO hardcode..., fixme later
    # error using AidiClient..model_registry.save_infer_model
    _EXCEPTIONS = [
        "_",
        "horizon_plugin_profiler",
        "horizon_plugin_pytorch",
        "torch",
        "mmcv",
        "mmdet",
        "pytorch",
        "distutils",
    ]
    assert cfg.endswith(".py"), "Only support .py config NOW!"
    config_dir = os.path.dirname(cfg)
    old_moduld = set(sys.modules.keys())
    sys.path.insert(0, config_dir)
    import_module(os.path.basename(cfg)[:-3])
    new_moduld = set(sys.modules.keys())
    config_deps = new_moduld - old_moduld
    deps = set()
    for dep in config_deps:
        dep = sys.modules[dep]
        # only trace single py file or pkg (with __init__.py) deps
        if not hasattr(dep, "__file__"):
            continue
        if dep.__file__ is not None:
            base_deps = dep.__name__.split(".")[0]
            dep = import_module(base_deps)
            path = dep.__file__
            if not path:
                continue
            # filter dep by exist py deps
            _skip = False
            for exist_dep in exist_py_deps:
                if exist_dep.__name__ in dep.__name__:
                    _skip = True
                    break
                if any([dep.__name__.startswith(i) for i in _EXCEPTIONS]):
                    _skip = True
            if _skip:
                continue
            if dep in deps:
                continue
            if path.endswith("__init__.py"):
                print(
                    f"Trace config module Dependence: {dep.__name__}, "
                    f"Package Path: {os.path.dirname(dep.__file__)}"
                )
                deps.add(dep)
            else:
                if os.path.abspath(path) != os.path.abspath(cfg):
                    print(
                        f"Trace config module Dependence: {dep.__name__}, "
                        f"File Path: {dep.__file__}"
                    )
                    deps.add(dep)
    return list(deps)


def create_model(
    client: AIDIClient,
    model_name_version: str,
    project: str,
    tags: ModelTags = None,
    desc: str = None,
    wait_finish=True,
    wait_timeout=200,
):
    # check if model name already exists
    if not client.model_registry.exist(model_name_version):
        client.model_registry.create(
            name=model_name_version,
            project=project,
            desc=desc,
            task_type=tags.algo_type,
            library=tags.framework,
        )
        logger.debug(f"add model {model_name_version}")

    if wait_finish or False:
        tic = time.time()
        finish = client.model_registry.exist(model_name_version)
        while not finish:
            if wait_timeout is not None and time.time() - tic > wait_timeout:
                raise TimeoutError(
                    f"!!![{model_name_version}]Time out creating aidi model card."  # noqa
                )
            if int(time.time() - tic) % 20 == 0:
                finish = client.model_registry.exist(f"{model_name_version}")
                logger.info(
                    f"[{model_name_version}]waiting for create aidi model card: {time.time() - tic}s"  # noqa
                )
                time.sleep(1)

    return client.model_registry.load(model_name_version)


def main(
    config: str,
    project: str,
    publish_version: str = None,
    model_params: str = None,
    disable_inference: bool = False,
):
    cfg = Config.fromfile(config)
    config_deps = trace_config_deps(cfg["model_config"], cfg["py_deps"])
    model_name = cfg["model_name"]
    model_version = publish_version
    model_name_version = f"{model_name}:{model_version}"

    tags = cfg["model_tags"]
    tags = ModelTags.from_dict(tags)
    spec = ModelFileSpec(
        model_file=cfg["model_config"],
    )
    if model_params is not None:
        param_file = model_params
    else:
        param_file = cfg.get("param_file")
    if param_file is not None:
        spec.add_attachment("checkpoint_file", param_file)
    if cfg.get("readme_file", None) is not None:
        readme_file = cfg.get("readme_file")
        if os.path.isfile(readme_file):
            spec.add_attachment("readme_file", readme_file)

    model_cls = cfg.get("model_cls", HatModel)
    model = model_cls(
        model_file_spec=spec,
    )

    if not client.model_registry.exist(model_name_version):
        create_model(
            client=client,
            model_name_version=model_name_version,
            project=project,
            tags=tags,
            desc=cfg.get("model_desc", ""),
        )
    # dump infer files to local model cache
    model_detail = client.model_registry.save_infer_model(
        name=model_name_version,
        model_cls=model_cls,
        model_file_spec=spec,
        tags=tags,
        py_deps=cfg.get("py_deps", []) + config_deps,
        docker_image=cfg["docker_image"],
        desc=cfg.get("model_desc", ""),
        instance_config=cfg.get("instance_config", None),
    )
    # prepare package for infer deploy
    if not disable_inference:
        docker_image = cfg["docker_image"] or get_docker_image()
        client.model_registry.inference_module.acquire_pkg(
            base_image=docker_image,
            model_name=model_detail.name,
            model_version=model_detail.version_name,
        )
    print("Save model success!")

    return model


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--project",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--publish-version",
        type=str,
        required=False,
        default=None,
        help="if specify version, will overwrite model_version in config",
    )
    parser.add_argument(
        "--model-params",
        type=str,
        required=False,
        default=None,
        help="if specify params, will overwrite params_path in config.model_info",  # noqa
    )
    parser.add_argument(
        "--disable-inference",
        required=False,
        action="store_true",
        help="if disable_inference, just save model without launch service",
    )

    args, argv = parser.parse_known_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    pipeline = main(**vars(args))
