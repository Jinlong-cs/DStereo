import argparse
import hashlib
import json
import logging
import multiprocessing
import os
import os.path as osp
import shutil
import subprocess
import time
from typing import Dict, List

from tools.deploy.compile_standalone import compile_standalone

import horizon_plugin_pytorch as horizon
import torch
import yaml
from aidisdk import AIDIClient
from easydict import EasyDict
from file_helper import CacheFile, upload_to_gallery, zip_dir
from hatbc.utils import _as_list
from termcolor import cprint

from hat.registry import RegistryContext, build_from_registry
from hat.utils.config import Config
from hat.utils.dag_node import EnvironContainer, ModuleContainer
from projects.pilot.configs.project_utils.enum import version_matcher

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

pt_results = {}


def check_and_convert(config, publish_version, sub_project):
    """
    Update model descs and convert it to IntInfer model.
    """
    for compiled_name, cfg_i in config.models.items():
        if cfg_i.get("compiled_file"):
            logger.warning(f"Skip Convert compiled: {cfg_i.compiled_file}")
            continue

        update_envrion = {}

        if "model_setting" in cfg_i:
            update_envrion["HAT_PILOT_MODEL_SETTING"] = cfg_i.model_setting

        # 1. update dpp thresh and detection thresh for descs in cfg.
        update_cfg = cfg_i.get("update_cfg", {})
        if update_cfg is not None:
            update_envrion["HAT_PILOT_MODEL_THRESH"] = json.dumps(update_cfg)

        split_flag = cfg_i.get("split_flag", None)
        if split_flag:
            update_envrion["HAT_PILOT_SPLIT_FLAG"] = split_flag

        with ModuleContainer(), EnvironContainer(update_envrion):
            model_name = (
                f"{compiled_name}_{config.march}_"
                f"{sub_project}_{publish_version}"
            )

            model_cfg = Config.fromfile(cfg_i.cfg_path)

            pt_file = os.path.join(
                model_cfg.ckpt_dir,
                f"int_infer_{model_name}-deploy-checkpoint-last.pt",
            )

            pt_results[compiled_name] = pt_file

            if os.path.exists(pt_file) and not cfg_i.get("override", False):
                logger.warning(f"Skip {model_name}")
                continue

            # 2. modify cfg of int_infer stage
            model_cfg["int_checkpoint"]["save_hash"] = False
            model_cfg["int_checkpoint"][
                "name_prefix"
            ] = f"int_infer_{model_name}-"
            model_cfg["int_infer_trainer"]["model_convert_pipeline"][
                "converters"
            ][1]["checkpoint_path"] = cfg_i.model_address

            # 3. generate int_infer model according to modified qat model
            horizon.march.set_march(config.march)
            model_cfg["march"] = config.march

            with RegistryContext():
                trainer = model_cfg.int_infer_trainer
                trainer = build_from_registry(trainer)
                trainer.fit()


def compile_local(config, publish_version, sub_project, release: bool = False):
    """
    Compile model accoridng to configs in local environment.
    """
    publish_tag = f"{config.publish_name}-{publish_version}"
    out_dir = osp.abspath(osp.join(config.output_dir, publish_tag))
    if osp.exists(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    hbm_out_dir = osp.join(out_dir, "hbms")
    os.makedirs(hbm_out_dir, exist_ok=True)
    output_packs: Dict[str, List[str]] = {}
    for compiled_name, cfg_i in config.models.items():
        model_name = f"{compiled_name}_{config.march}_{sub_project}_{publish_version}"  # noqa

        if cfg_i.get("compiled_file"):
            hbm = osp.join(hbm_out_dir, osp.basename(cfg_i.compiled_file))
            with CacheFile(url=cfg_i.compiled_file) as cache_file:
                shutil.copyfile(cache_file.name, hbm)
            logger.warning(f"Skip Compile compiled: {cfg_i.compiled_file}")
        else:
            hbm = os.path.join(hbm_out_dir, f"{model_name}.hbm")

        for output_pack in _as_list(
            cfg_i.get("output_packs", config.get("compiled_hbm_name"))
        ):
            if output_pack not in output_packs:
                output_packs[output_pack] = []
            output_packs[output_pack].append(hbm)

        if cfg_i.get("compiled_file"):
            continue

        t1 = time.time()
        opt = config.optimization_level
        compile_standalone(
            in_file=pt_results[compiled_name],
            input_size=cfg_i.input_shape,
            name=model_name,
            opt=opt,
            march=config.march,
            perf_only=not cfg_i.get("override", False),
            output=hbm_out_dir,
            jobs=cfg_i.jobs_num,
            input_key=cfg_i.input_key,
            torch_native=cfg_i.get("torch_native", "False"),
            input_source=cfg_i.input_source,
            input_layout=cfg_i.input_layout,
            output_layout=cfg_i.output_layout,
            debug=cfg_i.get("debug", False),
            dump_cmp_info=cfg_i.get("dump_cmp_info", False),
            extra_args=cfg_i.get("extra_args", ""),
            verify_model=release,
        )
        for suffix in ["pt", "hbm", "hbir"]:
            src_file = os.path.join(hbm_out_dir, f"model_opt_{opt}.{suffix}")
            if os.path.exists(src_file):
                os.rename(
                    src_file,
                    os.path.join(hbm_out_dir, f"{model_name}.{suffix}"),
                )

        logger.info(f"compile cost {(time.time() - t1):.2f} sec")

    # pack model
    perf_out_dir = osp.join(out_dir, "perf")
    os.makedirs(perf_out_dir, exist_ok=True)
    output_packs_md5: Dict[str, str] = {}
    for name, hbms in output_packs.items():
        outpath = os.path.join(out_dir, name)
        subprocess.run(
            f"hbdk-pack --output {outpath}"
            + f" --tag {config.desc}-{publish_version} "
            + " ".join(hbms),
            shell=True,
        )
        subprocess.run(
            f"hbdk-perf {outpath} --output-dir {perf_out_dir}",
            shell=True,
        )

        with open(outpath, "rb") as fin:
            output_packs_md5[name] = hashlib.md5(fin.read()).hexdigest()
    # dump package config
    with open(osp.join(out_dir, "config.yaml"), "w") as fout:
        yaml.safe_dump(
            data={
                n: [osp.relpath(p, out_dir) for p in ps]
                for n, ps in output_packs.items()
            },
            stream=fout,
        )

    with open(osp.join(out_dir, "md5.json"), "w") as fout:
        json.dump(output_packs_md5, fout, indent=4, separators=(",", ":"))

    # zip model
    package_file = osp.join(out_dir, f"{publish_tag}.zip")
    zip_dir(out_dir, package_file, True)
    if release and config.get("gallery_config"):
        with open(package_file, "rb") as fin:
            md5 = hashlib.md5(fin.read()).hexdigest()

        gallery_config = config.gallery_config
        pre_dir = osp.abspath(os.getcwd())
        os.chdir(osp.dirname(package_file))
        package_file = upload_to_gallery(
            results=package_file,
            version=f"{publish_tag}_{md5[:4]}",
            group=gallery_config.group,
            project=gallery_config.project,
            username="pilot.runner",
            password="!Kj70Ic7feMh%ER$",
        )
        os.chdir(pre_dir)

    cprint(package_file, color="green")


def upload_to_aidiexp(config, publish_version, sub_project, overwrite=False):
    client = AIDIClient()

    for compiled_name, cfg_i in config.models.items():
        if cfg_i.get("compiled_file"):
            logger.error(f"NotImplementedError for compiled_file {cfg_i}")
            continue

        model_name = f"{compiled_name}_{config.march}_{sub_project}_{publish_version}"  # noqa

        task_type = cfg_i.task_type

        # create experimental model
        if not client.model.exist(model_name):
            client.model.set_public_authority()
            client.model.create(model_name=model_name, task_type=task_type)

        if not client.model.exist(model_name, model_version=publish_version):
            model_version_item = client.model.create_version(
                model_name=model_name,
                model_version=publish_version,
                framework="PyTorch",
                version_tags=[publish_version, "PyTorch"],
            )
            logger.info(f"created model version {str(model_version_item)}")

        if client.model.exist(
            model_name, model_version=publish_version, stage="int_infer"
        ):
            if overwrite:
                try:
                    client.model.delete(
                        model_name,
                        model_version=publish_version,
                        stage="int_infer",
                    )
                except Exception as e:
                    logger.error(
                        f"{model_name}+{publish_version} overwrite failed"
                    )
                    raise e
            else:
                raise ValueError(
                    f"Model {model_name}+{publish_version} already exists,"
                )

        client.model.upload_checkpoint(
            model_name=model_name,
            model_version=publish_version,
            stage="int_infer",
            model_file=pt_results[compiled_name],
        )


def async_compile_task(task):
    task.wait()
    compile_model_url = task.get_download_url()
    logger.info("------------------- compile task success! -----------------")
    logger.info(f"Compile model url: {compile_model_url}")
    return compile_model_url


def compile_publish_aidi(
    config, publish_version, sub_project, return_compile_url, release=False
):
    client = AIDIClient()

    # create publish model
    publish_name = config.publish_name
    if not release:
        publish_name += "_" + client.session.current_user

    if not client.modelpublish.exist(publish_name=publish_name):
        client.modelpublish.create(
            publish_name=publish_name, desc=config.desc
        ).create_version(
            publish_name, publish_version=publish_version, commit="a commit"
        )

    if not client.modelpublish.exist(
        publish_name=publish_name, publish_version=publish_version
    ):
        client.modelpublish.create_version(
            publish_name=publish_name,
            publish_version=publish_version,
            commit=config.get("commit", None),
        )

    # model compile
    for compiled_name, cfg_i in config.models.items():
        if cfg_i.get("compiled_file"):
            logger.error(f"NotImplementedError for compiled_file {cfg_i}")
            continue

        model_name = (
            f"{compiled_name}_{config.march}_{sub_project}_{publish_version}"
        )

        output_packs = cfg_i.get(
            "output_packs", config.get("compiled_hbm_name")
        )
        output_layout = cfg_i.get("output_layout", config.get("output_layout"))
        client.modelpublish.set_compile_parameters(
            input_shape=cfg_i.input_shape,
            input_type=cfg_i.input_type,
            optimization_level=int(config.optimization_level[1:]),
            plugin_version=config.plugin_version,
            input_source=cfg_i.input_source,
            cpu=cfg_i.get("jobs_num"),
            input_key=cfg_i.input_key,
            extra_args=cfg_i.get("extra_args", ""),
            torch_native=cfg_i.get("torch_native", "False"),
            input_layout=cfg_i.get("input_layout"),
            output_layout=output_layout,
        ).append_model_to_compile(
            client.model.finditem(
                model_name,
                publish_version,
                "int_infer",
            ),
            compiled_name=compiled_name,
            output_packs=_as_list(output_packs) if output_packs else None,
        )

    compile_task = client.modelpublish.compile(
        publish_name=publish_name,
        publish_version=publish_version,
        hbcc_version=config.hbcc_version,
        framework_version=torch.__version__.split("+")[0],
        hbdk_internal=True,
        march=config.march,
        tag_template="${model_name}-V${model_version}.hbm",
        optimization_level=int(config.optimization_level[1:]),
    )

    if return_compile_url:
        logger.info(
            "-------------------- waiting compile task ------------------"
        )
        pool = multiprocessing.Pool(processes=1)
        wait_task = pool.apply_async(async_compile_task, (compile_task,))
        pool.close()
        return wait_task


def trace_and_compile(
    sub_project,
    publish_version,
    compile_mode,
    overwrite,
    return_compile_url=False,
    release=False,
):
    assert version_matcher.search(publish_version) is not None
    pub_config = Config.fromfile(
        os.path.join(
            os.path.dirname(__file__),
            f"cfg/pub_cfg_{sub_project}.py",
        ),
    )
    pub_config = EasyDict(pub_config)

    os.environ["HAT_TRAINING_STEP"] = "int_infer"

    check_and_convert(pub_config, publish_version, sub_project)

    if compile_mode == "local":
        compile_local(pub_config, publish_version, sub_project, release)
    elif compile_mode == "aidi":
        upload_to_aidiexp(
            pub_config, publish_version, sub_project, overwrite=overwrite
        )
        return compile_publish_aidi(
            pub_config,
            publish_version,
            sub_project,
            return_compile_url=return_compile_url,
            release=release,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sub-project", type=str, required=True)
    parser.add_argument("--publish-version", type=str, required=True)
    parser.add_argument(
        "--compile-mode",
        type=str,
        default="local",
        help="choose compile in aidi or local, default is local.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="overwrite existed model items in aidi exp model",
    )
    parser.add_argument(
        "--release",
        action="store_true",
        help="whether is release model",
    )
    parser.add_argument(
        "--return-compile-url",
        action="store_true",
        help="wait for compile task to complete, and return compile model url,"
        "only work for aidi compile",
    )
    args = parser.parse_args()
    trace_and_compile(**vars(args))
