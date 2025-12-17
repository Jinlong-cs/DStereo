import argparse
import json
import logging
import multiprocessing
import os
import subprocess
import time

import horizon_plugin_pytorch as horizon
import torch
from aidisdk import AIDIClient
from easydict import EasyDict

from hat.registry import RegistryContext, build_from_registry
from hat.utils.config import Config

logger = logging.getLogger(__name__)


pt_results = {}


def check_and_convert(config, publish_version):
    """
    Update model descs and convert it to IntInfer model.
    """
    for compiled_name, cfg_i in config.models.items():
        # model_name = f"{compiled_name}_{config.march}_{publish_version}"
        model_name = f"{compiled_name}_{config.march}"

        if "model_setting" in cfg_i:
            os.environ["HAT_TL_MODEL_SETTING"] = cfg_i.model_setting

        # 1. update dpp thresh and detection thresh for descs in cfg.
        update_cfg = cfg_i.get("update_cfg", {})
        if update_cfg:
            os.environ["HAT_TL_MODEL_THRESH"] = json.dumps(update_cfg)
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
        model_cfg["int_checkpoint"]["name_prefix"] = f"int_infer_{model_name}-"
        # model_cfg["int_infer_trainer"]["model_convert_pipeline"]["converters"][
        #     1
        # ]["checkpoint_path"] = cfg_i.model_address

        # 3. generate int_infer model according to modified qat model
        horizon.quantization.march = config.march
        model_cfg["march"] = config.march

        with RegistryContext():
            trainer = model_cfg.int_infer_trainer
            trainer = build_from_registry(trainer)
            trainer.fit()


def compile_local(config, publish_version):
    """
    Compile model accoridng to configs in local environment.
    """

    hbms = []
    for compiled_name, cfg_i in config.models.items():
        model_name = f"{compiled_name}_{config.march}_{publish_version}"

        out_dir = config.output_dir
        if out_dir != "" and not os.path.exists(out_dir):
            os.makedirs(out_dir)

        hbm = os.path.join(
            out_dir,
            f"{model_name}_{config.optimization_level}.hbm",
        )

        hbms.append(hbm)

        if os.path.exists(hbm) and not cfg_i.get("override", False):
            logger.warning(f"skip compile {model_name}")
            continue

        # load model file
        model = torch.jit.load(pt_results[compiled_name])

        example_inputs = None
        input_size = cfg_i.input_shape

        if cfg_i.input_type == "dict":
            group_input_size = [
                list(map(int, _input_size.split("x")))
                for _input_size in input_size.split("^")
            ]
            assert len(group_input_size[0]) in [4, 5]

            if len(group_input_size) > 1:
                example_inputs = {}
                input_keys = cfg_i.input_key.split("^")
                for i in range(len(input_keys)):
                    gis = group_input_size[i]
                    assert len(gis) == 4
                    example_inputs[input_keys[i]] = torch.randn(gis)
            elif len(group_input_size[0]) == 5:
                input_size = group_input_size[0]
                sequence_length = input_size[0]
                example_inputs = {
                    cfg_i.input_key: [
                        torch.randn(input_size[1:])
                        for _ in range(sequence_length)
                    ]
                }
            elif len(group_input_size[0]) == 4:
                input_size = group_input_size[0]
                example_inputs = {cfg_i.input_key: torch.randn(input_size)}
            else:
                logger.error("param invalid")
                return
        else:
            size = [int(i) for i in input_size.split("x")]
            example_inputs = [torch.ones(size[0], size[1], size[2], size[3])]

        cpu_num = cfg_i.jobs_num
        logger.info("compiling model ...")
        extra_args = cfg_i.get("extra_args", None)
        if extra_args:
            extra_args = extra_args.strip().split(" ")
        t1 = time.time()
        horizon.quantization.compile_model(
            module=model.eval(),
            example_inputs=example_inputs,  # list or dict
            march=config.march,
            input_source=[cfg_i.input_source],  # pyramid
            hbm=hbm,
            name=compiled_name,
            input_layout=cfg_i.input_layout,
            output_layout=cfg_i.output_layout,
            opt=config.optimization_level,
            progressbar=True,
            jobs=cpu_num,
            extra_args=extra_args,
        )
        logger.info(f"compile cost {(time.time() - t1):.2f} sec")

    result = subprocess.run(
        "hbdk-pack --output "
        + os.path.join(out_dir, config.compiled_hbm_name)
        + f" --tag {config.desc} "
        + " ".join(hbms),
        shell=True,
    )
    logger.info(result)


def upload_to_aidiexp(config, publish_version, overwrite=False):
    client = AIDIClient()

    for compiled_name, cfg_i in config.models.items():
        # model_name = f"{compiled_name}_{config.march}_{publish_version}"
        model_name = f"{compiled_name}_{config.march}"

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
    config, publish_version, return_compile_url, release=False
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
        # model_name = f"{compiled_name}_{config.march}_{publish_version}"
        model_name = f"{compiled_name}_{config.march}"

        input_shape = cfg_i.input_shape
        input_layout = cfg_i.get("input_layout")
        output_layout = cfg_i.get("output_layout", config.output_layout)

        client.modelpublish.set_compile_parameters(
            input_shape=input_shape,
            input_type=cfg_i.input_type,
            optimization_level=int(config.optimization_level[1:]),
            plugin_version=config.plugin_version,
            input_source=cfg_i.input_source,
            cpu=cfg_i.get("jobs_num"),
            input_key=cfg_i.input_key,
            extra_args=cfg_i.get("extra_args", ""),
            input_layout=input_layout,
            output_layout=output_layout,
        ).append_model_to_compile(
            client.model.finditem(
                model_name,
                publish_version,
                "int_infer",
            ),
            compiled_name=compiled_name,
            output_packs=[config.compiled_hbm_name],
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

    pub_config = Config.fromfile(
        os.path.join(
            os.path.dirname(__file__),
            f"cfg/pub_cfg_{sub_project}.py",
        ),
    )
    pub_config = EasyDict(pub_config)

    os.environ["HAT_TRAINING_STEP"] = "int_infer"

    check_and_convert(pub_config, publish_version)

    if compile_mode == "local":
        compile_local(pub_config, publish_version)
    elif compile_mode == "aidi":
        upload_to_aidiexp(pub_config, publish_version, overwrite=overwrite)
        return compile_publish_aidi(
            pub_config,
            publish_version,
            return_compile_url=return_compile_url,
            release=release,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sub-project", type=str, default="tl"
    )  # required=True)
    parser.add_argument(
        "--publish-version", type=str, default="v0.0.1"
    )  # required=True)
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
        "--return-compile-url",
        action="store_true",
        help="wait for compile task to complete, and return compile model url,"
        "only work for aidi compile",
    )
    args = parser.parse_args()
    trace_and_compile(**vars(args))
