#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Compile pt (or hbir) and perf_hbm standalone, without config.

Supported compilation processes:
  1. pt --> hbir --> hbm (& perf result).
  2. pt --> hbir.  (--hbir-only)
  3. hbir --> hbm (& perf result).
  4. hbm --> perf result.  (--perf-only)
"""

import argparse
import json
import logging
import multiprocessing
import os
import subprocess
import time
from distutils.version import LooseVersion
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import IO, Optional, Union

import hbdk
import horizon_plugin_pytorch as horizon
import torch
from hbdk.compiler import compile_hbir_model
from hbdk.torch_script.tools import perf_hbm
from horizon_plugin_pytorch import __version__
from horizon_plugin_pytorch.quantization import compile_model, export_hbir
from termcolor import cprint

try:
    from hbdk.torch_script.tools import placeholder
except ImportError:
    placeholder = None

try:
    from horizon_plugin_pytorch.utils.serialization import (
        get_version_from_scriptmodule,
    )
except ImportError:
    get_version_from_scriptmodule = None

try:
    import wget
except ImportError:
    wget = None

logger = logging.getLogger()
IS_LOCAL = not os.path.exists("/running_package")


def plugin_version_check(saved_version: str):
    if saved_version is not None:
        saved_plugin_version = saved_version.split("+")[0].split(".")
        current_plugin_version = __version__.split("+")[0].split(".")
        if (
            current_plugin_version[0] == saved_plugin_version[0]
            and current_plugin_version[1] == saved_plugin_version[1]
        ):
            pass
        else:
            logger.warning(
                "plugin version in checkpoint is different from "
                "the current plugin version, this may cause changes "
                "in qat and quantized accuracy."
            )
    else:
        logger.warning(
            "The model has not plugin version information, "
            "we can not check plugin version in model."
        )


def pt_plugin_version_check(pt_file: str):
    if get_version_from_scriptmodule is not None:
        pt_version = get_version_from_scriptmodule(pt_file)
        plugin_version_check(pt_version)
    else:
        logger.warning(
            "Please update your horizon-plugin-pytorch. "
            "Plugin version should be >= 0.14.6. If not, "
            "we can not check plugin version in pt file."
        )


def load_script_module(
    pt_path: Union[str, IO, Path],
    map_location: Optional[str] = None,
    _extra_files: Optional[dict] = None,
    check_plugin_version: bool = False,
) -> torch.jit.ScriptModule:
    """Load torch scriptmodule from path.

    Args:
        pt_path: Provided path for torch scriptmodule.
        map_location: Target device for scriptmodule.
        _extra_files: The extra information given in the map.
        check_plugin_version: Whether to check plugin version.

    Returns:
        pt_model: torch.jit.ScriptModule object.
    """

    if LooseVersion(__version__) >= LooseVersion("1.5.0"):
        # horizon.jit.load check compatibility of pt version and current
        # horizon_plugin_pytorch version. It prints warnings if some ops are
        # incompatible in this two versions.
        pt_model = horizon.jit.load(pt_path, map_location, _extra_files)
        return pt_model

    logger.warning(
        "Please upgrade horizon_plugin_pytorch to v1.5.0 or later, which "
        "gives more detailed description for plugin version compatibility."
    )
    pt_model = torch.jit.load(pt_path, map_location, _extra_files)
    if check_plugin_version:
        pt_plugin_version_check(pt_path)
    return pt_model


def generate_input(shape, torch_native=False, dtype=None):
    if torch_native:
        assert placeholder, "hbdk version should >= 3.28"
        return placeholder(shape, torch_native=True)
    else:
        if dtype is None:
            return torch.randn(shape)
        else:
            return torch.zeros(shape, dtype=dtype)


def get_input_type(input_type):
    if input_type is None:
        return None
    type_dict = {
        "float32": torch.float32,
        "int8": torch.int8,
        "int16": torch.int16,
    }
    input_type = input_type.split("^")
    res_type = []
    for e_type in input_type:
        assert e_type in type_dict, f"type{e_type} is not supported"
        res_type.append(type_dict[e_type])
    return res_type


def get_example_input(key, input_size, torch_native=False, dtype=None):
    group_input_size = [
        list(map(int, current_input_size.split("x")))
        for current_input_size in input_size.split(",")
    ]
    assert len(group_input_size[0]) in [2, 4, 5]
    if len(group_input_size) > 1:
        example_inputs = dict()  # noqa: C408
        example_inputs[key] = []
        for current_input_size in group_input_size:
            assert len(current_input_size) in [4, 5]
            if len(current_input_size) == 4:
                example_inputs[key] += [
                    generate_input(
                        current_input_size, torch_native, dtype=dtype
                    )
                ]
            elif len(current_input_size) == 5:
                sequence_length = current_input_size[0]
                example_inputs[key] += [
                    generate_input(
                        current_input_size[1:], torch_native, dtype=dtype
                    )
                    for _ in range(sequence_length)
                ]
    elif len(group_input_size[0]) == 5:
        input_size = group_input_size[0]
        sequence_length = input_size[0]
        example_inputs = {
            key: [
                generate_input(input_size[1:], torch_native)
                for _ in range(sequence_length)
            ]
        }
    elif len(group_input_size[0]) == 4:
        input_size = group_input_size[0]
        example_inputs = {
            key: generate_input(input_size, torch_native, dtype=dtype)
        }
    elif len(group_input_size[0]) == 2:
        input_size = group_input_size[0]
        example_inputs = {
            key: generate_input(input_size, torch_native, dtype=dtype)
        }
    else:
        raise NotImplementedError
    return example_inputs


def dump_to_json_file(data, path):
    with open(path, "w") as f:
        json.dump(data, f)


def compile_standalone(
    in_file: str,
    input_size: str,
    name: str = None,
    opt: str = "O0",
    march: str = "bayes",
    hbir_only: bool = False,
    perf_only: bool = False,
    output: str = "tmp_compile",
    debug: bool = False,
    jobs: int = 4,
    input_key: str = "img",
    torch_native: str = "False",
    input_source: str = "pyramid",
    input_layout: str = None,
    output_layout: str = None,
    extra_args: str = "",
    input_type: str = None,
    dump_cmp_info: bool = False,
    verify_model: bool = False,
):
    """Compile pt model function.

    Args:
        in_file: input pt or hbir file.
        input_size: input image size, [Nx]BxCxHxW, N is for sequence length or
                    BxCxHxW,BxCxHxW..., separate multiple input_size with `^`.
        name: hbm name.
        opt: optimize level, available options are
             O0, O1, O2, O3, ddr, fast, balance.
        march: march name.
        hbir_only: do export hbir only, skip compile and perf.
        perf_only: do perf only, skip export hbir and compile.
        output: output folder.
        debug: compile model with debug option,
               show layer_details in perf result.
        jobs: number of threads launched during compiler optimization.
              Default 0 means to use all available hardware concurrency.
        input_key: dict key name in input data to model,
                   default is `img`. Separate multiple input with `^`.
        torch_native: Whether to use torch-native,
                   default False.Separate multiple input_size with `^`.
        input_source: input source used by hbdk, default is `pyramid`.
        input_type: input dtype used by hbdk, default is None means `float32`,
                   Separate multiple input_type with `^`.
        input_layout: specify input layout of model,
                      users can view it through `hbdk-cc -h optional`.
        output_layout: specify output layout of model,
                      users can view it through `hbdk-cc -h optional`.
        extra_args: extra args, like `--max-time-per-fc 1000`.
        dump_cmp_info: Whether to dump compilation settings and output info.
    """

    out_dir = output
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    if in_file.startswith("http://"):

        file_url = in_file
        in_file_path = os.path.join(out_dir, os.path.basename(file_url))
        if os.path.exists(in_file_path):
            os.system(f"rm {in_file_path}")
        cprint(f"downloading to {in_file_path} ...", "green")
        wget.download(file_url, out=in_file_path, bar=None)
    else:
        in_file_path = in_file

    if str(in_file_path).endswith(".hbir"):
        is_hbir_input = True
    elif str(in_file_path).endswith(".pt"):
        is_hbir_input = False
        # load pt_file
        model = load_script_module(in_file_path, map_location="cpu")
    else:
        raise ValueError(
            f"input file should be .pt or .hbir file, but get {in_file_path}"
        )

    input_sizes = input_size.split(
        "^"
    )  # Separate multiple input_size with `^`
    input_keys = input_key.split("^")  # Separate multiple input_keys with `^

    torch_native = torch_native.split(
        "^"
    )  # Separate multiple input_keys with `^

    if len(torch_native) != 1:
        assert len(input_sizes) == len(torch_native)
    else:
        # extend the length of torch_native,equal to the length of input_sizes
        torch_native = torch_native * (len(input_sizes))

    assert len(input_sizes) == len(input_keys)
    input_type = get_input_type(input_type)
    if input_type is None:
        input_type = [None] * (len(input_sizes))

    example_inputs = {}
    for input_size_tmp, input_key_tmp, t_native, e_type in zip(
        input_sizes, input_keys, torch_native, input_type
    ):
        example_inputs.update(
            get_example_input(
                input_key_tmp, input_size_tmp, eval(t_native), dtype=e_type
            )
        )
    inputs_contains_list = True in [
        isinstance(example_inputs[key], list) for key in example_inputs
    ]
    if inputs_contains_list:
        for key in example_inputs:
            if not isinstance(example_inputs[key], list):
                example_inputs[key] = [example_inputs[key]]

    extra_args = extra_args.split()
    if extra_args:
        cprint(f"extra args are {extra_args}", "green")

    hbm = os.path.join(out_dir, f"model_opt_{opt}.hbm")
    hbir = os.path.join(out_dir, f"model_opt_{opt}.hbir")
    compiled_pt = os.path.join(out_dir, f"model_opt_{opt}.pt")
    if IS_LOCAL:
        cpu_num = jobs
    else:
        cpu_num = multiprocessing.cpu_count()
    cprint(f"use {cpu_num} cpus to compile", "green")

    # compile_info: save compile settings and out files
    compile_info = {}
    env = {
        "horizon_plugin_pytorch": __version__,
        "hbdk": hbdk.__version__,
        "hbir_only": hbir_only,
        "perf_only": perf_only,
    }
    compile_params = {
        "in_file": in_file,
        "input_sizes": input_sizes,
        "input_keys": input_keys,
        "torch_native": torch_native,
        "input_type": input_type,
        "march": march,
        "input_source": input_source,
        "output_hbm": hbm,
        "name": name,
        "input_layout": input_layout,
        "output_layout": output_layout,
        "debug": debug,
        "opt": opt,
        "jobs": cpu_num,
        "extra_args": extra_args,
    }

    compile_info["env"] = env
    compile_info["params"] = compile_params

    if os.path.exists(hbm) and perf_only:
        cprint(f"{hbm} exists, skip compile", "green")
    else:
        if opt == "balance":
            raise NotImplementedError
        if opt == "O3":
            cprint(
                "You choose O3 optimize level, which may cost up "
                "to minutes, but high FPS. Refer to http://wiki.h"
                "obot.cc/pages/viewpage.action?pageId=186764640 "
                "for more details.",
                "yellow",
            )
        if not is_hbir_input:
            cprint("exporting hbir ...", "green")
            result = export_hbir(
                module=model.eval(),
                example_inputs=example_inputs,
                hbir=hbir,
                march=march,
            )
            if hbir_only:
                compile_info["out_files"] = {"hbir": hbir}
                if dump_cmp_info:
                    dump_to_json_file(
                        data=compile_info,
                        path=os.path.join(out_dir, "_compile_info.json"),
                    )
                cprint("export hbir only, skip compile and perf.", "green")
                return

        cprint("compiling model ...", "green")

        compile_params = dict(  # noqa C406
            march=march,
            input_source=[input_source],
            hbm=hbm,
            name=name,
            input_layout=input_layout,
            output_layout=output_layout,
            debug=debug,
            opt=opt,
            progressbar=True,
            jobs=cpu_num,
            extra_args=extra_args,
        )

        t1 = time.time()
        if is_hbir_input:
            result = compile_hbir_model(
                model=in_file_path,  # hbir file
                visualize=False,
                **compile_params,
            )
        else:
            result = compile_model(
                module=model.eval(),
                example_inputs=example_inputs,
                # # not used yet
                # balance_factor=2,
                **compile_params,
            )
        cprint(f"compile cost {(time.time() - t1):.2f} sec", "yellow")

    cprint("perf model ...", "green")
    result = perf_hbm(  # noqa: F841
        hbm=hbm,
        layer_details=debug,
        out_dir=out_dir,
    )

    if verify_model:
        cprint("verify model ...", "green")
        with TemporaryDirectory(suffix=f"verify_{name}") as tmpdir:
            subprocess.run(
                "hbdk-model-verifier"
                + f" --hbm {hbm}"
                + f" --model-pt {compiled_pt}"
                + f" --local-work-path {tmpdir}"
                + " --skip-bpu",
                shell=True,
            )

    compile_info["out_files"] = {
        "hbm": hbm,
        "compiled_pt": compiled_pt,
        "hbir": hbir,
        "perf_json": os.path.join(out_dir, name + ".json"),
        "perf_html": os.path.join(out_dir, name + ".html"),
    }
    if dump_cmp_info:
        dump_to_json_file(
            data=compile_info,
            path=os.path.join(out_dir, "_compile_standalone_info.json"),
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "in_file",
        type=str,
        help="input pt or hbir file",
    )
    parser.add_argument(
        "--input-size",
        type=str,
        required=True,
        help="input image size, [Nx]BxCxHxW, N is for sequence length or BxCxHxW,BxCxHxW..., separate multiple input_size with `^`",  # noqa: E501
    )
    parser.add_argument("--name", default=None, type=str, help="hbm name")
    parser.add_argument(
        "--opt",
        default="O0",
        type=str,
        help="optimize level, available options are O0, O1, O2, O3, ddr, fast, balance.",  # noqa: E501
    )  # yapf: disable
    parser.add_argument(
        "--march",
        required=True,
        type=str,
        help="march name",
    )  # noqa
    parser.add_argument(
        "--hbir-only",
        dest="hbir_only",
        action="store_true",
        help="do export hbir only, skip compile and perf",
    )
    parser.add_argument(
        "--perf-only",
        dest="perf_only",
        action="store_true",
        help="do perf only, skip export hbir and compile",
    )
    parser.add_argument(
        "--output", default="tmp_compile", type=str, help="output folder"
    )
    parser.add_argument(
        "--debug",
        dest="debug",
        action="store_true",
        help="compile model with debug option, show layer_details in perf result",  # noqa: E501
    )  # yapf: disable
    parser.add_argument(
        "--jobs",
        default=4,
        type=int,
        help="number of threads launched during compiler optimization."
        " Default 0 means to use all available hardware concurrency.",
    )
    parser.add_argument(
        "--input-key",
        default="img",
        type=str,
        help="dict key name in input data to model, default is `img`. Separate multiple input with `^` ",  # noqa
    )
    parser.add_argument(
        "--torch-native",
        default="False",
        type=str,
        help="Whether to use torch-native,default False.Separate multiple input_size with `^`",  # noqa
    )
    parser.add_argument(
        "--input-source",
        default="pyramid",
        type=str,
        help="input source used by hbdk, default is `pyramid`",
    )
    parser.add_argument(
        "--input-type",
        default=None,
        type=str,
        help="input type used by hbdk, default is `float32`. Separate multiple input-type with `^`",  # noqa
    )
    parser.add_argument(
        "--input-layout",
        default=None,
        choices=["NHWC", "NCHW", "BPU_RAW"],
        type=str,
        help="specify input layout of model, users can view it through `hbdk-cc -h optional`",  # noqa
    )
    parser.add_argument(
        "--output-layout",
        type=str,
        help="specify output layout of model, users can view it through `hbdk-cc -h optional`. Separate multiple layout with `^`",  # noqa
    )
    parser.add_argument(
        "--extra-args",
        default="",
        type=str,
        help="extra args, like `--max-time-per-fc 1000`",
    )

    parser.add_argument(
        "--dump-cmp-info",
        default=False,
        action="store_true",
        help="dump compile setting and files info",
    )

    args = parser.parse_args()

    compile_standalone(
        in_file=args.in_file,
        input_size=args.input_size,
        name=args.name,
        opt=args.opt,
        march=args.march,
        hbir_only=args.hbir_only,
        perf_only=args.perf_only,
        output=args.output,
        debug=args.debug,
        jobs=args.jobs,
        input_key=args.input_key,
        torch_native=args.torch_native,
        input_source=args.input_source,
        input_layout=args.input_layout,
        output_layout=args.output_layout,
        extra_args=args.extra_args,
        input_type=args.input_type,
        dump_cmp_info=args.dump_cmp_info,
    )
