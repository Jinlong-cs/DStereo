#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Pack hbm standalone, without config."""

import argparse
import json
import logging
import os
import subprocess
from typing import List, Optional

from hbdk.torch_script.tools import perf_hbm
from termcolor import cprint

try:
    import wget
except ImportError:
    wget = None


logger = logging.getLogger()


def run_cmd(cmd):
    print(cmd)
    try:
        subprocess.check_output(cmd, stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as e:
        logger.error("Meet Exception: {}".format(e.stdout.decode()))


def pack_standalone(
    input_hbm_list: List[str],
    output_dir: str,
    tag_name: Optional[str] = None,
    debug: bool = False,
    dump_cmp_info: bool = False,
):

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    input_lists = []
    for input_name in input_hbm_list:
        if input_name.startswith("http://"):
            input_url = input_name
            input_file = os.path.join(output_dir, os.path.basename(input_url))
            if os.path.exists(input_file):
                os.system(f"rm {input_file}")
            cprint(f"downloading to {input_file} ...", "green")
            wget.download(input_url, out=input_file, bar=None)
            input_lists.append(input_file)
        else:
            input_lists.append(input_name)

    hbm = os.path.join(output_dir, "pack_model.hbm")
    disas_json = os.path.join(output_dir, "model_info.json")

    pack_cmd = ["hbdk-pack"]
    pack_cmd += input_lists
    pack_cmd += ["-o", hbm]
    if tag_name is not None:
        pack_cmd += ["--tag", tag_name]
    run_cmd(pack_cmd)

    disas_cmd = ["hbdk-disas", hbm, "-o", disas_json, "--json"]
    run_cmd(disas_cmd)

    cprint("perf model ...", "green")
    result = perf_hbm(  # noqa: F841
        hbm=hbm,
        layer_details=debug,
        out_dir=output_dir,
    )

    if dump_cmp_info:

        # pack_info: save pack settings and out files
        pack_info = {}
        import hbdk

        env = {
            "hbdk": hbdk.__version__,
            "pack_cmd": str(pack_cmd),
            "disas_cmd": str(disas_cmd),
        }
        pack_info = {
            "env": env,
            "out_files": {
                "hbm": hbm,
                "disas_json": disas_json,
                "perf_html": [],
                "perf_json": [],
            },
        }

        # perf result
        file_list = os.listdir(output_dir)
        for f in file_list:
            if f.endswith("html"):
                perf_html = os.path.join(output_dir, f)
                perf_json = os.path.join(output_dir, f.replace("html", "json"))

                pack_info["out_files"]["perf_html"].append(perf_html)
                pack_info["out_files"]["perf_json"].append(perf_json)

        with open(os.path.join(output_dir, "_pack_hbm_info.json"), "w") as f:
            json.dump(pack_info, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-hbm-list",
        type=str,
        nargs="+",
        help="Required hbm file.",
    )
    parser.add_argument(
        "--output", default="tmp_compile", type=str, help="output folder"
    )
    parser.add_argument(
        "--tag-name", default=None, type=str, help="tag name of output hbm"
    )
    parser.add_argument(
        "--debug",
        dest="debug",
        action="store_true",
        help="show layer_details in perf result",
    )

    # add parser related to aidi tracking.
    parser.add_argument(
        "--dump-cmp-info",
        action="store_true",
        help="enable aidi tracking",
    )

    args = parser.parse_args()

    try:
        pack_standalone(
            input_hbm_list=args.input_hbm_list,
            output_dir=args.output,
            tag_name=args.tag_name,
            debug=args.debug,
            dump_cmp_info=args.dump_cmp_info,
        )
    except Exception as e:
        logger.error(f"compile failed! {str(e)}")
        raise e
