import argparse
import os
import subprocess

import torch

from hat.utils.config import Config

pub_cfg_root = os.path.join(os.path.dirname(__file__), "cfg")

pip_ext = "-i https://pypi.hobot.cc/simple --extra-index-url=https://pypi.hobot.cc/hobot-local/simple --trusted-host pypi.hobot.cc"  # noqa


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sub-project",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--publish-version",
        type=str,
        required=True,
    )
    args = parser.parse_args()

    pub_cfg_path = os.path.join(pub_cfg_root, f"pub_cfg_{args.sub_project}.py")
    pub_cfg = Config.fromfile(pub_cfg_path)
    plugin_version = pub_cfg["plugin_version"]
    hbcc_version = pub_cfg["hbcc_version"].replace("v", "")

    build_cmd_pattern = "pip3 install --user -U {}=={} {} " + f"{pip_ext}"
    torch_version, cuda_sign = torch.__version__.split("+")
    torch_sign = "torch" + torch_version.replace(".", "")

    build_pkg_cmd = [
        # plugin_pytorch
        build_cmd_pattern.format(
            "horizon_plugin_pytorch",
            f"{plugin_version}+{cuda_sign}.{torch_sign}",
            f"-f https://art-internal.hobot.cc/artifactory/custom-algo-pypi/horizon-plugin-pytorch/{cuda_sign}/{torch_sign} --trusted-host art-internal.hobot.cc",  # noqa
        ),
        # hbdk
        build_cmd_pattern.format("hbdk", hbcc_version, ""),
        # hdbk-model-verifier
        build_cmd_pattern.format("hbdk-model-verifier", hbcc_version, ""),
    ]

    for cmd in build_pkg_cmd:
        print(cmd)
        subprocess.check_call(cmd, shell=True)
