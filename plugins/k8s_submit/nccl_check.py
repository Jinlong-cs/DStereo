import argparse
import logging
import os
import subprocess
from typing import Optional

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def run(cmd, node):
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            check=True,
            env=os.environ.copy(),
        )
    except subprocess.CalledProcessError as e:
        logger.warning(
            f"Node {node}: subprocess({e.cmd}) failed({e.returncode})! {e.output}.\n"  # noqa E501
        )
        result = None

    return result


def check_gpu_driver_version(node):
    print(f"CHECK GPU DRIVER ON NODE {node}:\n")

    result = run(
        cmd="nvidia-smi --query-gpu=driver_version --format=csv,noheader",
        node=node,
    )

    if result:
        driver_version = result.stdout.strip().split("\n")
        print(f"GPU driver on node {node}: {driver_version}")
    else:
        logger.warning(f"GPU driver check failed on node {node}")


def run_nccl_test(node, gpus):
    print(f"CHECK NCCL_TEST ON NODE {node}:\n")

    nccl_test_lib_dir = "./nccl-tests"

    # download nccl-test
    if not os.path.exists(nccl_test_lib_dir):
        cmd = "hdfs dfs -get hdfs://hobot-bigdata/user/mengyang.duan/nccl-tests.tgz ./ && tar -xf nccl-tests.tgz"  # noqa E501
        run(cmd=cmd, node=node)

    # build nccl-test
    if not os.path.exists(os.path.join(nccl_test_lib_dir, "build")):
        cmd = "cd nccl-tests && make CUDA_HOME=/usr/local/cuda"
        run(cmd=cmd, node=node)

    # nccl-test
    nccl_test_cmd = f"./nccl-tests/build/all_gather_perf -b 8 -e 128M -f 2 -g {gpus}"
    result = run(cmd=nccl_test_cmd, node=node)
    if result:
        print(f"{result.stdout}")


def main(
    node: Optional[str] = None,
    check_driver: bool = False,
    gpus: Optional[int] = None,
    nccl_test: bool = False,
):
    if node is None:
        node = os.getenv("HOST_NODE", None)

    print("=" * 50 + f"START CHECKING ON NODE {node}" + "=" * 50)

    if check_driver:
        check_gpu_driver_version(node)

    if nccl_test:
        run_nccl_test(node=node, gpus=gpus)

    print("=" * 50 + f"FINISH CHECKING ON NODE {node}" + "=" * 50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--node",
        type=str,
        required=False,
        help="node ip or hostname",
    )
    parser.add_argument(
        "--driver",
        action="store_true",
        default=False,
        help="whether to check gpu driver",
    )
    parser.add_argument(
        "--ngpus",
        type=int,
        required=False,
        help="number of gpus on per node",
    )
    parser.add_argument(
        "--nccl",
        action="store_true",
        default=False,
        help="whether to do nccl-test",
    )
    args = parser.parse_args()
    main(
        node=args.node,
        check_driver=args.driver,
        gpus=args.ngpus,
        nccl_test=args.nccl,
    )
