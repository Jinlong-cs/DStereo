import argparse
import subprocess
from typing import List, Optional


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--docker-file-dir",
        type=str,
        required=True,
        help="dir path of .Dockerfile",
    )

    parser.add_argument(
        "--docker-file", type=str, required=True, help="name of .Dockerfile"
    )

    parser.add_argument(
        "--docker-names",
        type=str,
        required=True,
        nargs="+",
        help="names of docker to push.",
    )

    parser.add_argument(
        "--build-args",
        type=str,
        required=False,
        default=None,
        nargs="+",
        help="args used in build docker, like `TAG_NAME=1.0`",
    )

    parser.add_argument(
        "--no-cache",
        action="store_true",
        default=False,
        help="whether not use cache when build docker",
    )

    return parser.parse_args()


def build_docker(
    docker_file_dir: str,
    docker_file_name: str,
    docker_names: List[str],
    build_args: Optional[List[str]] = None,
    no_cache: bool = False,
):
    """Build docker by using buildkit tools.

    Args:
        docker_file_dir: Dir path of dockerfile.
        docker_file_name: Name of dockerfile.
        docker_names: Names of docker to push.
        build_args: Build args used when build docker.
    """

    buildkit_cmd = (
        f"buildctl build "
        f"--frontend dockerfile.v0 "
        f"--local context=. "
        f"--local dockerfile={docker_file_dir} "
        f"--opt filename={docker_file_name} "
    )
    if build_args:
        for arg in build_args:
            buildkit_cmd += f"--opt build-arg:{arg} "

    if no_cache:
        buildkit_cmd += "--no-cache "

    docker_names = ",".join(docker_names)
    buildkit_cmd += (
        f"--output " f"type=image," f'\\"name={docker_names}\\",' f"push=true"
    )

    print(buildkit_cmd)
    subprocess.check_call(buildkit_cmd, shell=True)


if __name__ == "__main__":
    args = parse_args()

    build_docker(
        docker_file_dir=args.docker_file_dir,
        docker_file_name=args.docker_file,
        docker_names=args.docker_names,
        build_args=args.build_args,
        no_cache=args.no_cache,
    )
