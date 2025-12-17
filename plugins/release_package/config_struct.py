from dataclasses import dataclass
from typing import List, Optional


@dataclass
class DockerEnv(object):
    """The env for docker.

    Args:
        docker_url: The url for docker.
        save_name: The save name for docker image.
        remove_image: Whether to remove docker image after saved.
        build_docker: Whether to build docker.
        create_dockerfile: The script for create dockerfile.
        dockerfile_path: The path of dockerfile.
        workspace: The workspace for building docker.
    """

    docker_url: str
    save_name: str
    remove_image: bool = True
    build_docker: bool = True
    create_dockerfile: Optional[str] = None
    dockerfile_path: Optional[str] = None
    workspace: str = "."

    def __post_init__(self):
        if self.build_docker:
            assert (
                self.dockerfile_path is not None
            ), "You must set dockerfile_path when build_docker is True!"


@dataclass
class PackEnv(object):
    """The env for pack package.

    Args:
        pack_sources: The package name.
        pack_targets: The code strip script.
        enc_package: Whether to encrypt package.
        password: The password of package.
        encrypt_type: The encrypt_type for encrypt package.
    """

    pack_sources: List[str] = None
    pack_targets: List[str] = None
    enc_package: bool = True
    password: str = None
    encrypt_type: str = "des3"

    def __post_init__(self):
        if self.enc_package:
            assert (
                self.password is not None
            ), "You must set password when enc_package is True."


@dataclass
class RunDokcerEnv(object):
    """The env for run docker.

    Args:
        mount_sources: The source dir lists for mount.
        mount_targets: The target dir lists for mount.
        test_package: Whether to test package.
    """

    mount_sources: Optional[List[str]] = None
    mount_targets: Optional[List[str]] = None
    test_package: bool = True


@dataclass
class ReleasePackageEnv(object):
    """The env for release package.

    Args:
        name: The package name.
        code_strip: The code strip script.
        docker_env: The env for docker.
        run_docker_env: The env for run docker.
        pack_env: The env for pack.
    """

    name: str
    code_strip: str
    docker_env: DockerEnv
    run_docker_env: RunDokcerEnv
    pack_env: Optional[PackEnv] = None
