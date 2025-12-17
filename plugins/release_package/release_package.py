"""
Copyright (c) Horizon Robotics. All rights reserved.

Release Package Tools, include the following steps:
    1. code strip.
    2. create dockerfile. (optional)
    3. build docker. (optional)
    3. save and remove docker image.
    4. copy sources and pack package.
    5. unpack package.
    6. load docker image.
    7. mount source and run docker.
    8. test(optional).

"""
import argparse
import os
import subprocess
import sys

from hat.utils.config import Config


def run_cmd(cmd, errorlog):
    try:
        subprocess.check_call(cmd, shell=True)
    except subprocess.CalledProcessError:
        print(errorlog)
        sys.exit()


def copy(src, dst):
    """Copy file or folder to dst."""
    dst_dir = os.path.dirname(dst)
    if os.path.isdir(src):
        if dst.endswith("/"):
            dst_dir = os.path.dirname(dst_dir)
        dst = dst_dir

    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)
    cmd = f"cp -rf {src} {dst}"
    errorlog = f"copy {src} to {dst} error."
    run_cmd(cmd, errorlog)


def code_strip(code_strip_cmd):
    """Get the code and make doc."""
    print("start code strip and make docs!")
    errorlog = "code strip or make html error," "see more in log!"
    run_cmd(code_strip_cmd, errorlog)

    print("code strip and make docs success!")


def build_docker(
    dockerfile_path, docker_url, workspace=".", create_dockerfile=None
):
    """Build docker."""
    # create dockerfile and build docker

    if create_dockerfile is not None:
        print("start create dockerfile!")
        cmd = f"sh {create_dockerfile} {dockerfile_path}"
        errorlog = "Create dockerfile error, check your script!"
        run_cmd(cmd, errorlog)
        print("create dockerfile success!")
    print("start build docker!")
    cmd = f"docker build -f {dockerfile_path} -t {docker_url} {workspace}"
    errorlog = "Build docker error, check your config!"
    run_cmd(cmd, errorlog)
    print("build docker success!")


def save_docker(docker_name, docker_url, remove_image=True):
    """Save docker."""
    print("start save docker!")
    cmd = f"docker save -o {docker_name} {docker_url}"
    run_cmd(cmd, "Save docker error.")
    print("save docker success!")
    if remove_image:
        remove_docker_image(docker_url)


def load_docker(docker_dir):
    """Load docker."""
    print("start load docker!")
    cmd = f"docker load -i {docker_dir}"
    run_cmd(cmd, "Load docker error.")
    print("load docker success!")


def remove_docker_image(docker_url):
    """Remove docker image."""
    print("start remove docker!")
    cmd = f"docker rmi {docker_url}"
    run_cmd(cmd, "Remove docker error.")
    print("remove docker success!")


def pack_package(package_path, package_dir, password=None, encrypt_type=None):
    """Pack package."""
    print("start pack package!")
    dirname = os.path.dirname(package_dir)
    basename = os.path.basename(package_dir)
    files = f"-C {dirname} {basename} "

    if password is not None:
        cmd = (
            f"tar -czvf - {files} | openssl {encrypt_type} "
            f"-salt -k {password} -out {package_path}"
        )
    else:
        cmd = f"tar -czvf {package_path} {files}"
    errorlog = "Pack package error, check your config!"
    run_cmd(cmd, errorlog)
    print("pack package success!")


def unpack_package(
    package_path, unpackdir=None, password=None, encrypt_type=None
):
    """Unpack package."""
    print("start unpack package!")
    if password is not None:
        cmd = (
            f"openssl {encrypt_type} -d -k {password} "
            f"-salt -in {package_path} | tar xzvf -"
        )
    else:
        cmd = f"tar -xzvf {package_path}"
    if unpackdir is not None:
        if not os.path.exists(unpackdir):
            os.makedirs(unpackdir)
        cmd += f" -C {unpackdir}"
    errorlog = "Unpack package error, check your config!"
    run_cmd(cmd, errorlog)
    print("unpack package success!")


def run_docker(
    docker_url, test_package=False, mount_sources=None, mount_targets=None
):
    """Run docker."""
    print("start run docker!")
    cmd = "docker run -it  --gpus all --shm-size 8gb "
    if mount_sources is not None:
        assert len(mount_sources) == len(
            mount_targets
        ), "The length of mount_sources and mount_targets mismatch."
        for source, target in zip(mount_sources, mount_targets):
            if source.startswith("/"):
                source_dir = source
            else:
                source_dir = os.path.join(os.getcwd(), source)
            cmd += f"-v {source_dir}:{target} "

    cmd += f"{docker_url} "

    if test_package:
        user_name = os.getlogin()
        uid = os.getuid()
        add_user_script = f"sh {mount_targets[0]} {user_name} {uid}"
        test_script = f"su {user_name} -c 'sh {mount_targets[1]}'"
        cmd += f'/bin/bash -c "{add_user_script} && {test_script}"'
    else:
        cmd += '/bin/bash  -c "exit"'

    errorlog = "Run docker error, see more in log!"
    run_cmd(cmd, errorlog)
    print("run docker success!")


def main(args):

    args.target_dir = os.path.abspath(args.target_dir)
    if not os.path.exists(args.target_dir):
        os.makedirs(args.target_dir)
    else:
        if len(os.listdir(args.target_dir)) != 0:
            print(f"{args.target_dir} is not empty!! Please clean it first")
            return
    cfg = Config.fromfile(args.release_config)

    release_package_env = cfg.get("release_package_env", None)
    assert (
        release_package_env is not None
    ), "You must set release_package_env in release_config!"

    package_name = release_package_env.name

    package_dir = os.path.join(args.target_dir, package_name)

    os.makedirs(package_dir)

    # code stripping and make doc
    code_strip_script = release_package_env.code_strip
    code_strip_cmd = f"sh {code_strip_script}"
    code_strip(code_strip_cmd)

    docker_env = release_package_env.docker_env
    docker_url = docker_env.docker_url
    if docker_env.build_docker:
        # build docker
        dockerfile_path = docker_env.dockerfile_path
        workspace = docker_env.workspace
        create_dockerfile = docker_env.create_dockerfile
        build_docker(
            dockerfile_path=dockerfile_path,
            docker_url=docker_url,
            workspace=workspace,
            create_dockerfile=create_dockerfile,
        )
    else:
        print("strip docker build!!")

    docker_save_name = docker_env.save_name
    remove_image = docker_env.remove_image

    # save docker
    docker_save_dir = os.path.join(package_dir, docker_save_name)
    save_docker(docker_save_dir, docker_url, remove_image)

    # pack package
    pack_env = release_package_env.pack_env
    password = None
    encrypt_type = "des3"
    if pack_env is not None:
        pack_sources = pack_env.pack_sources
        pack_targets = pack_env.pack_targets
        if pack_sources is not None:
            assert len(pack_sources) == len(
                pack_targets
            ), "The length of mount_sources and pack_targets mismatch."
            for pack_source, pack_target in zip(pack_sources, pack_targets):
                copy(pack_source, os.path.join(package_dir, pack_target))
        if pack_env.enc_package:
            password = pack_env.password
            encrypt_type = pack_env.encrypt_type

    package_path = os.path.join(args.target_dir, package_name + ".tar.gz")

    pack_package(package_path, package_dir, password, encrypt_type)

    # unpack package

    unpackdir = os.path.join(args.target_dir, "unpack_" + package_name)

    unpack_package(package_path, unpackdir, password, encrypt_type)

    # load docker
    docker_dir = os.path.join(unpackdir, package_name, docker_save_name)

    load_docker(docker_dir)

    # run docker
    run_docker_env = release_package_env.run_docker_env

    test_package = run_docker_env.test_package
    mount_sources = run_docker_env.mount_sources
    mount_targets = run_docker_env.mount_targets

    run_docker(
        docker_url=docker_url,
        test_package=test_package,
        mount_sources=mount_sources,
        mount_targets=mount_targets,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--release-config", type=str, required=True)
    parser.add_argument("--target-dir", type=str, default="./release")
    args = parser.parse_args()
    main(args)
