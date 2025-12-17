from plugins.release_package.config_struct import (
    DockerEnv,
    PackEnv,
    ReleasePackageEnv,
    RunDokcerEnv,
)

docker_env = DockerEnv(
    docker_url="docker.hobot.cc/dlp/hat:toolchain",
    save_name="toolchain_docker.tar",
    remove_image=True,
    build_docker=True,
    create_dockerfile="example/create_dockerfile.sh",
    dockerfile_path="example/dockerfile",
    workspace="../../",
)

pack_env = PackEnv(
    pack_sources=[
        "../../dev/api_generator",
        "../../dev",
    ],
    pack_targets=[
        "tools_pack/api_generator",
        "dev",
    ],
    enc_package=True,
    password="123456789",
    encrypt_type="des3",
)

run_docker_env = RunDokcerEnv(
    mount_sources=[
        "./example/add_user.sh",
        "./example/test_package.sh",
        "/horizon-bucket",
    ],
    mount_targets=[
        "/add_user.sh",
        "/test_package.sh",
        "/horizon-bucket",
    ],
    test_package=True,
)

release_package_env = ReleasePackageEnv(
    name="toolchain_package",
    code_strip="example/code_strip.sh",
    docker_env=docker_env,
    pack_env=pack_env,
    run_docker_env=run_docker_env,
)
