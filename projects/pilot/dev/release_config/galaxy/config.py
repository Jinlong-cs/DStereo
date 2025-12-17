from plugins.release_package.config_struct import (
    DockerEnv,
    PackEnv,
    ReleasePackageEnv,
    RunDokcerEnv,
)

docker_env = DockerEnv(
    docker_url="docker.hobot.cc/imagesys/hat:pilot-runtime-cu111-galaxy-whitebox-test",  # noqa
    save_name="galaxy_whitebox.tar",
    remove_image=True,
    build_docker=True,
    create_dockerfile="galaxy/create_dockerfile.sh",
    dockerfile_path="galaxy/dockerfile",
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
        "./galaxy/add_user.sh",
        "./galaxy/whitebox_test.sh",
        "/home/users/wentao.xie/test/whitebox/pilot_data_raw",
        "/home/users/wentao.xie/test/whitebox/galaxy_x3c_side_lmdb_datasets.py",  # noqa
        "/home/users/wentao.xie/test/whitebox/galaxy_0233_rear_lmdb_datasets.py",  # noqa
        "/home/users/wentao.xie/test/whitebox/image_fail_seg_lmdb_datasets.py",  # noqa
        "/horizon-bucket",
    ],
    mount_targets=[
        "/add_user.sh",
        "/whitebox_test.sh",
        "/pilot_data_raw",
        "/release/projects/pilot/configs/datasets/galaxy_x3c_side_lmdb_datasets.py",  # noqa
        "/release/projects/pilot/configs/datasets/galaxy_0233_rear_lmdb_datasets.py",  # noqa
        "/release/projects/pilot/configs/datasets/image_fail_seg_lmdb_datasets.py",  # noqa
        "/horizon-bucket",
    ],
    test_package=True,
)

release_package_env = ReleasePackageEnv(
    name="galaxy_whitebox",
    code_strip="galaxy/code_strip.sh",
    docker_env=docker_env,
    pack_env=pack_env,
    run_docker_env=run_docker_env,
)
