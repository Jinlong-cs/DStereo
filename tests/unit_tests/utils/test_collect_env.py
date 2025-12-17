from hat.utils import collect_env


def test_collect_requirements_env_info():
    data = []
    libs = ["horizon_plugin_pytorch", "hbdk"]
    collect_env.collect_requirements_env_info(data)
    tmp_keys = []
    assert len(data) > 0
    for (k, _) in data:
        tmp_keys.append(k)
    assert sorted(libs) == sorted(tmp_keys)


def test_collect_env_info():
    env_str = collect_env.collect_env_info()
    env_infos = [
        "sys.platform",
        "Python",
        "PyTorch",
        "Torchvision",
        "HAT",
        "GPU available",
    ]
    for info in env_infos:
        assert info in env_str
