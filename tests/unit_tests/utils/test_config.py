import os

import horizon_plugin_pytorch as horizon
import pytest

from hat.utils.config import Config, filter_configs


def test_config(tmpdir):
    tmp_file = os.path.join(tmpdir, "base.py")
    with open(tmp_file, "w") as f:
        f.write("x = 2")

    cfg = Config.fromfile(tmp_file)
    assert cfg.x == 2


def test_config_with_same_name(tmpdir):
    test = os.path.join(tmpdir, "test.py")
    base_1 = os.path.join(tmpdir, "base.py")

    tmpdir_1 = os.path.join(tmpdir, "test")
    os.makedirs(tmpdir_1)
    base_2 = os.path.join(tmpdir_1, "base.py")

    with open(base_1, "w") as f:
        f.write("x = 2")

    with open(test, "w") as f:
        f.write("from hat.utils import Config\n")
        f.write("a = Config.fromfile('%s')\n" % (base_1))
        f.write("assert a.x == 2\n")
        f.write("cfg = Config.fromfile('%s')\n" % (base_2))

    with open(base_2, "w") as f:
        f.write("from hat.utils import Config\n")
        f.write("cfg = Config.fromfile('%s')\n" % (base_1))
        f.write("assert cfg.x == 2")

    cfg = Config.fromfile(test)  # noqa: F841


def test_config_with_common_object():
    d1 = {"aaa": "aaa"}
    d2 = {"aaa": d1, "bbb": "bbb"}
    d3 = {"a": d1, "b": d2}

    cfg_dict = Config(cfg_dict=d3)

    assert id(cfg_dict.a) == id(cfg_dict.b["aaa"])


def test_merge_from_list_or_dict(tmpdir):
    tmp_file = os.path.join(tmpdir, "base.py")
    with open(tmp_file, "w") as f:
        f.write("device_ids = [0, 1, 2, 3]\n")
        f.write("network = 'resnet18_cls'\n")
    cfg = Config.fromfile(tmp_file)
    assert isinstance(cfg, Config)
    assert len(cfg) > 0

    # test for list
    opts_list = ["device_ids", "0,1"]
    cfg.merge_from_list_or_dict(opts_list, overwrite=True)
    assert cfg.device_ids == [0, 1]

    # test for dict
    opts_dict = {
        "network": "'resnet50_cls'",
        "model.backbone.num_classes": "10",
    }
    cfg.merge_from_list_or_dict(opts_dict, overwrite=False)
    assert (
        cfg.network == "resnet18_cls"
        and cfg.model["backbone"]["num_classes"] == 10
    )
    cfg.merge_from_list_or_dict(opts_dict, overwrite=True)
    assert (
        cfg.network == "resnet50_cls"
        and cfg.model["backbone"]["num_classes"] == 10
    )


@pytest.mark.parametrize(
    ["filename"],
    [
        pytest.param("test.yaml"),
        pytest.param("test.yml"),
    ],
)
def test_config_yaml(tmpdir, filename):
    tmp_file = os.path.join(tmpdir, filename)
    with open(tmp_file, "w") as f:
        f.write("a: 1")
    cfg = Config.fromfile(tmp_file)
    assert cfg.a == 1


def test_qat_mode(tmpdir):
    tmp_file = os.path.join(tmpdir, "base.py")
    with open(tmp_file, "w") as f:
        f.write("qat_mode='with_bn'\n")
    cfg = Config.fromfile(tmp_file)
    horizon.qat_mode.set_qat_mode(cfg.get("qat_mode"))
    qat_mode = horizon.qat_mode.get_qat_mode()
    assert qat_mode == "with_bn"
    cfg.qat_mode = "fuse_bn"
    horizon.qat_mode.set_qat_mode(cfg.get("qat_mode"))
    qat_mode = horizon.qat_mode.get_qat_mode()
    assert qat_mode == "fuse_bn"


def test_config2json():
    config_path = "tests/data/resnet18_with_deploy_model.py"
    cfg = Config.fromfile(config_path)
    cfg = filter_configs(cfg, return_dict=False)
    assert cfg.dump_json()


def test_dumpjson(tmpdir):
    test = os.path.join(tmpdir, "test_dumpjson.py")
    with open(test, "w") as f:
        f.write("from collections import OrderedDict\n")
        f.write("a = OrderedDict()\n")
        f.write("b = (1, 2)\n")
        f.write("import numpy as np\n")
        f.write("c = np.array([1,2,3])\n")
        f.write("a[b] = c\n")
    config = Config.fromfile(test)
    config = filter_configs(config, return_dict=False)
    config.dump_json()
