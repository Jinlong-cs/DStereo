import pytest
import torch

from hat.registry import build_from_registry
from hat.utils.checkpoint import load_checkpoint, load_state_dict
from hat.utils.package_helper import check_packages_available


@pytest.mark.parametrize(
    ["url", "enable_tracking"],
    [
        pytest.param(
            "http://fm-min-du.alitrain.hogpu.cc/plat_gpu/\
fsd_multitask_v0.0.1-20210429-004913-COPY3/output/models/fsd_multitask\
/checkpoint-0012.pth.tar",
            False,
        ),
        pytest.param(
            "http://fm-mengyang-duan.train.hogpu.cc/HAT_TEST/\
ut_checkpoint/float-checkpoint-best-ef67e7d8.pth.tar",
            False,
        ),
    ],
)
def test_loadcheckpoint(url, enable_tracking):
    checkpoint = load_checkpoint(
        url, check_hash=False, enable_tracking=enable_tracking
    )
    assert isinstance(checkpoint, dict)
    assert "state_dict" in checkpoint


def test_load_checkpoint_with_state_dict_update():
    def _update_state_dict(state_dict):
        state_dict = {"hat." + k: v for k, v in state_dict.items()}
        return state_dict

    remote_url = "http://fm-min-du.alitrain.hogpu.cc/plat_gpu/fsd_multitask_v0.0.1-20210429-004913-COPY3/output/models/fsd_multitask/checkpoint-0012.pth.tar"  # noqa
    checkpoint = load_checkpoint(
        remote_url, state_dict_update_func=_update_state_dict, check_hash=False
    )
    assert isinstance(checkpoint, dict)
    assert "state_dict" in checkpoint
    assert "hat." in list(checkpoint["state_dict"].keys())[0]


def test_load_state_dict():
    cfg = dict(
        type="MobileNetV1",
        alpha=1.0,
        num_classes=100,
        bn_kwargs={},
    )
    state_dict = build_from_registry(cfg).state_dict()
    model = build_from_registry(cfg)

    # test load compile state dict
    if check_packages_available("torch>=2.0", raise_exception=False):

        compiled_model = torch.compile(model)
        compiled_state_dict = torch.compile(
            build_from_registry(cfg)
        ).state_dict()  # noqa E501

        # compile_model load compiled state_dict
        compiled_model = load_state_dict(compiled_model, state_dict)

        # compile_model load common state_dict
        compiled_model = load_state_dict(compiled_model, state_dict)

        # common model load compiled state_dict
        model = load_state_dict(model, compiled_state_dict)

    model = load_state_dict(model, state_dict)
    assert model is not None
