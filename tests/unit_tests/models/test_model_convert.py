import copy

import horizon_plugin_pytorch as horizon
import pytest
import torch
from torch import nn

from hat.models.backbones.mobilenetv1 import MobileNetV1
from hat.models.model_convert.converters import (
    Float2Calibration,
    Float2QAT,
    LoadCheckpoint,
    QAT2Quantize,
    QATFusePartBN,
    Torch2Compile,
)
from hat.models.model_convert.pipelines import ModelConvertPipeline
from hat.models.structures.classifier import Classifier
from hat.utils import qconfig_manager
from hat.utils.global_var import global_dict
from hat.utils.package_helper import check_packages_available

try:
    from torch import _dynamo as torch_dynamo
except ImportError:
    torch_dynamo = None

try:
    from horizon_plugin_pytorch.quantization.qconfig_template import (
        default_calibration_qconfig_setter,
        default_qat_qconfig_setter,
    )
except ImportError:
    default_qat_qconfig_setter = None
    default_calibration_qconfig_setter = None


@pytest.fixture
def float_model():
    backbone = MobileNetV1(num_classes=1000, bn_kwargs={})
    model = Classifier(backbone=backbone)
    return model


@pytest.fixture
def calibration_model(float_model):
    float_model.fuse_model()
    qconfig_manager.set_default_qconfig()
    float_model.qconfig = qconfig_manager.get_default_calibration_qconfig()
    float_model.set_calibration_qconfig()
    return float_model


@pytest.fixture
def qat_model(float_model):
    float_model.fuse_model()
    qconfig_manager.set_default_qconfig()
    float_model.qconfig = qconfig_manager.get_default_qat_qconfig()
    float_model.set_qconfig()
    horizon.quantization.prepare_qat(float_model, inplace=True)
    return float_model


@pytest.fixture
def qat_with_bn_model(float_model):
    horizon.qat_mode.set_qat_mode("with_bn")
    float_model.fuse_model()
    qconfig_manager.set_default_qconfig()
    float_model.qconfig = qconfig_manager.get_default_qat_qconfig()
    float_model.set_qconfig()
    horizon.quantization.prepare_qat(float_model, inplace=True)
    return float_model


@pytest.fixture
def quantize_model(qat_model):
    horizon.quantization.convert(qat_model.eval(), inplace=True)
    return qat_model


@pytest.mark.parametrize("convert_mode", ("eager", "fx"))
@pytest.mark.parametrize(
    "example_inputs, qconfig_setter",
    [
        (None, None),
        (
            torch.randn((1, 3, 224, 224)),
            default_qat_qconfig_setter,
        ),
    ],
)
def test_float2qat(float_model, convert_mode, example_inputs, qconfig_setter):
    model_convert = Float2QAT(
        convert_mode=convert_mode,
        example_inputs=example_inputs,
        qconfig_setter=qconfig_setter,
    )
    qat_model = model_convert(float_model)
    assert isinstance(qat_model, nn.Module)


@pytest.mark.parametrize("convert_mode", ("eager", "fx"))
@pytest.mark.parametrize(
    "example_inputs, qconfig_setter",
    [
        (None, None),
        (
            torch.randn((1, 3, 224, 224)),
            default_calibration_qconfig_setter,
        ),
    ],
)
def test_float2calibration(
    float_model, convert_mode, example_inputs, qconfig_setter
):
    model_convert = Float2Calibration(
        convert_mode=convert_mode,
        example_inputs=example_inputs,
        qconfig_setter=qconfig_setter,
    )
    calibration_model = model_convert(float_model)
    assert isinstance(calibration_model, nn.Module)


@pytest.mark.parametrize(
    "qat_fuse_patterns,regex,strict",
    [
        (["^.*backbone"], True, False),
        (["backbone"], False, False),
        (["^.*backbone"], True, True),
    ],
)
def test_qat2qat_part_fuse_bn(
    qat_with_bn_model, qat_fuse_patterns, regex, strict
):
    _qat_with_bn_model = copy.deepcopy(qat_with_bn_model)
    model_convert = QATFusePartBN(
        qat_fuse_patterns=qat_fuse_patterns,
        regex=regex,
        strict=strict,
    )
    qat_part_fuse_bn = model_convert(_qat_with_bn_model)
    assert isinstance(qat_part_fuse_bn, nn.Module)

    # Check if the model still has bn.
    def _check_bn_exist(module):
        for m in module.children():
            if isinstance(m, nn.modules.batchnorm._BatchNorm):
                raise AssertionError("Converted model still has bn!")

    gen = model_convert.get_match_method
    for _, m in gen(qat_part_fuse_bn, qat_fuse_patterns, strict):
        _check_bn_exist(m)

    # reset qat mode to default
    horizon.qat_mode.set_qat_mode("fuse_bn")


@pytest.mark.parametrize("convert_mode", ("eager",))
@pytest.mark.parametrize(
    "preserve_qat_mode_dict",
    (
        None,
        {"prefixes": ("backbone.mod1.0.0",)},
        {"types": (horizon.nn.qat.ConvReLU2d,)},
    ),
)
@pytest.mark.parametrize(
    ["fast_mode", "use_cutlass"],
    [
        pytest.param(False, False),
        pytest.param(True, False),
        pytest.param(False, True),
        pytest.param(True, True),
    ],
)
def test_qat2quantize(
    qat_model, convert_mode, preserve_qat_mode_dict, fast_mode, use_cutlass
):
    fast_mode_available = check_packages_available(
        "horizon_plugin_pytorch>=1.6.3",
        raise_exception=False,
    )
    use_cutlass_available = check_packages_available(
        "horizon_plugin_pytorch>=1.10.1",
        raise_exception=False,
    )

    model_convert = QAT2Quantize(
        convert_mode=convert_mode,
        preserve_qat_mode_dict=preserve_qat_mode_dict,
        fast_mode=fast_mode,
        use_cutlass=use_cutlass,
    )

    if (fast_mode and not fast_mode_available) or (
        use_cutlass and not use_cutlass_available
    ):
        with pytest.raises(ModuleNotFoundError):
            quantize_model = model_convert(qat_model)
    else:
        quantize_model = model_convert(qat_model)
        assert isinstance(quantize_model, nn.Module)
        if preserve_qat_mode_dict is not None:
            assert (
                type(quantize_model.backbone.mod1[0][0])
                == horizon.nn.qat.ConvReLU2d
            )


def test_load_checkpoint(float_model):
    state = {"state_dict": float_model.state_dict()}
    torch.save(state, "float-checkpoint-best-load.pth.tar")
    LC = LoadCheckpoint("float-checkpoint-best-load.pth.tar")
    model = LC(float_model)
    assert isinstance(model, nn.Module)
    assert "model_checkpoint" in global_dict


@pytest.mark.parametrize("convert_mode", ("eager", "fx"))
@pytest.mark.parametrize(
    "preserve_qat_mode_dict",
    (
        None,
        {"prefixes": ("backbone.mod1.0.0",)},
    ),
)
def test_model_convert_pipeline(
    float_model, convert_mode, preserve_qat_mode_dict
):
    state = {"state_dict": float_model.state_dict()}
    ckpt_name = "float-checkpoint-best-{}.pth.tar".format(convert_mode)
    torch.save(state, ckpt_name)
    convert_pipeline = ModelConvertPipeline(
        qat_mode="fuse_bn",
        converters=[
            LoadCheckpoint(ckpt_name),
            Float2QAT(convert_mode=convert_mode),
            QAT2Quantize(
                convert_mode=convert_mode,
                preserve_qat_mode_dict=preserve_qat_mode_dict,
            ),
        ],
    )
    model = convert_pipeline(float_model)
    assert isinstance(model, nn.Module)
    assert "model_checkpoint" in global_dict
    if preserve_qat_mode_dict is not None:
        assert type(model.backbone.mod1[0][0]) == horizon.nn.qat.ConvReLU2d


@pytest.mark.skipif(
    not (
        torch_dynamo
        and check_packages_available(
            "torch>=2.0",
            raise_exception=False,
        )
    ),
    reason="Need torch>=2.0",
)
def test_torch2_compile(float_model):
    compiler = Torch2Compile()
    compiled_model = compiler(float_model)

    assert isinstance(compiled_model, torch_dynamo.eval_frame.OptimizedModule)

    submodule_compiler = Torch2Compile(
        compile_submodules="backbone", regex=True
    )
    submodule_compiled_model = submodule_compiler(float_model)
    assert isinstance(submodule_compiled_model, nn.Module)
    assert isinstance(
        submodule_compiled_model.backbone,
        torch_dynamo.eval_frame.OptimizedModule,
    )
