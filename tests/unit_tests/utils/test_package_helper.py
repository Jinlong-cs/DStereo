import operator

import pytest
import torch

from hat.utils.package_helper import (
    check_packages_available,
    compare_version,
    require_packages,
)

torch_version = torch.__version__


@pytest.mark.parametrize(
    ["package_name", "op", "version", "except_ret"],
    [
        pytest.param("torch", ">=", torch_version, True),
        pytest.param("torch", "==", torch_version, True),
        pytest.param("torch", ">", "100000.0.1", False),
        pytest.param("torch", "!=", "100000.0.1", True),
        pytest.param("torch", operator.le, "100000.0.1", True),
        pytest.param("torch", operator.lt, "100000.0.1", True),
    ],
)
def test_compare_version(package_name, op, version, except_ret):
    ret = compare_version(package_name, op, version, return_version=False)
    assert ret == except_ret


@pytest.mark.parametrize(
    [
        "modules",
        "raise_exception",
        "except_ret",
    ],
    [
        pytest.param(["torch"], True, True),
        pytest.param([f"torch>={torch_version}"], False, True),
        pytest.param(["torch>=10000.0"], True, False),
        pytest.param(
            [f"torch>={torch_version}", "torchvision"],
            False,
            True,
        ),
    ],
)
def test_check_packages_available(modules, raise_exception, except_ret):
    if raise_exception and not except_ret:
        with pytest.raises(ModuleNotFoundError):
            check_packages_available(*modules, raise_exception=raise_exception)
    else:
        ret = check_packages_available(
            *modules, raise_exception=raise_exception
        )
        assert ret == except_ret


@pytest.mark.parametrize(
    [
        "modules",
        "raise_exception",
        "except_ret",
    ],
    [
        pytest.param(["torch"], True, True),
        pytest.param([f"torch>={torch_version}"], False, True),
        pytest.param(["torch>=10000.0"], True, False),
        pytest.param(
            [f"torch>={torch_version}", "torchvision", "torch.nn"],
            False,
            True,
        ),
    ],
)
def test_requires(modules, raise_exception, except_ret):
    @require_packages(*modules, raise_exception=raise_exception)
    def test_func():
        import torch

        _torch_version = torch.__version__
        print(_torch_version)

    class TestCls:
        @require_packages(*modules, raise_exception=raise_exception)
        def __init__(self) -> None:
            import torch
            import torchvision

            _torch_version = torch.__version__
            _vision_version = torchvision.__version__
            print(_torch_version, _vision_version)

    if raise_exception and not except_ret:
        with pytest.raises(ModuleNotFoundError):
            test_func()

        with pytest.raises(ModuleNotFoundError):
            TestCls()

    else:
        # maybe import fail, but not raise ModuleNotFoundError,
        # log warning only
        test_func()
        TestCls()
