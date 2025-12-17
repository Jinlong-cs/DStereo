import numba
import pytest
from numba import cuda

from hat.utils.numba_helper import numba_jit

try:
    from numba.cuda.dispatcher import CUDADispatcher
except (ImportError, AttributeError):
    CUDADispatcher = None

try:
    from numba.cuda.compiler import DeviceFunctionTemplate
except ImportError:
    DeviceFunctionTemplate = None


@pytest.mark.skipif(
    not (CUDADispatcher or DeviceFunctionTemplate),
    reason="requiring HAT_BUCKET bucket",
)
def test_numba_jit():
    @numba_jit(device=cuda.is_available())
    def numba_add(a, b, out=None):
        if out:
            out = a + b
        else:
            return a + b

    if cuda.is_available():
        # for different numba version
        if CUDADispatcher:
            assert isinstance(numba_add, CUDADispatcher)
        elif DeviceFunctionTemplate:
            assert isinstance(numba_add, DeviceFunctionTemplate)
    else:
        print(numba_add)
        assert isinstance(numba_add, numba.core.registry.CPUDispatcher)
