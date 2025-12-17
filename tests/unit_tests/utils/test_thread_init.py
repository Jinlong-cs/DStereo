import os

import cv2
import pytest
import torch

from hat.utils.thread_init import init_num_threads


@pytest.mark.parametrize(
    [
        "torch_num_threads",
        "opencv_num_threads",
        "openblas_num_threads",
        "mkl_num_threads",
        "omp_num_threads",
    ],
    [
        pytest.param(5, 6, 7, 4, 3),
        pytest.param(None, 2, 8, 5, 7),
        pytest.param(13, None, 4, 3, 6),
        pytest.param(11, 23, None, 4, 6),
        pytest.param(14, 22, 31, None, 5),
        pytest.param(11, 23, 13, 5, None),
        pytest.param(None, None, None, None, None),
    ],
)
def test_init_num_threads(
    torch_num_threads,
    opencv_num_threads,
    openblas_num_threads,
    mkl_num_threads,
    omp_num_threads,
):
    if torch_num_threads is not None:
        str_num_torch = str(torch_num_threads)
        os.environ["TORCH_NUM_THREADS"] = str_num_torch
    else:
        if "TORCH_NUM_THREADS" in os.environ:
            os.environ.pop("TORCH_NUM_THREADS")

    if opencv_num_threads is not None:
        str_num_opencv = str(opencv_num_threads)
        os.environ["OPENCV_NUM_THREADS"] = str_num_opencv
    else:
        if "OPENCV_NUM_THREADS" in os.environ:
            os.environ.pop("OPENCV_NUM_THREADS")

    if openblas_num_threads is not None:
        str_num_openblas = str(openblas_num_threads)
        os.environ["OPENBLAS_NUM_THREADS"] = str_num_openblas
    else:
        if "OPENBLAS_NUM_THREADS" in os.environ:
            os.environ.pop("OPENBLAS_NUM_THREADS")

    if mkl_num_threads is not None:
        str_num_mkl = str(mkl_num_threads)
        os.environ["MKL_NUM_THREADS"] = str_num_mkl
    else:
        if "MKL_NUM_THREADS" in os.environ:
            os.environ.pop("MKL_NUM_THREADS")

    if omp_num_threads is not None:
        str_num_omp = str(omp_num_threads)
        os.environ["OMP_NUM_THREADS"] = str_num_omp
    else:
        if "OMP_NUM_THREADS" in os.environ:
            os.environ.pop("OMP_NUM_THREADS")

    init_num_threads()

    if openblas_num_threads is not None:
        assert os.environ.get("OPENBLAS_NUM_THREADS") == str(
            openblas_num_threads
        )

    if mkl_num_threads is not None:
        assert os.environ.get("MKL_NUM_THREADS") == str(mkl_num_threads)

    if omp_num_threads is not None:
        assert os.environ.get("OMP_NUM_THREADS") == str(omp_num_threads)

    if torch_num_threads is not None:
        assert torch.get_num_threads() == torch_num_threads

    if opencv_num_threads is not None:
        assert cv2.getNumThreads() == opencv_num_threads
