import os

from hat.utils.gpu_affinity import set_affinity


def test_gpu_affinity():
    mode = [
        "none",
        "socket",
        "socket_single",
        "socket_single_unique",
        "socket_unique_interleaved",
        "socket_unique_contiguous",
    ]

    for m in mode:
        affinity = set_affinity(0, 2, mode=m)
        print(affinity)
        assert affinity is not None

        # reset
        os.sched_setaffinity(0, tuple(range(os.cpu_count())))
