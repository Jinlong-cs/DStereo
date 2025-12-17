import subprocess
import time
import warnings

from .utils import psutil_kills

try:
    import pynvml
except ImportError:
    pynvml = None


__all__ = ["check_time_info", "check_cpu_info", "check_gpu_info"]


def check_time_info(pid, begin_time):
    running_time = time.time() - begin_time

    # limit time to 60 min
    if running_time > 60 * 60:
        if pid is not None:
            psutil_kills(pid)
        raise RuntimeError("The running time is limited to 60 min !!!")


def check_cpu_info(pid):
    used_file = "/sys/fs/cgroup/memory/memory.usage_in_bytes"
    limit_file = "/sys/fs/cgroup/memory/memory.limit_in_bytes"

    used = int(
        subprocess.check_output(["cat", used_file]).decode("ascii").strip()
    )
    used = used / 1024 ** 3

    total = int(
        subprocess.check_output(["cat", limit_file]).decode("ascii").strip()
    )
    total = total / 1024 ** 3

    # cpu memory is elastic, just throw warning here
    if total - used < 0.01:
        warnings.warn(f"CPU memory will reach the limit, {used} / {total}")


def check_gpu_info(pid):
    if pynvml is None:
        return

    pynvml.nvmlInit()
    gpu_count = pynvml.nvmlDeviceGetCount()
    for index in range(gpu_count):
        handle = pynvml.nvmlDeviceGetHandleByIndex(index)
        meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
        total = meminfo.total / 1024 ** 2
        used = meminfo.used / 1024 ** 2

        # limit gpu to `100M less than max value`.
        if total - used < 100:
            if pid is not None:
                psutil_kills(pid)
            raise RuntimeError(
                f"GPU{index} memory has 100M left to reach the limit!!!"
            )
