# Copyright (c) Horizon Robotics. All rights reserved.
import logging
import os
import sys
import time
from abc import ABC

from hat.utils.distributed import get_dist_info
from hat.utils.logger import LOG_DIR, get_monitor_logger

__all__ = ["PackType"]


class PackType(ABC):
    """
    Data type interface class.

    Args:
        fixed_read_data: If reuse a fixed read data .
        log_speed: If logger the read data speed.
        kwargs: Kwargs of PackType.
    """

    def __init__(
        self,
        fixed_read_data: bool = False,
        log_speed: bool = True,
        **kwargs,
    ):
        self.fixed_read_data = fixed_read_data
        self.log_speed = log_speed
        self.fixed_data = None
        self.ncalls = 0
        self.read_size = 0
        self.read_time = 0
        self.hat_monitor_interval = 600
        self.start_time = None
        self.set_logger = False
        self.logger = None
        self.rank = 0
        if int(os.getenv("HAT_MONITOR_RANK_ZERO", 0)) == 0:
            self.rank, _ = get_dist_info()

    def open(self):
        """Open the data file."""
        pass

    def close(self):
        """Close the data file."""
        pass

    def write(self, idx: int, record: bytes):
        """Write record into data file by idx."""
        pass

    def read(self, idx):
        if self.logger is None:
            self.logger = get_monitor_logger(
                __name__,
                os.path.join(LOG_DIR, "data-profiler"),
                logging.INFO,
                rank=self.rank,
            )
        if self.start_time is None:
            self.start_time = time.time()
        if self.log_speed:
            stime = time.time()
            res = self._read(idx)
            self.read_time += time.time() - stime
            self.read_size += sys.getsizeof(res)
            if (
                time.time() - self.start_time
                > float(os.getenv("HAT_MONITOR_INTERVAL", self.hat_monitor_interval))
                or int(os.getenv("HAT_MONITOR_ALL_TIME", 0)) == 1
            ):
                read_size = self.read_size / 1024 / 1024
                speed = read_size / self.read_time
                if not hasattr(self, "uri"):
                    self.uri = None
                self.logger.info(
                    f"process[{os.getpid()}] "
                    f"file[{self.uri}] idx[{idx}] : Reading speed is "
                    f"{'%.2f' % speed} MB/sec"
                )
                self.read_size = 0
                self.read_time = 0
                self.start_time = time.time()
            return res
        else:
            return self._read(idx)

    def _read(self, idx):
        if self.fixed_read_data:
            if self.fixed_data is None:
                self.fixed_data = self.read_idx(idx)
            return self.fixed_data
        else:
            return self.read_idx(idx)

    def read_idx(self, idx):
        """Read the idx-th data."""
        pass

    def reset(self):
        """Reset the data file operator."""
        pass

    def get_keys(self):
        """Get keys for read."""
        pass

    def __del__(self):
        """Recycle resources."""
        self.close()

    def __len__(self):
        """Get the length."""
        pass
