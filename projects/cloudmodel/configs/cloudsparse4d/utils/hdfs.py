import hashlib
import logging
import os
import subprocess

import numpy as np

logger = logging.getLogger()


def process_command(cmd_str):
    p = subprocess.Popen(cmd_str, shell=True)
    res, err = p.communicate()
    rc = p.returncode
    if not rc:
        logger.info(f"Finished: {cmd_str}")
    else:
        logger.error(err)


class HDFSDownloader(object):
    def __init__(self, filename, cache_path=".hdfs_cache", clean=False):
        """HDFSDownloader

        Args:
            filename: str, filename to download
            cache_path, str, cache path to download the file
            clean, bool, clean the download caches after using.
        """
        self.filename = filename
        self.cache_path = cache_path
        self.clean = clean

    def __enter__(self):
        hdfs_path = self.filename
        os.makedirs(self.cache_path, exist_ok=True)
        fname = os.path.join(self.cache_path, os.path.basename(self.filename))
        if not os.path.exists(fname):
            cmd = f"hdfs dfs -get {hdfs_path} {self.cache_path}"
            process_command(cmd)
        else:
            logger.info(f"Cache file {fname} exists.")
        assert os.path.exists(fname), "Failed to download file."
        return fname

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.clean:
            cmd = f"rm -rf {self.cache_path}"
            process_command(cmd)


def get_hdfs_file(filename, cache_path=".hdfs_cache", reader="np"):
    assert reader in ["np"], f"reader {reader} not supported yet."
    with HDFSDownloader(filename, cache_path) as fname:
        data = np.load(fname)
    return data
