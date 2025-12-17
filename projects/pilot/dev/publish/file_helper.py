import logging
import os
import os.path as osp
import re
import subprocess
import zipfile
from tempfile import NamedTemporaryFile, _TemporaryFileWrapper
from typing import Iterator, Optional

import requests
from hatbc.filestream.utils import url_to_local_path

logger = logging.getLogger(__name__)


class CacheFile:
    def __init__(self, url: str):
        self.url = url
        self._cache_file: _TemporaryFileWrapper = None

    @property
    def filename(self) -> str:
        return osp.basename(self.url)

    @property
    def cache_file(self) -> _TemporaryFileWrapper:
        if self._cache_file is None:
            if self.url.startswith("http"):
                self._cache_file = NamedTemporaryFile(
                    suffix=osp.basename(self.url)
                )
                req = requests.get(url=self.url)
                if not req.ok:
                    raise FileNotFoundError(self.url)
                self._cache_file.write(req.content)
                self._cache_file.flush()
            else:
                filepath = osp.abspath(url_to_local_path(self.url))
                self._cache_file = _TemporaryFileWrapper(
                    file=open(filepath, mode="rb"),
                    name=filepath,
                    delete=False,
                )

        return self._cache_file

    def is_same(self, url: str) -> bool:
        return url == self.url

    @property
    def file(self):
        return self.cache_file.file

    @property
    def name(self) -> str:
        return self.cache_file.name

    def __enter__(self):
        self.cache_file.__enter__()
        return self

    def __exit__(self, exc, value, tb):
        result = self.cache_file.__exit__(exc=exc, value=value, tb=tb)
        return result

    def __iter__(self) -> Iterator:
        return self.cache_file.__iter__()


def zip_dir(src_dir: str, dst: str, zip_self: Optional[bool] = True) -> str:
    src_dir = osp.abspath(src_dir)
    dst = osp.abspath(dst)
    assert osp.isdir(src_dir)

    src_files = []
    for dirpath, _, filenames in os.walk(src_dir):
        for filename in filenames:
            src_files.append(osp.abspath(osp.join(dirpath, filename)))

    pre_dir = os.getcwd()
    if zip_self:
        work_dir = osp.abspath(osp.dirname(src_dir))
    else:
        work_dir = osp.abspath(src_dir)
    os.chdir(work_dir)

    os.makedirs(osp.dirname(dst), exist_ok=True)
    with zipfile.ZipFile(dst, "w", zipfile.ZIP_DEFLATED) as z:
        for src_file in src_files:
            z.write(src_file, osp.relpath(src_file, work_dir))

    os.chdir(pre_dir)

    return dst


def upload_to_gallery(
    results,
    version,
    group,
    project,
    username,
    password,
):
    init_cmd = f"""
    (echo "{username}"
    sleep 1
    echo "{password}"
    sleep 1
    )|gallery-cli login
    gallery-cli init -p {project} -g {group}
    """
    pattern = re.compile("http.*")
    upload_cmd = f"gallery-cli upload -f {results} --version {version}"
    subprocess.check_call(init_cmd, shell=True)
    rets = subprocess.check_output(upload_cmd, shell=True)
    logger.info("Finish uploading")
    download_url = str(rets, encoding="utf-8")
    logger.info(download_url)
    download_url = pattern.findall(download_url)[0]
    return download_url
