# Copyright (c) Horizon Robotics, All rights reserved.
"""AVSPEECH(多模语音)Info文件读取模块.

目前该模块下共有五个类:

BaseInfoList      : Info文件基类.
MMASRInfoList     : 多模ASR的Info文件列表类.
MMASRJsonInfoList : 多模ASR的Json格式的Info文件列表类.
MMCMDInfoList     : 多模命令词的Info文件列表类.
LipMoveInfoList   : 唇动检测的Info文件列表类.
UttParser         : 音频Utt分析器.

其中, MMASRInfoList、MMASRJsonInfoList、MMCMDInfoList 和 LipMoveInfoList
继承 BaseInfoList, 用于读取Info文件并按照规则过滤.
UttParser 用于对音频的Utt进行正则匹配, 获取 dataset、batch、mic 等信息.
"""

import json
import logging
import os
import pathlib
import re
from typing import Any, AnyStr, Dict, Match, Optional, Tuple, Union

from hat.data.datasets.avspeech.parse_utils import (
    parse_id_map,
    parse_monophone,
    parse_video_info,
)
from hat.registry import OBJECT_REGISTRY
from hat.utils.aidi import IS_LOCAL
from hat.utils.filesystem import get_filesystem

logger = logging.getLogger(__name__)


class BucketAdaptor(object):
    in_cluster = not IS_LOCAL
    local_bucket_prefix = ["/bucket/output", "/jfs-hdfs", "/horizon-bucket"]
    remote_bucket_prefix = ["/bucket/input", "/bucket/output"]
    bucket_prefix_dict = {}
    if in_cluster:
        for prefix in remote_bucket_prefix:
            if os.path.exists(prefix):
                for bucket in os.listdir(prefix):
                    bucket_prefix_dict[bucket] = prefix

    @staticmethod
    def adapt(path: str):
        if not BucketAdaptor.in_cluster:
            for prefix in BucketAdaptor.remote_bucket_prefix:
                if path.startswith(prefix):
                    if "/jfs-hdfs" in path:
                        path = path[len(prefix) :]
                    else:
                        path = "/horizon-bucket" + path[len(prefix) :]
            return path
        for prefix in BucketAdaptor.local_bucket_prefix:
            if path.startswith(prefix):
                if prefix == "/jfs-hdfs":
                    bucket_rel_path = path
                    bucket_name = "jfs-hdfs"
                else:
                    bucket_rel_path = path[len(prefix) :]
                    bucket_name = bucket_rel_path.lstrip("/").split("/")[0]
                path = (
                    BucketAdaptor.bucket_prefix_dict[bucket_name]
                    + bucket_rel_path
                )
                break

        return path


class BaseInfoList(object):
    """BaseInfoList.

    Info文件基类.

    Args:
        left_limit: 左边界限制,即小于该时长的会被过滤. 单位秒, 默认是0.1.
        right_limit: 右边界限制,即大于该时长的会被过滤. 单位秒, 默认是6.0.
    """

    def __init__(self, left_limit: float = 0.1, right_limit: float = 6.0):
        self._left_limit = left_limit
        self._right_limit = right_limit

    def __getitem__(self, index: int) -> dict:
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError

    def duration_filter(self, duration: float, line: str) -> bool:
        """duration_filter.

        检查目标时长是否符合限制条件.

        Args:
            duration: 需要被检查的时长.
            line: 被检查的数据行内容, 用于被过滤时打印log.

        Returns:
            是否被过滤. 被过滤返回True, 不被过滤返回False.
        """

        if duration <= 0.0:
            msg = f"错误的INFO行(时长为负): {line}"
        elif duration < self._left_limit:
            msg = f"错误的INFO行(<{self._left_limit}): {line}"
        elif duration > self._right_limit:
            msg = f"错误的info行(>{self._right_limit}): {line}"
        else:
            msg = None
        if msg is not None:
            logger.debug(msg)
        return msg is not None


@OBJECT_REGISTRY.register
class MMASRInfoList(BaseInfoList):
    """MMASRInfoList.

    多模ASR的Info文件列表类.

    Args:
        path: info文件的路径, 支持本地、hdfs和bucket路径.
        left_limit: 左边界限制, 即小于该时长的会被过滤. 单位秒, 默认是0.1.
        right_limit: 右边界限制, 即大于该时长的会被过滤. 单位秒, 默认是6.0.
    """

    def __init__(self, path: Union[str, pathlib.Path], **kwargs):
        super().__init__(**kwargs)
        self._info_path = path
        self._fork()

    def _fork(self):  # type: ignore
        fs = get_filesystem(self._info_path)
        with fs.open(self._info_path, "r", encoding="utf-8") as fr:

            def callback(line: str) -> Optional[Dict[str, Any]]:
                # 内部解析数据的接口
                utt, *text, beg, end = line.strip().split()
                beg, end = float(beg), float(end)
                duration = end - beg
                if self.duration_filter(duration, line):
                    return None
                ret = {
                    "seg_utt": utt,
                    "seg_beg": beg,
                    "seg_end": end,
                    "length": duration,
                    "text": "".join(text),
                }
                return ret

            info_iter = (callback(line) for line in fr)
            self.infos = [ret for ret in info_iter if ret is not None]

    def __getitem__(self, index: int) -> dict:
        return self.infos[index]

    def __len__(self) -> int:
        return len(self.infos)

    def __getstate__(self):  # type: ignore
        state = self.__dict__.copy()
        del state["infos"]
        return state

    def __setstate__(self, state):  # type: ignore
        self.__dict__ = state
        self._fork()

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__ + ":"
        repr_str += f"path=.../{pathlib.Path(self._info_path).name}"
        repr_str += f", left_limit={self._left_limit :.3f}"
        repr_str += f", right_limit={self._right_limit :.3f}"
        repr_str += f", lens={len(self)}"
        return repr_str


@OBJECT_REGISTRY.register
class MMASRJsonInfoList(BaseInfoList):
    """MMASRJsonInfoList.

    多模ASR的Json格式的Info文件列表类.

    Args:
        path: json格式的info文件的路径.
        left_limit: 左边界限制, 即小于该时长的会被过滤. 单位秒, 默认是0.1.
        right_limit: 右边界限制, 即大于该时长的会被过滤. 单位秒, 默认是6.0.
    """

    def __init__(self, path: Union[str, pathlib.Path], **kwargs):
        super().__init__(**kwargs)
        self._info_path = path
        self._fork()

    def _fork(self):
        fs = get_filesystem(self._info_path)
        with fs.open(self._info_path, "r", encoding="utf-8") as fr:

            def callback(line: str) -> Optional[Dict[str, Any]]:
                # 每条数据解析接口
                item = json.loads(line.strip())
                length = item["end"] - item["start"]
                # 时长检查
                if self.duration_filter(length, line):
                    return None
                ret = {
                    "text": item["label"],
                    "seg_beg": item["start"],
                    "seg_end": item["end"],
                    "length": length,
                }

                # 开发机和集群路径适配
                mp4_path: str = item["mp4_path"]
                wav_path: str = item["wav_path"]
                if not IS_LOCAL:
                    jfs_prefix_list = ["/bucket/input", "/bucket/output", ""]
                    for jfs_prefix in jfs_prefix_list:
                        if os.path.exists(f"{jfs_prefix}/jfs-hdfs"):
                            break
                    if mp4_path.startswith("/jfs-hdfs/"):
                        mp4_path = f"{jfs_prefix}{mp4_path}"
                    if wav_path.startswith("/jfs-hdfs/"):
                        wav_path = f"{jfs_prefix}{wav_path}"
                ret["mp4_path"] = mp4_path
                ret["wav_path"] = wav_path
                return ret

            info_iter = (callback(line) for line in fr)
            self.infos = [ret for ret in info_iter if ret is not None]

    def __getitem__(self, index: int) -> dict:
        return self.infos[index]

    def __len__(self) -> int:
        return len(self.infos)

    def __getstate__(self):  # type: ignore
        state = self.__dict__.copy()
        del state["infos"]
        return state

    def __setstate__(self, state):  # type: ignore
        self.__dict__ = state
        self._fork()

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__ + ":"
        repr_str += f"path=.../{pathlib.Path(self._info_path).name}"
        repr_str += f", left_limit={self._left_limit :.3f}"
        repr_str += f", right_limit={self._right_limit :.3f}"
        repr_str += f", lens={len(self)}"
        return repr_str


@OBJECT_REGISTRY.register
class MMCMDInfoList(BaseInfoList):
    """MMCMDInfoList.

    多模命令词的Info文件列表类.

    Args:
        info_path: 多模命令词info文件的路径.
        monophone_path: 样本utt和单音素序列的对应文件路径.
        id_map_path: 样本utt和特征rec时间戳的对应文件路径.
    """

    def __init__(self, info_path: str, monophone_path: str, id_map_path: str):
        self._info_path = info_path
        self._monophone_path = monophone_path
        self._id_map_path = id_map_path
        self._fork()

    def _fork(self):  # type: ignore
        utt_to_video_info = parse_video_info(self._info_path)
        utt_to_monophone = parse_monophone(self._monophone_path)
        utt_to_idxs = parse_id_map(self._id_map_path)
        infos = []
        filtered_count = 0
        for utt, idxs in utt_to_idxs.items():
            if utt not in utt_to_video_info:
                filtered_count += len(idxs)
                continue
            if utt not in utt_to_monophone:
                filtered_count += len(idxs)
                continue

            info = utt_to_video_info[utt]
            info["tokens"] = utt_to_monophone[utt]["monophone"]
            info["rec_idx"] = idxs
            infos.append(info)
        self.infos = infos

        logging.warning(f"Total Filtered : {filtered_count}")

    def __getitem__(self, index: int) -> dict:
        return self.infos[index]

    def __len__(self) -> int:
        return len(self.infos)

    def __getstate__(self):  # type: ignore
        state = self.__dict__.copy()
        del state["infos"]
        return state

    def __setstate__(self, state):  # type: ignore
        self.__dict__ = state
        self._fork()


@OBJECT_REGISTRY.register
class LipMoveInfoList(BaseInfoList):
    """LipMoveInfoList.

    唇动检测任务的Info文件列表类.

    Args:
        path: 唇动检测info文件的路径.
    """

    def __init__(self, path: str):
        self._info_path = path
        self._fork()

    def _fork(self):  # type: ignore
        fs = get_filesystem(self._info_path)
        with fs.open(self._info_path, "r", encoding="utf-8") as fr:

            def callback(line: str) -> Optional[Dict[str, Any]]:
                lsp = line.strip().split(maxsplit=5)
                utt, text, seg_beg, seg_end, vad_beg, vad_end = lsp[:6]
                length = float(seg_end) - float(seg_beg)
                ret = {
                    "seg_utt": utt,
                    "seg_beg": float(seg_beg),
                    "seg_end": float(seg_end),
                    "vad_beg": float(vad_beg),
                    "vad_end": float(vad_end),
                    "text": text,
                    "length": length,
                }
                return ret

            info_iter = (callback(line) for line in fr)
            self.infos = [ret for ret in info_iter if ret is not None]

    def __getitem__(self, index: int) -> dict:
        return self.infos[index]

    def __len__(self) -> int:
        return len(self.infos)

    def __getstate__(self):  # type: ignore
        state = self.__dict__.copy()
        del state["infos"]
        return state

    def __setstate__(self, state):  # type: ignore
        self.__dict__ = state
        self._fork()


class UttParser(object):
    """UttParser.

    音频Utt分析器, 将Utt进行正则匹配, 得到dataset、batch、mic等信息.
    """

    _DATASET_BATCH_PATTERNS = {
        "standard": re.compile(
            "(J2MM|S202DA|IPC_halo2|IPC_halo0|DATATANG|FYZ|BYD|QR|Qianren|MM_General).*?_(batch[0-9]{2})"  # noqa: E501
        ),
        "HAOKAN": re.compile("(batch[0-9]{4})"),
    }

    _UTT_PASER_REGEXS = {
        "J2MM": "(?P<dataset>J2MM.*?)_(?P<batch>batch[0-9]{2})-(?P<snt>[0-9]{5}_[0-9]{2}_[0-9]{2})(_(?P<mic>mic[0-9]+))?(_(?P<cam>cam[0-9]+))?(_(?P<index>[0-9]*))?",  # noqa: E501
        "IPC_halo0": "(?P<dataset>IPC_halo0.*?)_(?P<batch>batch[0-9]{2})-(?P<snt>((od)|(ho)|[0-9])[0-9]+_[0-9]{2}_[0-9]+)(_(?P<mic>mic[0-9]+))?(_(?P<cam>cam[0-9]+))?(_(?P<index>[0-9]*))?",  # noqa: E501
        "S202DA": "(?P<dataset>S202DA.*?)_(?P<batch>batch[0-9]{2})-(?P<snt>[0-9]{5}_[0-9]{2}_[0-9]{2})(_(?P<mic>mic[0-9]+))?(_(?P<cam>cam[0-9]+))?(_(?P<index>[0-9]*))?",  # noqa: E501
        "DATATANG": "(?P<dataset>DATATANG)_(?P<batch>batch[0-9]{2})-(?P<snt>G[0-9]{4}_[0-9]{2}_[0-9]{4})(_(?P<mic>mic[0-9]+))?(_(?P<cam>cam[0-9]+))?(_(?P<index>[0-9]*))?",  # noqa: E501
        "HAOKAN": "(?P<batch>batch[0-9]{4})-(?P<snt>11[0-9]{4})(_(?P<index>[0-9]*))?",  # noqa: E501
        "IPC_halo2": "(?P<dataset>IPC_halo2)_(?P<batch>batch[0-9]{2})_(?P<snt>[0-9]{5}_[0-9]{2}_[0-9]{2})(_(?P<mic>mic[0-9]+))?(_(?P<cam>cam[0-9]+))?(_(?P<index>[0-9]*))?",  # noqa: E501
        "FYZ": "(?P<dataset>FYZ.*?)_(?P<batch>batch[0-9]{2})-(?P<snt>((batch[0-9]{2}_)*[A-Z][0-9]{4}_[0-9]{2}_[0-9]{2}|[A-Z][0-9]{4}))(_(?P<mic>mic[0-9]+))?(_(?P<cam>cam[0-9]+))?(_(?P<index>[0-9]*))?",  # noqa: E501
        "BYD": "(?P<dataset>BYD.*?)_(?P<batch>batch[0-9]{2})-(?P<snt>((od)|(ho)|[0-9])[0-9]+_[0-9]{2}_[0-9]+)(_(?P<mic>mic[0-9]+))?(_(?P<cam>cam[0-9]+))?(_(?P<index>[0-9]*))?",  # noqa: E501
        "QR": "(?P<dataset>QR.*?)_(?P<batch>batch[0-9]{2})-(?P<snt>((od)|(ho))[0-9]{6}_[0-9]{2}_[0-9]{2,3})(_(?P<mic>mic[0-9]+))?(_(?P<cam>cam[0-9]+))?(_(?P<index>[0-9]*))?",  # noqa: E501
        "Qianren": "(?P<dataset>Qianren.*?)_(?P<batch>batch[0-9]{2})-(?P<snt>(((hy)[0-9]{6,7})|([0-9]{4}))_[0-9]{2}_[0-9]{2,3})(_(?P<mic>mic[0-9]+))?(_(?P<cam>cam[0-9]+))?(_(?P<index>[0-9]*))?",  # noqa: E501
        "MM_General": "(?P<dataset>MM_General.*?)_(?P<batch>batch[0-9]{2})-(?P<snt>((mp)[0-9]{6})_[0-9]{2}_[0-9]{3})(_(?P<mic>mic[0-9]+))?(_(?P<cam>cam[0-9]+))?(_(?P<index>[0-9]*))?",  # noqa: E501
    }

    def __init__(self):  # type: ignore[no-untyped-def]
        self.parsers = {}

    def match(self, utt: str) -> Optional[Match[AnyStr]]:
        dataset, batch = self.parse_dataset_and_batch(utt)
        if dataset not in self.parsers:
            self.parsers[dataset] = re.compile(self._UTT_PASER_REGEXS[dataset])
        matched = self.parsers[dataset].match(utt)
        if matched is None:
            logging.warning(f"[{utt}] can't be matched.")
        return matched

    def __getstate__(self):  # type: ignore
        state = self.__dict__.copy()
        del state["parsers"]
        return state

    def __setstate__(self, state):  # type: ignore
        self.__dict__ = state
        self.parsers = {}

    @classmethod
    def parse_dataset_and_batch(cls, info_key: str) -> Tuple[str, str]:
        dataset, batch = "unknown", "unknown"
        for key, pattern in cls._DATASET_BATCH_PATTERNS.items():
            db_and_batch = pattern.findall(info_key)
            if len(db_and_batch) != 1:
                continue
            if key == "HAOKAN":
                dataset, batch = "HAOKAN", db_and_batch[0]
                break
            else:
                dataset, batch = db_and_batch[0]
                break
        return dataset, batch
