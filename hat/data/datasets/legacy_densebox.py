"""legacy densebox dataset."""
import json
import os
import zlib
from distutils.version import LooseVersion

import cv2
import horizon_plugin_pytorch
import numpy as np
import torch.utils.data as data

from hat.core.anno_ts_utils import ImageRecord, get_img_idx_to_img_record_map
from hat.data.datasets.utils import img_to_rgb
from hat.registry import OBJECT_REGISTRY
from hat.utils.apply_func import _as_numpy as as_numpy
from hat.utils.pack_type.mxrecord import (
    MXRecord,
    MXRecordIO,
    unpack,
    unpack_img,
)
from hat.utils.pack_type.recordio_pb2 import RecordUnit

if LooseVersion(horizon_plugin_pytorch.__version__) >= LooseVersion("1.0.0"):
    HAT_LEGACYDENSEBOX_AVAILABLE = True
else:
    HAT_LEGACYDENSEBOX_AVAILABLE = False


def unpack_legacy_densebox_img_record(raw_record, to_rgb=False):
    header, s = unpack(raw_record)
    densebox_rec_flag = int(header.label)
    assert densebox_rec_flag == 1, (
        "hat.data.datasets."
        "LegacyDenseBoxImageRecordDataset only support the record."
        "header.label is mxnet::io::densebox::DENSEBOX_RECORD_IO_CONTENT"
        "when with_seg_label is True."
    )
    len_img = int.from_bytes(s[:8], "little")

    img_buf = s[8 : len_img + 8]
    imgs = cv2.imdecode(
        np.frombuffer(img_buf, dtype=np.uint8), flags=cv2.IMREAD_COLOR
    )
    if to_rgb:
        imgs = img_to_rgb(imgs)
    if densebox_rec_flag in [-1, -2]:
        return header, [imgs], img_buf
    elif densebox_rec_flag == 1:
        anno_compressed_bytes = s[len_img + 24 :]
        uncompressed_label = zlib.decompress(anno_compressed_bytes)
        c = int.from_bytes(uncompressed_label[0:4], "little")
        w = int.from_bytes(uncompressed_label[4:8], "little")
        h = int.from_bytes(uncompressed_label[8:12], "little")
        anno = (
            np.frombuffer(uncompressed_label[12:], dtype=np.float32)
            .reshape(c, w, h)
            .astype(np.int8)
        )
        return header, [anno, imgs], img_buf


@OBJECT_REGISTRY.register
class LegacyDenseBoxImageRecordDataset(data.Dataset):
    """
    A dataset that can read densebox image record.

    Args:
        rec_path: Image record path.
        anno_path: Annotation path.
        rec_idx_file_path: Image record index file path.
            if None use:rec_path + '.idx'
        read_only: Whether output raw content, by default False
        to_rgb: Whether output image in rgb color, by default True
        with_img_buf: Whether the raw content with img buf.
        with_seg_label: Whether the raw content with segmentation label.
        seg_label_dtype: The output data type of segmentation label.
        image_processing_backend: image processing backend.
            choose opencv, imageio, PIL, turbojpeg.
    """

    def __init__(
        self,
        rec_path: str,
        anno_path: str,
        read_only: bool = False,
        with_img_buf: bool = False,
        with_seg_label: bool = False,
        to_rgb: bool = True,
        rec_idx_file_path: str = None,
        seg_label_dtype: type = np.uint8,
        image_processing_backend: str = "opencv",
    ):
        assert (
            HAT_LEGACYDENSEBOX_AVAILABLE
        ), "horizon_plugin_pytorch >= 1.0.0 is required."
        self.read_only = read_only
        self.to_rgb = to_rgb
        self.with_img_buf = with_img_buf
        self.seg_label_dtype = seg_label_dtype
        self.with_seg_label = with_seg_label
        if rec_idx_file_path is None:
            rec_idx_file_path = rec_path + ".idx"

        if not os.path.exists(rec_idx_file_path):
            recode_tmp = MXRecordIO(rec_path, "r")
            recode_tmp.create_idx_file(rec_idx_file_path)
            recode_tmp.close()
        self._rec_dataset = MXRecord(
            uri=rec_path, idx_path=rec_idx_file_path, writable=False
        )

        self._anno_dataset = self._get_anno_dataset(anno_path)
        if isinstance(self._anno_dataset, dict):
            assert len(self._rec_dataset) == len(self._anno_dataset.keys())
        else:
            assert len(self._rec_dataset) == len(self._anno_dataset)

        self.image_processing_backend = image_processing_backend

    def _decoder(self, raw_record, anno):
        if self.read_only:
            return raw_record, anno
        else:
            imgs, raw_img = self._img_record_decoder(raw_record)
            if isinstance(anno, (str, bytes)):
                anno = self._anno_record_decoder(anno)
            if len(imgs) == 1:
                return imgs[0], raw_img, anno
            else:
                if self.seg_label_dtype is None:
                    seg_label = imgs[0][0]
                else:
                    seg_label = imgs[0][0].astype(self.seg_label_dtype)
                return imgs[1], raw_img, (anno, seg_label)

    def _anno_record_decoder(self, record):
        _, s = unpack(record)
        rec_data = RecordUnit()
        rec_data.ParseFromString(s)
        rec_data = rec_data.body
        assert len(rec_data.data) in [2, 3]
        assert len(rec_data.extra) == 0
        image_record = ImageRecord(
            init_dict=json.loads(bytes.decode(rec_data.data[1].value)),
            force_utf8=False,
        )
        if len(rec_data.data) == 3:
            parsing_map = cv2.imdecode(
                np.frombuffer(rec_data.data[2].value, dtype=np.uint8), flags=-1
            )
            return image_record, parsing_map
        else:
            return image_record

    def _img_record_decoder(self, raw_record):
        if self.with_seg_label:
            _, imgs, raw_img = unpack_legacy_densebox_img_record(
                raw_record, to_rgb=self.to_rgb
            )
        else:
            _, imgs, raw_img = unpack_img(
                raw_record,
                with_img_buf=True,
                backend=self.image_processing_backend,
            )
            if self.to_rgb:
                imgs = img_to_rgb(imgs)
            imgs = [imgs]

        assert len(imgs) in [1, 2]
        imgs = [as_numpy(img_i) for img_i in imgs]
        return imgs, raw_img

    def __getitem__(self, idx):
        raw_record = self._rec_dataset.read(idx)

        if isinstance(self._anno_dataset, dict):
            anno = self._anno_dataset[idx]
        else:
            anno = self._anno_dataset.read(idx)

        if self._decoder is None:
            return raw_record, anno
        else:
            img, img_buf, anno = self._decoder(raw_record, anno)
            if self.with_img_buf:
                return img, img_buf, anno
            else:
                return img, anno

    def __len__(self):
        return len(self._rec_dataset)

    def _get_anno_dataset(self, anno_path):

        ext = os.path.splitext(anno_path)[1]

        if ext in [".json"]:

            idx2anno_map = {}

            img_idx2anno_map = get_img_idx_to_img_record_map(anno_path)
            idx = 0
            while True:
                raw_record = self._rec_dataset.record.read()
                if raw_record is None:
                    break
                header, _ = unpack(raw_record)
                idx2anno_map[idx] = img_idx2anno_map[header.id]
                idx += 1
            self._rec_dataset.reset()
            return idx2anno_map

        elif ext in [".rec", ".pb_rec"]:

            anno_idx_path = anno_path + ".idx"
            if not os.path.exists(anno_idx_path):
                recode_tmp = MXRecordIO(self.data_path, "r")
                recode_tmp.create_idx_file(anno_idx_path)
                recode_tmp.close()
            return MXRecord(
                uri=anno_path, idx_path=anno_idx_path, writable=False
            )

        else:
            raise ValueError("Invalid anno_path %s" % anno_path)
