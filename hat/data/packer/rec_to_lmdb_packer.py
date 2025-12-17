import os
import pickle
from typing import Optional

import torch.utils.data as data

from hat.data.datasets.data_packer import Packer
from hat.registry import OBJECT_REGISTRY
from hat.utils.pack_type.mxrecord import unpack


class RecIdxToLmdbPacker(Packer):
    """
    Idx packer for packing rec to lmdb.

    Args:
        raw_dataset: raw_dataset returns img anno in bytes.
        decoded_dataset: decoded returns img anno in decoded format.
        total_len: total number of items to be packed.
        output_dir: path to save idx lmdb.
        num_workers: number of workers used for packing.
        start_idx: which idx to start packing, default 0.
    """

    def __init__(
        self,
        raw_dataset: data.Dataset,
        decoded_dataset: data.Dataset,
        total_len: int,
        output_dir: str,
        num_workers: int,
        start_idx: Optional[int] = 0,
    ):
        self.raw_dataset = raw_dataset
        self.decoded_dataset = decoded_dataset
        self.start_idx = start_idx
        super(RecIdxToLmdbPacker, self).__init__(
            output_dir,
            total_len,
            "lmdb",
            num_workers,
        )

    def pack_data(self, idx):
        curr_idx = idx + self.start_idx
        key = str(curr_idx).encode("ascii")

        return [str(curr_idx), key]

    def _write(self, idx, data):
        self.pack_file.write(data[0], data[1])


class RecImgToLmdbPacker(Packer):
    """
    Img packer for packing rec to lmdb.

    Args:
        raw_dataset: raw_dataset returns img anno in bytes.
        decoded_dataset: decoded returns img anno in decoded format.
        total_len: total number of items to be packed.
        output_dir: path to save img lmdb.
        num_workers: number of workers used for packing.
        start_idx: which idx to start packing, default 0.
        with_seg: true if given datasets are segmentation datasets,
        default false.
    """

    def __init__(
        self,
        raw_dataset: data.Dataset,
        decoded_dataset: data.Dataset,
        total_len: int,
        output_dir: str,
        num_workers: int,
        start_idx: Optional[int] = 0,
        with_seg: Optional[bool] = False,
        unpack_raw_img: Optional[bool] = False,
    ):
        self.raw_dataset = raw_dataset
        self.decoded_dataset = decoded_dataset
        self.start_idx = start_idx
        self.with_seg = with_seg
        self.unpack_raw_img = unpack_raw_img
        super(RecImgToLmdbPacker, self).__init__(
            output_dir,
            total_len,
            "lmdb",
            num_workers,
        )

    def pack_data(self, idx):
        curr_idx = idx + self.start_idx
        key = str(idx)
        raw_img, _raw_anno = self.raw_dataset[curr_idx]
        if self.with_seg:
            header, raw_img_string = unpack(raw_img)
            len_img = int.from_bytes(raw_img_string[:8], "little")
            img_bytes = raw_img_string[8 : len_img + 8]
        else:
            if self.unpack_raw_img:
                _, img_bytes = unpack(raw_img)
            else:
                img_bytes = raw_img

        return [key, img_bytes]

    def _write(self, idx, data):
        self.pack_file.write(data[0], data[1])


class RecAnnoToLmdbPacker(Packer):
    """
    Anno packer for packing rec to lmdb.

    Args:
        raw_dataset: raw_dataset returns img anno in bytes.
        decoded_dataset: decoded returns img anno in decoded format.
        total_len: total number of items to be packed.
        output_dir: path to save anno lmdb.
        num_workers: number of workers used for packing.
        start_idx: which idx to start packing, default 0.
        with_seg: true if given datasets are segmentation datasets,
        default false.
    """

    def __init__(
        self,
        raw_dataset: data.Dataset,
        decoded_dataset: data.Dataset,
        total_len: int,
        output_dir: str,
        num_workers: int,
        start_idx: Optional[int] = 0,
        with_seg: Optional[bool] = False,
    ):
        self.raw_dataset = raw_dataset
        self.decoded_dataset = decoded_dataset
        self.start_idx = start_idx
        self.with_seg = with_seg
        super(RecAnnoToLmdbPacker, self).__init__(
            output_dir,
            total_len,
            "lmdb",
            num_workers,
        )

    def pack_data(self, idx):
        curr_idx = idx + self.start_idx
        key = str(idx)

        if self.with_seg:
            raw_img, raw_anno = self.raw_dataset[curr_idx]
            _, decoded_anno = self.decoded_dataset[curr_idx]
            decoded_anno = decoded_anno[0]
            if not isinstance(decoded_anno, dict):
                decoded_anno = decoded_anno.to_dict()
            header, raw_img_string = unpack(raw_img)
            len_img = int.from_bytes(raw_img_string[:8], "little")
            anno_compressed_bytes = raw_img_string[len_img + 24 :]
            anno = {"seg_label_img_format_bytes": anno_compressed_bytes}
            anno.update(decoded_anno)
        else:
            img, anno = self.decoded_dataset[curr_idx]
            if isinstance(anno, tuple):
                anno = anno[1]
            if not isinstance(anno, dict):
                anno = anno.to_dict()
        anno = pickle.dumps(anno, protocol=4)
        return [key, anno]

    def _write(self, idx, data):
        self.pack_file.write(data[0], data[1])


@OBJECT_REGISTRY.register
class RecToLmdbPacker:
    """
    Packer used to pack rec dataset into lmdb dataset.

    Args:
        raw_dataset: raw_dataset returns img anno in bytes.
        decoded_dataset: decoded returns img anno in decoded format.
        output_dir: root path to save three lmdb datasets.
        num_workers: number of workers used for packing.
        length: pack length, default length of given dataset(whole dataset).
        start_idx: which idx to start packing, default 0.
        with_seg: true if given datasets are segmentation datasets,
        default false.
    """

    def __init__(
        self,
        raw_dataset: data.Dataset,
        decoded_dataset: data.Dataset,
        output_dir: str,
        num_workers: int,
        length: Optional[int] = None,
        start_idx: Optional[int] = 0,
        with_seg: Optional[bool] = False,
        unpack_raw_img: Optional[bool] = False,
    ):
        self.raw_dataset = raw_dataset
        self.decoded_dataset = decoded_dataset
        self.output_dir = output_dir
        self.num_workers = num_workers
        self.queue_maxsize = num_workers * 512
        self.start_method = "spawn"
        self.with_seg = with_seg
        self.unpack_raw_img = unpack_raw_img
        if length is None:
            self.total_len = len(self.raw_dataset)
        else:
            self.total_len = length
        self.start_idx = start_idx

    @property
    def idx_packer(self):
        if not hasattr(self, "_idx_packer"):
            self._idx_packer = RecIdxToLmdbPacker(
                self.raw_dataset,
                self.decoded_dataset,
                self.total_len,
                os.path.join(self.output_dir, "idx"),
                self.num_workers,
                self.start_idx,
            )
        return self._idx_packer

    @property
    def img_packer(self):
        if not hasattr(self, "_img_packer"):
            self._img_packer = RecImgToLmdbPacker(
                self.raw_dataset,
                self.decoded_dataset,
                self.total_len,
                os.path.join(self.output_dir, "img"),
                self.num_workers,
                self.start_idx,
                self.with_seg,
                self.unpack_raw_img,
            )
        return self._img_packer

    @property
    def anno_packer(self):
        if not hasattr(self, "_anno_packer"):
            self._anno_packer = RecAnnoToLmdbPacker(
                self.raw_dataset,
                self.decoded_dataset,
                self.total_len,
                os.path.join(self.output_dir, "anno"),
                self.num_workers,
                self.start_idx,
                self.with_seg,
            )
        return self._anno_packer

    def pack_img(self):
        img_packer = self.img_packer
        img_packer()

    def pack_anno(self):
        anno_packer = self.anno_packer
        anno_packer()

    def pack_idx(self):
        idx_packer = self.idx_packer
        idx_packer()
