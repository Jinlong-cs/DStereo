import ctypes
import json
import logging
import os
import pickle
import zlib
from multiprocessing import Pool
from typing import Any, List, Optional

import cv2

from hat.core.compose_transform import Compose
from hat.data.datasets.data_packer import Packer
from hat.registry import OBJECT_REGISTRY

__all__ = [
    "DetSeg2DPacker",
]

logger = logging.getLogger(__name__)


class DetSeg2DAnnoPacker(Packer):
    def __init__(
        self,
        data_queue: list,
        output_dir: str,
        num_workers: int,
    ) -> None:
        self.data_queue = data_queue
        self.packed = 0

        super(DetSeg2DAnnoPacker, self).__init__(
            output_dir,
            len(data_queue),
            "lmdb",
            num_workers,
        )

    def pack_data(self, idx: int) -> list:
        _img_path, anno = self.data_queue[idx]
        anno_original, _anno_transformed = anno
        if "image_uuid" in anno_original:
            key = anno_original["image_uuid"]
        else:
            key = str(idx)

        if "parsing_map_urls" in _anno_transformed:
            label_map_org = cv2.imread(
                _anno_transformed["parsing_map_urls"], cv2.IMREAD_GRAYSCALE
            )
            if label_map_org is None:
                raise RuntimeError("can not read label map")

            byte_map = label_map_org.astype("float32").tostring()
            org_str = bytes(ctypes.c_uint(1))
            org_str = org_str + bytes(ctypes.c_uint(label_map_org.shape[0]))
            org_str = org_str + bytes(ctypes.c_uint(label_map_org.shape[1]))
            org_str = org_str + byte_map
            compressed_str = zlib.compress(org_str)

            _anno_transformed["seg_label_img_format_bytes"] = compressed_str

        _anno_transformed = pickle.dumps(_anno_transformed)

        return [key, _anno_transformed]

    def _write(self, idx: int, data: list) -> None:
        self.packed += 1
        self.pack_file.write(data[0], data[1])


class DetSeg2DImgPacker(Packer):
    def __init__(
        self,
        data_queue: list,
        output_dir: str,
        num_workers: int,
    ) -> None:
        self.data_queue = data_queue
        self.packed = 0
        super(DetSeg2DImgPacker, self).__init__(
            output_dir,
            len(self.data_queue),
            "lmdb",
            num_workers,
        )

    def pack_data(self, idx: int) -> list:
        img_path, anno = self.data_queue[idx]
        anno_original, _anno_transformed = anno
        if "image_uuid" in anno_original:
            key = anno_original["image_uuid"]
        else:
            key = str(idx)

        with open(
            img_path,
            "rb",
        ) as image:
            image_bytes = image.read()
        self.packed += 1
        return [key, image_bytes]

    def _write(self, idx: int, data: list) -> None:
        self.pack_file.write(data[0], data[1])


class DetSeg2DIdxPacker(Packer):
    def __init__(
        self,
        data_queue: list,
        output_dir: str,
        num_workers: int,
    ) -> None:
        self.data_queue = data_queue

        super(DetSeg2DIdxPacker, self).__init__(
            output_dir,
            len(self.data_queue),
            "lmdb",
            num_workers,
        )
        self.packed = 0

    def pack_data(self, idx: int) -> list:
        _img_path, anno = self.data_queue[idx]
        anno_original, _anno_transformed = anno
        if "image_uuid" in anno_original:
            key = anno_original["image_uuid"]
        else:
            key = str(idx)

        return [str(idx), key.encode("ascii")]

    def _write(self, idx: int, data: list) -> None:
        self.packed += 1
        self.pack_file.write(data[0], data[1])


@OBJECT_REGISTRY.register
class DetSeg2DPacker:
    def __init__(
        self,
        image_anno_pairs: List[tuple],
        output_dir: str,
        num_workers: int,
        anno_transform: Optional[Any] = None,
    ) -> None:
        self.image_anno_pairs = image_anno_pairs
        self.output_dir = output_dir
        self.num_workers = num_workers
        self.queue_maxsize = num_workers * 512
        self.data_queue: List[tuple] = []
        self.start_method = "fork"

        if anno_transform is not None and isinstance(anno_transform, list):
            self.anno_transform = Compose(anno_transform)
        else:
            self.anno_transform = anno_transform
        self.preprocess_anno_list()

    @property
    def idx_packer(self) -> DetSeg2DIdxPacker:
        if not hasattr(self, "_idx_packer"):
            self._idx_packer = DetSeg2DIdxPacker(
                self.data_queue,
                os.path.join(self.output_dir, "idx"),
                self.num_workers,
            )
        return self._idx_packer

    @property
    def img_packer(self) -> DetSeg2DImgPacker:
        if not hasattr(self, "_img_packer"):
            self._img_packer = DetSeg2DImgPacker(
                self.data_queue,
                os.path.join(self.output_dir, "img"),
                self.num_workers,
            )
        return self._img_packer

    @property
    def anno_packer(self) -> DetSeg2DAnnoPacker:
        if not hasattr(self, "_anno_packer"):
            self._anno_packer = DetSeg2DAnnoPacker(
                self.data_queue,
                os.path.join(self.output_dir, "anno"),
                self.num_workers,
            )
        return self._anno_packer

    def preprocess_anno_list(self) -> None:
        if self.anno_transform is not None:
            anno_list = self.image_anno_pairs
            if self.num_workers > 0:
                with Pool(self.num_workers) as p:
                    transformed_annos = p.map(self.transfrom_anno, anno_list)

            else:
                transformed_annos = list(map(self.transfrom_anno, anno_list))

            for img_dir_url, item in transformed_annos:
                if item[1] is not None:
                    self.data_queue.append((img_dir_url, item))
        else:
            for img, anno in self.image_anno_pairs:
                self.data_queue.append((img, (anno, anno)))

    def transfrom_anno(self, x: Any) -> tuple:
        return (x[0], (x[1], self.anno_transform((x[0], x[1]))))

    def pack_idx(self) -> None:
        idx_packer = self.idx_packer
        idx_packer()

    def pack_img(self) -> None:
        img_packer = self.img_packer
        img_packer()

    def pack_anno(self) -> None:
        anno_packer = self.anno_packer
        anno_packer()


@OBJECT_REGISTRY.register
class DetSeg2DPackerLocal(DetSeg2DPacker):
    def __init__(
        self,
        folder_anno_pairs: List[tuple],
        output_dir: str,
        num_workers: int,
        anno_transform: Optional[Any] = None,
    ) -> None:

        if anno_transform is not None:
            image_anno_pairs = self.convert_folder_pairs_to_img_pairs(
                folder_anno_pairs
            )
        else:
            image_anno_pairs = folder_anno_pairs
        super(DetSeg2DPackerLocal, self).__init__(
            image_anno_pairs, output_dir, num_workers, anno_transform
        )

    def convert_folder_pairs_to_img_pairs(self, folder_anno_pairs):
        image_anno_pairs = []
        for img_dir_url, anno_file_url in folder_anno_pairs:
            with open(anno_file_url) as f:
                for line in f.readlines():
                    anno = json.loads(line)
                    img_path = os.path.join(img_dir_url, anno["image_key"])
                    if os.path.exists(img_path):
                        image_anno_pairs.append((img_path, anno))
                    else:
                        continue
        return image_anno_pairs
