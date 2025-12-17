import logging
import multiprocessing
import os
import time
from typing import Optional

import cv2
import msgpack
import numpy as np
from torch.multiprocessing import ProcessContext

from hat.data.datasets.data_packer import Packer
from .utils import norm_bbox

logger = logging.getLogger(__name__)


__all__ = ["BasePacker"]


class BasePacker(Packer):
    """Basic class for image lmdb packer.

    Crop image based on expanded bbox without any resizing. In this way, we can
    keep the original image pixel and flip image before random crop.

    Inherit this class and rewrite `_pack_anno`, `_pack_mask` etc.

    `vis_anno` can be implemented so that we can visualize 100 images with
    annotation to check the pack results.

    TODO: crop target size but not use expand_ratio.

    Args:
        anno_path: The path annotations file with img_path bbox label.
        src_type: the source type to pack, must be image, mask or anno.
        save_dir: Save path for packed file.
        crop_img: if pack image with crop.
        expand_ratio: bbox expand ratio.
        norm_method: bbox expand method. Only `longside ratio` and
            `longsie_square` are supported.
        num_workers: Num workers for reading data using multiprocessing.
        pack_type: The file type for packing.
        num_samples: the number of samples you want to pack. You
            will pack all the samples if num_samples is None.
    """

    def __init__(
        self,
        anno_path: str,
        src_type: str,
        save_dir: str,
        crop_img: Optional[bool] = True,
        expand_ratio: Optional[float] = 2.0,
        norm_method: Optional[str] = "longside_square",
        num_workers: Optional[int] = 8,
        pack_type: Optional[str] = "lmdb",
        num_samples: Optional[int] = None,
        **kwargs,
    ):
        self._parse_anno(anno_path)
        self.expand_ratio = expand_ratio
        self.norm_method = norm_method
        self.crop_img = crop_img
        self.src_type = src_type
        self.png_enc_param = [cv2.IMWRITE_PNG_COMPRESSION, 1]
        self.save_dir = save_dir
        self.vis_dir = os.path.join(save_dir, "vis")
        os.makedirs(self.vis_dir, exist_ok=True)
        if num_samples is None:
            num_samples = len(self.annos)
        super(BasePacker, self).__init__(
            save_dir, num_samples, pack_type, num_workers, **kwargs
        )

    def _parse_anno(self, anno_path):
        self.annos = [line.strip("\n").split() for line in open(anno_path)]

    def _process_image(self, raw_img, bbox):
        img = raw_img.copy()
        rects, expand = norm_bbox(
            bbox, self.norm_method, self.expand_ratio, img.shape
        )
        # clip extreme rects
        img_crop = img[rects[1] : rects[3], rects[0] : rects[2], :]
        rects = [rects[i] + expand[i] for i in range(4)]
        raw_width = rects[2] - rects[0]
        raw_height = rects[3] - rects[1]
        if self.norm_method == "longside_ratio":
            raw_shape = (raw_height, raw_width, 3)
        elif self.norm_method == "longside_square":
            length = max(raw_width, raw_height)
            raw_shape = (length, length, 3)
        else:
            raise NotImplementedError
        img_expand = np.zeros(raw_shape)
        img_expand[
            -expand[1] : raw_height - expand[3],
            -expand[0] : raw_width - expand[2],
            :,
        ] = img_crop
        return img_expand

    def pack_data(self, idx):
        if self.src_type == "image":
            pack_data = self._pack_image(idx)
        elif self.src_type == "anno":
            pack_data = self._pack_anno(idx)
        elif self.src_type == "mask":
            pack_data = self._pack_mask(idx)
        else:
            assert "src_type must in [image, anno, mask]"
        return msgpack.packb(pack_data, use_bin_type=True)

    def _pack_mask(self, idx):
        raise NotImplementedError()

    def _pack_anno(self, idx):
        raise NotImplementedError()

    def _pack_image(self, idx):
        anno = self.annos[idx]
        assert os.path.isfile(anno[0]), anno[0]
        raw_img = cv2.imread(anno[0])
        bbox = list(map(float, anno[1:5]))
        img = self._process_image(raw_img, bbox)
        img_en = cv2.imencode(".jpg", img)[1]
        pack_data = np.asarray(img_en).astype(np.uint8).tobytes()
        if idx < 100:
            self.vis_anno(
                idx=idx, img=img.copy(), anno=anno, bbox=bbox, raw_img=raw_img
            )
        return pack_data

    def vis_anno(self, **kwargs):
        # Rewrite this func and visualize your annotaion information.
        return kwargs["img"]

    def _mp_process(self):
        ctx = multiprocessing.get_context(self.start_method)
        queue = ctx.Queue(self.queue_maxsize)

        rank = int(self.num_blocks / self.num_workers)
        if rank * self.num_workers != self.num_blocks:
            rank += 1
        data_idx = list(range(self.num_blocks))

        read_process = [
            ctx.Process(
                target=_read_worker,
                args=(self, data_idx[i * rank : (i + 1) * rank], queue),
            )
            for i in range(self.num_workers)
        ]
        error_queues = [
            multiprocessing.SimpleQueue() for i in range(self.num_workers)
        ]

        write_process = ctx.Process(target=_write_worker, args=(self, queue))

        for p in read_process:
            p.start()

        error_context = ProcessContext(read_process, error_queues)
        write_process.start()

        error_context.join()

        for p in read_process:
            p.join()

        queue.put(None)
        write_process.join()

        for p in read_process + [write_process]:
            if p.is_alive():
                p.terminate()


def _read_worker(packer, data_idx, queue):
    """
    Read data and put into queue by idx of q_in.

    Processed data by packer.pack_data.

    Args:
        packer (Packer): Instantiated packer.
        data_idx (list): Packing index.
        queue (multiprocess.Queue): Queue for packing data.

    """

    try:
        for idx in data_idx:
            block = packer.pack_data(idx)
            if block is not None:
                queue.put((idx, block))
    except Exception as e:
        queue.put(None)
        raise e


def _write_worker(packer, queue):
    """
    Write processed data into packer.dataset.

    Args:
        packer (Packer): Instantiated packer.
        queue (multiprocess.Queue): Queue for packing data.
    """

    idx = 0
    pre_time = time.time()
    while True:
        data = queue.get()
        if data is None:
            # try to write length for dataset
            packer._write_length(idx)
            break

        q_idx, q_data = data
        packer._write(q_idx, q_data)
        idx += 1
        if idx % 10000 == 0:
            cur_time = time.time()
            logger.info(f"time: {cur_time - pre_time}, count: {idx}")
            pre_time = cur_time
