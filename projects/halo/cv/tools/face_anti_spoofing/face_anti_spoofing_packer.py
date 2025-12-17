import argparse
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
from hat.utils.logger import init_logger
from projects.halo.cv.tools.packer.utils import norm_bbox

logger = logging.getLogger(__name__)


class FaceAntiSpoofingPacker(Packer):
    """

    Args:
        anno_path: The path annotations file with img_path bbox label.
        src_type: the source type to pack, must be image, mask or anno.
        save_dir: Save path for packed file.
        debug: Save image for debug or not.
        crop_img: if pack image with crop.
        expand_ratio: bbox expand ratio.
        car_type: car type for each dataset. Defaults to unknown.
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
        debug: bool = False,
        crop_img: Optional[bool] = True,
        expand_ratio: Optional[float] = 2.0,
        car_type: Optional[str] = "unknown",
        target_size: Optional[int] = 256,
        norm_method: Optional[str] = "longside_square",
        num_workers: Optional[int] = 8,
        pack_type: Optional[str] = "lmdb",
        num_samples: Optional[int] = None,
        **kwargs,
    ):
        self.annos = [line.strip("\n").split() for line in open(anno_path)]
        self.expand_ratio = expand_ratio
        self.norm_method = norm_method
        self.crop_img = crop_img
        self.src_type = src_type
        self.car_type = car_type
        self.png_enc_param = [cv2.IMWRITE_PNG_COMPRESSION, 1]
        self.save_dir = save_dir
        self.debug = debug
        self.target_size = target_size

        if num_samples is None:
            num_samples = len(self.annos)
        super(FaceAntiSpoofingPacker, self).__init__(
            save_dir, num_samples, pack_type, num_workers, **kwargs
        )

    def _process_image(self, raw_img, bbox):
        img = raw_img.copy()
        rects, expand = norm_bbox(
            bbox, self.norm_method, self.expand_ratio, img.shape
        )
        # clip extreme rects
        img_h, img_w, _ = img.shape
        rects[0] = 0 if rects[0] < 0 else rects[0]
        rects[1] = 0 if rects[1] < 0 else rects[1]
        rects[2] = img_w - 1 if rects[2] >= img_w else rects[2]
        rects[3] = img_h - 1 if rects[3] >= img_h else rects[3]
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
        img = cv2.resize(img_expand, (self.target_size, self.target_size))
        return img

    def pack_data(self, idx):
        anno = self.annos[idx]
        if self.src_type == "image":
            img_path = anno[0].replace("/bucket/output", "/horizon-bucket")
            img_path = img_path.replace("/bucket/input", "/horizon-bucket")
            bbox = list(map(float, anno[1:5]))
            img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
            if self.crop_img:
                img = self._process_image(img, bbox)
            if self.debug:
                cv2.imwrite(f"{self.save_dir}/{idx:0>6d}.jpg", img)
            img_en = cv2.imencode(".jpg", img)[1]
            pack_data = np.asarray(img_en).astype(np.uint8).tobytes()
        elif self.src_type == "anno":
            label = int(anno[-1])
            pack_data = {"liveness": label, "car_type": self.car_type}
        else:
            assert "src_type must in [image, anno]"

        return msgpack.packb(pack_data, use_bin_type=True)

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


def parse_args():
    parser = argparse.ArgumentParser(
        description="Pack face anti-spoofing dataset."
    )
    parser.add_argument(
        "--anno-path",
        required=True,
        help="The path annotations file.",
    )
    parser.add_argument(
        "--src-type",
        required=True,
        help="src_type must be `image` or `anno`.",
    )
    parser.add_argument(
        "--save-dir",
        required=True,
        help="The directory for result of packer.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="save 100 image for debug",
    )
    parser.add_argument(
        "--expand-ratio",
        default=2.0,
        help="bbox expand ratio.",
    )
    parser.add_argument(
        "--target-size",
        default=256,
        help="target image size.",
    )
    parser.add_argument(
        "--norm-method",
        default="longside_square",
        help="bbox expand method, `longside ratio` or `longsie_square`",
    )
    parser.add_argument(
        "--pack-type",
        default="lmdb",
        help="The target pack type for result of packer",
    )
    parser.add_argument(
        "--car-type",
        required=True,
        help="Car type for every dataset. e.g. unknown/CD569/....",
    )
    parser.add_argument(
        "--num-workers",
        default=16,
        help="The number of workers to load image.",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    init_logger("work_dirs/hat_logss/face_antispoofing_dataset_packer")
    pack_path = os.path.join(
        args.save_dir,
        "%s_%s" % (args.src_type, args.pack_type),
    )
    packer = FaceAntiSpoofingPacker(
        anno_path=args.anno_path,
        src_type=args.src_type,
        save_dir=pack_path,
        debug=args.debug,
        target_size=args.target_size,
        crop_img=True,
        car_type=args.car_type,
        expand_ratio=args.expand_ratio,
        norm_method=args.norm_method,
        num_workers=int(args.num_workers),
        pack_type=args.pack_type,
        num_samples=100 if args.debug else None,
    )
    packer()
