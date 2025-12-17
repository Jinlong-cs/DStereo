import argparse
import json
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

logger = logging.getLogger(__name__)


class Face3dDatasetPacker(Packer):
    """
    Create face3d dataset.

    Args:
        anno_path (str): The path annotations json file.
        src_type (str): the source type to pack, must be image, mask or anno.
        save_dir (str): Save path for packed file.
        crop_img (bool): if pack image with crop.
        expand_ratio (float): bbox expand ratio.
        norm_method (str): bbox expand method. Only `longside ratio` and
            `longsie_square` are supported.
        num_workers (int): Num workers for reading data using multiprocessing.
        pack_type (str): The file type for packing.
        num_samples (int): the number of samples you want to pack. You
            will pack all the samples if num_samples is None.
    """

    def __init__(
        self,
        anno_path: str,
        src_type: str,
        save_dir: str,
        crop_img: Optional[bool] = True,
        expand_ratio: Optional[float] = 1.0,
        expand_size_wh: Optional[str] = "",
        norm_method: Optional[str] = "longside_square",
        num_workers: Optional[int] = 8,
        pack_type: Optional[str] = "lmdb",
        num_samples: Optional[int] = None,
        **kwargs,
    ):
        self.annos = [json.loads(line.strip("\n")) for line in open(anno_path)]
        self.expand_ratio = expand_ratio
        self.norm_method = norm_method
        self.crop_img = crop_img
        self.src_type = src_type
        self.png_enc_param = [cv2.IMWRITE_PNG_COMPRESSION, 1]
        size_list = expand_size_wh.split("x")
        if len(size_list) == 1:
            if expand_size_wh.isnumeric():
                self.expand_size = [int(expand_size_wh), int(expand_size_wh)]
            else:
                self.expand_size = None
        elif len(size_list) == 2:
            self.expand_size = list(map(int, size_list))
        else:
            raise ValueError(
                f"Not supported expand_size_wh: {expand_size_wh}. "
            )

        if num_samples is None:
            num_samples = len(self.annos)
        super(Face3dDatasetPacker, self).__init__(
            save_dir, num_samples, pack_type, num_workers, **kwargs
        )

    def _expand_bbox(self, bbox, shape):
        if self.expand_size is None:
            expand_type = "ratio"
        else:
            expand_type = "size"
        img_h, img_w = shape[0], shape[1]
        ori_w = bbox[2] - bbox[0]
        ori_h = bbox[3] - bbox[1]
        if expand_type == "ratio":
            if self.norm_method == "longside_ratio":
                exp_w = ori_w * (self.expand_ratio - 1.0) / 2
                exp_h = ori_h * (self.expand_ratio - 1.0) / 2
            elif self.norm_method == "longside_square":
                length = max(ori_h, ori_w)
                exp_w = (self.expand_ratio * length - ori_w) / 2.0
                exp_h = (self.expand_ratio * length - ori_h) / 2.0
            else:
                raise ValueError(
                    f"Not supported norm_method: {self.norm_method}."
                )
        else:
            exp_w = (self.expand_size[0] - ori_w) / 2.0
            exp_h = (self.expand_size[1] - ori_h) / 2.0

        x1 = max(bbox[0] - exp_w, 0)
        x2 = min(bbox[2] + exp_w, img_w)
        y1 = max(bbox[1] - exp_h, 0)
        y2 = min(bbox[3] + exp_h, img_h)
        new_bbox = list(map(int, [x1, y1, x2, y2]))

        return new_bbox

    def _process_image(self, raw_img, bbox):
        img = raw_img.copy()
        new_bbox = self._expand_bbox(bbox, img.shape)
        if self.expand_size is None:
            img = raw_img[new_bbox[1] : new_bbox[3], new_bbox[0] : new_bbox[2]]
        else:
            img = raw_img[
                new_bbox[1] : new_bbox[1] + self.expand_size[1],
                new_bbox[0] : new_bbox[0] + self.expand_size[0],
            ]
            pad_size = np.zeros((len(img.shape), 2), dtype=np.int64)
            pad_size[0, 1] = self.expand_size[1] - img.shape[0]
            pad_size[1, 1] = self.expand_size[0] - img.shape[1]
            img = np.pad(img, pad_size, "constant")
            assert list(img.shape[:2]) == self.expand_size[::-1]

        return img

    def _process_ldmk(self, ldmk, bbox, shape):
        new_bbox = self._expand_bbox(bbox, shape)
        x1, y1, _, _ = new_bbox
        bbox = [bbox[0] - x1, bbox[1] - y1, bbox[2] - x1, bbox[3] - y1]
        if ldmk is not None:
            ldmk = np.array(ldmk, dtype=np.float32).reshape(-1, 3)
            ldmk[:, 0] = ldmk[:, 0] - x1
            ldmk[:, 1] = ldmk[:, 1] - y1
        return ldmk, bbox, new_bbox

    def pack_data(self, idx):
        anno = self.annos[idx]
        if self.src_type == "image":
            img = cv2.cvtColor(cv2.imread(anno["img_path"]), cv2.COLOR_BGR2RGB)
            if self.crop_img:
                img = self._process_image(img, anno["bbox"][:4])
            img_en = cv2.imencode(".jpg", img)[1]
            pack_data = np.asarray(img_en).astype(np.uint8).tobytes()

        elif self.src_type == "mask":
            gt_mask = cv2.imread(anno["mask_path"], 0)
            if self.crop_img:
                gt_mask = self._process_image(gt_mask, anno["bbox"][:4])
            mask_en = cv2.imencode(".png", gt_mask, self.png_enc_param)[1]
            pack_data = np.asarray(mask_en).astype(np.uint8).tobytes()

        elif self.src_type == "anno":
            bbox = anno["bbox"][:4]
            if "ldmk_path" in anno:
                gt_ldmk = np.load(anno["ldmk_path"]).astype(np.float32)
            else:
                gt_ldmk = None
            if anno.get("fitting_ldmk3d") is None:
                gt_ldmk3d = None
            else:
                gt_ldmk3d = (
                    np.load(anno["fitting_ldmk3d"]).astype(np.float32).tolist()
                )
            if self.crop_img:
                img = cv2.imread(anno["img_path"])
                raw_img_shape = [img.shape[0], img.shape[1], 3]
                gt_ldmk, bbox, crop_bbox = self._process_ldmk(
                    gt_ldmk, bbox, img.shape
                )
            pack_data = {
                "img_path": anno["img_path"],
                "gt_bboxes": bbox,
                "gt_ldmk": None if gt_ldmk is None else gt_ldmk.tolist(),
                "gt_ldmk3d": gt_ldmk3d,
                "crop_bbox": crop_bbox,
                "save_crop": self.crop_img,
                "raw_img_shape": raw_img_shape,
                "gt_pose": anno.get("gt_pose", None),
                "intrinsic": anno.get("intrinsic", None),
                "distortion": anno.get("distortion", None),
                "global_pose": anno.get("global_pose", None),
                "shape_param": anno.get("shape_param", None),
                "transl": anno.get("transl", None),
                "jaw_pose": anno.get("jaw_pose", None),
                "exp_param": anno.get("exp_param", None),
                "eye3d_left": anno.get("eye3d_left", None),
                "eye3d_right": anno.get("eye3d_right", None),
            }
        else:
            assert "src_type must in [image, mask, anno]"

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
        if idx % 100 == 0:
            cur_time = time.time()
            logger.info(f"time: {cur_time - pre_time}, count: {idx}")
            pre_time = cur_time


def parse_args():
    parser = argparse.ArgumentParser(description="Pack face3d dataset.")
    parser.add_argument(
        "--anno-path",
        required=True,
        help="The path annotations json file.",
    )
    parser.add_argument(
        "--src-type",
        required=True,
        help="src_type must be `image`, `mask` or `anno`.",
    )
    parser.add_argument(
        "--save-dir",
        required=True,
        help="The directory for result of packer.",
    )
    parser.add_argument(
        "--crop-img",
        action="store_true",
        help="pack image with crop",
    )
    parser.add_argument(
        "--expand-ratio",
        default=2.0,
        help="bbox expand ratio.",
    )
    parser.add_argument(
        "--expand-size-wh",
        default="",
        help="bbox expand size.",
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
        "--num-workers",
        default=16,
        help="The number of workers to load image.",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    init_logger("work_dirs/hat_logss/face3d_dataset_packer")
    pack_path = os.path.join(
        args.save_dir,
        "%s_%s" % (args.src_type, args.pack_type),
    )

    packer = Face3dDatasetPacker(
        args.anno_path,
        args.src_type,
        pack_path,
        args.crop_img,
        args.expand_ratio,
        args.expand_size_wh,
        args.norm_method,
        int(args.num_workers),
        args.pack_type,
        None,
    )
    packer()
