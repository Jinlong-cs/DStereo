import argparse
import json
import logging
import multiprocessing
import os
import time
from typing import Optional, Tuple

import cv2
import msgpack
import numpy as np
from torch.multiprocessing import ProcessContext

from hat.data.datasets.data_packer import Packer
from hat.utils.logger import init_logger

logger = logging.getLogger(__name__)

# spin joints - cockpit21 joints
# 0 - 7, 1 - 6, 2 - 5, 3 - 2, 4 - 3
# 5 - 4, 6 - 14, 7 - 13, 8 - 12, 9 - 8
# 10 - 9, 11 - 10, 19 - 20, 20 - 18
# 21 - 17, 22 - 19, 23 - 16
# fmt: off
SPIN_JOINTS_IDX = [
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
    10, 11, 19, 20, 21, 22, 23
]
COCKPIT_JOINTS_IDX = [
    7, 6, 5, 2, 3, 4, 14, 13, 12,
    8, 9, 10, 20, 18, 17, 19, 16
]
# fmt: on


class Human3dDatasetPacker(Packer):
    """
    Create human3d dataset.

    Args:
        anno_path: The path annotations json file.
        src_type: the source type to pack, must be image, mask or anno.
        save_dir: Save path for packed file.
        img_dir: Img path to load img.
        expand_ratio: bbox expand ratio. Defaults to 1.0.
        pack_shape: The final shape of pack data. Defaults to (256, 256).
        norm_method: bbox expand method. Only `longside ratio` and
            `longsie_square` are supported. Defaults to 'longside_square'.
        num_workers: Num workers for reading data using multiprocessing.
            Defaults to 8.
        pack_type: The file type for packing. Defaults to 'lmdb'.
        num_samples: the number of samples you want to pack. You
            will pack all the samples if num_samples is None. Defaults to None.
    """

    def __init__(
        self,
        anno_path: str,
        src_type: str,
        save_dir: str,
        img_dir: str,
        expand_ratio: Optional[float] = 1.0,
        pack_shape: Tuple[int, int] = (256, 256),
        norm_method: Optional[str] = "longside_square",
        num_workers: Optional[int] = 8,
        pack_type: Optional[str] = "lmdb",
        num_samples: Optional[int] = None,
        **kwargs,
    ):
        self.annos = [json.loads(line) for line in open(anno_path, "r")][0]
        self.src_type = src_type
        self.img_dir = img_dir
        self.expand_ratio = expand_ratio
        self.width, self.height = pack_shape
        self.norm_method = norm_method
        self.png_enc_param = [cv2.IMWRITE_PNG_COMPRESSION, 1]

        if num_samples is None:
            num_samples = len(self.annos)
        super(Human3dDatasetPacker, self).__init__(
            save_dir, num_samples, pack_type, num_workers, **kwargs
        )

    def _expand_rects(
        self, bbox, img_shape, expand_ratio=2.0, norm_method=None
    ):
        ori_h = bbox[3] - bbox[1]
        ori_w = bbox[2] - bbox[0]
        if norm_method == "longside_ratio":
            exp_h = ori_h * (expand_ratio - 1.0) / 2.0
            exp_w = ori_w * (expand_ratio - 1.0) / 2.0
        elif norm_method == "longside_square":
            length = max(ori_h, ori_w)
            exp_h = (expand_ratio * length - ori_h) / 2.0
            exp_w = (expand_ratio * length - ori_w) / 2.0

        new_bbox = [0, 0, 0, 0]
        expand = [0, 0, 0, 0]

        if bbox[0] - exp_w < 0:
            new_bbox[0] = 0
            expand[0] = bbox[0] - exp_w
        else:
            new_bbox[0] = bbox[0] - exp_w
        if bbox[1] - exp_h < 0:
            new_bbox[1] = 0
            expand[1] = bbox[1] - exp_h
        else:
            new_bbox[1] = bbox[1] - exp_h
        if bbox[2] + exp_w > img_shape[1]:
            new_bbox[2] = img_shape[1]
            expand[2] = bbox[2] + exp_w - img_shape[1]
        else:
            new_bbox[2] = bbox[2] + exp_w
        if bbox[3] + exp_h > img_shape[0]:
            new_bbox[3] = img_shape[0]
            expand[3] = bbox[3] + exp_h - img_shape[0]
        else:
            new_bbox[3] = bbox[3] + exp_h

        new_bbox = list(map(int, new_bbox))
        expand = list(map(int, expand))

        return new_bbox, expand

    def _get_shape_scale(self, shape, rects):
        img_h, img_w = shape
        rects_1, expand = self._expand_rects(
            rects,
            shape,
            expand_ratio=self.expand_ratio,
            norm_method=self.norm_method,
        )

        rects_1[0] = 0 if rects_1[0] < 0 else rects_1[0]
        rects_1[1] = 0 if rects_1[1] < 0 else rects_1[1]
        rects_1[2] = img_w - 1 if rects_1[2] >= img_w else rects_1[2]
        rects_1[3] = img_h - 1 if rects_1[3] >= img_h else rects_1[3]

        rects_2 = [rects_1[i] + expand[i] for i in range(len(rects_1))]
        raw_width = rects_2[2] - rects_2[0]
        raw_height = rects_2[3] - rects_2[1]
        if self.norm_method == "longside_ratio":
            raw_shape = (raw_height, raw_width, 3)
        elif self.norm_method == "longside_square":
            length = max(raw_height, raw_width)
            raw_shape = (length, length, 3)
        else:
            raise ValueError("Not supported norm method.")

        return raw_shape, rects_2

    def _process_image(self, img, rects):
        img_h, img_w, _ = img.shape
        rects_1, expand = self._expand_rects(
            rects,
            img.shape,
            expand_ratio=self.expand_ratio,
            norm_method=self.norm_method,
        )

        # 2. clip extreme rects
        rects_1[0] = 0 if rects_1[0] < 0 else rects_1[0]
        rects_1[1] = 0 if rects_1[1] < 0 else rects_1[1]
        rects_1[2] = img_w - 1 if rects_1[2] >= img_w else rects_1[2]
        rects_1[3] = img_h - 1 if rects_1[3] >= img_h else rects_1[3]

        # 3. crop image with rects and expand it
        img_crop = img[rects_1[1] : rects_1[3], rects_1[0] : rects_1[2], :]
        rects_2 = [rects_1[i] + expand[i] for i in range(len(rects_1))]
        raw_width = rects_2[2] - rects_2[0]
        raw_height = rects_2[3] - rects_2[1]
        if self.norm_method == "longside_ratio":
            raw_shape = (raw_height, raw_width, 3)
        elif self.norm_method == "longside_square":
            length = max(raw_height, raw_width)
            raw_shape = (length, length, 3)
        else:
            raise ValueError("Not supported norm method.")

        img_expand = np.zeros(raw_shape)
        img_expand[
            -expand[1] : raw_height - expand[3],
            -expand[0] : raw_width - expand[2],
            :,
        ] = img_crop  # noqa

        # 4. resize
        assert img_expand.shape[0] > 0 and img_expand.shape[1] > 0
        img_resize = cv2.resize(img_expand, (self.width, self.height))
        img = img_resize
        return img

    def _process_ldmk(self, lmks, rects_2, raw_shape):
        num_lmks = len(lmks)
        spin_joints = np.zeros((24, 3), dtype=np.float32)
        lmks = np.array(lmks)

        lmks = lmks.reshape((num_lmks, 3))
        lmks[:, 0] -= rects_2[0]
        lmks[:, 1] -= rects_2[1]
        lmks[:, 0] *= self.width / raw_shape[1]
        lmks[:, 1] *= self.height / raw_shape[0]
        spin_joints[SPIN_JOINTS_IDX] = lmks[COCKPIT_JOINTS_IDX]
        lmks_label = spin_joints.tolist()
        return lmks_label

    def _concat_ldmk(self, ldmk, ldmk_attr):
        attr_array = np.zeros((ldmk.shape[0], 1))
        # ldmk attr:
        # -1: ignore, 0: invisble, 1: occluded, 2: full visible
        for i in range(len(ldmk_attr)):
            if ldmk_attr[i]["point_label"]["ignore"] == "yes":
                attr_array[i] = -1
            elif ldmk_attr[i]["point_label"]["ignore"] == "no":
                if ldmk_attr[i]["point_label"]["occlusion"] == "invisible":
                    attr_array[i] = 0
                elif ldmk_attr[i]["point_label"]["occlusion"] == "occluded":
                    attr_array[i] = 1
                elif (
                    ldmk_attr[i]["point_label"]["occlusion"] == "full_visible"
                ):
                    attr_array[i] = 2
        ldmk = np.concatenate((ldmk, attr_array), axis=-1)
        return ldmk

    def pack_data(self, idx):
        anno = self.annos[idx]
        img_name = anno["image_name"]

        if self.src_type == "image":
            img = cv2.cvtColor(
                cv2.imread(os.path.join(self.img_dir, img_name)),
                cv2.COLOR_BGR2RGB,
            )
            img = self._process_image(img, anno["box"])
            img_en = cv2.imencode(".png", img, self.png_enc_param)[1]
            pack_data = np.asarray(img_en).astype(np.uint8).tobytes()

        elif self.src_type == "anno":
            raw_shape, rects_2 = self._get_shape_scale(
                anno["img_shape"], anno["box"]
            )
            gt_ldmk = anno.get("keypoints", None)
            if gt_ldmk is not None:
                gt_ldmk = np.array(anno["keypoints"])
                gt_ldmk_attr = anno["keypoints_attr"]
                gt_ldmk = self._concat_ldmk(gt_ldmk, gt_ldmk_attr)
                gt_ldmk = self._process_ldmk(gt_ldmk, rects_2, raw_shape)
            openpose = anno.get("openpose", None)
            if openpose is not None:
                openpose = self._process_ldmk(openpose, rects_2, raw_shape)
            pack_data = {
                "img_name": img_name,
                "gt_ldmk": gt_ldmk,
                "gt_smpl_pose": anno.get("pose", None),
                "gt_smpl_shape": anno.get("shape", None),
                "has_smpl": anno.get("has_smpl", None),
                "gt_ldmk_3d": anno.get("pose3d", None),
                "gender": anno.get("gender", None),
                "openpose": openpose,
            }
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
        if idx % 100 == 0:
            cur_time = time.time()
            logger.info(f"time: {cur_time - pre_time}, count: {idx}")
            pre_time = cur_time


def parse_args():
    parser = argparse.ArgumentParser(description="Pack face3d dataset.")
    parser.add_argument(
        "--anno-path",
        required=True,
        help="The path annotations npz file.",
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
        "--expand-ratio",
        default=2.0,
        help="bbox expand ratio.",
    )
    parser.add_argument(
        "--norm-method",
        default="longside_square",
        help="bbox expand method, `longside ratio` or `longsie_square`",
    )
    parser.add_argument(
        "--img-dir",
        required=True,
        help="The directory of img path to load.",
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
    init_logger("work_dirs/hat_logss/human3d_dataset_packer")
    pack_path = os.path.join(
        args.save_dir,
        "%s_%s" % (args.src_type, args.pack_type),
    )

    packer = Human3dDatasetPacker(
        args.anno_path,
        args.src_type,
        pack_path,
        args.img_dir,
        args.expand_ratio,
        norm_method=args.norm_method,
        num_workers=int(args.num_workers),
        pack_type=args.pack_type,
        num_samples=None,
    )
    packer()
