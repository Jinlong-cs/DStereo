"""Pack emotion."""

import argparse
import codecs
import logging
from typing import Optional

import cv2
import msgpack
import numpy as np

from hat.data.datasets.data_packer import Packer
from hat.utils.logger import init_logger

logger = logging.getLogger(__name__)


class EmotionPacker(Packer):
    """
    EmotionPacker is used for packing emotion dataset to target format.

    Args:
        label_path: The label path of emotion data.
        save_dir: The saving directory of the pack data.
        num_workers: The num workers for reading data
            using multiprocessing.
        pack_type: The file type for packing.
        num_samples: the number of samples you want to pack. You
            will pack all the samples if num_samples is None.
    """

    def __init__(
        self,
        label_path: str,
        save_dir: str,
        num_workers: Optional[int] = 8,
        pack_type: Optional[str] = "lmdb",
        num_samples: Optional[int] = None,
        **kwargs
    ):
        self.label_info = self.load_label(label_path)

        if num_samples is None:
            num_samples = len(self.label_info)

        super(EmotionPacker, self).__init__(
            save_dir, num_samples, pack_type, num_workers, **kwargs
        )

    def load_label(self, path):
        label_info = []
        for line in codecs.open(path, "r", "utf-8"):
            line = line.strip()
            lsp = line.split()
            img_id = lsp[0]
            img_path = lsp[1]
            emotion_label_id = int(lsp[2])
            obj = {
                "img_id": img_id,
                "img_path": img_path,
                "emotion_label_id": emotion_label_id,
            }
            label_info.append(obj)
        return label_info

    def pack_data(self, idx):

        cur_label_info = self.label_info[idx]
        img_path = cur_label_info["img_path"]
        emotion_label_id = cur_label_info["emotion_label_id"]
        img_id = np.asarray(idx, dtype=np.int64).tobytes()
        image = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        image = cv2.imencode(".jpg", image)[1].tobytes()
        label = np.asarray(emotion_label_id, dtype=np.float32).tobytes()
        return msgpack.packb(img_id + label + image)


def parse_args():
    parser = argparse.ArgumentParser(description="Pack emotion dataset.")
    parser.add_argument(
        "--label_path",
        required=True,
        help="label_path.",
    )
    parser.add_argument(
        "--save_dir",
        required=True,
        help="save_dir.",
    )
    parser.add_argument(
        "--pack-type",
        required=True,
        help="The pack data type for result of packer",
    )
    parser.add_argument(
        "--num-workers",
        default=20,
        help="The number of workers to load image.",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()

    init_logger("work_dirs/hat_logss/emotion_packer")

    print("Loading dataset from {}".format(args.label_path))

    packer = EmotionPacker(
        args.label_path,
        args.save_dir,
        int(args.num_workers),
        args.pack_type,
    )
    packer()
