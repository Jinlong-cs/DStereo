import argparse
import os

import numpy as np
import tqdm

from hat.registry import build_from_registry
from hat.utils.config import Config


def save_raw(imgs, saved_path, index, interval, sample_number, current_number):
    stop = False
    if not os.path.exists(saved_path):
        os.mkdir(saved_path)

    for img in imgs:
        if index % interval == 0:
            img = (
                img.unsqueeze(0).permute(0, 2, 3, 1).numpy().astype(np.float32)
            )  # noqa
            raw_path = os.path.join(saved_path, str(index) + ".raw")
            img.tofile(raw_path)
            current_number += 1
        index += 1
        if current_number >= sample_number:
            stop = True
            break

    return index, stop, current_number


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Config of task.",
    )
    parser.add_argument(
        "--saved-path",
        default="./saved_img",
        type=str,
        help="Directory path of image to save.",
    )
    parser.add_argument(
        "--interval",
        default=1,
        type=int,
        help="Sample interval of image in dataset.",
    )
    parser.add_argument(
        "--sample-number",
        default=100,
        type=int,
        help="Total sample number of image.",
    )
    args = parser.parse_args()

    cfg = Config.fromfile(args.config)
    data_loader = build_from_registry(cfg["data_loader"])
    # print(len(data_loader))

    index = 0
    current_number = 0
    for data in tqdm.tqdm(data_loader):
        index, stop, current_number = save_raw(
            data["img"],
            args.saved_path,
            index,
            args.interval,
            args.sample_number,
            current_number,
        )
        if stop:
            break
