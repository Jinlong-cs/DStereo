# Copyright (c) Horizon Robotics. All rights reserved.

import argparse
import os
import re
from multiprocessing import Pool
from typing import Callable, Sequence

import cv2
import tqdm

from hat.utils.video import encoding_frame_to_video


def _read(frame):
    return cv2.imread(frame)


def match_order_image_to_video(
    images_path: str,
    video_output_path: str,
    video_fps: int,
    video_size: Sequence[int] = None,
    match_pattern: str = None,
    order_func: Callable = None,
    read_worker: int = 0,
):
    images_path = images_path
    video_output_path = video_output_path
    video_fps = video_fps
    video_size = video_size
    match_pattern = match_pattern
    if match_pattern is not None:
        matcher = re.compile(match_pattern)
    else:
        matcher = None
    if order_func is None:
        order_func = sorted
    order_func = order_func
    img_file_list = []
    for file in os.listdir(images_path):
        if os.path.splitext(file)[-1] in [".jpg", ".png"]:
            if matcher is None or matcher.match(file):
                img_file_list.append(os.path.join(images_path, file))
    img_file_list = order_func(img_file_list)
    if read_worker > 0:
        pool = Pool(read_worker)
        img_list = [pool.apply_async(_read, (f,)) for f in img_file_list]
        img_list = [i.get() for i in tqdm.tqdm(img_list, desc="Reading...")]
    else:
        img_list = [
            _read(i) for i in tqdm.tqdm(img_file_list, desc="Reading...")
        ]
    encoding_frame_to_video(
        img_list,
        output_path=video_output_path,
        fps=video_fps,
        size=video_size,
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--images-dir",
        type=str,
        required=True,
        help="dir of images to be encoded.",
    )
    parser.add_argument(
        "--video-path",
        type=str,
        required=True,
        help="save path of encode video.",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=15,
        help="the fps of video.",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=None,
        help="resolution of video.",
    )
    parser.add_argument(
        "--match-patter",
        type=str,
        default=None,
        help="regular matching pattern.",
    )
    parser.add_argument(
        "--num-worker",
        type=int,
        default=0,
        help="num worker of read image.",
    )
    parser.add_argument(
        "--order-type",
        type=str,
        required=False,
        default="ascending",
        help="order type of image. Support ascending or descending.",
        choices=["ascending", "descending"],
    )
    known_args, _ = parser.parse_known_args()
    return known_args


if __name__ == "__main__":
    args = parse_args()
    if args.order_type == "ascending":
        order_func = sorted
    elif args.order_type == "descending":
        order_func = lambda x: sorted(x, reverse=True)
    else:
        raise TypeError
    match_order_image_to_video(
        args.images_dir,
        args.video_path,
        args.fps,
        video_size=args.size,
        match_pattern=args.match_patter,
        order_func=order_func,
        read_worker=args.num_worker,
    )
