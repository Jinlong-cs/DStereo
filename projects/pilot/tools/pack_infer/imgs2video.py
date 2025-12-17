import argparse
import imghdr
import logging
import os
import os.path as osp
from multiprocessing import Pool
from tempfile import NamedTemporaryFile
from typing import Callable, List, Optional

import cv2
from moviepy.editor import VideoFileClip
from tqdm import tqdm

logger = logging.getLogger(__file__)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--imgs-root", type=str, required=True)
    parser.add_argument("--out-root", type=str, default=None)
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--multi-process", action="store_true", default=False)
    parser.add_argument("--del-imgs", action="store_true", default=False)

    return parser.parse_args()


def imgs2video(
    images: List[str],
    video_file: Optional[str] = "video.mp4",
    sort_key: Optional[Callable] = None,
    fps: Optional[int] = 15,
    del_imgs: Optional[bool] = False,
) -> str:
    images = sorted(
        filter(lambda img: imghdr.what(img) is not None, images),
        key=sort_key,
    )

    # 读取第一张图片以获取视频的尺寸
    frame = cv2.imread(images[0])
    height, width, layers = frame.shape

    with NamedTemporaryFile(suffix=osp.basename(video_file)) as tmpfile:
        # 创建视频写入对象
        video = cv2.VideoWriter(
            tmpfile.name, cv2.VideoWriter_fourcc(*"DIVX"), fps, (width, height)
        )

        # 遍历图片列表，将每张图片读取并写入视频
        for image in tqdm(images, total=len(images), desc=video_file):
            video.write(cv2.imread(image))

        # 释放视频写入对象
        video.release()

        # 视频压缩
        clip = VideoFileClip(tmpfile.name)
        # clip = clip.resize(height=360)
        # 写入压缩后的视频
        clip.write_videofile(video_file)
        clip.close()

    if del_imgs:
        [os.remove(img) for img in images]

    return video_file


def packinfer2video(
    imgs_root: str,
    out_root: Optional[str] = None,
    fps: Optional[int] = 15,
    multi_process: Optional[bool] = False,
    del_imgs: Optional[bool] = False,
):
    if out_root is None:
        out_root = imgs_root
    os.makedirs(out_root, exist_ok=True)

    pool = None
    if multi_process:
        pool = Pool(processes=os.cpu_count())

    results = []
    for cur_root, _, cur_files in os.walk(imgs_root):
        cur_images = [osp.join(cur_root, f) for f in cur_files]
        if not cur_images:
            continue

        cur_out_root = osp.join(out_root, osp.relpath(cur_root, imgs_root))
        os.makedirs(cur_out_root, exist_ok=True)
        kwds = dict(  # noqa
            images=cur_images,
            video_file=osp.join(cur_out_root, "video.mp4"),
            fps=fps,
            del_imgs=del_imgs,
        )
        if pool is not None:
            pool.apply_async(
                func=imgs2video,
                kwds=kwds,
                callback=lambda res: results.append(res),
            )
        else:
            video_file = imgs2video(**kwds)
            results.append(video_file)

    if pool is not None:
        pool.close()
        pool.join()

    logger.info(results)


if __name__ == "__main__":
    args = parse_args()

    packinfer2video(**vars(args))
