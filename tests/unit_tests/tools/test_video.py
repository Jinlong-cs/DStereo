import os
import subprocess

from tools.data.video import match_order_image_to_video

from tests import HAT_BUCKET_PATH


def test_image_to_video_encoding():
    imgs_path = os.path.join(
        HAT_BUCKET_PATH, "unit_test_data/utils/test_video_encoding/data"
    )
    video_file = "test1.avi"
    match_order_image_to_video(
        imgs_path,
        video_file,
        5,
    )
    assert os.path.exists(video_file)

    video_file = "test2.avi"
    match_order_image_to_video(
        imgs_path,
        video_file,
        5,
        match_pattern="165187.*",
        read_worker=16,
    )
    assert os.path.exists(video_file)

    video_file = "test3.avi"
    match_order_image_to_video(
        imgs_path,
        video_file,
        5,
        match_pattern="165187.*",
        order_func=lambda x: sorted(x, reverse=True),
    )
    assert os.path.exists(video_file)

    cmd = f"""
    export PYTHONPATH=`pwd`:$PYTHONPATH
    python3 tools/data/video.py  --images-dir {imgs_path} \
                            --video-path test4.avi \
                            --fps 5 \
                            --order-type descending \
    """
    subprocess.check_call(cmd, shell=True)
