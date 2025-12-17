import os

import cv2

from hat.utils.video import encoding_frame_to_video
from tests import HAT_BUCKET_PATH


def test_encoding_frame_to_video():
    test_img_dir = os.path.join(
        HAT_BUCKET_PATH, "unit_test_data/utils/test_video_encoding/data"
    )
    video_file = "test.avi"
    img_list = []
    for root, _, files in os.walk(test_img_dir):
        for file in files:
            if file.endswith(".jpg"):
                img = cv2.imread(os.path.join(root, file))
                img_list.append(img[:, :, ::-1])
    encoding_frame_to_video(img_list, video_file, 5)
    assert os.path.exists(video_file)
