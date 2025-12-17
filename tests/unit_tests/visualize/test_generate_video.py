import os
import shutil

import cv2
import numpy as np

from hat.visualize.generate_video import GenerateVideo, draw_ground


def test_draw_ground():

    image = np.random.randint(0, 255, (2160, 3840, 3), dtype=np.uint8)
    camera_calib = {
        "camera_x": 0,
        "camera_y": 0,
        "camera_z": 1.538,
        "center_u": 1919.9285888671875,
        "center_v": 1085.44140625,
        "distort": [0 for _ in range(8)],
        "focal_u": 2418.281494140625,
        "focal_v": 2418.281494140625,
        "pitch": 0.0009087863419741011,
        "roll": -0.008198394482362016,
        "yaw": 0.005041685210280009,
        "vcs": {
            "rotation": [0 for _ in range(3)],
            "translation": [0 for _ in range(3)],
        },
    }
    ret_img = draw_ground(image=image, camera_calib=camera_calib)
    assert ret_img.shape == (2160, 3840, 3)


def test_generate_video(tmpdir):

    save_dir = os.path.join(tmpdir, "example-imgs")
    os.makedirs(save_dir, exist_ok=True)

    for i in range(10):
        img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        cv2.imwrite(os.path.join(save_dir, f"test_{i}.png"), img)

    video = GenerateVideo(save_dir=save_dir)
    video(video_name="test.mp4", image_dir=save_dir)

    assert len(os.listdir(save_dir)) == 11
    assert "test.mp4" in os.listdir(save_dir)

    shutil.rmtree(save_dir)
