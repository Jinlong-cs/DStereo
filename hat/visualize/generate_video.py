# Copyright (c) Horizon Robotics. All rights reserved.

import logging
import os

import cv2
import numpy as np

from hat.core.virtual_camera.utils import transform_euler2rotMat
from hat.registry import OBJECT_REGISTRY

__all__ = ["GenerateVideo"]

logger = logging.getLogger(__name__)


def draw_ground(
    image: np.array,
    camera_calib: dict,
    line_color: tuple = (0, 0, 255),
    thickness: int = 2,
):
    """Visualize ground level.

    Args:
        image: Image with shape `H, W, 3`.
        camera_calib: Parameter dict of camera, should contain
        `focal_u, focal_v, center_u, center_v, pitch, roll, yaw,
        camera_x, camera_y, camera_z`.
        line_color: You can provide line colors like
            (255, 0, 0), (0, 255, 0)...
        thickness: The thinkness of line.
    Returns:
        The ploted image.
    """

    forward_list = [1, 5, 10, 20, 30, 45, 60]
    lateral_list = [-6, -4, -2, -1, 0, 1, 2, 4, 6]

    # set K
    K = [
        [camera_calib["focal_u"], 0, camera_calib["center_u"]],
        [0, camera_calib["focal_v"], camera_calib["center_v"]],
        [0, 0, 1],
    ]
    K = np.array(K)

    # set R
    R = transform_euler2rotMat(
        [camera_calib["roll"], camera_calib["pitch"], camera_calib["yaw"]]
    )

    # set T
    t = [
        [camera_calib["camera_x"]],
        [camera_calib["camera_y"]],
        [camera_calib["camera_z"]],
    ]

    # Hcam2local
    Hcam2local = np.vstack(
        (np.hstack((R, np.array(t))), np.array([[0, 0, 0, 1]]))
    )
    # Hcam2opencvcam
    Hcam2opencvcam = np.array(
        [[0, -1, 0, 0], [0, 0, -1, 0], [1, 0, 0, 0], [0, 0, 0, 1]]
    )
    # opencvcam2img
    opencvcam2img = K

    local2img = (
        opencvcam2img @ (Hcam2opencvcam @ np.linalg.inv(Hcam2local))[:3]
    )
    gnd2img = local2img[:, [0, 1, 3]]

    def _pt_img(mat, pt):
        pt_img = np.dot(mat, np.array(pt))
        return (pt_img[0] / pt_img[2], pt_img[1] / pt_img[2])

    horizon_lines = [
        [
            _pt_img(gnd2img, [forward, lateral_list[0], 1]),
            _pt_img(gnd2img, [forward, lateral_list[-1], 1]),
            (forward, lateral_list[0]),
            (forward, lateral_list[-1]),
        ]
        for forward in forward_list
    ]  # [point1(x,y), point2(x,y), point1_text, point2_text]
    vertical_lines = [
        [
            _pt_img(gnd2img, [forward_list[0], lateral, 1]),
            _pt_img(gnd2img, [forward_list[-1], lateral, 1]),
            (forward_list[0], lateral),
            (forward_list[-1], lateral),
        ]
        for lateral in lateral_list
    ]
    lines = horizon_lines + vertical_lines
    for line in lines:
        cv2.line(
            image,
            (int(line[0][0]), int(line[0][1])),
            (int(line[1][0]), int(line[1][1])),
            color=line_color,
            lineType=cv2.LINE_AA,
            thickness=thickness,
        )
        cv2.putText(
            image,
            "{}".format(line[2]),
            (int(line[0][0]), int(line[0][1])),
            cv2.FONT_HERSHEY_PLAIN,
            1,
            color=line_color,
        )
        cv2.putText(
            image,
            "{}".format(line[3]),
            (int(line[1][0]), int(line[1][1])),
            cv2.FONT_HERSHEY_PLAIN,
            1,
            color=line_color,
        )

    fp_x = int(gnd2img[0][0] / gnd2img[2][0])
    fp_y = int(gnd2img[1][0] / gnd2img[2][0])
    cv2.line(
        image,
        (fp_x - 10, fp_y),
        (fp_x + 10, fp_y),
        color=line_color,
        lineType=cv2.LINE_AA,
        thickness=thickness,
    )
    cv2.line(
        image,
        (fp_x, fp_y - 10),
        (fp_x, fp_y + 10),
        color=line_color,
        lineType=cv2.LINE_AA,
        thickness=thickness,
    )

    return image


@OBJECT_REGISTRY.register
class GenerateVideo(object):
    """
    GenerateVideo is a tool for visualize that generate video for saved images.

    Args:
        size (tuple): Shape of each images, (h, w).
        fps (int): The fps of the video.
        save_dir (str): Directory to save video.
    """

    def __init__(
        self,
        size: tuple = None,
        fps: int = 1,
        save_dir: str = "./",
        log_interval: int = 50,
    ):
        super().__init__()
        self.size = size
        self.fps = fps
        self.save_dir = save_dir
        self.log_interval = log_interval
        os.makedirs(self.save_dir, exist_ok=True)

    def __call__(self, video_name, image_dir):

        filelist = sorted(os.listdir(image_dir))
        assert len(filelist)
        h, w, _ = cv2.imread(os.path.join(image_dir, filelist[0])).shape
        if not self.size:
            self.size = (w, h)
        assert self.size == (w, h)

        logger.info(f"start genetrate video {video_name}")

        video = cv2.VideoWriter(
            os.path.join(self.save_dir, video_name),
            cv2.VideoWriter_fourcc("m", "p", "4", "v"),
            self.fps,
            self.size,
        )

        for idx, item in enumerate(filelist):
            if item.endswith(".png") or item.endswith(".jpg"):
                item = os.path.join(image_dir, item)
                img = cv2.imread(item).astype("uint8")
                video.write(img)
            if idx % self.log_interval == 0:
                logger.info(f"{idx} / {len(filelist)}")

        video.release()
        # no windows, don't need destroy!
        # comment to fix cicd error!
        # cv2.destroyAllWindows()
