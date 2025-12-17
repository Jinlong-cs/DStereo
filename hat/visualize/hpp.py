import os

import cv2
import numpy as np

from hat.registry import OBJECT_REGISTRY

__all__ = ["HPPViz"]


@OBJECT_REGISTRY.register
class HPPViz(object):
    def __init__(
        self,
        img_saved_path: str = None,
        gt_color: tuple = (0, 255, 0),
        pred_color: tuple = (0, 0, 255),
        radius: int = 3,
        thickness: int = 2,
    ):
        """HPP Visualization.

        Args:
            img_saved_path (str, optional): visualized image saved path.
                Defaults to None.
            gt_color (tuple, optional): gt color. Defaults to (0, 255, 0).
            pred_color (tuple, optional): pred color. Defaults to (0, 0, 255).
            radius (int, optional): circle radius. Defaults to 3.
            thickness (int, optional): circle thickness. Defaults to 2.
        """
        self.gt_color = gt_color
        self.pred_color = pred_color
        self.radius = radius
        self.thickness = thickness
        self.img_saved_path = img_saved_path

    def draw_odometry_on_picture(self, img, points, color):
        if not isinstance(img, np.ndarray):
            img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
        for x, y in points:
            if x < 0 or y < 0:
                break
            cv2.circle(
                img,
                (int(x), int(y)),
                radius=self.radius,
                thickness=self.thickness,
                color=color,
            )
        return img

    def __call__(self, img_path, img_name, gt_points, pred_points):
        img = cv2.imread(os.path.join(img_path, img_name))
        img = self.draw_odometry_on_picture(img, gt_points, self.gt_color)
        img = self.draw_odometry_on_picture(img, pred_points, self.pred_color)
        os.makedirs(self.img_saved_path, exist_ok=True)
        cv2.imwrite(os.path.join(self.img_saved_path, img_name), img)
