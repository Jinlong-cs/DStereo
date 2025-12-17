from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn

from hat.core.data_struct.base_struct import Points2D
from hat.registry import OBJECT_REGISTRY

__all__ = ["HPPDecoder"]


@OBJECT_REGISTRY.register
class HPPDecoder(nn.Module):
    def __init__(
        self,
        feat_stride: int = 8,
        img_shape: tuple = (512, 256),
        point_thresh: float = 0.65,
        to_hatbc_msg: bool = False,
        task_name: str = None,
    ):
        """Convert the output of model, dense map, to 2d points on the original image.  # noqa.

        Args:
            feat_stride (int, optional): feature stride. Defaults to 8.
            img_shape (tuple, optional): image shape. Defaults to (512, 256).
            point_thresh (float, optional): confidence that higher point
                threshold selected as foreground. Defaults to 0.65.
        """
        super(HPPDecoder, self).__init__()
        self.feat_stride = feat_stride
        self.point_thresh = point_thresh
        self.img_width = img_shape[0]
        self.img_height = img_shape[1]
        self.grid_location = self.get_grid_location()
        self.to_hatbc_msg = to_hatbc_msg
        self.task_name = task_name

    def get_grid_location(self):
        grid_w = self.img_width // self.feat_stride
        grid_h = self.img_height // self.feat_stride
        grid_location = np.zeros((grid_h, grid_w, 2))
        x = np.arange(grid_w)
        y = np.arange(grid_h)
        xx, yy = np.meshgrid(x, y, indexing="xy")
        grid_location[:, :, 0] = xx
        grid_location[:, :, 1] = yy
        return grid_location

    def map2points(self, confidence, offset):
        confidence = confidence[0]
        offset = offset.transpose(1, 2, 0)
        mask = confidence > self.point_thresh
        # mask = mask.squeeze()
        grid = self.grid_location[mask]
        filtered_offset = offset[mask]
        filtered_conf = confidence[mask]
        absoluate_x = []
        absoluate_y = []
        conf = []
        for i in range(len(grid)):
            # convert offset to absolute coordinates
            point_x = int(
                (filtered_offset[i][0] + grid[i][0]) * self.feat_stride
            )
            point_y = int(
                (filtered_offset[i][1] + grid[i][1]) * self.feat_stride
            )
            if (
                point_x > self.img_width
                or point_x < 0
                or point_y > self.img_height
                or point_y < 0
            ):
                continue
            absoluate_x.append(point_x)
            absoluate_y.append(point_y)
            conf.append(filtered_conf[i])

        if len(absoluate_x) == 0 or len(absoluate_y) == 0:
            absoluate_x = np.array([0])
            absoluate_y = np.array([0])
        else:
            absoluate_x = np.array(absoluate_x) * 1.0
            absoluate_y = np.array(absoluate_y) * 1.0

        coordinates = np.concatenate(
            [absoluate_x[:, None], absoluate_y[:, None]], axis=1
        )
        conf = np.array(conf)
        out = {"coordinate": coordinates, "conf": conf}
        return out

    def dict_to_points2d(self, model_outs):
        if not isinstance(model_outs, dict):
            raise TypeError
        assert "coordinate" in model_outs
        assert "conf" in model_outs
        coordinates = model_outs["coordinate"]
        conf = model_outs["conf"]
        num_samples = len(conf)
        ret = []
        for i in range(num_samples):
            ret.append(
                Points2D(
                    points=torch.from_numpy(coordinates[i]),
                    scores=torch.from_numpy(conf[i])
                    if len(coordinates[i]) > 1
                    else torch.tensor([0], dtype=torch.float64),
                    cls_idxs=torch.zeros(conf[i].shape)
                    if len(coordinates[i]) > 1
                    else torch.zeros(1),
                )
            )
        return ret

    def forward(self, inputs, targets):
        res = {"img_name": targets["img_name"], "coordinate": [], "conf": []}
        batch_size = len(res["img_name"])
        confidences = inputs["confidences"][-1]
        offsets = inputs["offsets"][-1]
        confidences = confidences.cpu().numpy()
        offsets = offsets.cpu().numpy()
        for idx in range(batch_size):
            conf = confidences[idx]
            offset = offsets[idx]
            outs = self.map2points(conf, offset)
            res["coordinate"].append(outs["coordinate"])
            res["conf"].append(outs["conf"])

        if self.to_hatbc_msg:
            hatbc_ret = OrderedDict()
            hatbc_ret[self.task_name] = self.dict_to_points2d(res)
            return hatbc_ret
        return res
