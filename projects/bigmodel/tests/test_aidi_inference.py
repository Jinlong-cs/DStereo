import os

from plugins.aidi_inference.tests.base import BaseTestModel

import numpy as np
import pytest
import torch
from aidisdk import AIDIClient

client = AIDIClient(endpoint="http://aidi.hobot.cc")


# TODO(yilin.xiong, ?): to verify consistency deployed model & local model #
class TestModelMVT4D(BaseTestModel):
    def setup_class(self):
        os.environ[
            "HAT_AIDI_INFERENCE_CONFIG"
        ] = "projects/bigmodel/configs/mvt4d/mvt4d_aidi_deploy.py"
        super().setup_class(self)

    def results_compare(self, src, dst):
        num_batch = len(src)
        diff = []
        is_pass = True

        def value_diff(x, y):
            if isinstance(x, torch.Tensor):
                return float(abs(x - y))
            else:
                return [float(abs(_x - _y)) for _x, _y in zip(x, y)]

        for i in range(num_batch):
            _diff = dict(  # noqa
                x=[],
                y=[],
                z=[],
                w=[],
                h=[],
                l=[],
                yaw=[],
                score=[],
                cls=[],
            )
            consistent = None
            perception_local = src[i].get_instances(topics="instance")
            perception_aidi = dst[i].get_instances(topics="instance")
            if len(perception_local) == len(perception_aidi):
                for inst_local, inst_aidi in zip(
                    perception_local, perception_aidi
                ):
                    box3d_local = inst_local.get_bbox3ds(topics="3d_box")[0]
                    box3d_aidi = inst_aidi.get_bbox3ds(topics="3d_box")[0]
                    cls_local = inst_local.get_attributes(topics="category")[0]
                    cls_aidi = inst_aidi.get_attributes(topics="category")[0]
                    _diff["x"].append(value_diff(box3d_local.x, box3d_aidi.x))
                    _diff["y"].append(value_diff(box3d_local.y, box3d_aidi.y))
                    _diff["z"].append(value_diff(box3d_local.z, box3d_aidi.z))
                    _diff["w"].append(
                        value_diff(box3d_local.width, box3d_aidi.width)
                    )
                    _diff["h"].append(
                        value_diff(box3d_local.height, box3d_aidi.height)
                    )
                    _diff["l"].append(
                        value_diff(box3d_local.length, box3d_aidi.length)
                    )
                    _diff["yaw"].append(
                        value_diff(box3d_local.yaw, box3d_aidi.yaw)
                    )
                    _diff["score"].append(
                        value_diff(box3d_local.score, box3d_aidi.score)
                    )
                    _diff["cls"].append(
                        value_diff(cls_local.value, cls_aidi.value)
                    )
                if np.mean(np.array(list(_diff.values()))) < 1e-3:
                    consistent = True
                else:
                    consistent = False
            else:
                consistent = False
            _diff["consistent"] = consistent
            diff.append(_diff)
            if not consistent:
                is_pass = False
        return diff, is_pass


if __name__ == "__main__":
    pytest.main(["-s", f"{__file__}"])
