import os

from plugins.aidi_inference.tests.base import BaseTestModel

import numpy as np
import pytest
from hatbc.message import BBox2D, CameraFrame, Image, Instance
from PIL import Image as pil_image

from hat.utils.config import Config


@pytest.mark.skip(reason="This is not designed for test.Just as a demo")
class TestVehicleAttributeClassificationModel(BaseTestModel):
    def setup_class(self):
        self.config = Config.fromfile(
            os.path.join(
                os.path.dirname(__file__),
                "../aidi_configs/classification/aidi_config_singletask_vehicle_attribute_classification.py",
            )
        )
        self.record = {}
        example_input, num_batch = self.make_test_images()
        self.record["num_batch"] = num_batch
        self.example_input = example_input

    def deserialize_single(self, result: CameraFrame):
        return CameraFrame

    def results_compare(self, src, dst):
        assert len(src) == len(dst)
        for perceptions_s, perceptions_d in zip(src, dst):
            assert len(perceptions_s.attributes) == len(
                perceptions_d.attributes
            )
            assert perceptions_s.topic == perceptions_d.topic
            for ins_s, ins_d in zip(
                perceptions_s.attributes, perceptions_d.attributes
            ):
                assert ins_s.value == ins_d.value
                diff = ins_s.score - ins_d.score
                is_pass = diff < 1e-3
                if not is_pass:
                    return diff, False
        return diff, True

    @staticmethod
    def make_test_images():
        img = pil_image.open(
            "/horizon-bucket/auto_jenkins_test/hdflow_workspace/test_modelzoo/ADAS_20220315-142130_023_4__29219_1647325476006_0.jpg"
            # noqa
        ).convert("RGB")

        img = np.array(img.resize((960, 640)), dtype=np.uint8)

        bbox = BBox2D(
            data=[599.5196, 287.7713, 699.1089, 350.1020],
            topic="vehicle_detection",
            score=0.9,
        )
        vehicle1 = Instance(
            topic="vehicle_detection",
            bbox2ds=[bbox],
        )
        vehicle2 = Instance(
            topic="vehicle_detection",
            bbox2ds=[bbox],
        )
        vehicle3 = Instance(
            topic="vehicle_detection",
            bbox2ds=[bbox],
        )
        frame1 = CameraFrame(
            image=Image(data=img, layout="hwc", color_space="rgb"),
            perceptions=[vehicle1, vehicle2],
        )
        frame2 = CameraFrame(
            image=Image(data=img, layout="hwc", color_space="rgb"),
            perceptions=[vehicle3],
        )

        example_input = [frame1, frame2]
        num_batch = len(frame1.perceptions) + len(frame2.perceptions)
        return example_input, num_batch


if __name__ == "__main__":
    pytest.main(["-s", "-x", f"{__file__}"])
