import os.path

from plugins.aidi_inference.tests.base import BaseTestModel

import numpy as np
import pytest
from hatbc.message import CameraFrame, Image
from PIL import Image as pil_image

from hat.utils.config import Config


@pytest.mark.skip(reason="This is not designed for test.Just as a demo")
class TestDetectionModel(BaseTestModel):
    def setup_class(self):
        self.config = Config.fromfile(
            os.path.join(
                os.path.dirname(__file__),
                "../aidi_configs/detection/aidi_config_multitask_detection.py",
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
            assert len(perceptions_s) == len(perceptions_d)
            for ins_s, ins_d in zip(perceptions_s, perceptions_d):
                bbox_s = ins_s.bbox2ds[0]
                bbox_d = ins_d.bbox2ds[0]
                assert ins_s.topic == ins_d.topic
                assert (bbox_s.score - bbox_d.score).abs() < 0.1
                diff = bbox_s.data.data - bbox_d.data.data
                is_pass = diff.abs().mean() < 1
                if not is_pass:
                    return diff, False
        return diff, True

    @staticmethod
    def make_test_images():
        img = pil_image.open(
            "/horizon-bucket/adas/big_model/tmp/demo/ADAS_20230519-231507_102_5__1118816_1684509483033_0.jpg"  # noqa
        ).convert("RGB")

        img1 = np.array(img.resize((3840, 2176)), dtype=np.uint8)
        img2 = np.array(img, dtype=np.uint8)

        frame1 = CameraFrame(
            image=Image(data=img1, layout="hwc", color_space="rgb")
        )
        frame2 = CameraFrame(
            image=Image(data=img2, layout="hwc", color_space="rgb")
        )

        example_input = [frame1, frame2]
        num_batch = len(example_input)
        return example_input, num_batch


if __name__ == "__main__":
    pytest.main(["-s", "-x", f"{__file__}"])
