import os

from plugins.aidi_inference.tests.base import BaseTestModel

import numpy as np
import pytest
from hatbc.message import CameraFrame, Image
from PIL import Image as pil_image

from hat.utils.config import Config


@pytest.mark.skip(reason="This is not designed for test.Just as a demo")
class TestSegmentationModel(BaseTestModel):
    def setup_class(self):
        self.config = Config.fromfile(
            os.path.join(
                os.path.dirname(__file__),
                "../aidi_configs/segmentation/aidi_config_singletask_parsing40cls.py",
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
        for s, d in zip(src, dst):
            seg_s = s.data.data
            seg_d = d.data.data
            diff = seg_s != seg_d
            is_pass = diff.sum() / (seg_s == seg_d).sum() < 1e-3
            if not is_pass:
                return diff, False
        return diff, True

    @staticmethod
    def make_test_images():
        img = pil_image.open(
            "/horizon-bucket/adas/big_model/tmp/demo/ADAS_20210304-165302_633_0__93449_1614848188663_0.jpg"  # noqa
        ).convert("RGB")

        img = np.array(img.resize((2048, 1280)), dtype=np.uint8)

        frame = CameraFrame(
            image=Image(data=img, layout="hwc", color_space="rgb")
        ).to_proto()

        example_input = [frame]
        num_batch = len(example_input)
        return example_input, num_batch


if __name__ == "__main__":
    pytest.main(["-s", "-x", f"{__file__}"])
