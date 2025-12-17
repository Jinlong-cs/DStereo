import os

from plugins.aidi_inference.tests.base import BaseTestModel

import pytest
from hatbc.message import CameraFrame, Image

from hat.utils.config import Config


@pytest.mark.skip(reason="This is not designed for test.Just as a demo")
class TestWorkConditionClassificationModel(BaseTestModel):
    def setup_class(self):
        self.config = Config.fromfile(
            os.path.join(
                os.path.dirname(__file__),
                "../aidi_configs/classification/aidi_config_multitask_work_condition_classification.py",
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
        img_url_1 = "/horizon-bucket/adas/lina.shen/hdflow_pack_image/7864/ADAS_20210628-205629_679_0_3071/ADAS_20210628-205629_679_0_3071__58796_1624885158637_0.jpg"
        img_url_2 = "/horizon-bucket/auto_jenkins_test/hdflow_workspace/test_modelzoo/ADAS_20220315-142130_023_4__29219_1647325476006_0.jpg"
        img_url_3 = "/horizon-bucket/adas/big_model/tmp/demo/ADAS_20210304-165302_633_0__93449_1614848188663_0.jpg"

        with open(img_url_1, "rb") as rf:
            img_buf_1 = rf.read()
        with open(img_url_2, "rb") as rf:
            img_buf_2 = rf.read()
        with open(img_url_3, "rb") as rf:
            img_buf_3 = rf.read()

        frame1 = CameraFrame(image=Image(data=img_buf_1))
        frame2 = CameraFrame(image=Image(data=img_buf_2))
        frame3 = CameraFrame(image=Image(data=img_buf_3))
        example_input = [frame1, frame2, frame3]
        num_batch = len(example_input)
        return example_input, num_batch


if __name__ == "__main__":
    pytest.main(["-s", "-x", f"{__file__}"])
