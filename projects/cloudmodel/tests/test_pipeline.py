import os
import subprocess

import pytest


class TestTrainPipeline:
    @pytest.mark.parametrize(
        ["task_name"],
        [
            pytest.param("detection.vehicle_detection"),
            pytest.param(
                "detection.vehicle_detection,detection.cyclist_detection"
            ),
            pytest.param("segmentation.semantic_parsing_40cls_segmentation"),
            pytest.param("classification.vehicle_attribute_classification"),
        ],
    )
    def test_training_perception_2d(self, task_name):
        dir_curr = os.path.dirname(__file__)
        prj_root = os.path.join(dir_curr, "../../..")
        config_file_path = os.path.join(
            dir_curr, "../configs/perception_2d/cloudmodel_entry_trainval.py"
        )

        cmd = f"""cd {prj_root}
CLOUDMODEL_PERCEPTION2D_TASKS={task_name} python3 tools/train.py --config {config_file_path} --stage float -ids 0  --pipeline-test
"""  # noqa
        print(
            f"\n----------------------check {os.path.basename(config_file_path)} training --------------------"  # noqa
        )
        print(cmd)
        subprocess.check_call(cmd, shell=True)
        print(
            f"\n----------------------end check {os.path.basename(config_file_path)} training --------------------"  # noqa
        )

    @pytest.mark.parametrize(
        ["task_name"],
        [
            pytest.param("detection.vehicle_detection"),
            pytest.param("segmentation.semantic_parsing_40cls_segmentation"),
        ],
    )
    def test_aidi_eval_perception_2d(self, task_name):
        dir_curr = os.path.dirname(__file__)
        prj_root = os.path.join(dir_curr, "../../..")
        config_file_path = os.path.join(
            dir_curr, "../configs/perception_2d/cloudmodel_entry_aidieval.py"
        )

        cmd = f"""cd {prj_root}
export PROJECT_ID=PD20230003
CLOUDMODEL_PERCEPTION2D_TASKS={task_name} python3 tools/predict.py --config {config_file_path} --stage float -ids 0  --pipeline-test
"""  # noqa
        print(
            f"\n----------------------check {os.path.basename(config_file_path)} aidi eval --------------------"  # noqa
        )
        print(cmd)
        subprocess.check_call(cmd, shell=True)
        print(
            f"\n----------------------end check {os.path.basename(config_file_path)} aidi eval --------------------"  # noqa
        )


if __name__ == "__main__":
    pytest.main(["-sv", __file__])
