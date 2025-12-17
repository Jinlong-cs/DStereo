from typing import List

from .base import ReleasePath

__all__ = ["landmark"]


class Landmark(ReleasePath):
    @property
    def models(self) -> List:
        return [
            "hat/models/structures/landmark/*",
            "hat/models/backbones/mixvargenet.py",
            "hat/models/backbones/vargnetv2.py",
            "hat/models/backbones/hrnet.py",
            "hat/models/backbones/mobilenetv2.py",
            "hat/models/backbones/snapdragon/*",
            "hat/models/base_modules/basic_mixvargenet_module.py",
            "hat/models/base_modules/basic_vargnet_module.py",
            "hat/models/losses/landmark/*",
            "hat/models/necks/group_fpn.py",
            "hat/models/task_modules/landmark/*",
        ]

    @property
    def transforms(self) -> List:
        return [
            "hat/data/transforms/landmark.py",
        ]

    @property
    def datasets(self) -> List:
        return [
            "hat/data/datasets/landmark_dataset.py",
        ]

    @property
    def metrics(self) -> List:
        return [
            "hat/metrics/landmark.py",
        ]

    @property
    def tests(self) -> List:
        return [
            "tests/unit_tests/data/transforms/test_landmark.py",
            "tests/unit_tests/metrics/test_coco_ldmk.py",
            "tests/unit_tests/metrics/test_landmark_metric.py",
            "tests/unit_tests/models/losses/test_landmark_loss.py",
            "tests/unit_tests/models/necks/test_group_fpn.py",
            "tests/unit_tests/models/structures/landmark/*",
            "tests/unit_tests/models/task_modules/landmark/*",
            "tests/unit_tests/models/backbones/snapdragon/*",
            "tests/unit_tests/data/transforms/test_human_landmark.py",
            "tests/unit_tests/data/datasets/test_ldmk_dataset.py",
        ]


landmark = Landmark().get_file_list()
