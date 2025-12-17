from typing import List

from .base import ReleasePath

__all__ = ["human3d"]


class Human3D(ReleasePath):
    @property
    def models(self) -> List:
        return [
            "hat/models/losses/human3d/*",
            "hat/models/structures/human3d/*",
            "hat/models/task_modules/human3d/*",
        ]

    @property
    def transforms(self) -> List:
        return [
            "hat/data/transforms/detection.py",
            "hat/data/transforms/landmark.py",
        ]

    @property
    def datasets(self) -> List:
        return [
            "hat/data/datasets/human3d_dataset.py",
        ]

    @property
    def metrics(self) -> List:
        return [
            "hat/metrics/human3d_metric.py",
        ]

    @property
    def samplers(self) -> List:
        return []

    @property
    def tests(self) -> List:
        return [
            "tests/unit_tests/data/datasets/test_human3d_dataset.py",
            "tests/unit_tests/data/datasets/test_human3d_packer.py",
            "tests/unit_tests/metrics/test_hand3d_metric.py",
            "tests/unit_tests/models/losses/test_human3d_loss.py",
            "tests/unit_tests/models/structures/human3d/*",
            "tests/unit_tests/models/task_modules/human3d/*",
        ]


human3d = Human3D().get_file_list()
