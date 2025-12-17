from typing import List

from .base import ReleasePath

__all__ = ["gaze"]


class Gaze(ReleasePath):
    @property
    def models(self):
        return [
            "hat/models/losses/gaze/*",
            "hat/models/structures/gaze/*",
            "hat/models/structures/gaze/gaze_model.py",
            "hat/models/task_modules/gaze/*",
            "hat/models/task_modules/gaze/gaze_head.py",
            "hat/models/task_modules/PCCR_gaze/*",
        ]

    @property
    def transforms(self) -> List:
        return [
            "hat/data/transforms/gaze/*",
        ]

    @property
    def datasets(self) -> List:
        return [
            "hat/data/datasets/gaze/*",
        ]

    @property
    def metrics(self) -> List:
        return [
            "hat/metrics/gaze_metric.py",
        ]

    @property
    def tests(self) -> List:
        return [
            "tests/unit_tests/data/datasets/gaze/*",
            "tests/unit_tests/data/transforms/test_gaze.py",
            "tests/unit_tests/metrics/test_gaze.py",
            "tests/unit_tests/models/structures/gaze/*",
        ]


gaze = Gaze().get_file_list()
