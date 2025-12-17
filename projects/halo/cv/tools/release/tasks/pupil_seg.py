from typing import List

from .base import ReleasePath

__all__ = ["pupil_seg"]


class PupilSeg(ReleasePath):
    @property
    def models(self) -> List:
        return [
            "hat/models/structures/pupil_segmentation.py",
            "hat/models/task_modules/pupil_segmentation/*",
            "hat/models/losses/pupil_segmentation/*",
        ]

    @property
    def transforms(self) -> List:
        return [
            "hat/data/transforms/detection.py",
            "hat/data/transforms/faceid.py",
            "hat/data/transforms/landmark.py",
            "hat/data/transforms/pupil_segmentation/*",
        ]

    @property
    def datasets(self) -> List:
        return [
            "hat/data/datasets/pupil_segmentation_dataset.py",
        ]

    @property
    def metrics(self) -> List:
        return [
            "hat/metrics/pupil_segmentation_metric.py",
            "hat/metrics/mean_iou.py",
        ]

    @property
    def tests(self) -> List:
        return [
            "tests/unit_tests/models/structures/test_pupil_segmentation.py",
            "tests/unit_tests/models/task_modules/pupil_segmentation/*",
            "tests/unit_tests/models/losses/pupil_segmentation/*",
            "tests/unit_tests/data/datasets/test_pupil_segmentation_dataset.py",
            "tests/unit_tests/data/transforms/test_detection.py",
            "tests/unit_tests/data/transforms/test_landmark.py",
            "tests/unit_tests/data/transforms/test_faceid.py",
            "tests/unit_tests/data/transforms/pupil_segmentation/*",
            "tests/unit_tests/callbacks/pupil_segmentation/*",
            "tests/unit_tests/metrics/test_pupil_segmentation_metric.py",
            "tests/unit_tests/metrics/test_mean_iou.py",
        ]


pupil_seg = PupilSeg().get_file_list()
