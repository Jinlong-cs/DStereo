from typing import List

from .base import ReleasePath

__all__ = ["eye_status"]


class EyeStatus(ReleasePath):
    @property
    def models(self) -> List:
        return [
            "hat/models/task_modules/eye_ldmk_status/*",
            "hat/models/task_modules/iris/*",
            "hat/models/structures/eye_status_classifier.py",
            "hat/models/losses/cross_entropy_loss.py",
            "hat/models/losses/smooth_l1_loss.py",
        ]

    @property
    def transforms(self) -> List:
        return [
            "hat/data/transforms/detection.py",
            "hat/data/transforms/landmark.py",
            "hat/data/transforms/common.py",
        ]

    @property
    def datasets(self) -> List:
        return [
            "hat/data/datasets/eye_status_dataset.py",
        ]

    @property
    def metrics(self) -> List:
        return [
            "hat/metrics/acc.py",
            "hat/metrics/eye_status_metrics.py",
        ]

    @property
    def samplers(self) -> List:
        return [
            "hat/data/samplers/dist_eye_status_balance_sampler.py",
        ]

    @property
    def tests(self) -> List:
        return [
            "tests/unit_tests/models/task_modules/eye_ldmk_status/*",
            "tests/unit_tests/models/task_modules/iris/*",
            "tests/unit_tests/models/structures/test_eye_status_classifier.py",
            "tests/unit_tests/models/losses/test_cross_entropy_loss.py",
            "tests/unit_tests/models/losses/test_smooth_l1_loss.py",
            "tests/unit_tests/data/datasets/test_eye_status_dataset.py",
            "tests/unit_tests/data/transforms/test_detection.py",
            "tests/unit_tests/data/transforms/test_landmark.py",
            "tests/unit_tests/data/transforms/test_common.py",
            "tests/unit_tests/metrics/test_acc.py",
            "tests/unit_tests/metrics/test_eye_status_metric.py",
            "tests/unit_tests/data/samplers/test_dist_eye_status_balance_sampler.py",
        ]


eye_status = EyeStatus().get_file_list()
