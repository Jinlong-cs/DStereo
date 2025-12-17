from typing import List

from .base import ReleasePath

__all__ = ["face_attr"]


class FaceAttr(ReleasePath):
    """Face attribute and face anti-spoofing.

    Owner: @yisu.zhou
    """

    @property
    def models(self) -> List:
        return [
            "hat/models/losses/fas_multisigmoid_loss.py",
            "hat/models/losses/age_cost_sensitive_loss.py",
            "hat/models/losses/cross_entropy_loss.py",
            "hat/models/structures/faceattr_classifier.py",
            "hat/models/structures/fas_classifier.py",
            "hat/models/task_modules/faceid_encoder/vargnet_v2.py",
            "hat/models/task_modules/faceattr/*",
            "hat/models/task_modules/fas/*",
            "hat/models/backbones/vargnetv2.py",
        ]

    @property
    def transforms(self) -> List:
        return [
            "hat/data/transforms/faceattr.py",
            "hat/data/transforms/face_fas.py",
            "hat/data/transforms/landmark.py",
            "hat/data/transforms/gaze/gaze.py",
            "hat/data/transforms/faceid.py",
            "hat/data/transforms/detection.py",
            "hat/data/transforms/common.py",
        ]

    @property
    def datasets(self) -> List:
        return [
            "hat/data/datasets/faceattr_dataset.py",
            "hat/data/datasets/fas_dataset.py",
        ]

    @property
    def metrics(self) -> List:
        return [
            "hat/metrics/faceattr_metrics.py",
            "hat/metrics/fas_metrics.py",
            "hat/metrics/acc.py",
        ]

    @property
    def samplers(self) -> List:
        return []

    @property
    def tests(self) -> List:
        return [
            "tests/unit_tests/data/datasets/test_faceattr_dataset.py",
            "tests/unit_tests/data/datasets/test_fas_dataset.py",
            "tests/unit_tests/data/transforms/test_fas.py",
            "tests/unit_tests/data/transforms/test_faceattr.py",
            "tests/unit_tests/metrics/test_faceattr_metrics.py",
            "tests/unit_tests/metrics/test_fas_metric.py",
        ]


face_attr = FaceAttr().get_file_list()
