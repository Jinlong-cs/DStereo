from typing import List

from .base import ReleasePath

__all__ = ["faceid"]


class FaceID(ReleasePath):
    @property
    def models(self) -> List:
        return [
            "hat/models/task_modules/faceid_encoder/*",
            "hat/models/structures/classifier.py",
            "hat/models/losses/dist_faceid_softmax.py",
            "hat/models/losses/faceid/*",
        ]

    @property
    def transforms(self) -> List:
        return [
            "hat/data/transforms/faceid.py",
            "hat/data/transforms/detection.py",
        ]

    @property
    def datasets(self) -> List:
        return [
            "hat/data/datasets/faceid_dataset.py",
        ]

    @property
    def metrics(self) -> List:
        return [
            "hat/metrics/faceid/*",
            "hat/metrics/faceid_gar_metric.py",
        ]

    @property
    def samplers(self) -> List:
        return [
            "hat/data/samplers/dist_faceid_balance_sampler.py",
        ]

    @property
    def tests(self) -> List:
        return [
            "tests/unit_tests/data/datasets/test_faceid_dataset.py",
            "tests/unit_tests/data/samplers/test_dist_faceid_balance_sampler.py",
            "tests/unit_tests/data/transforms/test_faceid.py",
            "tests/unit_tests/metrics/test_faceid_gar_metrics.py",
            "tests/unit_tests/models/losses/test_faceid_loss.py",
            "tests/unit_tests/models/losses/test_faceid_softmax.py",
            "tests/unit_tests/models/task_modules/faceid_encoder/test_faceid_encoder.py",
            "tests/unit_tests/models/structures/test_classifier.py",
            "tests/unit_tests/data/transforms/test_detection.py",
        ]


faceid = FaceID().get_file_list()
