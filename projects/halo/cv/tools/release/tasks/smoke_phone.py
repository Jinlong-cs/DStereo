from typing import List

from .base import ReleasePath

__all__ = ["smoke_phone"]


class SmokePhone(ReleasePath):
    @property
    def models(self) -> List:
        return [
            "hat/models/structures/smoke_cls_model.py",
            "hat/models/structures/smoke_kps_model.py",
            "hat/models/structures/phone_classifier.py",
            "hat/models/task_modules/smoke/*",
            "hat/models/task_modules/landmark/*",
            "hat/models/backbones/vargnetv2.py",
            "hat/models/base_modules/conv_module.py",
            "hat/models/losses/landmark/landmark_loss.py",
            "hat/models/losses/softmax_ce_loss.py",
            "hat/models/losses/utils.py",
        ]

    @property
    def transforms(self) -> List:
        return [
            "hat/data/transforms/smoke_cls_transform.py",
            "hat/data/transforms/face3d.py",
            "hat/data/transforms/landmark.py",
            "hat/data/transforms/detection.py",
        ]

    @property
    def datasets(self) -> List:
        return [
            "hat/data/datasets/smoke_cls_dataset.py",
            "hat/data/datasets/smoke_kps_dataset.py",
            "hat/data/datasets/phone_dataset.py",
            "hat/data/datasets/wenet_dataset.py",
            "hat/data/datasets/dataset_wrappers.py",
        ]

    @property
    def metrics(self) -> List:
        return [
            "hat/metrics/smoke_cls_metric.py",
            "hat/metrics/smoke_kps_metric.py",
            "hat/metrics/phone_metric.py",
            "hat/metrics/metric.py",
            "hat/metrics/landmark.py",
        ]

    @property
    def samplers(self) -> List:
        return []

    @property
    def tests(self) -> List:
        return [
            "tests/unit_tests/data/datasets/test_smoke_cls_dataset.py",
            "tests/unit_tests/data/datasets/test_smoke_kps_dataset.py",
            "tests/unit_tests/data/transforms/test_smoke_cls_transform.py",
            "tests/unit_tests/data/transforms/test_smoke_kps_transform.py",
            "tests/unit_tests/metrics/test_phone_metric.py",
            "tests/unit_tests/metrics/test_smoke_cls_metric.py",
            "tests/unit_tests/metrics/test_smoke_kps_metric.py",
            "tests/unit_tests/models/structures/test_smoke_cls_model.py",
            "tests/unit_tests/models/structures/test_smoke_kps_model.py",
            "tests/unit_tests/models/structures/test_phone_classifier.py",
            "tests/unit_tests/models/task_modules/smoke/*",
        ]


smoke_phone = SmokePhone().get_file_list()
