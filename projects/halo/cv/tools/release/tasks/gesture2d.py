from typing import List

from .base import ReleasePath

__all__ = ["gesture2d"]


class Gesture2D(ReleasePath):
    @property
    def models(self) -> List:
        return [
            "hat/models/structures/gesture/dyn_gesture_classifier.py",
            "hat/models/structures/gesture/multi_modality_classifier.py",
            "hat/models/task_modules/gesture/dyn_gest_head.py",
            "hat/models/task_modules/gesture/gest_multimodality_loss.py",
            "hat/models/task_modules/gesture/multi_modality_encoder.py",
            "hat/models/task_modules/gesture/multi_modality_head.py",
        ]

    @property
    def transforms(self) -> List:
        return [
            "hat/data/transforms/gesture/*",
        ]

    @property
    def datasets(self) -> List:
        return [
            "hat/data/datasets/roidb_act_dataset.py",
        ]

    @property
    def metrics(self) -> List:
        return []

    @property
    def samplers(self) -> List:
        return []

    @property
    def tests(self) -> List:
        return [
            "tests/unit_tests/models/structures/gesture/test_dyn_gesture_classifier.py",
            "tests/unit_tests/models/structures/gesture/test_mutimode_classifier.py",
            "tests/unit_tests/models/structures/gesture/test_sta_multimodality_classifier.py",
            "tests/unit_tests/data/datasets/test_ges_lmdb_dataset.py",
            "tests/unit_tests/models/task_modules/gestures/test_gest_multimodality_loss.py",
            "tests/unit_tests/models/task_modules/gestures/test_dyn_gesture_head.py",
            "tests/unit_tests/models/task_modules/gestures/test_multi_modality_encoder.py",
            "tests/unit_tests/models/task_modules/gestures/test_multimode_head.py",
            "tests/unit_tests/data/datasets/test_roidb_act_dataset.py",
        ]


gesture2d = Gesture2D().get_file_list()
