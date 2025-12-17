from typing import List

from .base import ReleasePath

__all__ = ["hand3d"]


class Hand3D(ReleasePath):
    @property
    def models(self) -> List:
        return [
            "hat/models/structures/hand3d/*",
            "hat/models/task_modules/hand3d/*",
            "hat/models/base_modules/basic_resnet_module.py",
            "hat/models/base_modules/conv_module.py",
            "hat/models/embeddings.py",
            "hat/models/losses/hand_bmcloss.py",
            "hat/models/losses/smooth_l1_loss.py",
            "hat/models/necks/fpn.py",
            "hat/models/backbones/resnet.py",
        ]

    @property
    def transforms(self) -> List:
        return [
            "hat/data/transforms/face3d.py",
            "hat/data/transforms/detection.py",
            "hat/data/transforms/common.py",
        ]

    @property
    def datasets(self) -> List:
        return [
            "hat/data/datasets/hand3d_lmdb_dataset.py",
            "hat/data/datasets/dataset_wrappers.py",
            "hat/data/datasets/split_dataset.py",
        ]

    @property
    def metrics(self) -> List:
        return [
            "hat/metrics/hand3d_metric.py",
            "hat/metrics/landmark.py",
        ]

    @property
    def samplers(self) -> List:
        return [
            "hat/data/samplers/stateful_sampler.py",
        ]

    @property
    def tests(self) -> List:
        return [
            "tests/unit_tests/data/datasets/test_hand3d_dataset.py",
            "tests/unit_tests/metrics/test_hand3d_metric.py",
            "tests/unit_tests/models/structures/hand3d/*",
            "tests/unit_tests/models/task_modules/hand3d/*",
        ]


hand3d = Hand3D().get_file_list()
