from typing import List

from .base import ReleasePath

__all__ = ["pnpnet"]


class Pnpnet(ReleasePath):
    @property
    def models(self) -> List:
        return [
            "hat/models/losses/l1_loss.py",
            "hat/models/structures/eye3d_pose.py",
        ]

    @property
    def transforms(self) -> List:
        return [
            "hat/data/transforms/eye3d_pose.py",
        ]

    @property
    def datasets(self) -> List:
        return [
            "hat/data/datasets/eye3d_pose_dataset.py",
        ]

    @property
    def metrics(self) -> List:
        return [
            "hat/metrics/eye3d_pose.py",
        ]

    @property
    def samplers(self) -> List:
        return []

    @property
    def tests(self) -> List:
        return [
            "tests/unit_tests/data/datasets/test_eye3d_pose_dataset.py",
            "tests/unit_tests/data/transforms/test_eye3d_pose.py",
            "tests/unit_tests/metrics/test_eye3d_pose.py",
            "tests/unit_tests/models/structures/test_eye3d_pose.py",
        ]


pnpnet = Pnpnet().get_file_list()
