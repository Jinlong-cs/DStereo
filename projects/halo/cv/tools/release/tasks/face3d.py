from .base import ReleasePath

__all__ = ["face3d"]


class Face3D(ReleasePath):
    @property
    def models(self):
        return [
            "hat/models/losses/face3d/*",
            "hat/models/backbones/mixvargenet.py",
            "hat/models/structures/face3d/*",
            "hat/models/task_modules/face3d/*",
        ]

    @property
    def datasets(self):
        return [
            "hat/data/datasets/face3d_dataset.py",
        ]

    @property
    def transforms(self):
        return [
            "hat/data/transforms/face3d.py",
            "hat/data/transforms/gaze/gaze.py",
            "hat/data/transforms/common.py",
            "hat/data/transforms/detection.py",
            "hat/data/transforms/faceattr.py",
        ]

    @property
    def metrics(self):
        return [
            "hat/metrics/face3d.py",
        ]

    @property
    def samplers(self):
        return []

    @property
    def dataloader(self):
        return []

    @property
    def tests(self):
        return [
            "tests/unit_tests/core/face3d/*",
            "tests/unit_tests/data/datasets/test_face3d_dataset.py",
            "tests/unit_tests/data/transforms/test_face3d.py",
            "tests/unit_tests/metrics/test_face3d_metric.py",
        ]


face3d = Face3D().get_file_list()
