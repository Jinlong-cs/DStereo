from .base import ReleasePath

__all__ = ["nlu"]


class Nlu(ReleasePath):
    @property
    def models(self):
        return [
            "hat/models/backbones/nlu_tcn.py",
            "hat/models/structures/nlu.py",
        ]

    @property
    def datasets(self):
        return [
            "hat/core/nlu/*",
            "hat/data/collates/collates.py",
            "hat/data/datasets/nlu_dataset.py",
        ]

    @property
    def transforms(self):
        return []

    @property
    def metrics(self):
        return [
            "hat/metrics/nlu_metric.py",
            "hat/metrics/nlu_sequence_metric.py",
        ]

    @property
    def samplers(self):
        return [
            "hat/data/samplers/*",
        ]

    @property
    def tests(self):
        return [
            "tests/unit_tests/data/datasets/test_nlu_dataset.py",
            "tests/unit_tests/metrics/test_nlu_metric.py",
            "tests/unit_tests/metrics/test_nlu_sequence_metric.py",
            "tests/unit_tests/models/backbones/test_nlu_tcn.py",
            "tests/unit_tests/models/structures/test_nlu.py",
        ]


nlu = Nlu().get_file_list()
