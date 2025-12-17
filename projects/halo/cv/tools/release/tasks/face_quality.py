from .base import ReleasePath

__all__ = ["face_quality"]


class FaceQuality(ReleasePath):
    @property
    def models(self):
        return [
            "hat/models/structures/image_classifier.py",
            "hat/models/task_modules/facequality/*",
            "hat/models/structures/multitask_graph_model.py",
        ]

    @property
    def datasets(self):
        return [
            "hat/data/datasets/dataset_wrappers.py",
            "hat/data/datasets/facequality_mtl_dataset.py",
        ]

    @property
    def transforms(self):
        return [
            "hat/data/transforms/landmark.py",
            "hat/data/transforms/faceid.py",
            "hat/data/transforms/detection.py",
            "hat/data/transforms/facequality_mtl.py",
            "hat/data/transforms/common.py",
        ]

    @property
    def metrics(self):
        return [
            "hat/metrics/acc.py",
        ]

    @property
    def samplers(self):
        return [
            "hat/data/samplers/dist_proportion_sampler.py",
        ]

    @property
    def dataloader(self):
        return [
            "hat/data/dataloaders/multitask_loader.py",
        ]

    @property
    def tests(self):
        return [
            "tests/unit_tests/models/structures/test_image_classifier.py",
            "tests/unit_tests/models/task_modules/facequality/*",
            "tests/unit_tests/models/structures/test_multitask_graph_model.py",
            "tests/unit_tests/data/datasets/test_dataset_wrappers.py",
            "tests/unit_tests/data/datasets/test_facequality_dataset.py",
            "tests/unit_tests/data/transforms/test_landmark.py",
            "tests/unit_tests/data/transforms/test_faceid.py",
            "tests/unit_tests/data/transforms/test_detection.py",
            "tests/unit_tests/data/transforms/test_facemtl.py",
            "tests/unit_tests/data/transforms/test_common.py",
            "tests/unit_tests/metrics/test_acc.py",
            "tests/unit_tests/data/samplers/test_dist_proportion_sampler.py",
            "tests/unit_tests/data/dataloaders/test_multitask_loader.py",
        ]


face_quality = FaceQuality().get_file_list()
