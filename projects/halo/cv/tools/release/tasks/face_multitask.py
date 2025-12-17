from .base import ReleasePath

__all__ = ["face_multitask"]


class FaceMultitask(ReleasePath):
    @property
    def models(self):
        return [
            "hat/models/structures/face3d/*",
            "hat/models/task_modules/face3d/*",
            "hat/models/task_modules/facemtl/*",
            "hat/models/necks/group_fpn.py",
            "hat/models/structures/landmark/*",
            "hat/models/task_modules/landmark/*",
            "hat/models/losses/landmark/*",
            "hat/models/structures/image_classifier.py",
            "hat/models/structures/multitask_graph_model.py",
        ]

    @property
    def datasets(self):
        return [
            "hat/data/datasets/face3d_dataset.py",
            "hat/data/datasets/landmark_dataset.py",
            "hat/data/datasets/dataset_wrappers.py",
            "hat/data/datasets/facequality_mtl_dataset.py",
        ]

    @property
    def transforms(self):
        return [
            "hat/data/transforms/detection.py",
            "hat/data/transforms/face3d.py",
            "hat/data/transforms/facemtl.py",
            "hat/data/transforms/landmark.py",
            "hat/data/transforms/faceid.py",
            "hat/data/transforms/faceattr.py",
            "hat/data/transforms/facequality_mtl.py",
            "hat/data/transforms/common.py",
        ]

    @property
    def metrics(self):
        return [
            "hat/metrics/face3d.py",
            "hat/metrics/landmark.py",
            "hat/metrics/acc.py",
            "hat/metrics/recall.py",
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
            "tests/unit_tests/models/structures/face3d/*",
            "tests/unit_tests/models/task_modules/face3d/*",
            "tests/unit_tests/models/task_modules/facemtl/*",
            "tests/unit_tests/models/necks/test_group_fpn.py",
            "tests/unit_tests/models/structures/landmark/*",
            "tests/unit_tests/models/task_modules/landmark/*",
            "tests/unit_tests/models/losses/test_landmark_loss.py",
            "tests/unit_tests/models/structures/test_image_classifier.py",
            "tests/unit_tests/models/structures/test_multitask_graph_model.py",
            "tests/unit_tests/data/datasets/test_face3d_dataset.py",
            "tests/unit_tests/data/datasets/test_ldmk_dataset.py",
            "tests/unit_tests/data/datasets/test_dataset_wrappers.py",
            "tests/unit_tests/data/datasets/test_facequality_dataset.py",
            "tests/unit_tests/data/transforms/test_detection.py",
            "tests/unit_tests/data/transforms/test_face3d.py",
            "tests/unit_tests/data/transforms/test_facemtl.py",
            "tests/unit_tests/data/transforms/test_landmark.py",
            "tests/unit_tests/data/transforms/test_faceid.py",
            "tests/unit_tests/data/transforms/test_faceattr.py",
            "tests/unit_tests/data/transforms/test_facequality_mtl.py",
            "tests/unit_tests/data/transforms/test_common.py",
            "tests/unit_tests/metrics/test_face3d_metric.py",
            "tests/unit_tests/metrics/test_landmark_metric.py",
            "tests/unit_tests/metrics/test_acc.py",
            "tests/unit_tests/metrics/test_recall.py",
            "tests/unit_tests/data/samplers/test_dist_proportion_sampler.py",
            "tests/unit_tests/data/dataloaders/test_multitask_loader.py",
        ]


face_multitask = FaceMultitask().get_file_list()
