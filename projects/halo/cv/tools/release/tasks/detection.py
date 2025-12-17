from .base import ReleasePath

__all__ = ["detection"]


class Detection(ReleasePath):
    @property
    def models(self):
        return [
            "hat/models/necks/fpn.py",
            "hat/models/necks/fix_channel.py",
            "hat/models/necks/pick_stride.py",
            "hat/models/task_modules/anchor_module.py",
            "hat/models/task_modules/roi_module.py",
            "hat/models/task_modules/rpn/*",
            "hat/models/task_modules/roi_modules/*",
            "hat/models/task_modules/track2d/roi_track_decoder.py",
            "hat/models/task_modules/track2d/roi_track_head.py",
            "hat/models/task_modules/track2d/roi_track_loss.py",
            "hat/models/task_modules/landmark/*",
            "hat/models/task_modules/person_position/*",
            "hat/models/base_modules/extend_container.py",
            "hat/models/base_modules/label_encoder.py",
            "hat/models/base_modules/roi_feat_extractors.py",
            "hat/models/base_modules/matcher.py",
            "hat/models/base_modules/postprocess/anchor_postprocess.py",
            "hat/models/base_modules/postprocess/rcnn_postprocess.py",
            "hat/models/base_modules/target/bbox_target.py",
            "hat/models/structures/multitask_graph_model.py",
            "hat/models/structures/detectors/two_stage.py",
            "hat/models/structures/detectors/three_stage.py",
            "hat/models/losses/cross_entropy_loss.py",
            "hat/models/losses/smooth_l1_loss.py",
            "hat/models/losses/hinge_loss.py",
            "hat/models/losses/landmark/landmark_loss.py",
        ]

    @property
    def datasets(self):
        return [
            "hat/data/metas.py",
            "hat/data/datasets/dataset_wrappers.py",
            "hat/data/datasets/roidb_detection_dataset.py",
            "hat/data/datasets/roidb_track_dataset.py",
            "hat/data/datasets/roidb_person_position_dataset.py",
        ]

    @property
    def transforms(self):
        return [
            "hat/data/transforms/detection.py",
            "hat/data/transforms/seq_transform.py",
            "hat/data/transforms/common.py",
            "hat/data/transforms/landmark.py",
        ]

    @property
    def metrics(self):
        return [
            "hat/metrics/metric_person_position.py",
            "hat/metrics/coco_ldmk.py",
        ]

    @property
    def samplers(self):
        return [
            "hat/data/samplers/dist_group_sampler.py",
        ]

    @property
    def tests(self):
        return [
            "tests/unit_tests/data/dataloaders/test_multitask_loader.py",
            "tests/unit_tests/data/datasets/test_roidb_detection_dataset.py",
            "tests/unit_tests/data/datasets/test_roidb_person_position_dataset.py",
            "tests/unit_tests/data/datasets/test_roidb_track_dataset.py",
            "tests/unit_tests/data/samplers/test_dist_group_sampler.py",
            "tests/unit_tests/data/transforms/test_detection.py",
            "tests/unit_tests/data/transforms/test_human_landmark.py",
            "tests/unit_tests/data/transforms/test_landmark.py",
            "tests/unit_tests/data/transforms/test_seq_transform.py",
            "tests/unit_tests/metrics/test_coco_ldmk.py",
            "tests/unit_tests/metrics/test_landmark_metric.py",
            "tests/unit_tests/metrics/test_metric_person_position.py",
            "tests/unit_tests/metrics/test_recall_precision.py",
            "tests/unit_tests/models/base_modules/test_loss_hard_negative_mining.py",
            "tests/unit_tests/models/base_modules/test_matcher.py",
            "tests/unit_tests/models/losses/test_hinge_loss.py",
            "tests/unit_tests/models/losses/test_landmark_loss.py",
            "tests/unit_tests/models/losses/test_smooth_l1_loss.py",
            "tests/unit_tests/models/losses/test_cross_entropy_loss.py",
            "tests/unit_tests/models/necks/test_fpn.py",
            "tests/unit_tests/models/task_modules/person_position/*",
            "tests/unit_tests/models/task_modules/roi_modules/test_rcnn_decoder.py",
            "tests/unit_tests/models/task_modules/roi_modules/test_rcnn_head.py",
            "tests/unit_tests/models/task_modules/roi_modules/test_roi_sampler.py",
            "tests/unit_tests/models/task_modules/roi_modules/test_share_conv_head.py",
            "tests/unit_tests/models/task_modules/rpn/*",
            "tests/unit_tests/models/task_modules/test_anchor_module.py",
            "tests/unit_tests/models/task_modules/test_roi_module.py",
        ]


detection = Detection().get_file_list()
