from .base import ReleasePath

__all__ = ["mmasr"]


class Mmasr(ReleasePath):
    @property
    def models(self):
        return [
            "hat/models/backbones/resnet.py",
            "hat/models/backbones/vargnetv2.py",
            "hat/models/backbones/timesformer.py",
            "hat/models/backbones/swin_transformer.py",
            "hat/models/base_modules/basic_resnet_module.py",
            "hat/models/base_modules/basic_vargnet_module.py",
            "hat/models/base_modules/basic_mixvargenet_module.py",
            "hat/models/base_modules/label_encoder.py",
            "hat/models/base_modules/basic_timesformer_module.py",
            "hat/models/base_modules/conv_module.py",
            "hat/models/base_modules/mlp_module.py",
            "hat/models/base_modules/transformer_attentions.py",
            "hat/models/base_modules/transformer_bricks.py",
            "hat/models/embeddings.py",
            "hat/models/frame_utils.py",
            "hat/models/losses/*",
            "hat/models/model_convert/*",
            "hat/models/necks/*",
            "hat/models/structures/avspeech/*",
            "hat/models/structures/carp_structure.py",
            "hat/models/structures/encoder_decoder.py",
            "hat/models/structures/lipmove_model.py",
            "hat/models/structures/mmvad_structure.py",
            "hat/models/task_modules/avspeech/*",
        ]

    @property
    def datasets(self):
        return [
            "hat/data/datasets/avspeech/*",
            "hat/data/datasets/avspeech_dataset.py",
            "hat/data/collates/collates.py",
            "hat/data/datasets/wenet_dataset.py",
            "hat/data/dataloaders/chain_dataloader.py",
        ]

    @property
    def transforms(self):
        return [
            "hat/data/transforms/avspeech/*",
            "hat/data/utils.py",
        ]

    @property
    def metrics(self):
        return [
            "hat/metrics/*",
            "hat/metrics/avspeech_metric.py",
            "hat/metrics/loss_show.py",
        ]

    @property
    def samplers(self):
        return [
            "hat/data/samplers/*",
        ]

    @property
    def tests(self):
        return [
            "tests/unit_tests/models/backbones/*",
            "tests/unit_tests/models/backbones/test_resnet.py",
            "tests/unit_tests/models/backbones/test_vargnet.py",
            "tests/unit_tests/models/base_modules/*",
            "tests/unit_tests/models/base_modules/test_basic_resnet_module.py",
            "tests/unit_tests/models/base_modules/test_basic_vargnet_module.py",
            "tests/unit_tests/models/base_modules/test_encoder_decoder.py",
            "tests/unit_tests/models/task_modules/avspeech/*",
            "tests/unit_tests/models/structures/test_encoder_decoder.py",
            "tests/unit_tests/models/structures/test_lipmove_model.py",
            "tests/unit_tests/models/structures/test_mmvad_structure.py",
            "tests/unit_tests/models/structures/test_nlu.py",
            "tests/unit_tests/models/losses/test_cross_entropy_loss.py",
            "tests/unit_tests/models/losses/test_l1_loss.py",
            "tests/unit_tests/models/losses/test_mse_loss.py",
            "tests/unit_tests/models/losses/test_smooth_l1_loss.py",
            "tests/unit_tests/models/losses/test_softmax_ce_loss.py",
            "tests/unit_tests/data/datasets/avspeech/*",
            "tests/unit_tests/data/datasets/test_carp_dataset.py",
            "tests/unit_tests/data/datasets/test_batch_transform_dataset.py",
            "tests/unit_tests/data/datasets/test_nlu_dataset.py",
            "tests/unit_tests/data/dataloaders/test_multitask_loader.py",
        ]


mmasr = Mmasr().get_file_list()
