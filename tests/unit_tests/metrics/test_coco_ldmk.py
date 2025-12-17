import pytest
import torch

from hat.registry import build_from_registry
from hat.utils.package_helper import check_packages_available
from tests.unit_tests.metrics.testers import MetricTester

JSON_PATH = "./tmp_orig_data/landmark/hat_test/human_keypoints_test\
/A11_RGB_IMS_valset_kps_person_head_face_hand_bottle_phone_gt.json"


@pytest.mark.skipif(
    not check_packages_available("pycocotools", raise_exception=False),
    reason="need pycocotools",
)
def test_coco_ldmk_metric():
    coco_ldmk_metric = build_from_registry(
        dict(type="COCOLdmkMetric", ann_file=JSON_PATH, cleanup=False)
    )

    batch = {
        "image_name": [
            "od002506_M_30_175_T_LM_S1_S_A11_imx290_20211203_nj_SUN_N_NH_NM_NM_NG_10000_UN_UN_UN_1_RGB_1920-1080_17d10c4_000000.jpg"  # noqa
        ],  # noqa
        "gt_boxes": torch.randn((1, 1, 4)),
    }
    preds = {
        "pred_ldmk": torch.tensor(
            [
                [
                    [
                        1382.766775790738,
                        878.952672114599,
                        0.7310383915901184,
                        1515.7399207463307,
                        1033.1990559673407,
                        0.2268187403678894,
                        1607.427048905892,
                        910.628268827961,
                        0.04130493476986885,
                        1195.1030627909702,
                        890.6588711676484,
                        0.7708529829978943,
                        1191.3497882753034,
                        1050.414054091899,
                        0.07226699590682983,
                        1172.0472341711697,
                        925.7774666850472,
                        0.03154166415333748,
                        1481.9604523424543,
                        527.7667103736106,
                        0.8421804904937744,
                        1508.2333717149968,
                        726.7720877084538,
                        0.7671985626220703,
                        1606.3546848955257,
                        896.1676702391571,
                        0.9053956866264343,
                        1620.2954179890585,
                        1016.6726571110647,
                        0.775341272354126,
                        1145.774315118217,
                        482.3191159814768,
                        0.7857683897018433,
                        1034.248451648312,
                        726.0834884401715,
                        0.7186225652694702,
                        1169.9025067896155,
                        908.5624689709266,
                        0.8461283445358276,
                        1225.129256599282,
                        969.159261630584,
                        0.9420687556266785,
                        1424.052791628001,
                        507.1087126241407,
                        0.8591914176940918,
                    ]
                ]
            ]
        ),
        "pred_scores": torch.ones((1, 1)),
        "pred_boxes": torch.tensor([[[0.0, 0.0, 1500.0, 1500.0]]]),
    }
    coco_ldmk_metric.update(batch, preds)
    coco_ldmk_metric.get()


@pytest.mark.skipif(
    not check_packages_available("pycocotools", raise_exception=False),
    reason="need pycocotools",
)
class TestDistRecallPrecision(MetricTester):
    def setup_class(self):
        super(TestDistRecallPrecision, self).setup_class(self)
        metric = build_from_registry(
            dict(type="COCOLdmkMetric", ann_file=JSON_PATH)  # noqa
        )

        batch_list = []
        for i in range(4):  # noqa B007
            batch = {
                "image_name": [
                    "od002506_M_30_175_T_LM_S1_S_A11_imx290_20211203_nj_SUN_N_NH_NM_NM_NG_10000_UN_UN_UN_1_RGB_1920-1080_17d10c4_000000.jpg"  # noqa
                ],  # noqa
                "gt_boxes": torch.randn((1, 1, 4)),
            }
            batch_list.append(batch)

        preds_list = []
        for i in range(4):  # noqa B007
            preds = {
                "pred_ldmk": torch.tensor(
                    [
                        [
                            [
                                1382.766775790738,
                                878.952672114599,
                                0.7310383915901184,
                                1515.7399207463307,
                                1033.1990559673407,
                                0.2268187403678894,
                                1607.427048905892,
                                910.628268827961,
                                0.04130493476986885,
                                1195.1030627909702,
                                890.6588711676484,
                                0.7708529829978943,
                                1191.3497882753034,
                                1050.414054091899,
                                0.07226699590682983,
                                1172.0472341711697,
                                925.7774666850472,
                                0.03154166415333748,
                                1481.9604523424543,
                                527.7667103736106,
                                0.8421804904937744,
                                1508.2333717149968,
                                726.7720877084538,
                                0.7671985626220703,
                                1606.3546848955257,
                                896.1676702391571,
                                0.9053956866264343,
                                1620.2954179890585,
                                1016.6726571110647,
                                0.775341272354126,
                                1145.774315118217,
                                482.3191159814768,
                                0.7857683897018433,
                                1034.248451648312,
                                726.0834884401715,
                                0.7186225652694702,
                                1169.9025067896155,
                                908.5624689709266,
                                0.8461283445358276,
                                1225.129256599282,
                                969.159261630584,
                                0.9420687556266785,
                                1424.052791628001,
                                507.1087126241407,
                                0.8591914176940918,
                            ]
                        ]
                    ]
                ),
                "pred_scores": torch.ones((1, 1)),
                "pred_boxes": torch.tensor([[[0.0, 0.0, 1500.0, 1500.0]]]),
            }
            preds_list.append(preds)
        self.args = (preds_list, batch_list, metric)

    def func_test(self, rank, preds, targets, metric):
        torch.cuda.set_device(rank)
        metric = metric.cuda()
        rank_pred = preds[rank]
        rank_pred_ = {
            "pred_ldmk": rank_pred["pred_ldmk"].cuda(),
            "pred_scores": rank_pred["pred_scores"].cuda(),
            "pred_boxes": rank_pred["pred_boxes"].cuda(),
        }
        target = targets[rank]
        rank_target = {
            "image_name": target["image_name"],
            "gt_boxes": target["gt_boxes"].cuda(),
        }
        metric.update(rank_target, rank_pred_)

        _, m_values = metric.get()
