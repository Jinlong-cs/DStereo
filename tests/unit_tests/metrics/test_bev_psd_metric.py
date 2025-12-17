import pytest
import torch

from hat.metrics.bev.bev_psd_metric import ANCBEVPSDMetric

try:
    import aidisdk
except ImportError:
    aidisdk = None


@pytest.mark.skipif(aidisdk is None, reason="need aidisdk")
def test_bevpsd_metric():
    iou_threshold = 0.5
    global_max_distance = 2
    local_max_distance = 1
    validation_bev_range_list = [
        (0, 0, 0, 0),
        (-3, -3, 6, 3),
        (-7, -7, 10, 7),
        (-10, -10, 14, 10),
        (-10, -10, 20, 10),
    ]
    resolution_m_per_grid = 0.2
    metric = ANCBEVPSDMetric(
        iou_threshold=iou_threshold,
        global_max_distance=global_max_distance,
        local_max_distance=local_max_distance,
        validation_bev_range_list=validation_bev_range_list,
        resolution_m_per_grid=resolution_m_per_grid,
        device="cpu",
        eval_result_path="./eval_metric.json",
    )
    annos_bev_psd_obj = {
        "global": torch.rand([1, 2, 16]) * 100,
        "local": torch.rand([1, 2, 22]) * 100,
    }
    decode_label = {
        "decode_global": [
            {
                "slot_junctions_locations_list": [
                    [
                        [51.63263702392578, 104.02135467529297],
                        [51.719818115234375, 117.26274871826172],
                        [24.779077529907227, 116.93089294433594],
                        [24.93161964416504, 103.6398696899414],
                    ],
                ],
                "slot_occupancy_list": [
                    0.9627720713615417,
                ],
                "slot_type_list": [
                    0,
                ],
                "slot_orientation_list": [
                    [-0.9987615942955017, -0.049751751124858856],
                ],
                "slot_score_list": [
                    0.26631274819374084,
                ],
            }
        ],
        "decode_local": [
            [
                {
                    "j0_location_list": [],
                    "j0_sline_angle_list": [],
                    "j0_type_list": [],
                    "j0_score_list": [],
                },
                {
                    "j1_location_list": [],
                    "j1_sline_angle_list": [],
                    "j1_type_list": [],
                    "j1_score_list": [],
                },
                {
                    "j2_location_list": [[23.07050895690918, 110.0]],
                    "j2_sline_angle_list": [[1.0, 4.4102143518611985e-18]],
                    "j2_type_list": [0.0008927699527703226],
                    "j2_score_list": [0.27685296535491943],
                },
                {
                    "j3_location_list": [[23.0, 110.0]],
                    "j3_sline_angle_list": [
                        [-1.8170259963312674e-08, 1.4100278349360451e-06]
                    ],
                    "j3_type_list": [0.0008490910404361784],
                    "j3_score_list": [0.27637171745300293],
                },
            ]
        ],
        "decode_slots": torch.rand([1, 2, 33]) * 100,
    }
    metric.update(annos_bev_psd_obj, decode_label)
    _, value = metric.get()
    assert len(value.tables)
