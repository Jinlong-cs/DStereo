import pytest
import torch

from hat.metrics.bev.bev_parkingrod_metric import ANCBEVParkingrodMetric

try:
    import aidisdk
except ImportError:
    aidisdk = None


@pytest.mark.skipif(aidisdk is None, reason="need aidisdk")
def test_bev_parkingrod_metric():
    distance_threshold = 5
    angle_threshold = 30
    resolution_m_per_grid = 0.2
    eval_vcs_range = (-10, -10, 20, 10)
    ego_vcs_range = (-1, -1, 4, 1)
    dep_intervals = (2, 6, 9)
    roi_vcs_range = (-12.8, -12.8, 25.6, 12.8)
    eval_parking_slot_rod = True
    ego_vcs_range_rod = (-0.5, -0.5, 3, 0.5)
    metric = ANCBEVParkingrodMetric(
        rod_distance_threshold=distance_threshold,
        strip_dictance_threshold=distance_threshold,
        angle_threshold=angle_threshold,
        resolution_m_per_grid=resolution_m_per_grid,
        eval_vcs_range=eval_vcs_range,
        dep_intervals=dep_intervals,
        vcs_range=roi_vcs_range,
        ego_vcs_range=ego_vcs_range,
        eval_parking_slot_rod=eval_parking_slot_rod,
        ego_vcs_range_rod=ego_vcs_range_rod,
    )
    annos_bev_parkingrod_obj = {
        "parking_rod": torch.rand([1, 2, 17]) * 100,
    }
    preds_label = torch.rand([1, 2, 6]) * 100
    metric.update(annos_bev_parkingrod_obj, preds_label)
    _, value = metric.get()
    assert len(value.tables)
