from hat.metrics.metric_lane_detection import CulaneF1Score
from tests.utils import gen_fake_np_data


def test_metric_culane_f1_score():
    metric = CulaneF1Score("test_culane_f1")
    preds = [
        [
            gen_fake_np_data((4, 2), "float32"),
            gen_fake_np_data((5, 2), "float32"),
        ]
    ]
    labels = [
        [
            gen_fake_np_data((3, 2), "float32"),
            gen_fake_np_data((4, 2), "float32"),
        ]
    ]
    metric.update(labels, preds)
    _, result = metric.get()
    assert result is not None
