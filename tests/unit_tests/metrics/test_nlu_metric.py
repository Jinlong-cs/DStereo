from hat.metrics.nlu_metric import NluEvalMetric


def test_nlu_metric():
    nlu_metric = NluEvalMetric("test_nlu")
    assert nlu_metric.name == "test_nlu"
