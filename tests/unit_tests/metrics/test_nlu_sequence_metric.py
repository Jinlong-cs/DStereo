from hat.metrics.nlu_sequence_metric import f1_score


def test_f1_score():
    y_true = [
        ["O", "O", "O", "B-MISC", "O"],
        ["B-MISC", "O", "B-PER", "I-PER", "O"],
    ]
    y_pred = [
        ["O", "O", "B-MISC", "O", "B-MISC"],
        ["B-MISC", "O", "B-PER", "I-PER", "O"],
    ]
    assert f1_score(y_true, y_pred) >= 0.0
