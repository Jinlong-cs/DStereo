from hat.evaluation import compare_real3d


def test_compare_real3d():
    eval_prediction = {
        "name": "real3d eval",
        "header": ["Camera+Category", "Recall"],
        "data": [
            {
                "Camera+Category": "front_0820+Pedestrian",
                "Recall": 0.8,
            }
        ],
    }
    diff_prediction = {
        "name": "real3d eval",
        "header": ["Camera+Category", "Recall"],
        "data": [
            {
                "Camera+Category": "front_0820+Pedestrian",
                "Recall": 0.85,
            }
        ],
    }
    tb = compare_real3d.compare(
        "eval", eval_prediction, "diff", diff_prediction
    )
    assert type(tb) == str
