from hat.evaluation import compare_bev_seg


def test_compare_bev_seg():
    eval_prediction = {
        "mean_iou": "0.3793",
        "mean_acc": "0.4570",
        "all_acc": "0.8820",
        "classes": [
            {"name": "class1", "iou": "0.8598", "acc": "0.9144"},
            {"name": "class2", "iou": "0.2634", "acc": "0.4267"},
            {"name": "class3", "iou": "0.0147", "acc": "0.0300"},
        ],
    }
    diff_prediction = {
        "mean_iou": "0.4215",
        "mean_acc": "0.5231",
        "all_acc": "0.9546",
        "classes": [
            {"name": "class1", "iou": "0.1214", "acc": "0.9144"},
            {"name": "class2", "iou": "0.3552", "acc": "0.4586"},
            {"name": "class3", "iou": "0.0147", "acc": "0.0012"},
        ],
    }
    tb = compare_bev_seg.compare(
        "eval", eval_prediction, "diff", diff_prediction
    )
    assert type(tb) == str
