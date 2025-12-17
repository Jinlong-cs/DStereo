from hat.metrics.kitti2d_detection import Kitti2DMetric


def test_kitti2d_detection():
    metric = Kitti2DMetric(
        anno_file="./tmp_orig_data/kitti2d/kitti_eval.json",
    )
    metric.pred_dict = metric.gt_dict
    for _, v in metric.pred_dict.items():
        for ele in v:
            ele["score"] = 0.0
    metric.CLS_DICT = {0: "Car"}
    names, values = metric.get()
    assert names is not None
    assert values is not None
    assert len(names) == len(values)
