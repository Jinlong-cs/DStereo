import torch

from hat.metrics.psd_metric import PSDMetric
from hat.models.task_modules.ipm_psd.psd_postprocess import PSDPostprocess


def test_psd_metric():
    iou_threshold = 0.5
    global_max_distance = 10
    local_max_distance = 10
    input_size = [896, 896]
    downsample_factor_globalslot_feature = 32
    downsample_factor_localjunction_feature = 4
    threshold_globalslot_feature = 0.18
    threshold_localjunction_feature = 0.25
    topk_globalslot_feature = 50
    topk_localjunction_feature = 30
    threshold_occupancy_feature = 0.5
    threshold_junction_type_feature = 0.5
    threshold_fuse_distance = 25
    global_kernel_size = 3
    local_kernel_size = 5
    draw_flag = False
    nms_distance_threshold = (20,)
    psd_postprocess = PSDPostprocess(
        input_size,
        downsample_factor_globalslot_feature,
        downsample_factor_localjunction_feature,
        threshold_globalslot_feature,
        threshold_localjunction_feature,
        topk_globalslot_feature,
        topk_localjunction_feature,
        threshold_occupancy_feature,
        threshold_junction_type_feature,
        threshold_fuse_distance,
        global_kernel_size,
        local_kernel_size,
        draw_flag,
        nms_distance_threshold=nms_distance_threshold,
    )
    pred = (
        torch.randn((1, 1, 28, 28)),
        torch.randn((1, 8, 28, 28)),
        torch.randn((1, 1, 28, 28)),
        torch.randn((1, 1, 28, 28)),
        torch.randn((1, 2, 28, 28)),
        torch.randn((1, 4, 224, 224)),
        torch.randn((1, 8, 224, 224)),
        torch.randn((1, 8, 224, 224)),
        torch.randn((1, 4, 224, 224)),
    )
    pred = psd_postprocess(pred)
    metric = PSDMetric(
        iou_threshold,
        global_max_distance,
        local_max_distance,
    )
    result_dict = dict(
        preds_label=pred["preds_label"],
        labels=[
            [
                [
                    [
                        126.93574999999998,
                        708.6539999999999,
                        -1,
                        -0.9999964833259583,
                        0.002632478717714548,
                        0,
                        253.972,
                        645.179,
                        253.64,
                        771.461,
                        0.131,
                        770.644,
                        0.0,
                        647.332,
                    ]
                ],
                [
                    [
                        253.972,
                        645.179,
                        253.64,
                        771.461,
                        0.131,
                        770.644,
                        0.0,
                        647.332,
                        -0.9999640583992004,
                        0.008476827293634415,
                        -0.9999948143959045,
                        -0.0032228140626102686,
                        0.9999948143959045,
                        0.0032228140626102686,
                        0.9999640583992004,
                        -0.008476827293634415,
                        1,
                        1,
                        0,
                        0,
                        -1,
                    ]
                ],
            ]
        ],
        img_scene=[0],
    )
    metric.update(result_dict)
    total_names, total_values = metric.get()
    assert len(total_names) == len(total_values)
