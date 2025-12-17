import torch

from hat.models.task_modules.ipm_psd.psd_postprocess import PSDPostprocess


def test_superpsd_postprocess():

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
    psd_postprocess(pred)
