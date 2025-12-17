import torch

from hat.models.task_modules.bev.postprocess import ANCBevPSDDecoder


def test_bevpsd_postprocess():
    input_size = [192, 128]
    downsample_factor_globalslot_feature = 4
    downsample_factor_localjunction_feature = 1
    threshold_globalslot_feature = 0.18
    threshold_localjunction_feature = 0.25
    topk_globalslot_feature = 50
    topk_localjunction_feature = 30
    threshold_occupancy_feature = 0.5
    threshold_junction_type_feature = 0.5
    threshold_fuse_distance = 5
    global_kernel_size = 1
    local_kernel_size = 1
    nms_distance_threshold = 7
    psd_postprocess = ANCBevPSDDecoder(
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
        nms_distance_threshold,
    )
    pred = (
        torch.randn((1, 1, 48, 32)),
        torch.randn((1, 8, 48, 32)),
        torch.randn((1, 1, 48, 32)),
        torch.randn((1, 1, 48, 32)),
        torch.randn((1, 2, 48, 32)),
        torch.randn((1, 4, 192, 128)),
        torch.randn((1, 8, 192, 128)),
        torch.randn((1, 8, 192, 128)),
        torch.randn((1, 4, 192, 128)),
    )
    psd_postprocess(pred)
