import torch

from hat.models.task_modules.bev.postprocess import ANCBEVParkingrodDecoder


def test_bevparkingrod_decoder():
    input_size = [192, 128]
    downsampling_factor_feature = 2
    score_threshold = 0.2
    topk = 200
    nms_distance_threshold = 5
    kernel_size = 3

    parkingrod_decoder = ANCBEVParkingrodDecoder(
        input_size=input_size,
        downsample_factor_feature=downsampling_factor_feature,
        score_threshold=score_threshold,
        topk=topk,
        nms_distance_threshold=nms_distance_threshold,
        kernel_size=kernel_size,
    )
    pred = (
        torch.randn((1, 1, 96, 64)),
        torch.randn((1, 4, 96, 64)),
    )
    parkingrod_decoder(pred)
