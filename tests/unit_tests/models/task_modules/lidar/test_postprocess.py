import torch

from hat.models.task_modules.lidar import AfdetPredict, AfsegPredict


def gen_fake_data_pred():
    return dict(
        hm=torch.zeros(1, 3, 128, 240, dtype=torch.float32),
        rhd=torch.zeros(1, 6, 128, 240, dtype=torch.float32),
        rot=torch.zeros(1, 2, 128, 240, dtype=torch.float32),
    )


def gen_fake_data_target():
    return dict(object_token="test")


def gen_fake_data_pred_seg():
    return [
        dict(
            map_seg_hm=torch.zeros(1, 2, 128, 128, dtype=torch.float32),
        )
    ]


def gen_fake_data_target_seg():

    return dict(
        metadata=range(1),
        map_seg_hm=torch.zeros(1, 2, 128, 128, dtype=torch.float32),
        seg_loss_mask=torch.zeros((1, 2, 128, 128), dtype=torch.int64),
    )


def test_AfdetPredict():

    maxpool_dict = dict(maxpool_kernel_size=[[3] * 3], topk=[200] * 3)
    model = AfdetPredict(num_classes=[3], maxpool_dict=maxpool_dict)

    pc_range = [-19.2, -72.0, -4.0, 96.0, 72.0, 2.3]
    voxel_size = [0.15, 0.15, 0.1]
    example = gen_fake_data_target()
    preds_dict = gen_fake_data_pred()
    voxel_size = list(voxel_size[:2])
    alpha = [[2.2, 2.1, 3.4]]

    results = model(
        example,
        preds_dict,
        pc_range,
        alpha,
    )

    out_box_num = sum(maxpool_dict["topk"])
    assert "label_preds" in results[0]
    assert "box3d_lidar" in results[0]
    assert "scores" in results[0]
    assert results[0]["box3d_lidar"].shape == (out_box_num, 7)


def test_AfsegPredict():

    model = AfsegPredict()

    example = gen_fake_data_target_seg()
    preds_dict = gen_fake_data_pred_seg()

    results = model(
        example,
        preds_dict,
    )

    assert "map_seg_hm" in results[0]
