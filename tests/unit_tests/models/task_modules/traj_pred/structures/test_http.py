import numpy as np
import pytest
import torch

from hat.models.backbones.vargnetv2 import VargNetV2
from hat.models.base_modules.roi_feat_extractors import CropperQAT
from hat.models.necks.unet import Unet
from hat.models.task_modules.traj_pred.heads.http_heads import (
    HeatmapHead,
    TrajRegHead,
)
from hat.models.task_modules.traj_pred.structures.http import (
    HTTP,
    EndpointEncoder,
    MaxPointsSampler,
    StateEncoder,
)


@pytest.mark.parametrize(
    [
        "channels",
        "endpoints",
    ],
    [
        pytest.param(
            [16, 32, 32],
            [torch.randn([10, 2, 1, 1]) for _ in range(10)],
        ),
        pytest.param(
            [
                16,
            ],
            [torch.randn([10, 20, 1, 1]) for _ in range(1)],
        ),
    ],
)
def test_enpoint_encoder(channels, endpoints):
    model = EndpointEncoder(
        bn_kwargs={"eps": 2e-5, "momentum": 0.1},
        channels=channels,
        input_channel=endpoints[0].shape[1],
    )
    output = model(endpoints)
    assert len(output) == len(endpoints)
    assert all([x.shape[0] == endpoints[0].shape[0] for x in output])
    assert all([x.shape[1] == channels[-1] for x in output])
    assert all([x.shape[2] == 1 for x in output])
    assert all([x.shape[3] == 1 for x in output])


@pytest.mark.parametrize(
    ["channels_seq", "channels_fc", "kernel_size", "state"],
    [
        pytest.param(
            [16, 16, 16],
            [32, 32],
            [3, 3, 3],
            torch.randn((10, 2, 1, 7)),
        ),
        pytest.param(
            [16, 16, 16, 32],
            [32, 32],
            [6, 6, 6, 5],
            torch.randn((10, 3, 1, 20)),
        ),
        pytest.param(
            [16, 16],
            [32, 32],
            [3, 3],
            torch.randn((10, 2, 1, 5)),
        ),
    ],
)
def test_state_encoder(channels_seq, channels_fc, kernel_size, state):
    model = StateEncoder(
        bn_kwargs={"eps": 2e-5, "momentum": 0.1},
        channels_seq=channels_seq,
        channels_fc=channels_fc,
        kernel_size=kernel_size,
        input_channel=state.shape[1],
    )
    output = model(state)
    assert output.shape[0] == state.shape[0]
    assert output.shape[1] == channels_fc[-1]
    assert output.shape[2] == 1
    assert output.shape[3] == 1


def test_max_points_sampler():
    def fn(x, inverse=False):
        return x

    sampler = MaxPointsSampler(
        sample_num=10,
        kernel_size=3,
        transformation_func=fn,
        update_radius=None,
        impossible_threshold=0.0,
        use_passable_mask=True,
    )
    heatmap = torch.zeros([5, 192, 192])

    ret_points = []
    ret_values = []
    for j in range(10):
        ret_points.append([])
        for i in range(5):
            x, y = 10 * j + i * 10 + 20, 10 * j + i * 10 + 20
            heatmap[i, x, y] = 1 - i * 0.1
            ret_points[-1].append([x, y])
        ret_values.append(np.array([0.1992, 0.1792, 0.1592, 0.1392, 0.1192]))

    points, values = sampler(heatmap)
    assert len(points) == len(values)
    for i in range(len(points)):
        assert np.all(
            np.array(ret_points[i]) == points[i].detach().cpu().numpy()
        )
        assert np.all(
            np.abs(ret_values[i] - values[i].detach().cpu().numpy()) < 1e-3
        )


def test_http():
    from hat.models.backbones.vargnetv2 import get_vargnetv2_stride2channels

    def fn(x, inverse=False):
        return x

    history_time_step = 20
    future_time_step = 30
    agent_feature_size = 11  # feature size to crop
    input_channel = 3 + history_time_step * 2
    bn_kwargs = dict(eps=2e-5, momentum=0.1)
    alpha = 1.0
    strides = [2, 4, 8, 16, 32]
    strides2channels = get_vargnetv2_stride2channels(alpha)
    channels = [strides2channels[x] for x in strides]
    agent_feature_channel = sum(channels)
    state_feature_channel = 32
    endpoint_feature_channel = 32

    model = HTTP(
        teacher_loss_weight=0.5,
        teacher_num=5,
        need_backward=False,
        backbone=VargNetV2(
            input_channels=3,
            num_classes=-1,
            bn_kwargs=bn_kwargs,
            alpha=alpha / 2,
            bias=True,
            include_top=False,
            flat_output=False,
        ),
        backbone_extra=VargNetV2(
            input_channels=input_channel - 3,
            num_classes=-1,
            bn_kwargs=bn_kwargs,
            alpha=alpha / 2,
            bias=True,
            include_top=False,
            flat_output=False,
        ),
        neck=Unet(
            stride2channels=strides2channels,
            in_strides=strides,
            out_strides=strides,
            factor=2,
            bn_kwargs=bn_kwargs,
        ),
        feature_cropper=CropperQAT(
            size=agent_feature_size,
            strides=strides,
        ),
        state_encoder=StateEncoder(
            bn_kwargs=bn_kwargs,
            channels_seq=[16, 16, 32, 32],
            kernel_size=[6, 6, 6, 5],
            channels_fc=[state_feature_channel],
        ),
        heatmap_head=HeatmapHead(
            bn_kwargs=bn_kwargs,
            input_channel=agent_feature_channel + state_feature_channel,
            channels=[512, 256, 128, 64],
            with_offset=True,
        ),
        sampler=MaxPointsSampler(
            sample_num=10,
            kernel_size=5,
            update_radius=-1,
            transformation_func=fn,
        ),
        endpoint_encoder=EndpointEncoder(
            bn_kwargs=bn_kwargs,
            channels=[16, 32, endpoint_feature_channel],
        ),
        trajectory_head=TrajRegHead(
            bn_kwargs=bn_kwargs,
            input_channel=agent_feature_channel + state_feature_channel,
            endpoint_feature_channel=endpoint_feature_channel,
            state_feature_channel=state_feature_channel,
            output_channel=future_time_step * 2,
            conv_channels=[256, 128, 128, 64, 64],
            paddings=[0] * 5,
            fc_channels=[64, 32, 32],
            confidence=True,
        ),
    )

    sample = {
        "raster_map": torch.randn([2, 3, 896, 896]),
        "rasterized_dynamic_data": torch.randn(
            [2, input_channel - 3, 896, 896]
        ),
        "history_state": torch.randn([6, history_time_step, 2]),
        "passable_mask": torch.ones([6, 1, 176, 176]),
        "future_traj": torch.randn([6, future_time_step, 2]),
        "batch_index": torch.Tensor([0, 0, 0, 0, 1, 1]),
        "history": torch.randn([6, history_time_step, 2]),
    }
    model_outs = model(sample)
    assert model_outs["heatmap"].shape[0] == 6
    assert model_outs["heatmap"].shape[1] == 176
    assert model_outs["heatmap"].shape[2] == 176

    assert model_outs["endpoint"].shape[0] == 6
    assert model_outs["endpoint"].shape[1] == 10
    assert model_outs["endpoint"].shape[2] == 2

    assert model_outs["trajectory_pred"].shape[0] == 6
    assert model_outs["trajectory_pred"].shape[1] == 10 + 5
    assert model_outs["trajectory_pred"].shape[2] == future_time_step
    assert model_outs["trajectory_pred"].shape[3] == 2

    assert model_outs["traj_pred_decoded"].shape[0] == 6
    assert model_outs["traj_pred_decoded"].shape[1] == 10
    assert model_outs["traj_pred_decoded"].shape[2] == future_time_step
    assert model_outs["traj_pred_decoded"].shape[3] == 2

    assert model_outs["confidence"].shape[0] == 6
    assert model_outs["confidence"].shape[1] == 10

    sample_trace = {
        "raster_map": torch.randn([2, 3, 896, 896]),
        "rasterized_dynamic_data": torch.randn(
            [2, input_channel - 3, 896, 896]
        ),
        "history_state": torch.randn([6, 2, 1, history_time_step]),
        "passable_mask": torch.ones([6, 1, 176, 176]),
    }

    rois = [[torch.randn(4, 4), torch.randn(2, 4)]] * len(strides)
    endpoints = [torch.randn(6, 2, 1, 1) for _ in range(5)]
    model(sample_trace, rois, endpoints)
