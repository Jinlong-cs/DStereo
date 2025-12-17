import math

import numpy as np

from hat.data.transforms.traj_pred.heatmap_based_traj_pred import (
    HWC2CHW,
    AsType,
    CropAndResize,
    DropUselessItems,
    FilteroutUselessAgents,
    RasterizedDynDataGenerator,
    RotateAugmentation,
    StateNormalize,
    TargetGenerator,
)


def test_transform_list():
    max_rotate_angle = math.pi
    raster_map_size = [512, 512]
    heatmap_size = [192, 192]

    def bev2heatmap(input, inverse=False):
        if not inverse:
            output = input / 2
            output = output + np.array(heatmap_size) / 2
        else:
            output = input - np.array(heatmap_size) / 2
            output = output * 2
        return output

    keys = [
        "future",
        "future_traj",
        "future_mask",
        "history",
        "history_mask",
        "history_state",
        "heatmap",
        "offset",
        "passable_mask",
    ]

    transforms = [
        StateNormalize(),
        RasterizedDynDataGenerator(
            max_rotate_angle=max_rotate_angle,
            history_mask_prob=-1,
        ),
        RotateAugmentation(max_rotate_angle=max_rotate_angle),
        CropAndResize(
            downsample_rate=1,
            output_size=raster_map_size,
            translation_aug=False,
        ),
        FilteroutUselessAgents(
            keys_to_filter=keys,
            ignore_cls=[5],
            min_pixel_to_edge=20,
            min_future_frame=10,
            min_history_frame=10,
            shift_threshold=20,
            max_stationary_num=0,
        ),
        TargetGenerator(
            heatmap_size=heatmap_size,
            bev2heatmap=bev2heatmap,
            variance=10,
            with_offset_map=True,
            gen_passable_mask=True,
        ),
        HWC2CHW(keys=["raster_map", "rasterized_dynamic_data", "offset"]),
        AsType(dtype="float32"),
        DropUselessItems(keys=["category"]),
    ]

    history_time_step = 20
    future_time_step = 10
    N = 10
    sample = {
        "raster_map": np.random.uniform(size=[1000, 1000, 3]),
        "history_state": np.random.uniform(size=[N, history_time_step, 2]),
        "history": np.random.uniform(
            size=[N, history_time_step, 2], low=450, high=550
        ),
        "history_mask": np.ones([N, history_time_step]),
        "future": np.random.uniform(size=[N, future_time_step, 2]),
        "future_mask": np.ones([N, future_time_step]),
        "category": np.random.uniform(high=10, size=N).round().astype("int32"),
    }

    for ts in transforms:
        sample = ts(sample)

    for key in keys:
        assert key in sample

    assert sample["raster_map"].shape[0] == 3
    assert sample["raster_map"].shape[1] == raster_map_size[0]
    assert sample["raster_map"].shape[2] == raster_map_size[1]
    assert sample["rasterized_dynamic_data"].shape[0] == history_time_step * 2
    assert sample["rasterized_dynamic_data"].shape[1] == raster_map_size[0]
    assert sample["rasterized_dynamic_data"].shape[2] == raster_map_size[1]
    assert sample["heatmap"].shape[0] == sample["history_state"].shape[0]
    assert sample["heatmap"].shape[1] == heatmap_size[0]
    assert sample["heatmap"].shape[2] == heatmap_size[1]
    assert sample["offset"].shape[0] == sample["history_state"].shape[0]
    assert sample["offset"].shape[1] == 2
    assert sample["offset"].shape[2] == heatmap_size[0]
    assert sample["offset"].shape[3] == heatmap_size[1]
    assert sample["passable_mask"].shape[0] == sample["history_state"].shape[0]
    assert sample["passable_mask"].shape[1] == heatmap_size[0]
    assert sample["passable_mask"].shape[2] == heatmap_size[1]
    assert "category" not in sample
