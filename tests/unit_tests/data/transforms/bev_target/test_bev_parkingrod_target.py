from hat.data.transforms.bev_parkingrod_target import (  # noqa
    ANCBEVParkingRodTargetGenerator,
)


def test_bevparkingrod_target():
    max_objs = 200
    len_gt_info = 13
    target_size = (96, 64)
    num_parkingrod_class = 3
    vcs_range = (-12.80, -12.80, 25.60, 12.80)
    rod_weight_dict = {0: 1.0, 1: 1.0, 2: 1.0}
    rod_resolution_m_per_grid = 0.4
    dilate_rate = 1
    data = {
        "annos_bev_parkingrod_obj": {},
        "gt_bev_parking_obj": {},
    }
    data["annos_bev_parkingrod_obj"]["parking_rod"] = [
        [
            32.042121,
            119,
            949514,
            1,
            32.359122,
            114.809680,
            31.725121,
            125.089348,
            1,
            1,
        ],
    ]
    target_keys = [
        "classification_obj",
        "endpoint_offset_obj",
    ]
    grad_keys = [
        "classification_weight_mask",
        "endpoint_offset_weight_mask",
    ]
    use_vis_mask = True
    visable_condition = {
        "visable_threshold": 1,
    }
    vismask_vcsrange_cfg = {
        (192, 128): (
            -12.80,
            -12.80,
            25.60,
            12.80,
        ),  # spatial res is 0.2m/pixel
        (448, 512): (-31.8, -76.8, 102.6, 76.8),  # spatial res is 0.6m/pixel
        (512, 512): (-30.0, -51.2, 72.4, 51.2),  # spatial res is 0.1m/pixel
        (1024, 1024): (-30.0, -51.2, 72.4, 51.2),  # spatial res is 0.1m/pixel
        (2048, 1536): (-51.2, -76.8, 153.6, 76.8),  # spatial res is 0.1m/pixel
        (2048, 1600): (-51.2, -80.0, 153.6, 80.0),  # spatial res is 0.1m/pixel
        # resize vismask in packed lmdb
        (1024, 768): (-51.2, -76.8, 153.6, 76.8),  # spatial res is 0.2m/pixel
        (1024, 800): (-51.2, -80.0, 153.6, 80.0),  # spatial res is 0.2m/pixel
    }
    with_psd = True
    lidar_distance_threshold = 1.4

    target = ANCBEVParkingRodTargetGenerator(
        max_objs=max_objs,
        len_gt_info=len_gt_info,
        target_size=target_size,
        roi_vcs_range=vcs_range,
        rod_weight_dict=rod_weight_dict,
        num_parkingrod_class=num_parkingrod_class,
        resolution_m_per_grid=rod_resolution_m_per_grid,
        dilate_rate=dilate_rate,
        use_vis_mask=use_vis_mask,
        visable_condition=visable_condition,
        with_psd=with_psd,
        vismask_vcsrange_cfg=vismask_vcsrange_cfg,
        lidar_distance_threshold=lidar_distance_threshold,
    )
    data = target(data)

    for tk in target_keys:
        assert tk in data["gt_bev_parkingrod_obj"].keys()
    for gk in grad_keys:
        assert gk in data["gt_bev_parkingrod_obj"].keys()
