dataset_ids = dict(
    bev_3d={
        "bev_3d_vehicle": [
            # 目前sd评测数据集需要提供dataset名称以及gt名称, 用来获取bucket路径
            # ("6040511", "ls11v_all_region_vehicle", "vehicle_gt_for_aidieval.json"),
            (
                "6040806",
                "side_img_bev_3d_vehicle",
                "vehicle_gt_for_aidieval_total.json",
            )
        ],
    },
)
