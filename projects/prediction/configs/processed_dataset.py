# Copyright (c) Horizon Robotics. All rights reserved.


from hat.data.collates.traj_pred_collates import (
    bev_gt_path_function,
    gt_path_func_for_data_pipeline_v3,
    lmdb_path_replace_func_for_data_pipeline_v3,
    lmdb_path_replace_func_for_data_pipeline_v5,
)

DENSETNT_UNITTEST_PREFIX = (
    "11_perception_prediction/02_user/yujun.zhang/hat_unittest"  # noqa: E501
)
HAT_UNITTEST_PREFIX = (
    "11_perception_prediction/02_user/shengzhe.dai/hat_unittest"  # noqa: E501
)
SD_DEMO_DEC2021_PREFIX = "11_perception_prediction/SD_demo_Dec2021"
SD_DEMO_APR2022_PREFIX = "11_perception_prediction/SD_demo_Apr2022"
VW_DEMO_JUN2022_PREFIX = "11_perception_prediction/VW_demo_June2022"
NAVI_DATA_APR2022_PREFIX = "11_perception_prediction/Navi_data_Apr2022"
TRAJ_BEHAV_PILOT_PREFIX = (
    "11_perception_prediction/02_user/zepei.sun/pilot/test"
)
TRAJ_BEHAV_PILOT_ROOT = "11_perception_prediction/02_user/zepei.sun/pilot"
TRAJ_BEHAV_PERREC_PREFIX = (
    "11_perception_prediction/03_pack_package/wenke.wang/TRAJ_DATA"
)
TRAJ_BEHAV_PERREC_PREFIX_2 = (
    "11_perception_prediction/03_pack_package/wenke.wang/TRAJ_DATA_2"
)
TRAJ_BEHAV_NDMFUS_PREFIX = (
    "11_perception_prediction/03_pack_package/wenke.wang/FUS_DATA"
)
TRAJ_BEHAV_PREFIX = "11_perception_prediction/03_pack_package/wenke.wang"
TRAJ_BEHAV_SD_ROOT = "11_perception_prediction/02_user/zepei.sun/SD"
BEHAV_SD_ROOT = "11_perception_prediction/03_pack_package/zepei.sun/sd_dataset"
BEHAV_HIGH_ROOT = (
    "11_perception_prediction/03_pack_package/yuhang.an/sd_dataset"
)

PERREC_June2022_PREFIX = "11_perception_prediction/02_user"
HIGH_FREQ_DATA_PREFIX = (
    "11_perception_prediction/02_user/minrui.xu/temporal_data"
)
LI_MAP_PREFIX = "11_perception_prediction/02_user/shaokai.li"
FPV_PREFIX = "/horizon-bucket/SD_Algorithm/11_perception_prediction/02_user/zhanbo01.li"  # noqa: E501
NUSCENES_DATASET = "11_perception_prediction/02_user/shengzhe.dai/nuscenes_tdt"
OPENSOURCE_DATASET = {
    "nuscenes": {
        "bucket": "J5FSD",
        "basic_train_pkl": f"{NUSCENES_DATASET}/data/pickles/pickle_test_train.pkl",  # noqa: E501
        "basic_val_pkl": f"{NUSCENES_DATASET}/data/pickles/pickle_test_val.pkl",  # noqa: E501
        "map_path_func": None,
        "data_token2path_mapping": f"{NUSCENES_DATASET}/maps/nusc_map_label",
    }
}
SD_DEMO_DATASET = {
    "multi-task_ndmfus": {
        "bucket": "SD_Algorithm",
        "NDMFUS_train_pkl": f"{TRAJ_BEHAV_SD_ROOT}/concat_pkl/20_0.8.4_0_0.0.1_13_71_4_71_multitask_urban/train_136497_ndmfus.pkl",  # noqa: E501
        "NDMFUS_val_pkl": f"{TRAJ_BEHAV_SD_ROOT}/concat_pkl/20_0.8.4_0_0.0.1_13_71_4_71_multitask_urban/test_40069_ndmfus.pkl",  # noqa: E501
        "map_path_func": gt_path_func_for_data_pipeline_v3,  # noqa: E501
        "data_token2path_mapping": {
            "H3165_20220421": [
                f"{TRAJ_BEHAV_NDMFUS_PREFIX}",
                "rasterized_map/4_71/bev_GT_bevGT_hde_test_v2.1",
            ],
            "H3165_20220422": [
                f"{TRAJ_BEHAV_NDMFUS_PREFIX}",
                "rasterized_map/4_71/bev_GT_bevGT_hde_test_v2.1",
            ],
            "H3165_20220423": [
                f"{TRAJ_BEHAV_NDMFUS_PREFIX}",
                "rasterized_map/4_71/bev_GT_bevGT_hde_test_v2.1",
            ],
            "H3165_20220424": [
                f"{TRAJ_BEHAV_NDMFUS_PREFIX}",
                "rasterized_map/4_71/bev_GT_bevGT_hde_test_v2.1",
            ],
            "H3165_20220425": [
                f"{TRAJ_BEHAV_NDMFUS_PREFIX}",
                "rasterized_map/4_71/bev_GT_bevGT_hde_test_v2.1",
            ],
            "H3165_20220426": [
                f"{TRAJ_BEHAV_NDMFUS_PREFIX}",
                "rasterized_map/4_71/bev_GT_bevGT_hde_test_v2.1",
            ],
        },
    },
    "multi-task_perrec": {
        "bucket": "SD_Algorithm",
        "basic_train_pkl": f"{TRAJ_BEHAV_SD_ROOT}/concat_pkl/all_scene_dataset/train_812122_all_scene_50ctxframes.pkl",  # noqa: E501
        "basic_val_pkl": f"{TRAJ_BEHAV_SD_ROOT}/concat_pkl/all_scene_dataset/test_84322_all_scene_50ctxframes.pkl",  # noqa: E501
        "map_path_func": gt_path_func_for_data_pipeline_v3,  # noqa: E501
        "data_token2path_mapping": {
            "H3165_20220509": [
                f"{TRAJ_BEHAV_PERREC_PREFIX}",
                "rasterized_map/3_70/bev_GT_bevGT_hde_test_v2.1",
            ],
            "H3165_20220421": [
                f"{TRAJ_BEHAV_PERREC_PREFIX}",
                "rasterized_map/3_70/bev_GT_bevGT_hde_test_v2.1",
            ],
            "H3165_20220422": [
                f"{TRAJ_BEHAV_PERREC_PREFIX}",
                "rasterized_map/3_70/bev_GT_bevGT_hde_test_v2.1",
            ],
            "H3165_20220423": [
                f"{TRAJ_BEHAV_PERREC_PREFIX}",
                "rasterized_map/3_70/bev_GT_bevGT_hde_test_v2.1",
            ],
            "H3165_20220424": [
                f"{TRAJ_BEHAV_PERREC_PREFIX}",
                "rasterized_map/3_70/bev_GT_bevGT_hde_test_v2.1",
            ],
            "H3165_20220425": [
                f"{TRAJ_BEHAV_PERREC_PREFIX}",
                "rasterized_map/3_70/bev_GT_bevGT_hde_test_v2.1",
            ],
            "H3165_20220426": [
                f"{TRAJ_BEHAV_PERREC_PREFIX}",
                "rasterized_map/3_70/bev_GT_bevGT_hde_test_v2.1",
            ],
            "H3165_20220427": [
                f"{TRAJ_BEHAV_PERREC_PREFIX}",
                "rasterized_map/3_70/bev_GT_bevGT_hde_test_v2.1",
            ],
            "H3165_20220428": [
                f"{TRAJ_BEHAV_PERREC_PREFIX}",
                "rasterized_map/3_70/bev_GT_bevGT_hde_test_v2.1",
            ],
            "H3165_20220429": [
                f"{TRAJ_BEHAV_PERREC_PREFIX}",
                "rasterized_map/3_70/bev_GT_bevGT_hde_test_v2.1",
            ],
            "H3165_20220502": [
                f"{TRAJ_BEHAV_PERREC_PREFIX}",
                "rasterized_map/3_70/bev_GT_bevGT_hde_test_v2.1",
            ],
            "H3165_20220503": [
                f"{TRAJ_BEHAV_PERREC_PREFIX}",
                "rasterized_map/3_70/bev_GT_bevGT_hde_test_v2.1",
            ],
            "H3165_20220504": [
                f"{TRAJ_BEHAV_PERREC_PREFIX}",
                "rasterized_map/3_70/bev_GT_bevGT_hde_test_v2.1",
            ],
            "H3165_20220505": [
                f"{TRAJ_BEHAV_PERREC_PREFIX}",
                "rasterized_map/3_70/bev_GT_bevGT_hde_test_v2.1",
            ],
            "H3165_20220506": [
                f"{TRAJ_BEHAV_PERREC_PREFIX}",
                "rasterized_map/3_70/bev_GT_bevGT_hde_test_v2.1",
            ],
            "H3165_20220507": [
                f"{TRAJ_BEHAV_PERREC_PREFIX}",
                "rasterized_map/3_70/bev_GT_bevGT_hde_test_v2.1",
            ],
            "UT1R3_20220719": [
                f"{TRAJ_BEHAV_PERREC_PREFIX}",
                "rasterized_map/3_70/bev_GT_bevGT_hde_test_v2.1",
            ],
            "UT1R3_20220720": [
                f"{TRAJ_BEHAV_PERREC_PREFIX}",
                "rasterized_map/3_70/bev_GT_bevGT_hde_test_v2.1",
            ],
            "UT1R3_20220721": [
                f"{TRAJ_BEHAV_PERREC_PREFIX}",
                "rasterized_map/3_70/bev_GT_bevGT_hde_test_v2.1",
            ],
            "UT0Q9_20220701": [
                f"{TRAJ_BEHAV_PERREC_PREFIX}",
                "rasterized_map/3_70/bev_GT_bevGT_hde_test_v2.1",
            ],
            "UT0Q9_20220702": [
                f"{TRAJ_BEHAV_PERREC_PREFIX}",
                "rasterized_map/3_70/bev_GT_bevGT_hde_test_v2.1",
            ],
            "UT0Q9_20220704": [
                f"{TRAJ_BEHAV_PERREC_PREFIX}",
                "rasterized_map/3_70/bev_GT_bevGT_hde_test_v2.1",
            ],
            "UT0Q9_20220706": [
                f"{TRAJ_BEHAV_PERREC_PREFIX}",
                "rasterized_map/3_70/bev_GT_bevGT_hde_test_v2.1",
            ],
            "UT0Q9_20220707": [
                f"{TRAJ_BEHAV_PERREC_PREFIX}",
                "rasterized_map/3_70/bev_GT_bevGT_hde_test_v2.1",
            ],
            "UT0Q9_20220708": [
                f"{TRAJ_BEHAV_PERREC_PREFIX}",
                "rasterized_map/3_70/bev_GT_bevGT_hde_test_v2.1",
            ],
            "UT0Q9_20220709": [
                f"{TRAJ_BEHAV_PERREC_PREFIX}",
                "rasterized_map/3_70/bev_GT_bevGT_hde_test_v2.1",
            ],
            "UT0Q9_20220710": [
                f"{TRAJ_BEHAV_PERREC_PREFIX}",
                "rasterized_map/3_70/bev_GT_bevGT_hde_test_v2.1",
            ],
            "UT0Q9_20220711": [
                f"{TRAJ_BEHAV_PERREC_PREFIX}",
                "rasterized_map/3_70/bev_GT_bevGT_hde_test_v2.1",
            ],
            "UT0Q9_20220712": [
                f"{TRAJ_BEHAV_PERREC_PREFIX}",
                "rasterized_map/3_70/bev_GT_bevGT_hde_test_v2.1",
            ],
            "UT0Q9_20220714": [
                f"{TRAJ_BEHAV_PERREC_PREFIX}",
                "rasterized_map/3_70/bev_GT_bevGT_hde_test_v2.1",
            ],
            "UT0Q9_20220715": [
                f"{TRAJ_BEHAV_PERREC_PREFIX}",
                "rasterized_map/3_70/bev_GT_bevGT_hde_test_v2.1",
            ],
            "UT0Q9_20220717": [
                f"{TRAJ_BEHAV_PERREC_PREFIX}",
                "rasterized_map/3_70/bev_GT_bevGT_hde_test_v2.1",
            ],
            "UT0Q9_20220718": [
                f"{TRAJ_BEHAV_PERREC_PREFIX}",
                "rasterized_map/3_70/bev_GT_bevGT_hde_test_v2.1",
            ],
            "UT0Q9_20220719": [
                f"{TRAJ_BEHAV_PERREC_PREFIX}",
                "rasterized_map/3_70/bev_GT_bevGT_hde_test_v2.1",
            ],
            "UT0Q9_20220721": [
                f"{TRAJ_BEHAV_PERREC_PREFIX}",
                "rasterized_map/3_70/bev_GT_bevGT_hde_test_v2.1",
            ],
            "UT3J5_20220607": [
                f"{TRAJ_BEHAV_PERREC_PREFIX}",
                "rasterized_map/3_70/bev_GT_bevGT_hde_test_v2.1",
            ],
            "UT3J5_20220608": [
                f"{TRAJ_BEHAV_PERREC_PREFIX}",
                "rasterized_map/3_70/bev_GT_bevGT_hde_test_v2.1",
            ],
            "UT3J5_20220712": [
                f"{TRAJ_BEHAV_PERREC_PREFIX}",
                "rasterized_map/3_70/bev_GT_bevGT_hde_test_v2.1",
            ],
            "UT3J5_20220713": [
                f"{TRAJ_BEHAV_PERREC_PREFIX}",
                "rasterized_map/3_70/bev_GT_bevGT_hde_test_v2.1",
            ],
            "UT3J5_20220714": [
                f"{TRAJ_BEHAV_PERREC_PREFIX}",
                "rasterized_map/3_70/bev_GT_bevGT_hde_test_v2.1",
            ],
            "UT3J5_20220920": [
                f"{TRAJ_BEHAV_PERREC_PREFIX}",
                "rasterized_map/3_70/bev_GT_bevGT_hde_test_v2.1",
            ],
            "UT3J5_20220921": [
                f"{TRAJ_BEHAV_PERREC_PREFIX}",
                "rasterized_map/3_70/bev_GT_bevGT_hde_test_v2.1",
            ],
            "UT3J5_20220922": [
                f"{TRAJ_BEHAV_PERREC_PREFIX}",
                "rasterized_map/3_70/bev_GT_bevGT_hde_test_v2.1",
            ],
            "UT3J5_20220923": [
                f"{TRAJ_BEHAV_PERREC_PREFIX}",
                "rasterized_map/3_70/bev_GT_bevGT_hde_test_v2.1",
            ],
            "UT0F3_20221019": [
                f"{TRAJ_BEHAV_PERREC_PREFIX}",
                "rasterized_map/3_70/bev_GT_bevGT_hde_test_v2.1",
            ],
            "UT0F3_20221020": [
                f"{TRAJ_BEHAV_PERREC_PREFIX}",
                "rasterized_map/3_70/bev_GT_bevGT_hde_test_v2.1",
            ],
            "H3165_20220325": [
                f"{TRAJ_BEHAV_PERREC_PREFIX}",
                "rasterized_map/3_70/bev_GT_bevGT_hde_test_v2.1",
            ],
            "H3165_20220228": [
                f"{TRAJ_BEHAV_PERREC_PREFIX}",
                "rasterized_map/3_70/bev_GT_bevGT_hde_test_v2.1",
            ],
            "H3165_20220227": [
                f"{TRAJ_BEHAV_PERREC_PREFIX}",
                "rasterized_map/3_70/bev_GT_bevGT_hde_test_v2.1",
            ],
            "H3165_20220226": [
                f"{TRAJ_BEHAV_PERREC_PREFIX}",
                "rasterized_map/3_70/bev_GT_bevGT_hde_test_v2.1",
            ],
            "H3165_20220301": [
                f"{TRAJ_BEHAV_PERREC_PREFIX}",
                "rasterized_map/3_70/bev_GT_bevGT_hde_test_v2.1",
            ],
            "UT0Q9_20220401": [
                f"{TRAJ_BEHAV_PERREC_PREFIX}",
                "rasterized_map/3_70/bev_GT_bevGT_hde_test_v2.1",
            ],
            "UT0Q9_20220527": [
                f"{TRAJ_BEHAV_PERREC_PREFIX}",
                "rasterized_map/3_70/bev_GT_bevGT_hde_test_v2.1",
            ],
            "UT0Q9_20220528": [
                f"{TRAJ_BEHAV_PERREC_PREFIX}",
                "rasterized_map/3_70/bev_GT_bevGT_hde_test_v2.1",
            ],
            "UT0Q9_20220529": [
                f"{TRAJ_BEHAV_PERREC_PREFIX}",
                "rasterized_map/3_70/bev_GT_bevGT_hde_test_v2.1",
            ],
            "UT0Q9_20220530": [
                f"{TRAJ_BEHAV_PERREC_PREFIX}",
                "rasterized_map/3_70/bev_GT_bevGT_hde_test_v2.1",
            ],
            "UT0Q9_20220531": [
                f"{TRAJ_BEHAV_PERREC_PREFIX}",
                "rasterized_map/3_70/bev_GT_bevGT_hde_test_v2.1",
            ],
            "UT0Q9_20220601": [
                f"{TRAJ_BEHAV_PERREC_PREFIX}",
                "rasterized_map/3_70/bev_GT_bevGT_hde_test_v2.1",
            ],
            "UT0Q9_20220314": [
                f"{TRAJ_BEHAV_PERREC_PREFIX}",
                "rasterized_map/3_70/bev_GT_bevGT_hde_test_v2.1",
            ],
            "UT1R3_20230207": [
                f"{TRAJ_BEHAV_PERREC_PREFIX_2}",
                "rasterized_map/3_70/bev_GT_bevGT_hde_test_v2.1",
            ],
            "UT1R3_20230214": [
                f"{TRAJ_BEHAV_PERREC_PREFIX_2}",
                "rasterized_map/3_70/bev_GT_bevGT_hde_test_v2.1",
            ],
            "UT1R3_20230208": [
                f"{TRAJ_BEHAV_PERREC_PREFIX_2}",
                "rasterized_map/3_70/bev_GT_bevGT_hde_test_v2.1",
            ],
        },
        "HWP_train_pkl": f"{TRAJ_BEHAV_SD_ROOT}/concat_pkl/20_0.8.4_0_0.0.1_12_70_3_70_HWP_50ctxframes_dist_restriction/train_285567_UT3J5_50ctxframes_dist_restriction.pkl",  # noqa: E501
        "HWP_val_pkl": f"{TRAJ_BEHAV_SD_ROOT}/concat_pkl/20_0.8.4_0_0.0.1_12_70_3_70_HWP_50ctxframes_dist_restriction/test_18670_UT3J5_50ctxframes_dist_restriction.pkl",  # noqa: E501
        "HWP_viz_pkl": f"{TRAJ_BEHAV_PERREC_PREFIX}/UT3J5_20220607_D/20220607-152904_855/traj_process_multitask/20_0.8.4_0_0.0.1_12_70_3_70_50contextframes_dist_restriction/pickles/pickle_test_train.pkl",  # noqa: E501
        "UP_train_pkl": f"{TRAJ_BEHAV_SD_ROOT}/concat_pkl/20_0.8.4_1_0.0.1_12_70_3_70_mutitask_urban_50ctxframes_dist_restriction/train_526555_H3165_50ctxframes_dist_restriction.pkl",  # noqa: E501
        "UP_val_pkl": f"{TRAJ_BEHAV_SD_ROOT}/concat_pkl/20_0.8.4_1_0.0.1_12_70_3_70_mutitask_urban_50ctxframes_dist_restriction/test_65652_H3165_50ctxframes_dist_restriction.pkl",  # noqa: E501
        "UP_viz_pkl": f"{TRAJ_BEHAV_PERREC_PREFIX}/H3165_20220421_D/20220421-113209_224/traj_process_multitask/20_0.8.4_1_0.0.1_12_70_3_70_50contextframes_dist_restriction/pickles/pickle_test_train.pkl",  # noqa: E501
        "full_scene_train_pkl": f"{TRAJ_BEHAV_SD_ROOT}/concat_pkl/all_scene_dataset/train_1276037_add_pilot_data.pkl",  # noqa: E501
        "full_scene_SH_train_pkl": f"{TRAJ_BEHAV_SD_ROOT}/concat_pkl/all_scene_dataset/train_1283202_add_shanghai.pkl",  # noqa: E501
        "full_scene_SH_pilot_train_pkl": f"{TRAJ_BEHAV_SD_ROOT}/concat_pkl/all_scene_dataset/train_1747117_add_pilot_shanghai.pkl",  # noqa: E501
        "full_scene_val_pkl": f"{TRAJ_BEHAV_SD_ROOT}/concat_pkl/all_scene_dataset/test_84322_all_scene_50ctxframes_dist_restriction.pkl",  # noqa: E501
        "full_scene_SH_val_pkl": f"{TRAJ_BEHAV_SD_ROOT}/concat_pkl/all_scene_dataset/test_167568_add_shanghai.pkl",  # noqa: E501
        "SH_train_pkl": f"{TRAJ_BEHAV_SD_ROOT}/concat_pkl/20_3.3.0_0_0.0.1_12_70_3_70_SH/train_471080_SH.pkl",  # noqa: E501,
        "SH_val_pkl": f"{TRAJ_BEHAV_SD_ROOT}/concat_pkl/20_3.3.0_0_0.0.1_12_70_3_70_SH/test_83246_SH.pkl",  # noqa: E501,
        "perrec_compare_train_pkl": f"{TRAJ_BEHAV_SD_ROOT}/concat_pkl/20_0.8.4_1_0.0.1_12_70_3_70_mutitask_urban/train_155762_perrec_compare.pkl",  # noqa: E501
        "perrec_compare_val_pkl": f"{TRAJ_BEHAV_SD_ROOT}/concat_pkl/20_0.8.4_1_0.0.1_12_70_3_70_mutitask_urban/test_46829_perrec_compare.pkl",  # noqa: E501
        "behav_full_train": f"{BEHAV_SD_ROOT}/behav_full_dataset/behav_full_train_1042555samples_online.pkl",  # noqa: E501,
        "behav_full_val": f"{BEHAV_SD_ROOT}/behav_full_dataset/behav_full_val_287425samples_online_tag_array.pkl",  # noqa: E501,
        "behav_BJ_val": f"{BEHAV_SD_ROOT}/behav_full_dataset/behav_BJ_val_99496samples_online_tag_array.pkl",  # noqa: E501,
        "behav_SH_val": f"{BEHAV_SD_ROOT}/behav_full_dataset/behav_SH_val_187929samples_online_tag_array.pkl",  # noqa: E501,
        "behav_viz": f"{TRAJ_BEHAV_PERREC_PREFIX_2}/UT1R3_20230204_D/20230204-153942_098/20_3.3.0_0_0.0.1_12_70_3_70_perrec_low/pickles/pickle_behav_seqdf.pkl",  # noqa: E501,
        "behav_train": f"{BEHAV_SD_ROOT}/behav_dataset_train",
        "behav_val": f"{BEHAV_SD_ROOT}/behav_dataset_val",
        "full_train": "/horizon-bucket/SD_Algorithm/11_perception_prediction/03_pack_package/wenke.wang/DATASET_PATH/PickledTdtDatasetV2/train_1203331.pkl",  # noqa: E501
        "full_val": "/horizon-bucket/SD_Algorithm/11_perception_prediction/03_pack_package/wenke.wang/DATASET_PATH/PickledTdtDatasetV2/val_167753.pkl",  # noqa: E501
        "small_train_densetnt": "/horizon-bucket/SD_Algorithm/11_perception_prediction/03_pack_package/wenke.wang/DATASET_PATH/PickledTdtDatasetV2/train_200293.pkl",  # noqa: E501
        "small_test_densetnt": "/horizon-bucket/SD_Algorithm/11_perception_prediction/03_pack_package/wenke.wang/DATASET_PATH/PickledTdtDatasetV2/val_33961.pkl",  # noqa: E501
        "10hz_train": "/horizon-bucket/SD_Algorithm/11_perception_prediction/03_pack_package/wenke.wang/DATASET_PATH/TenHZPkl/train_181078.pkl",  # noqa: E501
        "10hz_val": "/horizon-bucket/SD_Algorithm/11_perception_prediction/03_pack_package/wenke.wang/DATASET_PATH/TenHZPkl/val_27820.pkl",  # noqa: E501
        "fus_train": "/horizon-bucket/SD_Algorithm/11_perception_prediction/03_pack_package/wenke.wang/DATASET_PATH/PickledTdtDatasetV2/fus_train_276517.pkl",  # noqa: E501
        "fus_test": "/horizon-bucket/SD_Algorithm/11_perception_prediction/03_pack_package/wenke.wang/DATASET_PATH/PickledTdtDatasetV2/fus_test_77464.pkl",  # noqa: E501
        "densetnt_tiny_train": "/horizon-bucket/SD_Algorithm/11_perception_prediction/03_pack_package/wenke.wang/DATASET_PATH/PickledTdtDatasetV2/densetnt_tiny_train.pkl",  # noqa: E501
        "densetnt_tiny_test": "/horizon-bucket/SD_Algorithm/11_perception_prediction/03_pack_package/wenke.wang/DATASET_PATH/PickledTdtDatasetV2/densetnt_tiny_test.pkl",  # noqa: E501
        "full_val_sh": "/horizon-bucket/SD_Algorithm/11_perception_prediction/03_pack_package/wenke.wang/DATASET_PATH/PickledTdtDatasetV2/val_SH_82170.pkl",  # noqa: E501
        "full_val_bj": "/horizon-bucket/SD_Algorithm/11_perception_prediction/03_pack_package/wenke.wang/DATASET_PATH/PickledTdtDatasetV2/val_BJ_85583.pkl",  # noqa: E501
        "perrec_train": "/horizon-bucket/SD_Algorithm/11_perception_prediction/03_pack_package/wenke.wang/DATASET_PATH/PickledTdtDatasetV2/compare_perrec_train_274285.pkl",  # noqa: E501
        "perrec_test": "/horizon-bucket/SD_Algorithm/11_perception_prediction/03_pack_package/wenke.wang/DATASET_PATH/PickledTdtDatasetV2/compare_perrec_test_77011.pkl",  # noqa: E501
        "full_fus_train": "/horizon-bucket/SD_Algorithm/11_perception_prediction/03_pack_package/wenke.wang/DATASET_PATH/PickledTdtDatasetV2/full_fus_train_1624149.pkl",  # noqa: E501
        "full_fus_val": "/horizon-bucket/SD_Algorithm/11_perception_prediction/03_pack_package/wenke.wang/DATASET_PATH/PickledTdtDatasetV2/full_fus_val_113418.pkl",  # noqa: E501
        "full_fus_hdmap_train": "/horizon-bucket/SD_Algorithm/11_perception_prediction/03_pack_package/wenke.wang/DATASET_PATH/PickledTdtDatasetV2/full_fus_hdmap_train_75637.pkl",  # noqa: E501
        "full_fus_hdmap_val": "/horizon-bucket/SD_Algorithm/11_perception_prediction/03_pack_package/wenke.wang/DATASET_PATH/PickledTdtDatasetV2/full_fus_hdmap_val_63791.pkl",  # noqa: E501
        "his5_fut10_10hz_train": "/horizon-bucket/SD_Algorithm/11_perception_prediction/03_pack_package/wenke.wang/DATASET_PATH/TenHZPkl/train_his5_fut10_181078.pkl",  # noqa: E501
        "his5_fut10_10hz_val": "/horizon-bucket/SD_Algorithm/11_perception_prediction/03_pack_package/wenke.wang/DATASET_PATH/TenHZPkl/val_his5_fut10_27820.pkl",  # noqa: E501
        "perrec_low_train_behav": f"{BEHAV_HIGH_ROOT}/behav_dataset/behav_full_train_1048195samples_lowfreq.pkl",  # noqa: E501
        "perrec_low_val_behav": f"{BEHAV_HIGH_ROOT}/behav_dataset/behav_full_val_288393samples_lowfreq.pkl",  # noqa: E501
        "perrec_high_train_behav": f"{BEHAV_HIGH_ROOT}/behav_dataset/behav_full_train_1060440samples_highfreq.pkl",  # noqa: E501
        "perrec_high_val_behav": f"{BEHAV_HIGH_ROOT}/behav_dataset/behav_full_val_289108samples_highfreq.pkl",  # noqa: E501
        "data_mining_balance_train": "/horizon-bucket/SD_Algorithm/11_perception_prediction/03_pack_package/wenke.wang/DATASET_PATH/PickledTdtDatasetV2DataMining/train_197573.pkl",  # noqa: E501
        "data_mining_balance_val": "/horizon-bucket/SD_Algorithm/11_perception_prediction/03_pack_package/wenke.wang/DATASET_PATH/PickledTdtDatasetV2DataMining/val_18385.pkl",  # noqa: E501
        "full_fus_and_data_mining_balance_train": "/horizon-bucket/SD_Algorithm/11_perception_prediction/03_pack_package/wenke.wang/DATASET_PATH/FULL_FUS_AND_INTERACTIVE/train_1821722.pkl",  # noqa: E501
        "full_fus_and_data_mining_balance_val": "/horizon-bucket/SD_Algorithm/11_perception_prediction/03_pack_package/wenke.wang/DATASET_PATH/FULL_FUS_AND_INTERACTIVE/val_131803.pkl",  # noqa: E501
        "full_fus_data_train": "/horizon-bucket/SD_Algorithm/11_perception_prediction/02_user/shiqi.tan/traj_dataset_pkl_path/fus_version-hdmap_dataset_version1/fus_data_train_4375585.pkl",  # noqa: E501
        "full_fus_data_val": "/horizon-bucket/SD_Algorithm/11_perception_prediction/02_user/shiqi.tan/traj_dataset_pkl_path/fus_version-hdmap_dataset_version1/fus_data_val_440762.pkl",  # noqa: E501
        "full_fus_data_viz": "/horizon-bucket/SD_Algorithm/11_perception_prediction/02_user/shiqi.tan/traj_dataset_pkl_path/fus_version-hdmap_dataset_version1/fus_data_viz2.pkl",  # noqa: E501
        "full_fus_data_train_657w": "/horizon-bucket/SD_Algorithm/11_perception_prediction/02_user/shiqi.tan/traj_dataset_pkl_path/fus_version-hdmap_dataset_version1/fus_data_train_6579136.pkl",  # noqa: E501
        "full_fus_data_train_657w_6s": "/horizon-bucket/SD_Algorithm/11_perception_prediction/02_user/yujun.zhang/PickledTdtDatasetTest/fus_full_train_his6_600w.pkl",  # noqa: E501
        "full_fus_data_val_44w_6s": "/horizon-bucket/SD_Algorithm/11_perception_prediction/02_user/yujun.zhang/PickledTdtDatasetTest/fus_full_val_his6_600w.pkl",  # noqa: E501
        "fus_val_full_his6": "/horizon-bucket/SD_Algorithm/11_perception_prediction/02_user/yujun.zhang/PickledTdtDatasetTest/fus_val_full_VRU_his6.pkl",  # noqa: E501
        "fus_train_full_his6": "/horizon-bucket/SD_Algorithm/11_perception_prediction/02_user/yujun.zhang/PickledTdtDatasetTest/fus_train_full_his6.pkl",  # noqa: E501
        "small_fus_train": "/horizon-bucket/SD_Algorithm/11_perception_prediction/02_user/yujun.zhang/PickledTdtDatasetTest/fus_train_his2.pkl",  # noqa: E501
        "small_fus_val": "/horizon-bucket/SD_Algorithm/11_perception_prediction/02_user/yujun.zhang/PickledTdtDatasetTest/fus_val_his2.pkl",  # noqa: E501
    },
    "behav_perrec": {
        "bucket": "SD_Algorithm",
        "map_path_func": gt_path_func_for_data_pipeline_v3,  # noqa: E501
        "data_token2path_mapping": {
            "H3165_20220509": [
                f"{TRAJ_BEHAV_PERREC_PREFIX}",
                "rasterized_map/3_70/bev_GT_bevGT_hde_test_v2.1",
            ],
            "H3165_20220421": [
                f"{TRAJ_BEHAV_PERREC_PREFIX}",
                "rasterized_map/3_70/bev_GT_bevGT_hde_test_v2.1",
            ],
            "H3165_20220422": [
                f"{TRAJ_BEHAV_PERREC_PREFIX}",
                "rasterized_map/3_70/bev_GT_bevGT_hde_test_v2.1",
            ],
            "H3165_20220423": [
                f"{TRAJ_BEHAV_PERREC_PREFIX}",
                "rasterized_map/3_70/bev_GT_bevGT_hde_test_v2.1",
            ],
            "H3165_20220425": [
                "11_perception_prediction/03_pack_package/wenke.wang/September_2022",  # noqa: E501
                "rasterized_map/3_70/bev_GT_bevGT_hde_test_v2.1",
            ],
            "H3165_20220426": [
                "11_perception_prediction/03_pack_package/wenke.wang/September_2022",  # noqa: E501
                "rasterized_map/3_70/bev_GT_bevGT_hde_test_v2.1",
            ],
            "H3165_20220427": [
                f"{TRAJ_BEHAV_PERREC_PREFIX}",
                "rasterized_map/3_70/bev_GT_bevGT_hde_test_v2.1",
            ],
            "H3165_20220428": [
                "11_perception_prediction/03_pack_package/wenke.wang/September_2022",  # noqa: E501
                "rasterized_map/3_70/bev_GT_bevGT_hde_test_v2.1",
            ],
            "H3165_20220429": [
                "11_perception_prediction/03_pack_package/wenke.wang/September_2022",  # noqa: E501
                "rasterized_map/3_70/bev_GT_bevGT_hde_test_v2.1",
            ],
            "H3165_20220502": [
                "rasterized_map/3_70/bev_GT_bevGT_hde_test_v2.1",
            ],
            "H3165_20220503": [
                "rasterized_map/3_70/bev_GT_bevGT_hde_test_v2.1",
            ],
            "H3165_20220504": [
                "11_perception_prediction/03_pack_package/wenke.wang/September_2022",  # noqa: E501
                "rasterized_map/3_70/bev_GT_bevGT_hde_test_v2.1",
            ],
        },
        "experment_train_pkl_v1": f"{TRAJ_BEHAV_SD_ROOT}/MultiTask_Concat_Dataset/H3165_PERREC_experiments_datasets/train_79906_H3165_0421-0422.pkl",  # noqa: E501
        "experment_train_pkl_v2": f"{TRAJ_BEHAV_SD_ROOT}/MultiTask_Concat_Dataset/H3165_PERREC_experiments_datasets/train_114486_H3165_0421-0422_oversample.pkl",  # noqa: E501
        "handcrafted_val_pkl": f"{TRAJ_BEHAV_SD_ROOT}/MultiTask_Concat_Dataset/H3165_highquality_eval/pickle/pickle_test_train.pkl",  # noqa: E501
    },
    "Lidar128": {  # SD_demo_Dec2021
        # Configuration:
        # -- Lidar: pandar 128, use 128 float model (integrated to EVS)
        # -- Lidar post processing: use EVS tracking.
        # -- The dataset is processed by pipeline v2.
        # Note: we will set the newest dataset as the "basic_train_pkl"
        "bucket": "J5FSD",
        "basic_train_pkl": f"{SD_DEMO_DEC2021_PREFIX}/H3165_float128_v2_10days/pickle_test_train.pkl",  # noqa: E501
        "basic_val_pkl": f"{SD_DEMO_DEC2021_PREFIX}/H3165_float128_v2_10days/pickle_test_val.pkl",  # noqa: E501
        "map_path_func": bev_gt_path_function,  # noqa: E501
        "data_token2path_mapping": {
            "DG201_20210513": [
                f"{SD_DEMO_DEC2021_PREFIX}/superdrive_bev",
                "bev_GT_bevgt_loc_map_3_5_1020_odom_1.0.2_add_vl_1.3.0",
            ],
            "DG201_20210514": [
                f"{SD_DEMO_DEC2021_PREFIX}/superdrive_bev",
                "bev_GT_bevgt_loc_map_3_5_1020_odom_1.0.2_add_vl_1.3.0",
            ],
            "DG201_20210626": [
                f"{SD_DEMO_DEC2021_PREFIX}/superdrive_bev",
                "bev_GT_bevgt_loc_map_3_5_1020_odom_1.0.2_add_vl_1.3.0",
            ],
            "DG201_20210704": [
                f"{SD_DEMO_DEC2021_PREFIX}/superdrive_bev",
                "bev_GT_bevgt_loc_map_3_5_1020_odom_1.0.2_add_vl_1.3.0",
            ],
            "DG201_20210719": [
                f"{SD_DEMO_DEC2021_PREFIX}/superdrive_bev",
                "bev_GT_bevgt_loc_map_3_5_1020_odom_1.0.2_add_vl_1.3.0",
            ],
            "DG201_20210726": [
                f"{SD_DEMO_DEC2021_PREFIX}/superdrive_bev",
                "bev_GT_bevgt_loc_map_3_5_1020_odom_1.0.2_add_vl_1.3.0",
            ],
            "DG202_20210601": [
                f"{SD_DEMO_DEC2021_PREFIX}/superdrive_bev",
                "bev_GT_bevgt_loc_map_3_5_1020_odom_1.0.2_add_vl_1.3.0",
            ],
            "DG202_20210603": [
                f"{SD_DEMO_DEC2021_PREFIX}/superdrive_bev",
                "bev_GT_bevgt_loc_map_3_5_1020_odom_1.0.2_add_vl_1.3.0",
            ],
            "H3165_20211029": [
                f"{SD_DEMO_DEC2021_PREFIX}/superdrive_bev",
                "bev_GT_bevgt_loc_map_3_5_1020_odom_1.0.3_add_vl_1.3.0",
            ],
            "H3165_20211101": [
                f"{SD_DEMO_DEC2021_PREFIX}/superdrive_bev",
                "bev_GT_bevgt_loc_map_3_5_1020_odom_1.0.5_add_vl_1.3.0",
            ],
            "H3165_20211103": [
                f"{SD_DEMO_DEC2021_PREFIX}/superdrive_bev",
                "bev_GT_bevgt_loc_map_3_5_1020_odom_1.0.3_add_vl_1.3.0",
            ],
            "H3165_20211104": [
                f"{SD_DEMO_DEC2021_PREFIX}/superdrive_bev",
                "bev_GT_bevgt_loc_map_3_5_1207_odom_1.0.4_add_vl_1.3.0",
            ],
            "H3165_20211105": [
                f"{SD_DEMO_DEC2021_PREFIX}/superdrive_bev",
                "bev_GT_bevgt_loc_map_3_5_1020_odom_1.0.3_add_vl_1.3.0",
            ],
            "H3165_20211111": [
                f"{SD_DEMO_DEC2021_PREFIX}/superdrive_bev",
                "bev_GT_bevgt_loc_map_3_5_1207_odom_1.0.4_add_vl_1.3.0",
            ],
            "H3165_20211112": [
                f"{SD_DEMO_DEC2021_PREFIX}/superdrive_bev",
                "bev_GT_bevgt_loc_map_3_5_1207_odom_1.0.4_add_vl_1.3.0",
            ],
            "H3165_20211116": [
                f"{SD_DEMO_DEC2021_PREFIX}/superdrive_bev",
                "bev_GT_bevgt_loc_map_3_5_1207_odom_1.0.3_add_vl_1.3.0",
            ],
            "H3165_20211118": [
                f"{SD_DEMO_DEC2021_PREFIX}/superdrive_bev",
                "bev_GT_bevgt_loc_map_3_5_1020_odom_1.0.3_add_vl_1.3.0",
            ],
        },
        "navinet_data_token2path_mapping": {
            "DG201_20210513": [
                f"{SD_DEMO_DEC2021_PREFIX}/superdrive_bev",
                "navinetmap_bevgt_loc_map_3_5_1020_odom_1.0.2_add_vl_1.3.0",
            ],
            "DG201_20210514": [
                f"{SD_DEMO_DEC2021_PREFIX}/superdrive_bev",
                "navinetmap_bevgt_loc_map_3_5_1020_odom_1.0.2_add_vl_1.3.0",
            ],
            "DG201_20210704": [
                f"{SD_DEMO_DEC2021_PREFIX}/superdrive_bev",
                "navinetmap_bevgt_loc_map_3_5_1020_odom_1.0.2_add_vl_1.3.0",
            ],
            "DG202_20210601": [
                f"{SD_DEMO_DEC2021_PREFIX}/superdrive_bev",
                "navinetmap_bevgt_loc_map_3_5_1020_odom_1.0.2_add_vl_1.3.0",
            ],
            "DG202_20210603": [
                f"{SD_DEMO_DEC2021_PREFIX}/superdrive_bev",
                "navinetmap_bevgt_loc_map_3_5_1020_odom_1.0.2_add_vl_1.3.0",
            ],
            "DG201_20210626": [
                f"{SD_DEMO_DEC2021_PREFIX}/superdrive_bev",
                "navinetmap_bevgt_loc_map_3_5_1020_odom_1.0.2_add_vl_1.3.0",
            ],
            "DG201_20210719": [
                f"{SD_DEMO_DEC2021_PREFIX}/superdrive_bev",
                "navinetmap_bevgt_loc_map_3_5_1020_odom_1.0.2_add_vl_1.3.0",
            ],
            "DG201_20210726": [
                f"{SD_DEMO_DEC2021_PREFIX}/superdrive_bev",
                "navinetmap_bevgt_loc_map_3_5_1020_odom_1.0.2_add_vl_1.3.0",
            ],
            "H3165_20211029": [
                f"{SD_DEMO_DEC2021_PREFIX}/superdrive_bev",
                "navinetmap_bevgt_loc_map_3_5_1020_odom_1.0.2_add_vl_1.3.0",
            ],
        },
        # Below are the historical datasets:
        # -- updated December 22, 2021
        "H3165_172packs_train_pkl": f"{SD_DEMO_DEC2021_PREFIX}/H3165_float128_v2_10days/pickle_test_train.pkl",  # noqa: E501
        "H3165_172packs_val_pkl": f"{SD_DEMO_DEC2021_PREFIX}/H3165_float128_v2_10days/pickle_test_val.pkl",  # noqa: E501
        "H3165_172packs_urban_train_pkl": f"{SD_DEMO_DEC2021_PREFIX}/H3165_float128_v2_urban_lmdb/pickle/pickle_test_train.pkl",  # noqa: E501
        "H3165_172packs_urban_val_pkl": f"{SD_DEMO_DEC2021_PREFIX}/H3165_float128_v2_urban_lmdb/pickle/pickle_test_val.pkl",  # noqa: E501
        "H3165_172packs_urban_lmdb_path": f"{SD_DEMO_DEC2021_PREFIX}/H3165_float128_v2_urban_lmdb/lmdb_dataset/struct_road/",  # noqa: E501
        # -- updated long long ago, may be not useable
        "lmdb_174packs_train_pkl": f"{SD_DEMO_DEC2021_PREFIX}/IC_pandar128float_map3.5_vis_loc_1.0.2_v4/test_lmdb_pickle_IC_pandar128float_map3.5_vis_loc_1.0.2_v4_train.pkl",  # noqa: E501
        "lmdb_174packs_val_pkl": f"{SD_DEMO_DEC2021_PREFIX}/IC_pandar128float_map3.5_vis_loc_1.0.2_v4/test_lmdb_pickle_IC_pandar128float_map3.5_vis_loc_1.0.2_v4_val.pkl",  # noqa: E501
        # -- updated long long long ago, may be not useable
        "lmdb_70packs_train_pkl": f"{SD_DEMO_DEC2021_PREFIX}/IC_pandar128float_map3.5_vis_loc_1.0.2_part1/test_lmdb_pickle_IC_pandar128float_map3.5_vis_loc_1.0.2_part1_train.pkl",  # noqa: E501
        "lmdb_70packs_val_pkl": f"{SD_DEMO_DEC2021_PREFIX}/IC_pandar128float_map3.5_vis_loc_1.0.2_part1/test_lmdb_pickle_IC_pandar128float_map3.5_vis_loc_1.0.2_part1_val.pkl",  # noqa: E501
    },
    "Lidar3M1": {  # SD_demo_Apr2022
        # Configuration:
        # -- Lidar: 3M1, use fillback 3M1 model
        #           (fuel fillback or collecting on test vehicles)
        # -- Lidar post processing: same as the method on test vehicles.
        # -- The dataset is processed by pipeline v3 (3.1)
        # Note: we will set the newest dataset as the "basic_train_pkl"
        "bucket": "J5FSD",
        "basic_train_pkl": f"{SD_DEMO_APR2022_PREFIX}/record_512packs_week2_5_for_3.7.0/concat_512p_train.pkl",  # noqa: E501
        "basic_val_pkl": f"{SD_DEMO_APR2022_PREFIX}/record_385packs_for_3.5.0/concat_385p_val_juefei.pkl",  # noqa: E501
        "map_path_func": gt_path_func_for_data_pipeline_v3,  # noqa: E501
        "data_token2path_mapping": {
            "J4243_20220308": [
                f"{SD_DEMO_APR2022_PREFIX}/bev",
                "bev_GT_bevGT_test_v0.0",
            ],
            "J4243_20220301": [
                f"{SD_DEMO_APR2022_PREFIX}/bev",
                "bev_GT_bevGT_test_v0.0",
            ],
            "J4243_20220302": [
                f"{SD_DEMO_APR2022_PREFIX}/bev",
                "bev_GT_bevGT_test_v0.0",
            ],
            "J4243_20220303": [
                f"{SD_DEMO_APR2022_PREFIX}/bev",
                "bev_GT_bevGT_test_v0.0",
            ],
            "J4243_20220304": [
                f"{SD_DEMO_APR2022_PREFIX}/bev",
                "bev_GT_bevGT_test_v0.0",
            ],
            "J4243_20220228": [
                f"{SD_DEMO_APR2022_PREFIX}/bev",
                "bev_GT_bevGT_test_v0.0",
            ],
            "J4243_20220225": [
                f"{SD_DEMO_APR2022_PREFIX}/bev",
                "bev_GT_bevGT_test_v0.0",
            ],
            "J4243_20220224": [
                f"{SD_DEMO_APR2022_PREFIX}/bev",
                "bev_GT_bevGT_test_v0.0",
            ],
            "J4243_20220223": [
                f"{SD_DEMO_APR2022_PREFIX}/bev",
                "bev_GT_bevGT_test_v0.0",
            ],
            "J4243_20220218": [
                f"{SD_DEMO_APR2022_PREFIX}/bev",
                "bev_GT_bevGT_test_v0.0",
            ],
            "J4243_20220217": [
                f"{SD_DEMO_APR2022_PREFIX}/bev",
                "bev_GT_bevGT_test_v0.0",
            ],
            "J4243_20220215": [
                f"{SD_DEMO_APR2022_PREFIX}/bev",
                "bev_GT_bevGT_test_v0.0",
            ],
            "H3165_20220307": [
                f"{SD_DEMO_APR2022_PREFIX}/bev",
                "bev_GT_bevGT_hde_test",
            ],
            "J4243_20220313": [
                f"{SD_DEMO_APR2022_PREFIX}/bev",
                "bev_GT_bevGT_test_v0.0",
            ],
            "J4243_20220312": [
                f"{SD_DEMO_APR2022_PREFIX}/bev",
                "bev_GT_bevGT_test_v0.0",
            ],
            "J4243_20220314": [
                f"{SD_DEMO_APR2022_PREFIX}/bev",
                "bev_GT_bevGT_test_v0.0",
            ],
            "J4243_20220316": [
                f"{SD_DEMO_APR2022_PREFIX}/bev",
                "bev_GT_bevGT_test_v0.0",
            ],
            "H3165_20220228": [
                f"{SD_DEMO_APR2022_PREFIX}/bev",
                "bev_GT_bevGT_test_v0.0",
            ],
            "H3165_20220301": [
                f"{SD_DEMO_APR2022_PREFIX}/bev",
                "bev_GT_bevGT_hde_test",
            ],
            "H3165_20220302": [
                f"{SD_DEMO_APR2022_PREFIX}/bev",
                "bev_GT_bevGT_hde_test",
            ],
            "J4243_20220321": [
                f"{SD_DEMO_APR2022_PREFIX}/bev",
                "bev_GT_bevGT_test_v0.0",
            ],
            "J4243_20220323": [
                f"{SD_DEMO_APR2022_PREFIX}/bev",
                "bev_GT_bevGT_test_v0.0",
            ],
            "J4243_20220320": [
                f"{SD_DEMO_APR2022_PREFIX}/bev",
                "bev_GT_bevGT_test_v0.0",
            ],
            "H3165_20220227": [
                f"{SD_DEMO_APR2022_PREFIX}/bev",
                "bev_GT_bevGT_test_v0.0",
            ],
            "H3165_20220226": [
                f"{SD_DEMO_APR2022_PREFIX}/bev",
                "bev_GT_bevGT_test_v0.0",
            ],
            "H3165_20220326": [
                f"{SD_DEMO_APR2022_PREFIX}/bev",
                "bev_GT_bevGT_hde_test",
            ],
            "H3165_20220327": [
                f"{SD_DEMO_APR2022_PREFIX}/bev",
                "bev_GT_bevGT_hde_test",
            ],
            "H3165_20220328": [
                f"{SD_DEMO_APR2022_PREFIX}/bev",
                "bev_GT_bevGT_hde_test",
            ],
            "H3165_20220329": [
                f"{SD_DEMO_APR2022_PREFIX}/bev",
                "bev_GT_bevGT_hde_test",
            ],
            "H3165_20220330": [
                f"{SD_DEMO_APR2022_PREFIX}/bev",
                "bev_GT_bevGT_hde_test",
            ],
            "H3165_20220409": [
                f"{SD_DEMO_APR2022_PREFIX}/bev",
                "bev_GT_bevGT_hde_test_v2.1",
            ],
        },
        # Below are the historical datasets:
        # -- updated on Mar 11, 2022
        "104packs_train_pkl": f"{SD_DEMO_APR2022_PREFIX}/record_104packs_for_3.3.0/concat_104p_train.pkl",  # noqa: E501
        "siwei_val_pkl": f"{SD_DEMO_APR2022_PREFIX}/record_104packs_for_3.3.0/concat_104p_val.pkl",  # noqa: E501
        # -- updated on Mar 18, 2022
        "185packs_train_pkl": f"{SD_DEMO_APR2022_PREFIX}/record_185packs_for_3.4.0/concat_185p_train.pkl",  # noqa: E501
        # -- updated on Mar 25, 2022
        "juefei_val_pkl": f"{SD_DEMO_APR2022_PREFIX}/record_385packs_for_3.5.0/concat_385p_val_juefei.pkl",  # noqa: E501
        # -- updated on Apr 5, 2022
        "482packs_train_pkl": f"{SD_DEMO_APR2022_PREFIX}/record_482packs_week2_4_for_3.6.0/concat_482p_train.pkl",  # noqa: E501
        # -- updated on Apr 14, 2022
        "512packs_train_pkl": f"{SD_DEMO_APR2022_PREFIX}/record_512packs_week2_5_for_3.7.0/concat_512p_train.pkl",  # noqa: E501
    },
    "Bev3d": {
        # Configuration:
        # -- Bev3d: fusion 6v, model version > 3.4.0
        "bucket": "J5FSD",
        "basic_train_pkl": f"{SD_DEMO_APR2022_PREFIX}/record_110packs_bev3d/concat_110p_train.pkl",  # noqa: E501
        "basic_val_pkl": f"{SD_DEMO_APR2022_PREFIX}/record_110packs_bev3d/concat_110p_val.pkl",  # noqa: E501
        "map_path_func": gt_path_func_for_data_pipeline_v3,  # noqa: E501
        "data_token2path_mapping": {
            "H3165_20220331": [
                f"{SD_DEMO_APR2022_PREFIX}/bev",
                "bev_GT_bevGT_hde_test",
            ],
            "H3165_20220330": [
                f"{SD_DEMO_APR2022_PREFIX}/bev",
                "bev_GT_bevGT_hde_test",
            ],
            "H3165_20220407": [
                f"{SD_DEMO_APR2022_PREFIX}/bev",
                "bev_GT_bevGT_hde_test",
            ],
            "H3165_20220326": [
                f"{SD_DEMO_APR2022_PREFIX}/bev",
                "bev_GT_bevGT_hde_test",
            ],
            "J4243_20220331": [
                f"{SD_DEMO_APR2022_PREFIX}/bev",
                "bev_GT_bevGT_hde_test",
            ],
            "J4243_20220401": [
                f"{SD_DEMO_APR2022_PREFIX}/bev",
                "bev_GT_bevGT_hde_test",
            ],
            "J4243_20220402": [
                f"{SD_DEMO_APR2022_PREFIX}/bev",
                "bev_GT_bevGT_hde_test",
            ],
        },
        # Below are the historical datasets:
        # -- updated on Apr 8, 2022
        "110packs_train_pkl": f"{SD_DEMO_APR2022_PREFIX}/record_110packs_bev3d/concat_110p_train.pkl",  # noqa: E501
        "110packs_val_pkl": f"{SD_DEMO_APR2022_PREFIX}/record_110packs_bev3d/concat_110p_val.pkl",  # noqa: E501
    },
    "Lidar3M1_Navi": {
        # Configuration:
        # -- Lidar: 3M1, use fillback 3M1 model
        #           (fuel fillback or collecting on test vehicles)
        # -- Lidar post processing: same as the method on test vehicles.
        # -- The dataset is processed by pipeline v3 (3.1)
        # Note: we will set the newest dataset as the "basic_train_pkl"
        "bucket": "J5FSD",
        "basic_train_pkl": f"{NAVI_DATA_APR2022_PREFIX}/record_3M1_navi_J4243/concat_J4243_navis_train.pkl",  # noqa: E501
        "basic_val_pkl": f"{NAVI_DATA_APR2022_PREFIX}/separate_day/record_3M1_navi_H3165_0407/record_v2.1/pickles/pickle_test_train.pkl",  # noqa: E501
        "map_path_func": gt_path_func_for_data_pipeline_v3,  # noqa: E501
        "lmdb_path_replace_func": lmdb_path_replace_func_for_data_pipeline_v3,
        "data_token2path_mapping": {
            "J4243_20220413": [
                f"{LI_MAP_PREFIX}/superdrive_dataset/record_one",  # noqa: E501
                "bev_GT_bevGT_hde_test_v2.1",
            ],
            "J4243_20220417": [
                [
                    f"{LI_MAP_PREFIX}/superdrive_dataset/record_one",  # noqa: E501
                ],
                "bev_GT_bevGT_hde_test_v2.1",
            ],
            "J4243_20220414": [
                "11_perception_prediction/03_pack_package/wenke.wang/superdrive_dataset/2022_June_before_dataset",  # noqa: E501
                "bev_GT_bevGT_hde_test_v2.1",
            ],
            "J4243_20220402": [
                f"{SD_DEMO_APR2022_PREFIX}/bev",
                "bev_GT_bevGT_hde_test_v2.1",
            ],
            "J4243_20220401": [
                f"{SD_DEMO_APR2022_PREFIX}/bev",
                "bev_GT_bevGT_hde_test_v2.1",
            ],
            "J4243_20220331": [
                [
                    f"{SD_DEMO_APR2022_PREFIX}/bev",
                ],
                "bev_GT_bevGT_hde_test_v2.1",
            ],
            "J4243_20220320": [
                f"{SD_DEMO_APR2022_PREFIX}/bev",
                "bev_GT_bevGT_hde_test_v2.1",
            ],
            "H3165_20220326": [
                f"{SD_DEMO_APR2022_PREFIX}/bev",
                "bev_GT_bevGT_hde_test_v2.1",
            ],
            "H3165_20220327": [
                f"{SD_DEMO_APR2022_PREFIX}/bev",
                "bev_GT_bevGT_hde_test_v2.1",
            ],
            "H3165_20220328": [
                [
                    f"{SD_DEMO_APR2022_PREFIX}/bev",
                ],
                "bev_GT_bevGT_hde_test_v2.1",
            ],
            "H3165_20220329": [
                f"{SD_DEMO_APR2022_PREFIX}/bev",
                "bev_GT_bevGT_hde_test_v2.1",
            ],
            "H3165_20220330": [
                f"{SD_DEMO_APR2022_PREFIX}/bev",
                "bev_GT_bevGT_hde_test_v2.1",
            ],
            "H3165_20220331": [
                [
                    f"{SD_DEMO_APR2022_PREFIX}/bev",
                ],
                "bev_GT_bevGT_hde_test_v2.1",
            ],
            "H3165_20220407": [
                f"{SD_DEMO_APR2022_PREFIX}/bev",
                "bev_GT_bevGT_hde_test_v2.1",
            ],
            "H3165_20220409": [
                f"{SD_DEMO_APR2022_PREFIX}/bev",
                "bev_GT_bevGT_hde_test_v2.1",
            ],
        },
        # Below are the historical datasets:
        # AutoMultiAgentNaviDataset (J4243 7days, 155packs)
        "J4243_155packs_train_pkl": f"{NAVI_DATA_APR2022_PREFIX}/record_3M1_navi_J4243/concat_J4243_navis_train.pkl",  # noqa: E501
        # AutoMultiAgentNaviDataset (H3165 6days, 108packs)
        "H3165_108packs_train_pkl": f"{NAVI_DATA_APR2022_PREFIX}/record_3M1_navi_H3165/concat_H3165_navis_train.pkl",  # noqa: E501
        "navi_train_pkl": f"{NAVI_DATA_APR2022_PREFIX}/record_3M1_navi_J4243/concat_J4243_navis_train.pkl",  # noqa: E501
        # AutoMultiAgentNaviDataset (H3165 1day(0407), 6packs)
        "navi_val_pkl": f"{NAVI_DATA_APR2022_PREFIX}/record_3M1_navi_H3165_0407/record_v2.1/pickles/pickle_test_train.pkl",  # noqa: E501
    },
    "smallTest_f8": {
        # only include 2 packs
        "bucket": "J5FSD",
        "basic_train_pkl": f"{HIGH_FREQ_DATA_PREFIX}/../test/test_f8/pickles/pickle_test_train.pkl",  # noqa: E501
        "basic_val_pkl": f"{HIGH_FREQ_DATA_PREFIX}/../test/test_f8/pickles/pickle_test_train.pkl",  # noqa: E501
        "map_path_func": bev_gt_path_function,  # noqa: E501
        "data_token2path_mapping": {
            "H3165_20220226": [
                "11_perception_prediction/SD_demo_Apr2022/bev",
                "bev_GT_bevGT_test_v0.0",
            ],
            "H3165_20220227": [
                "11_perception_prediction/SD_demo_Apr2022/bev",
                "bev_GT_bevGT_test_v0.0",
            ],
            "H3165_20220228": [
                "11_perception_prediction/SD_demo_Apr2022/bev",
                "bev_GT_bevGT_test_v0.0",
            ],
        },
    },
}

VW_DEMO_DATASET = {  # VW_demo_Jun2022
    "Vision": {
        # Configuration:
        # -- Bev3d: fusion 6v, model version > 3.4.0
        "bucket": "J5FSD",
        "basic_train_pkl": f"{VW_DEMO_JUN2022_PREFIX}/fillback_May_23/concat_train.pkl",  # noqa: E501
        "basic_val_pkl": f"{VW_DEMO_JUN2022_PREFIX}/fillback_May_23/concat_val.pkl",  # noqa: E501
        "basic_eval_pkl": f"{VW_DEMO_JUN2022_PREFIX}/fusion6v_eval_20220421/concat_eval.pkl",  # noqa: E501
        "map_path_func": gt_path_func_for_data_pipeline_v3,  # noqa: E501
        "data_token2path_mapping": {
            "H3165_20220423": [
                f"{VW_DEMO_JUN2022_PREFIX}/superdrive_bev",
                "bev_GT_bevGT_hde_test_v2.1",
            ],
            "H3165_20220425": [
                f"{VW_DEMO_JUN2022_PREFIX}/superdrive_bev",
                "bev_GT_bevGT_hde_test_v2.1",
            ],
            "H3165_20220426": [
                f"{VW_DEMO_JUN2022_PREFIX}/superdrive_bev",
                "bev_GT_bevGT_hde_test_v2.1",
            ],
            "H3165_20220424": [
                f"{VW_DEMO_JUN2022_PREFIX}/superdrive_bev",
                "bev_GT_bevGT_hde_test_v2.1",
            ],
            "UT0Q9_20220313": [
                f"{VW_DEMO_JUN2022_PREFIX}/superdrive_bev",
                "bev_GT_bevGT_hde_test_v2.1",
            ],
            "UT0Q9_20220314": [
                f"{VW_DEMO_JUN2022_PREFIX}/superdrive_bev",
                "bev_GT_bevGT_hde_test_v2.1",
            ],
            "UT0Q9_20220315": [
                f"{VW_DEMO_JUN2022_PREFIX}/superdrive_bev",
                "bev_GT_bevGT_hde_test_v2.1",
            ],
            "UT0Q9_20220316": [
                f"{VW_DEMO_JUN2022_PREFIX}/superdrive_bev",
                "bev_GT_bevGT_hde_test_v2.1",
            ],
            "UT0Q9_20220318": [
                f"{VW_DEMO_JUN2022_PREFIX}/superdrive_bev",
                "bev_GT_bevGT_hde_test_v2.1",
            ],
            "UT0Q9_20220319": [
                f"{VW_DEMO_JUN2022_PREFIX}/superdrive_bev",
                "bev_GT_bevGT_hde_test_v2.1",
            ],
            "UT0Q9_20220320": [
                f"{VW_DEMO_JUN2022_PREFIX}/superdrive_bev",
                "bev_GT_bevGT_hde_test_v2.1",
            ],
            "H3165_20220421": [
                f"{VW_DEMO_JUN2022_PREFIX}/superdrive_bev",
                "bev_GT_bevGT_hde_test_v2.1",
            ],
            "H3165_20220422": [
                "11_perception_prediction/03_pack_package/wenke.wang/superdrive_dataset/2022_June_before_dataset",  # noqa: E501
                "bev_GT_bevGT_hde_test_v2.1",
            ],
            "H3165_20220602": [
                f"{VW_DEMO_JUN2022_PREFIX}/superdrive_bev",
                "bev_GT_bevgt_loc_map_3_5_1.0.2_add_vl",
            ],
            "H3165_20220604": [
                f"{VW_DEMO_JUN2022_PREFIX}/superdrive_bev",
                "bev_GT_bevgt_loc_map_3_5_1.0.2_add_vl",
            ],
            "UT0Q9_20220527": [
                f"{VW_DEMO_JUN2022_PREFIX}/superdrive_bev",
                "rasterized_map/0_0/bev_GT_bevGT_hde_test_v2.1",
            ],
            "UT0Q9_20220528": [
                f"{VW_DEMO_JUN2022_PREFIX}/superdrive_bev",
                "rasterized_map/0_0/bev_GT_bevGT_hde_test_v2.1",
            ],
        },
        # Below are the historical datasets:
        # -- updated on May 23, 2022
        "270packs_train_pkl": f"{VW_DEMO_JUN2022_PREFIX}/fillback_May_23/concat_train.pkl",  # noqa: E501
        "270packs_val_pkl": f"{VW_DEMO_JUN2022_PREFIX}/fillback_May_23/concat_val.pkl",  # noqa: E501
        # -- updated on June 2, 2022
        "0421_eval_packs": f"{VW_DEMO_JUN2022_PREFIX}/fusion6v_eval_20220421/concat_eval.pkl",  # noqa: E501
        # -- updated on June 7, 2022
        "vw_130_packs": f"{VW_DEMO_JUN2022_PREFIX}/UT0Q9_0527_0528/pickles/pickle_test_train.pkl",  # noqa: E501
        "vw_test_pkl": f"{VW_DEMO_JUN2022_PREFIX}/UT0Q9_0528_test/pickles/pickle_test_train.pkl",  # noqa: E501
        # -- updated on June 22, 2022
        "0421_0422_fusion_eval": f"{VW_DEMO_JUN2022_PREFIX}/fusion6v_eval_20220421_20220422/concat_eval.pkl",  # noqa: E501
        "0421_0422_lidar_eval": f"{VW_DEMO_JUN2022_PREFIX}/lidar_eval_20220421_20220422/concat_eval.pkl",  # noqa: E501
    },
    "Vision_Navi_highway": {
        # Configuration:
        # -- Bev3d: fusion 6v, model version > 3.4.0
        "bucket": "J5FSD",
        # basic_train_pkl: 0528(82)+0530(46)+0531(52)+0614(13)+0615(73)
        "basic_train_pkl": f"{NAVI_DATA_APR2022_PREFIX}/concat_fusion6v_5days_navi_pickle/concat_fusion6v_5days_266packs_navi_train.pkl",  # noqa: E501
        "basic_val_pkl": f"{NAVI_DATA_APR2022_PREFIX}/fusion6V/UT0Q9_20220601/pickles/pickle_test_train.pkl",  # noqa: E501
        "basic_eval_pkl": f"{NAVI_DATA_APR2022_PREFIX}/fusion6V/H3165_20220421/pickles/pickle_test_train.pkl",  # noqa: E501
        # "basic_eval_pkl": f"{VW_DEMO_JUN2022_PREFIX}/fusion6v_eval_20220421/concat_eval.pkl",  # noqa: E501
        "map_path_func": gt_path_func_for_data_pipeline_v3,  # noqa: E501
        "lmdb_path_replace_func": lmdb_path_replace_func_for_data_pipeline_v5,  # noqa: E501
        "data_token2path_mapping": {
            "UT0Q9_20220601": [
                f"{VW_DEMO_JUN2022_PREFIX}/superdrive_bev",
                "rasterized_map/0_0/bev_GT_bevGT_hde_test_v2.1",
            ],
            "UT0Q9_20220531": [
                f"{LI_MAP_PREFIX}/superdrive_dataset/dag_UT0Q9",  # noqa: E501
                "rasterized_map/0_0/bev_GT_bevGT_hde_test_v2.1",
            ],
            "UT0Q9_20220530": [
                f"{LI_MAP_PREFIX}/superdrive_dataset/dag_UT0Q9",  # noqa: E501
                "rasterized_map/0_0/bev_GT_bevGT_hde_test_v2.1",
            ],
            "UT0Q9_20220527": [
                f"{VW_DEMO_JUN2022_PREFIX}/superdrive_bev",
                "rasterized_map/0_0/bev_GT_bevGT_hde_test_v2.1",
            ],
            "UT0Q9_20220528": [
                f"{VW_DEMO_JUN2022_PREFIX}/superdrive_bev",
                "rasterized_map/0_0/bev_GT_bevGT_hde_test_v2.1",
            ],
            "UT0Q9_20220529": [
                f"{VW_DEMO_JUN2022_PREFIX}/superdrive_bev",
                "rasterized_map/0_0/bev_GT_bevGT_hde_test_v2.1",
            ],
            "UTHS6_20220615": [
                f"{LI_MAP_PREFIX}/superdrive_dataset/dag_UTHS6",  # noqa: E501
                "rasterized_map/0_0/bev_GT_bevGT_hde_test_v2.1",
            ],
            "UTHS6_20220614": [
                f"{LI_MAP_PREFIX}/superdrive_dataset/dag_UTHS6",  # noqa: E501
                "rasterized_map/0_0/bev_GT_bevGT_hde_test_v2.1",
            ],
            "H3165_20220421": [
                "11_perception_prediction/03_pack_package/wenke.wang/VW_demo_June",  # noqa: E501
                "rasterized_map/0_0/bev_GT_bevGT_hde_test_v2.1",
            ],
            "H3165_20220422": [
                "11_perception_prediction/03_pack_package/wenke.wang/superdrive_dataset/2022_June_before_dataset",  # noqa: E501
                "bev_GT_bevGT_hde_test_v2.1",
            ],
        },
    },
    "Vision_Navi_f8": {
        # Configuration:
        # -- Bev3d: fusion 6v, model version > 3.4.0
        "bucket": "J5FSD",
        "basic_train_pkl": f"{HIGH_FREQ_DATA_PREFIX}/train_data_322_packs_from_0315_to_0426/freq8/\
            pickle_test_train.pkl",
        "val_67packs_pkl": f"{HIGH_FREQ_DATA_PREFIX}/\
        test_data_67_packs_on_0421/freq8/pickle_test_train.pkl",
        "val_138packs_pkl": f"{HIGH_FREQ_DATA_PREFIX}/test_data_138_from_0421_to_0422/\
freq8/pickle_test_train.pkl",
        "basic_val_pkl": f"{HIGH_FREQ_DATA_PREFIX}/../test/\
        test_f8/pickles/pickle_test_train.pkl",
        "map_path_func": gt_path_func_for_data_pipeline_v3,
        "data_token2path_mapping": {
            "H3165_20220423": [
                f"{VW_DEMO_JUN2022_PREFIX}/superdrive_bev",
                "rasterized_map/0_0/bev_GT_bevGT_hde_test_v2.1",
            ],
            "H3165_20220424": [
                f"{VW_DEMO_JUN2022_PREFIX}/superdrive_bev",
                "rasterized_map/0_0/bev_GT_bevGT_hde_test_v2.1",
            ],
            "H3165_20220425": [
                f"{VW_DEMO_JUN2022_PREFIX}/superdrive_bev",
                "rasterized_map/0_0/bev_GT_bevGT_hde_test_v2.1",
            ],
            "H3165_20220426": [
                f"{VW_DEMO_JUN2022_PREFIX}/superdrive_bev",
                "rasterized_map/0_0/bev_GT_bevGT_hde_test_v2.1",
            ],
            "UT0Q9_20220315": [
                f"{VW_DEMO_JUN2022_PREFIX}/superdrive_bev",
                "rasterized_map/0_0/bev_GT_bevGT_hde_test_v2.1",
            ],
            "UT0Q9_20220316": [
                f"{VW_DEMO_JUN2022_PREFIX}/superdrive_bev",
                "rasterized_map/0_0/bev_GT_bevGT_hde_test_v2.1",
            ],
            "UT0Q9_20220318": [
                f"{VW_DEMO_JUN2022_PREFIX}/superdrive_bev",
                "rasterized_map/0_0/bev_GT_bevGT_hde_test_v2.1",
            ],
            "UT0Q9_20220319": [
                f"{VW_DEMO_JUN2022_PREFIX}/superdrive_bev",
                "rasterized_map/0_0/bev_GT_bevGT_hde_test_v2.1",
            ],
            "UT0Q9_20220320": [
                f"{VW_DEMO_JUN2022_PREFIX}/superdrive_bev",
                "rasterized_map/0_0/bev_GT_bevGT_hde_test_v2.1",
            ],
            "H3165_20220421": [
                f"{VW_DEMO_JUN2022_PREFIX}/superdrive_bev",
                "rasterized_map/0_0/bev_GT_bevGT_hde_test_v2.1",
            ],
            "H3165_20220422": [
                f"{VW_DEMO_JUN2022_PREFIX}/superdrive_bev",
                "rasterized_map/0_0/bev_GT_bevGT_hde_test_v2.1",
            ],
        },
    },
    "Vision_Navi": {
        # Configuration:
        # -- Bev3d: fusion 6v, model version > 3.4.0
        "bucket": "J5FSD",
        "basic_train_pkl": f"{VW_DEMO_JUN2022_PREFIX}/fusion6v_navi_pkl/vw_navi_v0.4.3_train/concat_fusion6v_9days_322packs_navi_train.pkl",  # noqa: E501
        "basic_val_pkl": f"{VW_DEMO_JUN2022_PREFIX}/fusion6v_navi_pkl/vw_navi_v0.4.3_eval_0421/concat_fusion6v_0421_67packs_navi_eval.pkl",  # noqa: E501
        "basic_eval_pkl": f"{VW_DEMO_JUN2022_PREFIX}/fusion6v_navi_pkl/vw_navi_v0.4.3_eval_0422/concat_fusion6v_0422_71packs_navi_eval.pkl",  # noqa: E501
        "basic_test_pkl": f"{VW_DEMO_JUN2022_PREFIX}/fusion6v_navi_pkl/vw_navi_v0.4.3_eval_0421_0422/concat_fusion6v_0421_0422_138packs_navi_eval.pkl",  # noqa: E501
        "map_path_func": gt_path_func_for_data_pipeline_v3,  # noqa: E501
        "lmdb_path_replace_func": lmdb_path_replace_func_for_data_pipeline_v5,  # noqa: E501
        "data_token2path_mapping": {
            "H3165_20220423": [
                f"{VW_DEMO_JUN2022_PREFIX}/superdrive_bev",
                "rasterized_map/0_0/bev_GT_bevGT_hde_test_v2.1",
            ],
            "H3165_20220424": [
                f"{VW_DEMO_JUN2022_PREFIX}/superdrive_bev",
                "rasterized_map/0_0/bev_GT_bevGT_hde_test_v2.1",
            ],
            "H3165_20220425": [
                f"{VW_DEMO_JUN2022_PREFIX}/superdrive_bev",
                "rasterized_map/0_0/bev_GT_bevGT_hde_test_v2.1",
            ],
            "H3165_20220426": [
                f"{VW_DEMO_JUN2022_PREFIX}/superdrive_bev",
                "rasterized_map/0_0/bev_GT_bevGT_hde_test_v2.1",
            ],
            "UT0Q9_20220315": [
                f"{VW_DEMO_JUN2022_PREFIX}/superdrive_bev",
                "rasterized_map/0_0/bev_GT_bevGT_hde_test_v2.1",
            ],
            "UT0Q9_20220316": [
                f"{VW_DEMO_JUN2022_PREFIX}/superdrive_bev",
                "rasterized_map/0_0/bev_GT_bevGT_hde_test_v2.1",
            ],
            "UT0Q9_20220318": [
                f"{VW_DEMO_JUN2022_PREFIX}/superdrive_bev",
                "rasterized_map/0_0/bev_GT_bevGT_hde_test_v2.1",
            ],
            "UT0Q9_20220319": [
                f"{VW_DEMO_JUN2022_PREFIX}/superdrive_bev",
                "rasterized_map/0_0/bev_GT_bevGT_hde_test_v2.1",
            ],
            "UT0Q9_20220320": [
                f"{VW_DEMO_JUN2022_PREFIX}/superdrive_bev",
                "rasterized_map/0_0/bev_GT_bevGT_hde_test_v2.1",
            ],
            "H3165_20220421": [
                f"{VW_DEMO_JUN2022_PREFIX}/superdrive_bev",
                "rasterized_map/0_0/bev_GT_bevGT_hde_test_v2.1",
            ],
            "H3165_20220422": [
                f"{VW_DEMO_JUN2022_PREFIX}/superdrive_bev",
                "rasterized_map/0_0/bev_GT_bevGT_hde_test_v2.1",
            ],
        },
    },
}
