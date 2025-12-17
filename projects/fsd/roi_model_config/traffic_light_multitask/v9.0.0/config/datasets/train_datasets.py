from easydict import EasyDict
from hatbc.filestream.bucket.client import get_gpfs_bucket_mount_root

bucket2mount_root = get_gpfs_bucket_mount_root()

root_bucket = bucket2mount_root.get("mono", None)
root = f"{root_bucket}"

datapaths = dict(
    traffic_light_lens=dict(
        train_batch_size_per_ctx=450,
        train_data_paths=[
            dict(
                rec_path=f"{root}/data/traffic_lens_det_roilist/20220809_220_lens_batch_all/20220811-111713/data.rec",  # noqa
                anno_path=f"{root}/fan.lv/data/traffic_lens_det_roilist/20220809_220_lens_batch_all/20220811-111713/data.anno.pb_rec",  # noqa
                roi_list_path=f"{root}/fan.lv/data/traffic_lens_det_roilist/20220809_220_lens_batch_all/20220811-111713/data.roilist.pb_rec",  # noqa
                sample_weight=39,
            ),
            dict(
                rec_path=f"{root}/data/traffic_lens_det_roilist/20220811_323_lens_batch_all/20220816-184600/data.rec",  # noqa
                anno_path=f"{root}/fan.lv/data/traffic_lens_det_roilist/20220811_323_lens_batch_all/20220816-184600/data.anno.pb_rec",  # noqa
                roi_list_path=f"{root}/fan.lv/data/traffic_lens_det_roilist/20220811_323_lens_batch_all/20220816-184600/data.roilist.pb_rec",  # noqa
                sample_weight=58,
            ),
            dict(
                rec_path=f"{root}/data/traffic_lens_det_roilist/20221024_820_lens_batch_all_online_roi/20221030-131812/data.rec",  # noqa
                anno_path=f"{root}/data/traffic_lens_det_roilist/20221024_820_lens_batch_all_online_roi/20221030-131812/data.anno.pb_rec",  # noqa
                roi_list_path=f"{root}/fan.lv/data/traffic_lens_det_roilist/20221024_820_lens_batch_all_online_roi/20221030-131812/data.roilist.pb_rec",  # noqa
                sample_weight=64,
            ),
            dict(
                rec_path=f"{root}/data/traffic_lens_det_roilist/20221024_10652_lens_batch_all_online_roi/20221026-024633/data.rec",  # noqa
                anno_path=f"{root}/data/traffic_lens_det_roilist/20221024_10652_lens_batch_all_online_roi/20221026-024633/data.anno.pb_rec",  # noqa
                roi_list_path=f"{root}/fan.lv/data/traffic_lens_det_roilist/20221024_10652_lens_batch_all_online_roi/20221026-024633/data.roilist.pb_rec",  # noqa
                sample_weight=35,
            ),
            dict(
                rec_path=f"{root}/data/traffic_lens_det_roilist/20230622_x8b_lens_det_batch_all/20230624-175618/data.rec",  # noqa
                anno_path=f"{root}/data/traffic_lens_det_roilist/20230622_x8b_lens_det_batch_all/20230624-175618/data.anno.pb_rec",  # noqa
                roi_list_path=f"{root}/fan.lv/data/traffic_lens_det_roilist/20230622_x8b_lens_det_batch_all/20230624-175618/data.roilist.pb_rec",  # noqa
                sample_weight=62,
            ),
            dict(
                rec_path=f"{root}/data/traffic_lens_det_roilist/20230622_badcase_day_lens_det_batch_all/20230624-132033/data.rec",  # noqa
                anno_path=f"{root}/data/traffic_lens_det_roilist/20230622_badcase_day_lens_det_batch_all/20230624-132033/data.anno.pb_rec",  # noqa
                roi_list_path=f"{root}/fan.lv/data/traffic_lens_det_roilist/20230622_badcase_day_lens_det_batch_all/20230624-132033/data.roilist.pb_rec",  # noqa
                sample_weight=5,
            ),
            dict(
                rec_path=f"{root}/data/traffic_lens_night_det_roilist/20230622_badcase_night_lens_batch_all/20230624-135623/data.rec",  # noqa
                anno_path=f"{root}/data/traffic_lens_night_det_roilist/20230622_badcase_night_lens_batch_all/20230624-135623/data.anno.pb_rec",  # noqa
                roi_list_path=f"{root}/fan.lv/data/traffic_lens_night_det_roilist/20230622_badcase_night_lens_batch_all/20230624-135623/data.roilist.pb_rec",  # noqa
                sample_weight=3,
            ),
            # sd 数据
            dict(
                rec_path=f"{root}/data/traffic_lens_det_roilist/20230519_sd_x8b_lens_det_batch_all/20230522-114919/data.rec",  # noqa
                anno_path=f"{root}/data/traffic_lens_det_roilist/20230519_sd_x8b_lens_det_batch_all/20230522-114919/data.anno.pb_rec",  # noqa
                roi_list_path=f"{root}/data/traffic_lens_det_roilist/20230519_sd_x8b_lens_det_batch_all/20230522-114919/data.roilist.pb_rec",  # noqa
                sample_weight=10,
            ),
            dict(
                rec_path=f"{root}/data/traffic_lens_det_roilist/20230622_sd_badcase_day_lens_det_batch_all/20230624-144023/data.rec",  # noqa
                anno_path=f"{root}/data/traffic_lens_det_roilist/20230622_sd_badcase_day_lens_det_batch_all/20230624-144023/data.anno.pb_rec",  # noqa
                roi_list_path=f"{root}/fan.lv/data/traffic_lens_det_roilist/20230622_sd_badcase_day_lens_det_batch_all/20230624-144023/data.roilist.pb_rec",  # noqa
                sample_weight=4,
            ),
            dict(
                rec_path=f"{root}/data/traffic_lens_night_det_roilist/20230622_sd_badcase_night_lens_det_batch_all/20230624-144915/data.rec",  # noqa
                anno_path=f"{root}/data/traffic_lens_night_det_roilist/20230622_sd_badcase_night_lens_det_batch_all/20230624-144915/data.anno.pb_rec",  # noqa
                roi_list_path=f"{root}/fan.lv/data/traffic_lens_night_det_roilist/20230622_sd_badcase_night_lens_det_batch_all/20230624-144915/data.roilist.pb_rec",  # noqa
                sample_weight=3,
            ),
            dict(
                rec_path=f"{root}/data/traffic_lens_det_roilist/20230519_sd_x8b_ov30_badcase_lens_det_batch_3/20230520-012856/data.rec",  # noqa
                anno_path=f"{root}/data/traffic_lens_det_roilist/20230519_sd_x8b_ov30_badcase_lens_det_batch_3/20230520-012856/data.anno.pb_rec",  # noqa
                roi_list_path=f"{root}/data/traffic_lens_det_roilist/20230519_sd_x8b_ov30_badcase_lens_det_batch_3/20230520-012856/data.roilist.pb_rec",  # noqa
                sample_weight=1,
            ),
        ],
        val_batch_size_per_ctx=10,
        val_data_paths=[
            dict(
                rec_path=f"{root}/data/2pe_traffic_light_lens_detection/zx_tl_len_det_2pe_attr_v3_0323_eval_20200604/data.rec",  # noqa
                anno_path=f"{root}/data/2pe_traffic_light_lens_detection/zx_tl_len_det_2pe_attr_v3_0323_eval_20200604/data.json",  # noqa
                sample_weight=10,
            ),
        ],
    ),
    traffic_light_color=dict(
        train_batch_size_per_ctx=280,
        train_data_paths=[
            dict(
                rec_path=f"{root}/data/traffic_light_color_cls/20230608_220_color_batch_all/20230608-201753/data.rec",  # noqa
                anno_path=f"{root}/data/traffic_light_color_cls/20230608_220_color_batch_all/20230608-201753/data.anno.pb_rec",  # noqa
                sample_weight=40,
            ),
            dict(
                rec_path=f"{root}/data/traffic_light_color_cls/20230608_323_color_batch_all/20230608-224508/data.rec",  # noqa
                anno_path=f"{root}/data/traffic_light_color_cls/20230608_323_color_batch_all/20230608-224508/data.anno.pb_rec",  # noqa
                sample_weight=65,
            ),
            dict(
                rec_path=f"{root}/data/traffic_light_color_cls/20230608_820_color_batch_all/20230609-002741/data.rec",  # noqa
                anno_path=f"{root}/data/traffic_light_color_cls/20230608_820_color_batch_all/20230609-002741/data.anno.pb_rec",  # noqa
                sample_weight=80,
            ),
            dict(
                rec_path=f"{root}/data/traffic_light_color_cls/20230608_10652_color_batch_all/20230608-224722/data.rec",  # noqa
                anno_path=f"{root}/data/traffic_light_color_cls/20230608_10652_color_batch_all/20230608-224722/data.anno.pb_rec",  # noqa
                sample_weight=25,
            ),
            dict(
                rec_path=f"{root}/data/traffic_light_color_cls/20230622_x8b_color_batch_all_cls7/20230621-224314/data.rec",  # noqa
                anno_path=f"{root}/data/traffic_light_color_cls/20230622_x8b_color_batch_all_cls7/20230621-224314/data.anno.pb_rec",  # noqa
                sample_weight=47,
            ),
            dict(
                rec_path=f"{root}/data/traffic_light_night_color_cls_badcase/20230622_badcase_night_color_batch_all_cls7/20230621-220234/data.rec",  # noqa
                anno_path=f"{root}/data/traffic_light_night_color_cls_badcase/20230622_badcase_night_color_batch_all_cls7/20230621-220234/data.anno.pb_rec",  # noqa
                sample_weight=5,
            ),
            dict(
                rec_path=f"{root}/data/traffic_light_color_cls_badcase/20230622_badcase_day_color_batch_all_cls7/20230621-165526/data.rec",  # noqa
                anno_path=f"{root}/data/traffic_light_color_cls_badcase/20230622_badcase_day_color_batch_all_cls7/20230621-165526/data.anno.pb_rec",  # noqa
                sample_weight=7,
            ),
            # sd 数据
            dict(
                rec_path=f"{root}/data/traffic_light_color_cls/20230608_sd_x8b_color_batch_all/20230609-002709/data.rec",  # noqa
                anno_path=f"{root}/data/traffic_light_color_cls/20230608_sd_x8b_color_batch_all/20230609-002709/data.anno.pb_rec",  # noqa
                sample_weight=7,
            ),
            # sd badcase 数据
            dict(
                rec_path=f"{root}/data/traffic_light_color_cls_badcase/20230622_sd_badcase_day_color_batch_all_cls7/20230621-192932/data.rec",  # noqa
                anno_path=f"{root}/data/traffic_light_color_cls_badcase/20230622_sd_badcase_day_color_batch_all_cls7/20230621-192932/data.anno.pb_rec",  # noqa
                sample_weight=2,
            ),
            dict(
                rec_path=f"{root}/data/traffic_light_night_color_cls_badcase/20230622_sd_badcase_night_color_batch_all_cls7/20230621-221510/data.rec",  # noqa
                anno_path=f"{root}/data/traffic_light_night_color_cls_badcase/20230622_sd_badcase_night_color_batch_all_cls7/20230621-221510/data.anno.pb_rec",  # noqa
                sample_weight=1,
            ),
            # sd 窄角数据
            dict(
                rec_path=f"{root}/data/traffic_light_color_cls/20230622_sd_x8b_ov30_badcase_color_batch_all_cls7/20230621-201346/data.rec",  # noqa
                anno_path=f"{root}/data/traffic_light_color_cls/20230622_sd_x8b_ov30_badcase_color_batch_all_cls7/20230621-201346/data.anno.pb_rec",  # noqa
                sample_weight=1,
            ),
        ],
        val_batch_size_per_ctx=10,
        val_data_paths=[
            dict(
                rec_path=f"{root}/data/2pe_traffic_light_color_cls/tl_shell_color_down_sample_back_side-2.0-val/data.rec",  # noqa
                anno_path=f"{root}/data/2pe_traffic_light_color_cls/tl_shell_color_down_sample_back_side-2.0-val/data.json",  # noqa
                sample_weight=10,
            ),
        ],
    ),
    traffic_light_dire=dict(
        train_batch_size_per_ctx=258,
        train_data_paths=[
            dict(
                rec_path=f"{root}/data/traffic_light_dire_cls/20230608_220_dire_batch_all/20230608-195011/data.rec",  # noqa
                anno_path=f"{root}/data/traffic_light_dire_cls/20230608_220_dire_batch_all/20230608-195011/data.anno.pb_rec",  # noqa
                sample_weight=30,
            ),
            dict(
                rec_path=f"{root}/data/traffic_light_dire_cls/20230608_323_dire_batch_all/20230608-194506/data.rec",  # noqa
                anno_path=f"{root}/data/traffic_light_dire_cls/20230608_323_dire_batch_all/20230608-194506/data.anno.pb_rec",  # noqa
                sample_weight=60,
            ),
            dict(
                rec_path=f"{root}/data/traffic_light_dire_cls/20230608_820_dire_batch_all/20230608-215436/data.rec",  # noqa
                anno_path=f"{root}/data/traffic_light_dire_cls/20230608_820_dire_batch_all/20230608-215436/data.anno.pb_rec",  # noqa
                sample_weight=70,
            ),
            dict(
                rec_path=f"{root}/data/traffic_light_dire_cls/20230608_323_dire_batch_all/20230608-201338/data.rec",  # noqa
                anno_path=f"{root}/data/traffic_light_dire_cls/20230608_323_dire_batch_all/20230608-201338/data.anno.pb_rec",  # noqa
                sample_weight=20,
            ),
            dict(
                rec_path=f"{root}/data/traffic_light_dire_cls/20230622_x8b_dire_batch_all/20230621-184239/data.rec",  # noqa
                anno_path=f"{root}/data/traffic_light_dire_cls/20230622_x8b_dire_batch_all/20230621-184239/data.anno.pb_rec",  # noqa
                sample_weight=40,
            ),
            dict(
                rec_path=f"{root}/data/traffic_light_dire_cls/20230622_badcase_night_dire_batch_all/20230621-221902/data.rec",  # noqa
                anno_path=f"{root}/data/traffic_light_dire_cls/20230622_badcase_night_dire_batch_all/20230621-221902/data.anno.pb_rec",  # noqa
                sample_weight=7,
            ),
            dict(
                rec_path=f"{root}/data/traffic_light_dire_cls/20230622_badcase_day_dire_batch_all/20230621-172327/data.rec",  # noqa
                anno_path=f"{root}/data/traffic_light_dire_cls/20230622_badcase_day_dire_batch_all/20230621-172327/data.anno.pb_rec",  # noqa
                sample_weight=10,
            ),
            # sd 数据
            dict(
                rec_path=f"{root}/data/traffic_light_dire_cls/20230608_sd_x8b_dire_batch_all/20230608-225408/data.rec",  # noqa
                anno_path=f"{root}/data/traffic_light_dire_cls/20230608_sd_x8b_dire_batch_all/20230608-225408/data.anno.pb_rec",  # noqa
                sample_weight=10,
            ),
            # sd badcase 数据
            dict(
                rec_path=f"{root}/data/traffic_light_dire_cls/20230622_sd_badcase_day_dire_batch_all/20230621-220246/data.rec",  # noqa
                anno_path=f"{root}/data/traffic_light_dire_cls/20230622_sd_badcase_day_dire_batch_all/20230621-220246/data.anno.pb_rec",  # noqa
                sample_weight=5,
            ),
            dict(
                rec_path=f"{root}/data/traffic_light_dire_cls/20230622_sd_badcase_night_dire_batch_all/20230621-223313/data.rec",  # noqa
                anno_path=f"{root}/data/traffic_light_dire_cls/20230622_sd_badcase_night_dire_batch_all/20230621-223313/data.anno.pb_rec",  # noqa
                sample_weight=5,
            ),
            # sd 窄角数据
            dict(
                rec_path=f"{root}/data/traffic_light_dire_cls/20230622_sd_x8b_ov30_badcase_dire_batch_all/20230621-224500/data.rec",  # noqa
                anno_path=f"{root}/data/traffic_light_dire_cls/20230622_sd_x8b_ov30_badcase_dire_batch_all/20230621-224500/data.anno.pb_rec",  # noqa
                sample_weight=1,
            ),
        ],
        val_batch_size_per_ctx=1,
        val_data_paths=[
            dict(
                rec_path=f"{root}/data/traffic_light_dire_cls/20230622_sd_x8b_ov30_badcase_dire_batch_all/20230621-224500/data.rec",  # noqa
                anno_path=f"{root}/data/traffic_light_dire_cls/20230622_sd_x8b_ov30_badcase_dire_batch_all/20230621-224500/data.anno.pb_rec",  # noqa
                sample_weight=1,
            ),
        ],
    ),
    traffic_light_category=dict(
        train_batch_size_per_ctx=240,
        train_data_paths=[
            dict(
                rec_path=f"{root}/data/traffic_light_type_cls/20220809_220_type_batch_all/20220809-155152/data.rec",  # noqa
                anno_path=f"{root}/fan.lv/data/traffic_light_type_cls/20220809_220_type_batch_all/20220809-155152/data.anno.pb_rec",  # noqa
                sample_weight=28,
            ),
            dict(
                rec_path=f"{root}/data/traffic_light_type_cls/20220811_323_type_batch_all/20220811-161700/data.rec",  # noqa
                anno_path=f"{root}/fan.lv/data/traffic_light_type_cls/20220811_323_type_batch_all/20220811-161700/data.anno.pb_rec",  # noqa
                sample_weight=53,
            ),
            dict(
                rec_path=f"{root}/data/traffic_light_type_cls/20221024_820_type_batch_lx_online_roi/20221025-000459/data.rec",  # noqa
                anno_path=f"{root}/data/traffic_light_type_cls/20221024_820_type_batch_lx_online_roi/20221025-000459/data.anno.pb_rec",  # noqa
                sample_weight=60,
            ),
            dict(
                rec_path=f"{root}/data/traffic_light_type_cls/20221024_10652_type_batch_all_online_roi/20221025-133446/data.rec",  # noqa
                anno_path=f"{root}/data/traffic_light_type_cls/20221024_10652_type_batch_all_online_roi/20221025-133446/data.anno.pb_rec",  # noqa
                sample_weight=24,
            ),
            dict(
                rec_path=f"{root}/data/traffic_light_type_cls/20230622_x8b_type_batch_all/20230621-153604/data.rec",  # noqa
                anno_path=f"{root}/data/traffic_light_type_cls/20230622_x8b_type_batch_all/20230621-153604/data.anno.pb_rec",  # noqa
                sample_weight=30,
            ),
            dict(
                rec_path=f"{root}/data/traffic_light_night_type_cls_badcase/20230622badcase_night_type_batch_all/20230621-160935/data.rec",  # noqa
                anno_path=f"{root}/data/traffic_light_night_type_cls_badcase/20230622badcase_night_type_batch_all/20230621-160935/data.anno.pb_rec",  # noqa
                sample_weight=3,
            ),
            dict(
                rec_path=f"{root}/data/traffic_light_type_cls_badcase/20230622_badcase_day_type_batch_all/20230621-170749/data.rec",  # noqa
                anno_path=f"{root}/data/traffic_light_type_cls_badcase/20230622_badcase_day_type_batch_all/20230621-170749/data.anno.pb_rec",  # noqa
                sample_weight=4,
            ),
            # sd 数据
            dict(
                rec_path=f"{root}/data/traffic_light_type_cls/20230531_sd_x8b_type_batch_all/20230531-235539/data.rec",  # noqa
                anno_path=f"{root}/data/traffic_light_type_cls/20230531_sd_x8b_type_batch_all/20230531-235539/data.anno.pb_rec",  # noqa
                sample_weight=4,
            ),
            # sd badcase数据
            dict(
                rec_path=f"{root}/data/traffic_light_type_cls_badcase/20230622_sd_badcase_day_type_batch_all/20230621-154606/data.rec",  # noqa
                anno_path=f"{root}/data/traffic_light_type_cls_badcase/20230622_sd_badcase_day_type_batch_all/20230621-154606/data.anno.pb_rec",  # noqa
                sample_weight=2,
            ),
            dict(
                rec_path=f"{root}/data/traffic_light_type_cls_badcase/20230622_sd_badcase_night_type_batch_all/20230621-211422/data.rec",  # noqa
                anno_path=f"{root}/data/traffic_light_type_cls_badcase/20230622_sd_badcase_night_type_batch_all/20230621-211422/data.anno.pb_rec",  # noqa
                sample_weight=2,
            ),
            # sd 窄角数据
            dict(
                rec_path=f"{root}/data/traffic_light_type_cls/20230622_sd_x8b_ov30_badcase_type_batch_all/20230621-155756/data.rec",  # noqa
                anno_path=f"{root}/data/traffic_light_type_cls/20230622_sd_x8b_ov30_badcase_type_batch_all/20230621-155756/data.anno.pb_rec",  # noqa
                sample_weight=1,
            ),
        ],
        val_batch_size_per_ctx=10,
        val_data_paths=[
            dict(
                rec_path=f"{root}/data/2pe_traffic_light_type_cls/tl_shell_type_2-2.0-val/data.rec",  # noqa
                anno_path=f"{root}/data/2pe_traffic_light_type_cls/tl_shell_type_2-2.0-val/data.json",  # noqa
                sample_weight=10,
            ),
        ],
    ),
    traffic_light_fp=dict(
        train_batch_size_per_ctx=115,
        train_data_paths=[
            dict(
                rec_path=f"{root}/data/traffic_light_fp_cls/20220809_220_fp_batch_all/20220809-155415/data.rec",  # noqa
                anno_path=f"{root}/fan.lv/data/traffic_light_fp_cls/20220809_220_fp_batch_all/20220809-155415/data.anno.pb_rec",  # noqa
                sample_weight=17,
            ),
            dict(
                rec_path=f"{root}/data/traffic_light_fp_cls/20220811_323_fp_batch_all/20220811-161955/data.rec",  # noqa
                anno_path=f"{root}/fan.lv/data/traffic_light_fp_cls/20220811_323_fp_batch_all/20220811-161955/data.anno.pb_rec",  # noqa
                sample_weight=26,
            ),
            dict(
                rec_path=f"{root}/data/traffic_light_fp_cls/20221024_820_fp_batch_lx_online_roi/20221024-235707/data.rec",  # noqa
                anno_path=f"{root}/data/traffic_light_fp_cls/20221024_820_fp_batch_lx_online_roi/20221024-235707/data.anno.pb_rec",  # noqa
                sample_weight=33,
            ),
            dict(
                rec_path=f"{root}/data/traffic_light_fp_cls/20221024_10652_fp_batch_all_online_roi/20221025-135632/data.rec",  # noqa
                anno_path=f"{root}/data/traffic_light_fp_cls/20221024_10652_fp_batch_all_online_roi/20221025-135632/data.anno.pb_rec",  # noqa
                sample_weight=12,
            ),
            dict(
                rec_path=f"{root}/data/traffic_light_fp_cls/20230622_x8b_fp_batch_all/20230621-153743/data.rec",  # noqa
                anno_path=f"{root}/data/traffic_light_fp_cls/20230622_x8b_fp_batch_all/20230621-153743/data.anno.pb_rec",  # noqa
                sample_weight=22,
            ),
            dict(
                rec_path=f"{root}/data/traffic_light_night_fp_cls/20230622_badcase_night_fp_batch_all/20230621-190902/data.rec",  # noqa
                anno_path=f"{root}/data/traffic_light_night_fp_cls/20230622_badcase_night_fp_batch_all/20230621-190902/data.anno.pb_rec",  # noqa
                sample_weight=3,
            ),
            dict(
                rec_path=f"{root}/data/traffic_light_fp_cls_badcase/20230622_badcase_day_fp_batch_all/20230621-183434/data.rec",  # noqa
                anno_path=f"{root}/data/traffic_light_fp_cls_badcase/20230622_badcase_day_fp_batch_all/20230621-183434/data.anno.pb_rec",  # noqa
                sample_weight=2,
            ),
            # sd 数据
            dict(
                rec_path=f"{root}/data/traffic_light_fp_cls/20230531_sd_x8b_fp_batch_all/20230601-004225/data.rec",  # noqa
                anno_path=f"{root}/data/traffic_light_fp_cls/20230531_sd_x8b_fp_batch_all/20230601-004225/data.anno.pb_rec",  # noqa
                sample_weight=2,
            ),
            # sd badcase
            dict(
                rec_path=f"{root}/data/traffic_light_fp_cls_badcase/20230622_sd_badcase_day_fp_batch_all/20230621-155012/data.rec",  # noqa
                anno_path=f"{root}/data/traffic_light_fp_cls_badcase/20230622_sd_badcase_day_fp_batch_all/20230621-155012/data.anno.pb_rec",  # noqa
                sample_weight=2,
            ),
            dict(
                rec_path=f"{root}/data/traffic_light_fp_cls_badcase/20230622_sd_badcase_night_fp_batch_all/20230621-160141/data.rec",  # noqa
                anno_path=f"{root}/data/traffic_light_fp_cls_badcase/20230622_sd_badcase_night_fp_batch_all/20230621-160141/data.anno.pb_rec",  # noqa
                sample_weight=1,
            ),
            # sd 窄角数据
            dict(
                rec_path=f"{root}/data/traffic_light_fp_cls/20230622_sd_x8b_ov30_badcase_fp_batch_all/20230621-214022/data.rec",  # noqa
                anno_path=f"{root}/data/traffic_light_fp_cls/20230622_sd_x8b_ov30_badcase_fp_batch_all/20230621-214022/data.anno.pb_rec",  # noqa
                sample_weight=1,
            ),
        ],
        val_batch_size_per_ctx=10,
        val_data_paths=[
            dict(
                rec_path=f"{root}/data/2pe_traffic_light_fp_cls/tl_shell_type_fp_vs_others-2.0-val/data.rec",  # noqa
                anno_path=f"{root}/data/2pe_traffic_light_fp_cls/tl_shell_type_fp_vs_others-2.0-val/data.json",  # noqa
                sample_weight=10,
            ),
        ],
    ),
    traffic_light_time=dict(
        train_batch_size_per_ctx=100,
        train_data_paths=[
            dict(
                rec_path=f"{root}/data/traffic_light_time_cls/20230417_x8b_timing_light/20230417-221156/data.rec",  # noqa
                anno_path=f"{root}/data/traffic_light_time_cls/20230417_x8b_timing_light/20230417-221156/data.anno.pb_rec",  # noqa
                sample_weight=20,
            ),
            dict(
                rec_path=f"{root}/data/traffic_light_time_cls/20230622_mono_x8b_timing_light/20230621-144833/data.rec",  # noqa
                anno_path=f"{root}/data/traffic_light_time_cls/20230622_mono_x8b_timing_light/20230621-144833/data.anno.pb_rec",  # noqa
                sample_weight=20,
            ),
            dict(
                rec_path=f"{root}/data/traffic_light_time_cls/20230417_820_timing_light/20230417-221445/data.rec",  # noqa
                anno_path=f"{root}/data/traffic_light_time_cls/20230417_820_timing_light/20230417-221445/data.anno.pb_rec",  # noqa
                sample_weight=13,
            ),
            dict(
                rec_path=f"{root}/data/traffic_light_time_cls/20230417_820_timing_light/20230418-022132/data.rec",  # noqa
                anno_path=f"{root}/data/traffic_light_time_cls/20230417_820_timing_light/20230418-022132/data.anno.pb_rec",  # noqa
                sample_weight=14,
            ),
            # mono badcase数据
            dict(
                rec_path=f"{root}/data/traffic_light_time_cls/20230622_mono_badcase_time_cls_batch_all/20230621-155536/data.rec",  # noqa
                anno_path=f"{root}/data/traffic_light_time_cls/20230622_mono_badcase_time_cls_batch_all/20230621-155536/data.anno.pb_rec",  # noqa
                sample_weight=2,
            ),
            # 多灯倒计时数据
            dict(
                rec_path=f"{root}/data/traffic_light_time_cls/20230531_sd_x8b_time_cls/20230601-002249/data.rec",  # noqa
                anno_path=f"{root}/data/traffic_light_time_cls/20230531_sd_x8b_time_cls/20230601-002249/data.anno.pb_rec",  # noqa
                sample_weight=30,
            ),
            # sd badcase数据
            dict(
                rec_path=f"{root}/data/traffic_light_time_cls/20230622_sd_badcase_day_time_cls/20230621-155220/data.rec",  # noqa
                anno_path=f"{root}/data/traffic_light_time_cls/20230622_sd_badcase_day_time_cls/20230621-155220/data.anno.pb_rec",  # noqa
                sample_weight=2,
            ),
            dict(
                rec_path=f"{root}/data/traffic_light_time_cls/20230622_sd_badcase_night_time_cls/20230621-191652/data.rec",  # noqa
                anno_path=f"{root}/data/traffic_light_time_cls/20230622_sd_badcase_night_time_cls/20230621-191652/data.anno.pb_rec",  # noqa
                sample_weight=2,
            ),
            dict(
                rec_path=f"{root}/data/traffic_light_time_cls/20230622_sd_x8b_ov30_time_cls/20230621-173129/data.rec",  # noqa
                anno_path=f"{root}/data/traffic_light_time_cls/20230622_sd_x8b_ov30_time_cls/20230621-173129/data.anno.pb_rec",  # noqa
                sample_weight=2,
            ),
        ],
        val_batch_size_per_ctx=10,
        val_data_paths=[
            dict(
                rec_path=f"{root}/data/traffic_light_time_cls/timing_light_x8b_validation/20221230-160653/data.rec",  # noqa
                anno_path=f"{root}/data/traffic_light_time_cls/timing_light_x8b_validation/20221230-160653/data.anno.pb_rec",  # noqa
                sample_weight=10,
            ),
        ],
    ),
)

datapaths = EasyDict(datapaths)
