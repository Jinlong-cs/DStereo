import os

rgb_length = 8


albu_params = [
    {
        "name": "Downscale",  # shift
        "scale_min": 0.4,
        "scale_max": 0.8,
        "p": 0.6,
    },
    {"name": "GaussianBlur", "blur_limit": (3, 7), "p": 0.2},
    {"name": "MotionBlur", "blur_limit": (3, 9), "p": 0.4},  # shift
    {
        "name": "JpegCompression",
        "quality_lower": 50,
        "quality_upper": 90,
        "p": 0.6,
    },
    {"name": "ToGray", "p": 0.3},
    {
        "name": "HueSaturationValue",
        "hue_shift_limit": 20,
        "sat_shift_limit": 30,
        "val_shift_limit": 15,
        "p": 0.6,
    },
    {
        "name": "RandomBrightnessContrast",
        "brightness_limit": 0.30,
        "contrast_limit": 0.20,
        "p": 1.0,
    },
    {"name": "FancyPCA", "alpha": 0.06, "p": 0.4},
    {"name": "GaussNoise", "var_limit": (10.0, 50.0), "mean": 0, "p": 0.6},
]

convert_label_map = {
    "1": 0,
    "2": 0,
    "6": -1,
    "8": 0,
    "9": 0,
    "10": 0,
    "11": 0,
    "12": 0,
    "13": 0,
    "14": 0,
    "15": 0,
    "16": 0,
    "18": 0,
    "19": 0,
    "20": -1,
    "21": 0,
    "22": 0,
    "23": 0,
    "24": 0,
    "25": 0,
    "26": 0,
    "27": 0,
    "28": 0,
    "29": 0,
    "30": -1,
    "31": 0,
    "32": 0,
    "33": 0,
    "34": 0,
    "35": 0,
    "36": 0,
    "37": 0,
    "38": 0,
    "39": 0,
    "40": 0,
    "41": 0,
    "42": 0,
    "43": 0,
    "44": 0,
    "45": 0,
    "46": 0,
    "47": 0,
    "48": 0,
    "49": 0,
    "50": -1,
    "51": -1,
    "52": 0,
    "53": 0,
    "54": 0,
    "55": 0,
    "56": 0,
    "57": 0,
    "58": 0,
    "59": 0,
    "60": 0,
    "61": 0,
    "62": 0,
    "63": 0,
    "64": 0,
    "65": 0,
    "66": 0,
    "67": 0,
    "68": -1,
}


def get_train_transforms():
    num_classes = 71
    seq_len = 16
    num_kps = 21
    use_3d_kps = False  # gluon.config.aug_params.use_3d_ldmk
    use_kps_score = True
    feat_ch = 4 if use_3d_kps else 3  # xyzs or xys
    if not use_kps_score:
        feat_ch -= 1  # remove score

    # flip # TODO
    flip_label_pairs = [[1, 2], [12, 13], [18, 19], [21, 22]]
    flip_label_dict = {}
    for label_pair in flip_label_pairs:
        assert isinstance(label_pair, list) and len(label_pair) == 2
        assert label_pair[0] not in flip_label_dict
        flip_label_dict[label_pair[0]] = label_pair[1]
        assert label_pair[1] not in flip_label_dict
        flip_label_dict[label_pair[1]] = label_pair[0]
    return [
        dict(
            # preprocess.py
            type="ActionClipDataPreProcess",
            act_kps_num_classes=num_classes,
            act_kps_seq_len=seq_len,
            num_kps=num_kps,
            use_3d_kps=use_3d_kps,
            use_kps_score=use_kps_score,
            act_img_seq_len=rgb_length,
            use_rgb_branch=True,
            use_replenish=True,
            repenish_max_frame=10000,
        ),
        dict(
            # affine.py
            type="ActionRotateKpsFrames",
            p_rotate=1,
            rotate_range=15,
            feat_ch=feat_ch,
            use_rgb_branch=True,
        ),
        dict(
            # landmark.py
            type="CropRecROI",
            crop_type="",
            target_shape=(128, 128, 3),
            base_roi=[48, 48, 208, 208],
            # optimized, gluonperson original 0.5, equivalent to 0.25 in HAT
            crop_jitter_range=0.16,
            center_shift_range=0.0,
            random_type="gaussian",
            crop_frames_prob_dist=[0.3, 0.7, 0.0],
            use_crop_limit=False,
        ),
        # add for static gesture
        dict(
            # frames_transform.py
            type="ActionImgClipAlbuTrans",
            albu_same_in_seq=True,
            albu_params=albu_params,
        ),
        dict(
            # affine.py
            type="ActionFlipKpsFrames",
            p_flip=0.5,
            feat_ch=feat_ch,
            flip_label_dict=flip_label_dict,
        ),
        dict(
            # label_transform.py
            type="ActionLabelMap",
            convert_label_map=convert_label_map,
        ),
        dict(
            # kps_transform.py
            type="ActionKpsReshape",
            num_kps=num_kps,
            feat_ch=feat_ch,
        ),
        dict(
            # kps_transform.py
            type="ActionKpsScoreNormalize",
            use_kps_score=use_kps_score,
            disable_kps_conf=False,
            use_3d_kps=use_3d_kps,
            kps_score_norm_scale=4.897640403536304,
        ),
        # Not used, just for demonstrate, p_smooth_kps=0
        dict(
            type="ActionKpsSmooth",
            p_smooth_kps=0,
            smooth_method="GaussianBlur",
        ),
        dict(
            # kps_transform.py
            type="ActionKpsNormalize",
            use_3d_kps=use_3d_kps,
            z_norm_type="z_abs_max",
            use_kps_score=use_kps_score,
            norm_type="center_norm",
            center_idxs=[9],
            center_type="fixed",
            multiscale_norm_ratio_list=[1.0],
            multi_ratio_channel="xyz",
            kps_num_list=[num_kps],
            fix_norm_len=None,
        ),
        dict(
            # kps_transform.py
            type="ActionKpsJitter",
            p_point_jitter=0.33,
            jitter_std_value=0.01,
        ),
        dict(
            # kps_transform.py
            type="ActionKpsRandScale",
            p_rand_scale=1.0,
            rand_scale_mu=1.0,
            rand_scale_sigma=0.1,
        ),
        dict(
            # kps_transform.py
            type="ActionKpsToTensor",
            tensor_layout="chw",
            use_3d_kps=use_3d_kps,
        ),
        dict(
            # label_transform.py
            type="ActionLabelToTensor",
            seq_len=seq_len,
            use_weight_box_vis_ratio=True,
        ),
        dict(
            # frames_transform.py
            type="ActionImgClipToTensor",
            to_yuv=True,
            tensor_layout="chw",
            img_layout="rgb",
            mean=(128.0, 128.0, 128.0),
            std=(128.0, 128.0, 128.0),
        ),
        dict(
            # preprocess.py
            type="ActionClipDataPostProcess",
            keep_keynames=[
                "act_label",
                # "seq_label",
                "label_weight",
                "clip_keypoints",
                "frames",
            ],
        ),
    ]


def get_test_transforms():

    num_classes = 71
    seq_len = 16
    num_kps = 21
    use_3d_kps = False
    use_kps_score = True
    feat_ch = 4 if use_3d_kps else 3  # xyzs or xys
    if not use_kps_score:
        feat_ch -= 1  # remove score

    return [
        dict(
            # from gluon.config.train.data_params.replenish
            type="ActionClipDataPreProcess",
            act_kps_num_classes=num_classes,
            act_kps_seq_len=seq_len,
            num_kps=num_kps,
            use_3d_kps=use_3d_kps,
            use_kps_score=use_kps_score,
            act_img_seq_len=8,
            use_rgb_branch=True,
            use_replenish=True,
            repenish_max_frame=10000,
        ),
        dict(
            # landmark.py
            type="CropRecROI",
            crop_type="",
            target_shape=(128, 128, 3),
            base_roi=[32, 32, 160, 160],
            crop_jitter_range=0.0,
            center_shift_range=0.0,
            random_type="gaussian",
            crop_frames_prob_dist=[0, 0, 1],
            use_crop_limit=False,
        ),
        dict(
            # label_transform.py
            type="ActionLabelMap",
            convert_label_map=convert_label_map,
        ),
        dict(
            type="ActionKpsReshape",
            num_kps=num_kps,
            feat_ch=feat_ch,
        ),
        dict(
            type="ActionKpsScoreNormalize",
            use_kps_score=use_kps_score,
            disable_kps_conf=False,
            use_3d_kps=use_3d_kps,
        ),
        dict(
            type="ActionKpsSmooth",
            p_smooth_kps=0,
            smooth_method="GaussianBlur",
        ),
        dict(
            type="ActionKpsNormalize",
            use_3d_kps=use_3d_kps,
            z_norm_type="z_abs_max",
            use_kps_score=use_kps_score,
            norm_type="center_norm",
            center_idxs=[9],
            center_type="fixed",
            multiscale_norm_ratio_list=[1.0],
            multi_ratio_channel="xyz",
            kps_num_list=[num_kps],
            fix_norm_len=None,
        ),
        dict(
            type="ActionKpsToTensor",
            tensor_layout="chw",
            use_3d_kps=use_3d_kps,
        ),
        dict(
            type="ActionLabelToTensor",
            seq_len=seq_len,
            use_weight_box_vis_ratio=False,
        ),
        # add for static gesture
        dict(
            # frames_transform.py
            type="ActionImgClipToTensor",
            to_yuv=True,
            tensor_layout="chw",
            img_layout="rgb",
            mean=(128.0, 128.0, 128.0),
            std=(128.0, 128.0, 128.0),
        ),
        dict(
            type="ActionClipDataPostProcess",
            keep_keynames=[
                "act_label",
                # "seq_label",
                "label_weight",
                "clip_keypoints",
                "frames",
            ],
        ),
    ]


def get_lmdb_train_data_list(small_data=True):
    white_box_train_datasets = [
        "H9_hand_gesture_210728_210729_210730_210802_210803_static_v2.1_ldmk2.5dv3.0.3",  # sta_train_1 # noqa
        "C281_hand_gesture_211108_211109_211110_211111_211112_static_v2.0_ldmk2.5dv3.0.3",  # sta_train_3 # noqa
        "GY_hand_gesture_MT_210801_210816_cam3_static_v2.1_ldmk2.5dv3.0.3",  # sta_train_8 # noqa
    ]

    white_box_val_datasets = [
        "CD569_hand_gesture_mute_front_IMS-RGB_v1_74506_static_v2.0_ldmk2.5dv3.0.3",  # sta_test_0 # noqa
        "E300_hand_gesture_static_IMS-RGB_front_all_static_v2.0_ldmk2.5dv3.0.3",  # sta_test_1 # noqa
        "CP_hand_gesture_A13_static_front_IMS_RGB_static_v2.0_ldmk2.5dv3.0.3",  # sta_test_2 # noqa
    ]

    # train and validation data list is from
    # gluon - scripts/action/act_kps/examples/hgr_cockpit/config_cockpit_halo_hgr_v3_static_v1.0.7_rgbkps.py # noqa
    all_train_dataset = [
        # sta_00
        # **********************  动态手势的一批数据  ********************** #
        # CA S202采集实车数据,720p, 动态左右挥手的正样本, 52W+6.6W+6.1W
        "Ca_hand_gesture_20200902_siyuan_dynamic_v2.0_ldmk2.5dv3.0.3",
        "Ca_hand_gesture_20200826_anno_20200901_dynamic_v2.0_ldmk2.5dv3.0.3",
        "Ca_hand_gesture_20200902_mengzhen_dynamic_v2.0_ldmk2.5dv3.0.3",
        "Ca_hand_gesture_20200907_dynamic_v2.0_ldmk2.5dv3.0.3",
        "Ca_hand_gesture_20200910_dynamic_v2.0_ldmk2.5dv3.0.3",
        "Ca_hand_gesture_20200915_dynamic_v2.0_ldmk2.5dv3.0.3",
        "Ca_hand_gesture_20200916_dynamic_v2.0_ldmk2.5dv3.0.3",
        "Ca_hand_gesture_20200920_dynamic_v2.0_ldmk2.5dv3.0.3",
        "Ca_hand_gesture_20200921_dynamic_v2.0_ldmk2.5dv3.0.3",
        "Ca_hand_gesture_20200922_dynamic_v2.0_ldmk2.5dv3.0.3",
        "Ca_hand_gesture_20200923_dynamic_v2.0_ldmk2.5dv3.0.3",
        # # 南京S202DA采集实车数据,960p, 动态手势左右挥手正样本, 200W+13W+13W
        # 'Ca_hand_gesture_20201109_anno_dynamic_v2.1_ldmk2.5dv3.0.3',
        # 'Ca_hand_gesture_20201119_anno_dynamic_v2.1_ldmk2.5dv3.0.3',
        # 'Ca_hand_gesture_20201120_anno_dynamic_v2.1_ldmk2.5dv3.0.3',
        # 'Ca_hand_gesture_20201202_anno_dynamic_v2.1_ldmk2.5dv3.0.3',
        # 'Ca_hand_gesture_20201203_anno_dynamic_v2.1_ldmk2.5dv3.0.3',
        # 'Ca_hand_gesture_20201204_anno_dynamic_v2.1_ldmk2.5dv3.0.3',
        # 'Ca_hand_gesture_20201207_anno_dynamic_v2.1_ldmk2.5dv3.0.3',
        # 'Ca_hand_gesture_20201208_anno_dynamic_v2.1_ldmk2.5dv3.0.3',
        # 'Ca_hand_gesture_20201209_anno_dynamic_v2.1_ldmk2.5dv3.0.3',
        # 致恺提供的一批抽烟打电话负样本数据,包含较多手型,用作手势的负样本, 23W
        "CaBg_hand_gesture_20201112zkdms_bg_dynamic_v2.0_ldmk2.5dv3.0.3",
        # 南京CD569采集的左右挥手正样本,1080P, 19W+1.5W+1.5W
        "Ca_hand_gesture_20210114_anno_dynamic_v2.0_ldmk2.5dv3.0.3",
        "Ca_hand_gesture_20210111_anno_dynamic_v2.0_ldmk2.5dv3.0.3",
        # 南京S202（S202DA）采集的抽烟打电话数据,用作手势负样本, 200W
        "CaBg_hand_gesture_20200305_dms_bg_dynamic_v2.0_ldmk2.5dv3.0.3",
        "CaBg_hand_gesture_20210118_dms_bg_dynamic_v2.0_ldmk2.5dv3.0.3",
        "CaBg_hand_gesture_20210125_dms_bg_dynamic_v2.0_ldmk2.5dv3.0.3",
        "CaBg_hand_gesture_20210201_dms_bg_dynamic_v2.0_ldmk2.5dv3.0.3",
        "CaBg_hand_gesture_20210222_dms_bg_dynamic_v2.0_ldmk2.5dv3.0.3",
        "CaBg_hand_gesture_20210315_dms_bg_dynamic_v2.0_ldmk2.5dv3.0.3",
        "CaBg_hand_gesture_20210329_bg_dynamic_v2.1_ldmk2.5dv3.0.3",
        # s202da自然驾驶误报数据
        "CaBg_hand_gesture_s202da_nds_total_rgb_dynamic_v2.0_ldmk2.5dv3.0.3",
        # **********************  动态手势的一批数据  ********************** #
        # **********************  x3数据  ********************** #
        # sta_01
        # # x3demo二期手势数据, 见 http://wiki.hobot.cc/pages/viewpage.action?pageId=143294581#id-%E6%89%8B%E5%8A%BF%E8%AF%86%E5%88%ABX3%E4%B8%BB%E5%8A%A8%E4%BA%A4%E4%BA%92%E4%BA%8C%E6%9C%9F%E6%95%B0%E6%8D%AE%E9%87%87%E9%9B%86-%E6%95%B0%E6%8D%AE%E8%BF%94%E5%9B%9E # noqa
        "X3_hand_gesture_20201107_static_v2.7_ldmk2.5dv3.0.3",
        "X3_hand_gesture_20201106_static_v2.7_ldmk2.5dv3.0.3",
        "X3_hand_gesture_20201120_ICPARK_static_v2.7_ldmk2.5dv3.0.3",
        "X3_hand_gesture_20201116_static_v2.7_ldmk2.5dv3.0.3",
        "X3_hand_gesture_20201109_static_v2.7_ldmk2.5dv3.0.3",
        "X3_hand_gesture_20201110_static_v2.7_ldmk2.5dv3.0.3",
        "X3_hand_gesture_20201120_static_v2.7_ldmk2.5dv3.0.3",
        # sta_02
        # 外部供应商手机采集的tcl x12手势数据,3个角度,模特为儿童
        # "x3tcl_hand_gesture_appen_20210205_static_v2.7_ldmk2.5dv3.0.3",
        # sta_03
        # # 北京9楼采集tcl手势数据第二批,见 http://wiki.hobot.cc/pages/viewpage.action?pageId=162902530#id-%E6%89%8B%E5%8A%BF%E8%AF%86%E5%88%ABX3%E4%B8%BB%E5%8A%A8%E4%BA%A4%E4%BA%92%E4%B8%89%E6%9C%9F%EF%BC%88TCLX12%EF%BC%89%E6%95%B0%E6%8D%AE%E9%87%87%E9%9B%86-%E7%AC%AC%E4%BA%8C%E6%89%B9  # noqa
        # # (1.0+0.55+0.27), 105W(AS)+97W(PRC)+97W(PRAC)
        "x3tcl_hand_gesture_v1.1_20210122_ok_ID1_10_static_v2.8_ldmk2.5dv3.0.3",
        "x3tcl_hand_gesture_v1.1_20210122_ok_ID11_20_static_v2.8_ldmk2.5dv3.0.3",
        "x3tcl_hand_gesture_v1.1_20210122_ok_ID21_30_static_v2.8_ldmk2.5dv3.0.3",
        "x3tcl_hand_gesture_v1.1_20210122_ok_ID31_40_static_v2.8_ldmk2.5dv3.0.3",
        "x3tcl_hand_gesture_v1.1_20210122_ok_ID41_50_static_v2.8_ldmk2.5dv3.0.3",
        "x3tcl_hand_gesture_v1.1_20210122_ok_ID51_60_static_v2.8_ldmk2.5dv3.0.3",
        "x3tcl_hand_gesture_v1.1_20210122_ok_ID61_70_static_v2.8_ldmk2.5dv3.0.3",
        "x3tcl_hand_gesture_v1.1_20210122_ok_ID71_74_static_v2.8_ldmk2.5dv3.0.3",
        # **********************  x3数据  ********************** #
        # sta_04
        # # 北京9楼采集的TCL手势数据第一批,见 http://wiki.hobot.cc/pages/viewpage.action?pageId=162902530#id-%E6%89%8B%E5%8A%BF%E8%AF%86%E5%88%ABX3%E4%B8%BB%E5%8A%A8%E4%BA%A4%E4%BA%92%E4%B8%89%E6%9C%9F%EF%BC%88TCLX12%EF%BC%89%E6%95%B0%E6%8D%AE%E9%87%87%E9%9B%86-%E7%AC%AC%E4%B8%80%E6%89%B9  # noqa
        # # (1.0+0.55+0.27), 72W(TU)+133W(VT)+131W(OK)+129W(MT)+253W(PL)+71W(LE)+69W(RI)
        "x3tcl_hand_gesture_v1.1_20210108_20210114_ID1_10_static_v2.8_ldmk2.5dv3.0.3",
        "x3tcl_hand_gesture_v1.1_20210108_20210114_ID11_20_static_v2.8_ldmk2.5dv3.0.3",
        "x3tcl_hand_gesture_v1.1_20210108_20210114_ID21_30_static_v2.8_ldmk2.5dv3.0.3",
        "x3tcl_hand_gesture_v1.1_20210108_20210114_ID31_40_static_v2.8_ldmk2.5dv3.0.3",
        "x3tcl_hand_gesture_v1.1_20210108_20210114_ID41_50_static_v2.8_ldmk2.5dv3.0.3",
        "x3tcl_hand_gesture_v1.1_20210108_20210114_ID51_60_static_v2.8_ldmk2.5dv3.0.3",
        "x3tcl_hand_gesture_v1.1_20210108_20210114_ID61_70_static_v2.8_ldmk2.5dv3.0.3",
        "x3tcl_hand_gesture_v1.1_20210108_20210114_ID71_80_static_v2.8_ldmk2.5dv3.0.3",
        "x3tcl_hand_gesture_v1.1_20210108_20210114_ID81_90_static_v2.8_ldmk2.5dv3.0.3",
        # sta_05
        # # 南京和北京采集的小批量静态手势误报负样本,手机采集, 76W(9f*1+nj*1)
        "x3tcl_hand_gesture_20210319_bg_nj_static_v2.9_ldmk2.5dv3.0.3",
        "x3tcl_hand_gesture_20210319_bg_9f_static_v2.9_ldmk2.5dv3.0.3",
        "x3tcl_hand_gesture_20210319_bg_nj_static_v2.9_ldmk2.5dv3.0.3",
        "x3tcl_hand_gesture_20210319_bg_9f_static_v2.9_ldmk2.5dv3.0.3",
        "x3tcl_hand_gesture_20210319_bg_nj_static_v2.9_ldmk2.5dv3.0.3",
        "x3tcl_hand_gesture_20210319_bg_9f_static_v2.9_ldmk2.5dv3.0.3",
        # sta_06
        # # 南京实车+工控机+800W非量产摄像头采集的数据,未清洗
        "CANJ_hand_gesture_210428_210507_210508_210510_210511_static_v2.1_ldmk2.5dv3.0.3",
        "CANJ_hand_gesture_210512_210513_210514_210515_210517_210518_210519_static_v2.1_ldmk2.5dv3.0.3",
        "CANJ_hand_gesture_210520_210521_210524_210526_210527_static_v2.1_ldmk2.5dv3.0.3",
        "CANJ_hand_gesture_210528_210529_210531_210601_210602_210603_static_v2.1_ldmk2.5dv3.0.3",
        "CANJ_hand_gesture_210611_210615_210616_static_v2.1_ldmk2.5dv3.0.3",
        "CANJ_hand_gesture_210617_210618_210621_210622_210623_210624_static_v2.1_ldmk2.5dv3.0.3",
        # sta_07
        # # 贵阳实车+工控机+800W非量产摄像头采集的数据,标注过,
        # # 17W(FH)+19W(TU)+22W(VT)+15W(OK)
        "GyFhTuVtOk_hand_gesture_9815_9816_static_v2.1_ldmk2.5dv3.0.3",
        # sta_08
        # 贵阳实车+工控机+800W非量产摄像头采集的静态手势专项负样本数据,包含一些人为预判和静态手势比较相似的动作, 843W
        "GYBG_hand_gesture_guiyang_20210710_static_bg_static_v2.1_ldmk2.5dv3.0.3",
        "GYBG_hand_gesture_guiyang_20210706_static_bg_static_v2.1_ldmk2.5dv3.0.3",
        "GYBG_hand_gesture_guiyang_20210705_static_bg_static_v2.1_ldmk2.5dv3.0.3",
        "GYBG_hand_gesture_guiyang_20210704_static_bg_static_v2.1_ldmk2.5dv3.0.3",
        "GYBG_hand_gesture_guiyang_20210709_static_bg_static_v2.1_ldmk2.5dv3.0.3",
        "GYBG_hand_gesture_guiyang_20210711_static_bg_static_v2.1_ldmk2.5dv3.0.3",
        "GYBG_hand_gesture_guiyang_20210707_static_bg_static_v2.1_ldmk2.5dv3.0.3",
        "GYBG_hand_gesture_guiyang_20210703_static_bg_static_v2.1_ldmk2.5dv3.0.3",
        "GYBG_hand_gesture_guiyang_20210712_static_bg_static_v2.1_ldmk2.5dv3.0.3",
        "GYBG_hand_gesture_guiyang_20210708_static_bg_static_v2.1_ldmk2.5dv3.0.3",
        # sta_09
        # H9 sta, H9静态手势训练集,未清洗
        # # 4W(FH)+4W(TU)+4W(VT)+4W(OK)
        "H9_hand_gesture_210721_210724_210726_210727_static_v2.1_ldmk2.5dv3.0.3",
        "H9_hand_gesture_210728_210729_210730_210802_210803_static_v2.1_ldmk2.5dv3.0.3",
        "H9_hand_gesture_210721_210724_210726_210727_static_v2.1_ldmk2.5dv3.0.3",
        "H9_hand_gesture_210728_210729_210730_210802_210803_static_v2.1_ldmk2.5dv3.0.3",
        "H9_hand_gesture_210721_210724_210726_210727_static_v2.1_ldmk2.5dv3.0.3",
        "H9_hand_gesture_210728_210729_210730_210802_210803_static_v2.1_ldmk2.5dv3.0.3",
        # sta_10
        # # H9采集的抽烟数据,用来作为手势的误报训练集, 50W
        "H9BG_hand_gesture_h9_smoke_static_v2.0_ldmk2.5dv3.0.3",
        # sta_11
        # # H9采集的静态手势误报负样本，(基于TR51_v0.8.1), 数据量47W+6.6W
        # H9采集的静态手势误报负样本，(基于TR51_v0.8.1), 数据量47W+6.6W, 清洗一轮后
        # ##### wangrui clean  #######
        "H9BG_hand_gesture_20210906-20210910_hgr_v2_sta_081_fp_h9_static_v2.1_ldmk2.5dv3.0.3",
        "H9BG_hand_gesture_20210914-20210916_hgr_v2_sta_081_fp_h9_static_v2.1_ldmk2.5dv3.0.3",
        # ##### wangrui clean  #######
        # sta_12
        # C281新增比6朝上,比6朝下训练集
        "C281_hand_gesture_211020_211021_211025_211026_211027_211028_211103_static_v2.0_ldmk2.5dv3.0.3",
        "C281_hand_gesture_211108_211109_211110_211111_211112_static_v2.0_ldmk2.5dv3.0.3",
        # # C281实车静态手势（比心,点赞,胜利）10 ID训练集
        "C281_hand_gesture_211206_211207_static_v2.0_ldmk2.5dv3.0.3",
        # sta_13
        # 静态手势比心/点赞/胜利/OK 正样本训练集
        # CD569 bj, CD569 北京紧急采集13ID训练数据,多人多手,未清洗
        # # 2W(FH)+0.7W(TU)+2W(VT)
        "CD569_hand_gesture_20211103_CD569_train_static_v2.0_ldmk2.5dv3.0.3",
        "CD569_hand_gesture_20211103_CD569_train_static_v2.0_ldmk2.5dv3.0.3",
        "CD569_hand_gesture_20211103_CD569_train_static_v2.0_ldmk2.5dv3.0.3",
        "CD569_hand_gesture_20211103_CD569_train_static_v2.0_ldmk2.5dv3.0.3",
        "CD569_hand_gesture_20211103_CD569_train_static_v2.0_ldmk2.5dv3.0.3",
        "CD569_hand_gesture_20211103_CD569_train_static_v2.0_ldmk2.5dv3.0.3",
        "CD569_hand_gesture_20211103_CD569_train_static_v2.0_ldmk2.5dv3.0.3",
        "CD569_hand_gesture_20211103_CD569_train_static_v2.0_ldmk2.5dv3.0.3",
        "CD569_hand_gesture_20211103_CD569_train_static_v2.0_ldmk2.5dv3.0.3",
        "CD569_hand_gesture_20211103_CD569_train_static_v2.0_ldmk2.5dv3.0.3",
        # sta_14
        # cd569 nj 20ID,CD569南京采集20ID静态手势训练数据,未清洗
        # # 7W(FH)+7W(TU)+7W(VT)+7W(OK)
        "CD569_hand_gesture_20211102_20211105_CD569_static-gesture_static_v2.0_ldmk2.5dv3.0.3",
        "CD569_hand_gesture_20211102_20211105_CD569_static-gesture_static_v2.0_ldmk2.5dv3.0.3",
        "CD569_hand_gesture_20211102_20211105_CD569_static-gesture_static_v2.0_ldmk2.5dv3.0.3",
        # sta_15
        # # CD569 静态手势比心/点赞/胜利/OK 负样本训练集
        # # CD569 fp, CD569采集静态手势误报负样本（根据TR51_0.7.3）, 38W
        "CD569BG_hand_gesture_20210827_569-2_badcase_static_v2.0_ldmk2.5dv3.0.3",
        "CD569BG_hand_gesture_20210827_569-2_badcase_static_v2.0_ldmk2.5dv3.0.3",
        "CD569BG_hand_gesture_20210827_569-2_badcase_static_v2.0_ldmk2.5dv3.0.3",
        # sta_16
        # 北京10楼采集6个ID误报负样本，cd569台架采集
        "CD569BG_hand_gesture_hgr_v2_sta_bg_rgbkps_hardcase_static_v2.0_ldmk2.5dv3.0.3",
        "CD569BG_hand_gesture_hgr_v2_sta_bg_rgbkps_hardcase_static_v2.0_ldmk2.5dv3.0.3",
        "CD569BG_hand_gesture_hgr_v2_sta_bg_rgbkps_hardcase_static_v2.0_ldmk2.5dv3.0.3",
        # sta_17
        # guangqi sta, 广汽静态手势训练集,6/8个车型有数据,其中A55数据只有1天（3ID）,未清洗
        # # 16W(FH)+16W(TU)+16W(VT)+16W(OK)
        "GuangQi_a11_0816-0825_hand_gesture_a11_static_v2.0_ldmk2.5dv3.0.3",
        "GuangQi_a13_0825-0903_hand_gesture_a13_static_v2.0_ldmk2.5dv3.0.3",
        "GuangQi_a20_0825-0903_hand_gesture_a20_static_v2.0_ldmk2.5dv3.0.3",
        "GuangQi_a29_0825-0903_hand_gesture_a29_static_v2.0_ldmk2.5dv3.0.3",
        "GuangQi_a55_0816_hand_gesture_a55_static_v2.0_ldmk2.5dv3.0.3",
        "GuangQi_a60_0816-0824_hand_gesture_a60_static_v2.0_ldmk2.5dv3.0.3",
        # sta_18
        # CD569+广汽A18+A57+A55 正样本 比心/点赞/胜利/OK
        "CD569_hand_gesture_CD569_211102_211105_static_v2.0_ldmk2.5dv3.0.3",
        # (3: 127399, 4: 127345, 5: 132944, 17: 127663)
        "GuangQi_hand_gesture_A18_A55_A57_211108_211116_static_v2.0_ldmk2.5dv3.0.3",
        # sta_19
        # # A11 hand with objects,A11采集静态手势负样本数据,包含手持物和手背朝前的负样本,未清洗, 76W
        "GuangQiBg_handobj_hand_gesture_20211101_A11_static-gesture_static_v2.0_ldmk2.5dv3.0.3",
        "GuangQiBg_handobj_hand_gesture_20211102_A11_static-gesture_static_v2.0_ldmk2.5dv3.0.3",
        "GuangQiBg_handobj_hand_gesture_20211027_A11_static-gesture_static_v2.0_ldmk2.5dv3.0.3",
        "GuangQiBg_handobj_hand_gesture_20211028_A11_static-gesture_static_v2.0_ldmk2.5dv3.0.3",
        "GuangQiBg_handobj_hand_gesture_20211029_A11_static-gesture_static_v2.0_ldmk2.5dv3.0.3",
        # sta_20
        # # # A13+A55+A57的误报负样本训练集, 150W
        "GuangQiBg_neg_hand_gesture_A13_211130_211201_211202_static_v2.0_ldmk2.5dv3.0.3",
        "GuangQiBg_neg_hand_gesture_A55_211130_211201_211202_static_v2.0_ldmk2.5dv3.0.3",
        "GuangQiBg_neg_hand_gesture_A57_211201_211202_static_v2.0_ldmk2.5dv3.0.3",
        # sta_21
        # 针对广汽 0.3.7版本分析的badcase，38W
        "BG_hand_gesture_v3_a11_a20_a60_exp0029_bg_20211216_static_v2.0_ldmk2.5dv3.0.3",
        "BG_hand_gesture_v3_a11_a20_a60_exp0029_bg_20211217_static_v2.0_ldmk2.5dv3.0.3",
        # sta_22
        # 广汽 A60+A11+A20 正样本 比心/点赞/胜利/OK
        # 60323   64407   64856   65408
        "GuangQi_hand_gesture_another_a11_a20_a60_20211220_20211221_static_v2.0_ldmk2.5dv3.0.3",
        "GuangQi_hand_gesture_A11_A20_A60_20211220_20211221_static_v2.0_ldmk2.5dv3.0.3",
        # sta_23
        # 广汽 A60+A11+A20 负样本
        "GuangQiBg_hand_gesture_A11_A20_A60_20211220_20211221_bg_static_v2.0_ldmk2.5dv3.0.3",
        "GuangQiBg_hand_gesture_A11_A20_A60_20211222_bg_static_v2.0_ldmk2.5dv3.0.3",
        "GuangQiBg_hand_gesture_A11_A20_A60_20211223_bg_static_v2.0_ldmk2.5dv3.0.3",
        # sta_24
        # # 广汽 A18+A13+A20/A29 过年时候采集的一批，背景衣服颜色多样化 正样本 比心/点赞/胜利/OK
        # "GuangQi_A13_A18_A20_hand_gesture_220129_220207_dynamic_v2.0_ldmk2.5dv3.0.3",
        # sta_25
        # A20实车体验数据（包含抽烟打电话和手势），数据量4.7W
        "GuangQiBg_expnds_hand_gesture_A20_static_IMS-RGB_static_v2.0_ldmk2.5dv3.0.3",
        "GuangQiBg_expnds_hand_gesture_A20_static_IMS-RGB_static_v2.0_ldmk2.5dv3.0.3",
        # # A55实车体验数据（包含抽烟打电话和手势），数据量0.8W
        "GuangQiBg_expnds_hand_gesture_A55_static_IMS-IR_static_v2.0_ldmk2.5dv3.0.3",
        # # A55实车体验数据（包含抽烟打电话和手势），数据量10W
        "GuangQiBg_expnds_hand_gesture_A55_static_IMS-RGB_static_v2.0_ldmk2.5dv3.0.3",
        # A29实车体验数据, 13W
        "GuangQiBg_expnds_hand_gesture_A29_static_IMS-RGB_static_v2.0_ldmk2.5dv3.0.3",
        # sta_26
        # A13 fp, A13采集的静态手势误报负样本数据（根据TR51_0.7.3）, 7W(0810*1+0811*1)
        "GuangQiBg_hand_gesture_A13_20210810_static_v2.0_ldmk2.5dv3.0.3",
        "GuangQiBg_hand_gesture_A13_20210811_static_v2.0_ldmk2.5dv3.0.3",
        "GuangQiBg_hand_gesture_A13_20210810_static_v2.0_ldmk2.5dv3.0.3",
        "GuangQiBg_hand_gesture_A13_20210811_static_v2.0_ldmk2.5dv3.0.3",
        # sta_27
        # 算法实车自测数据（攻击型，针对exp0029） A29+A55
        # 针对exp0029的A29实车badcase, 5W
        "GuangQiBg_hand_gesture_A29_train_20211203_static_v2.0_ldmk2.5dv3.0.3",
        "GuangQiBg_hand_gesture_A29_test_20211202_static_v2.0_ldmk2.5dv3.0.3",
        "GuangQiBg_hand_gesture_A29_train_20211203_static_v2.0_ldmk2.5dv3.0.3",
        "GuangQiBg_hand_gesture_A29_test_20211202_static_v2.0_ldmk2.5dv3.0.3",
        "GuangQiBg_hand_gesture_A29_train_20211203_static_v2.0_ldmk2.5dv3.0.3",
        "GuangQiBg_hand_gesture_A29_test_20211202_static_v2.0_ldmk2.5dv3.0.3",
        # 针对exp0029的A55实车badcase, 3W
        "GuangQiBg_hand_gesture_A55_imsir_train_20211204_static_v2.0_ldmk2.5dv3.0.3",
        "GuangQiBg_hand_gesture_A55_imsir_train_20211204_static_v2.0_ldmk2.5dv3.0.3",
        "GuangQiBg_hand_gesture_A55_imsir_train_20211204_static_v2.0_ldmk2.5dv3.0.3",
        # 针对exp0029的A55 ims-rgb实车badcase, 4W
        "GuangQiBg_hand_gesture_A55_imsrgb_train_20211204_static_v2.0_ldmk2.5dv3.0.3",
        "GuangQiBg_hand_gesture_A55_imsrgb_train_20211204_static_v2.0_ldmk2.5dv3.0.3",
        "GuangQiBg_hand_gesture_A55_imsrgb_train_20211204_static_v2.0_ldmk2.5dv3.0.3",
        # sta_28
        # 根利提供的手持物数据 C281+S311+H9
        "BG_hand_obj_hand_gesture_detect-common-object_static_v2.0_ldmk2.5dv3.0.3",
        "CP_bg_hand_gesture_handobj_210930_211013_C281_ov2311_imx290_train_static_v2.0_ldmk2.5dv3.0.3",
        "CP_bg_hand_gesture_handobj_211019_211022_S311_ov2311_ov2775_train_static_v2.0_ldmk2.5dv3.0.3",
        # **********************  仰坤提供的抽烟打电话数据  ********************** #
        # sta_29
        # There are 1244815 samples with label 0 for phone IR
        # There are 2249354 samples with label 0 for phone RGB
        # after down sample: 31w IR + 11w RGB
        "BG_hand_gesture_guangqi_a11_phone_IR_static_v2.1.1_ldmk2.5dv3.0.3",
        "BG_hand_gesture_guangqi_a13_phone_IR_static_v2.1.1_ldmk2.5dv3.0.3",
        "BG_hand_gesture_guangqi_a18_phone_IR_static_v2.1.1_ldmk2.5dv3.0.3",
        "BG_hand_gesture_guangqi_a20_phone_IR_static_v2.1.1_ldmk2.5dv3.0.3",
        "BG_hand_gesture_guangqi_a29_phone_IR_static_v2.1.1_ldmk2.5dv3.0.3",
        "BG_hand_gesture_guangqi_a55_phone_IR_static_v2.1.1_ldmk2.5dv3.0.3",
        "BG_hand_gesture_guangqi_a57_phone_IR_static_v2.1.1_ldmk2.5dv3.0.3",
        "BG_hand_gesture_guangqi_a60_phone_IR_static_v2.1.1_ldmk2.5dv3.0.3",
        "BG_hand_gesture_guangqi_a11_phone_RGB_static_v2.1.1_ldmk2.5dv3.0.3",
        "BG_hand_gesture_guangqi_a13_phone_RGB_static_v2.1.1_ldmk2.5dv3.0.3",
        "BG_hand_gesture_guangqi_a18_phone_RGB_static_v2.1.1_ldmk2.5dv3.0.3",
        "BG_hand_gesture_guangqi_a29_phone_RGB_static_v2.1.1_ldmk2.5dv3.0.3",
        "BG_hand_gesture_guangqi_a55_phone_RGB_static_v2.1.1_ldmk2.5dv3.0.3",
        "BG_hand_gesture_guangqi_a57_phone_RGB_static_v2.1.1_ldmk2.5dv3.0.3",
        "BG_hand_gesture_guangqi_a60_phone_RGB_static_v2.1.1_ldmk2.5dv3.0.3",
        # 4558388 RGB_smoke and 1376675 IR_smoke
        # after down sample: 30w RGB + 13w IR
        "BG_hand_gesture_a11_a13_a18_a20_a29_a55_a57_a60_RGB_smoke_static_v2.0.1_ldmk2.5dv3.0.3",
        "BG_hand_gesture_a11_a13_a18_a20_a29_a55_a57_a60_IR_smoke_static_v2.1.1_ldmk2.5dv3.0.3",
        # sta_30
        # 625306 smoke rgb-4cam-800w for cam3
        # after down sample: 31w
        "BG_hand_gesture_210526_210528_210529_210531_210601_210602_cam3_smokebg_static_v2.1.1_ldmk2.5dv3.0.3",
        "BG_hand_gesture_210604_210607_210608_210609_210610_210611_cam3_smokebg_static_v2.1.1_ldmk2.5dv3.0.3",
        "BG_hand_gesture_210615_210617_210618_210622_210623_210624_210625_210626_cam3_smokebg_static_v2.1.1_ldmk2.5dv3.0.3",
        "BG_hand_gesture_210621_210603_cam3_smokebg_static_v2.1.1_ldmk2.5dv3.0.3",
        # smoke neg (652485+2739833+1220050+3874659)
        # after down sample: 13w+27w+12w+38w
        "BG_hand_gesture_a11_a13_a18_a20_IR_smoke_neg_static_v2.1.1_ldmk2.5dv3.0.3",
        "BG_hand_gesture_a11_a13_a18_a20_RGB_smoke_neg_static_v2.0.1_ldmk2.5dv3.0.3",
        "BG_hand_gesture_a29_a55_a57_a60_IR_smoke_neg_static_v2.1.1_ldmk2.5dv3.0.3",
        "BG_hand_gesture_a29_a55_a57_a60_RGB_smoke_neg_static_v2.0.1_ldmk2.5dv3.0.3",
        # **********************  仰坤提供的抽烟打电话数据  ********************** #
        # sta_31
        # T18 正样本 比心/点赞/胜利/OK
        "CP_sta_hand_gesture_T18_220126_220127_220128_220129_220130_220207_static_v2.2.1_ldmk2.5dv3.0.6",
        "CP_sta_hand_gesture_T18_220208_220209_220211_220213_220214_220215_static_v2.2.1_ldmk2.5dv3.0.6",
        # MT
        # sta_07  MT
        "GY_hand_gesture_MT_210621_210629_cam3_static_v2.1_ldmk2.5dv3.0.3",
        "GY_hand_gesture_MT_210801_210816_cam3_static_v2.1_ldmk2.5dv3.0.3",
        "GY_hand_gesture_MT_210722_210731_cam3_static_v2.1_ldmk2.5dv3.0.3",
        # sta_32  MT
        "C281_A55_A60_hand_gesture_MT_220117_220223_dynamic_v2.0_ldmk2.5dv3.0.3",
        "C281_A55_A60_hand_gesture_MT_220117_220223_dynamic_v2.0_ldmk2.5dv3.0.3",
        # awesome
        # sta_32  awesome
        "CD569_C281_hand_gesture_awesome_220117_220223_static_v2.0_ldmk2.5dv3.0.3",
        "A55_A60_A18_A20_hand_gesture_awesome_220117_220223_static_v2.0_ldmk2.5dv3.0.3",
        # sta_33 e300
        "E300_hand_gesture_fhtuvtok_220228_220303_static_v2.0_ldmk2.5dv3.0.3",
        "E300_hand_gesture_fhtuvtok_220228_220303_static_v2.0_ldmk2.5dv3.0.3",
    ]

    all_val_dataset = [
        # CD569召回测试集
        "CP_hand_gesture_CD569_static_front_IMS_RGB_v2_static_v2.0_ldmk2.5dv3.0.3",
        "CP_hand_gesture_CD569_static_back_IMS_RGB_v2_static_v2.0_ldmk2.5dv3.0.3",
        "CP_hand_gesture_CD569_J2_static_front_IMS_RGB_v1_static_v2.0_ldmk2.5dv3.0.3",
        "CP_hand_gesture_CD569_J2_static_back_IMS_RGB_v1_static_v2.0_ldmk2.5dv3.0.3",
        # H9实车召回测试集
        "CP_hand_gesture_H9_static_back_IMS_IR_v1_static_v2.0_ldmk2.5dv3.0.3",
        "CP_hand_gesture_H9_static_back_IMS_RGB_v1_static_v2.0_ldmk2.5dv3.0.3",
        "CP_hand_gesture_H9_static_front_IMS_IR_v1_static_v2.0_ldmk2.5dv3.0.3",
        "CP_hand_gesture_H9_static_front_IMS_RGB_v1_static_v2.0_ldmk2.5dv3.0.3",
        # C281实车召回测试集
        # "CP_hand_gesture_C281_static_front_IMS_RGB_v1_static_v2.0_ldmk2.5dv3.0.3",
        # "CP_hand_gesture_C281_static_back_IMS_RGB_v1_static_v2.0_ldmk2.5dv3.0.3",
        "C281_hand_gesture_static_front_IMS-RGB_v1_c281_static_v2.5_ldmk2.5dv3.0.7",
        "C281_hand_gesture_static_back_IMS-RGB_v1_c281_static_v2.5_ldmk2.5dv3.0.7",
        # A13实车自测hardcase测试集
        "CP_bg_hand_gesture_A13_20211207_bg_hardcase_static_v2.0_ldmk2.5dv3.0.3",
        # 广汽实车体验和自然驾驶误报测试集
        "CP_bg_hand_gesture_A11_exp_nds_IMS_RGB_v1_static_v2.0_ldmk2.5dv3.0.3",
        "CP_bg_hand_gesture_A13_nds_IMS_RGB_v1_static_v2.0_ldmk2.5dv3.0.3",
        "CP_bg_hand_gesture_A18_exp_nds_IMS_RGB_v1_static_v2.0_ldmk2.5dv3.0.3",
        "CP_bg_hand_gesture_A18_nds_IMS_RGB_v1_static_v2.0_ldmk2.5dv3.0.3",
        "CP_bg_hand_gesture_A20_exp_nds_IMS_RGB_v1_static_v2.0_ldmk2.5dv3.0.3",
        "CP_bg_hand_gesture_A29_exp_nds_IMS_RGB_v1_static_v2.0_ldmk2.5dv3.0.3",
        "CP_bg_hand_gesture_A55_nds_IMS_IR_v1_static_v2.0_ldmk2.5dv3.0.3",
        "CP_bg_hand_gesture_A55_exp_nds_IMS_RGB_v1_static_v2.0_ldmk2.5dv3.0.3",
        "CP_bg_hand_gesture_A55_nds_IMS_RGB_v1_static_v2.0_ldmk2.5dv3.0.3",
        "CP_bg_hand_gesture_A57_exp_nds_IMS_RGB_v1_static_v2.0_ldmk2.5dv3.0.3",
        "CP_bg_hand_gesture_A60_exp_nds_front_IMS_RGB_v1_static_v2.0_ldmk2.5dv3.0.3",
        "CP_bg_hand_gesture_A60_nds_rgb_IMS_RGB_v1_static_v2.0_ldmk2.5dv3.0.3",
        # H9自然驾驶和实车体验误报测试集
        "CP_bg_hand_gesture_H9_exp_nds_IMS_RGB_v1_static_v2.0_ldmk2.5dv3.0.3",
        "CP_bg_hand_gesture_H9_exp_nds_IMS_IR_v1_static_v2.0_ldmk2.5dv3.0.3",
        "CP_bg_hand_gesture_H9_nds_IMS_RGB_v2_static_v2.0_ldmk2.5dv3.0.3",
        # CD569自然驾驶和实车体验误报测试集
        "CP_bg_hand_gesture_CD569_nds_rgb_IMS_RGB_v1_static_v2.0_ldmk2.5dv3.0.3",
        # 江淮S811实车体验和自然驾驶误报测试集
        "CP_bg_hand_gesture_S811_nds_IMS_RGB_static_v2.0_ldmk2.5dv3.0.3",
        # 广汽召回测试集
        "CP_hand_gesture_A11_static_front_IMS_RGB_static_v2.0_ldmk2.5dv3.0.3",
        "CP_hand_gesture_A11_static_back_IMS_RGB_static_v2.0_ldmk2.5dv3.0.3",
        "CP_hand_gesture_A13_static_front_IMS_IR_static_v2.0_ldmk2.5dv3.0.3",
        "CP_hand_gesture_A13_static_back_IMS_IR_static_v2.0_ldmk2.5dv3.0.3",
        "CP_hand_gesture_A13_static_front_IMS_RGB_static_v2.0_ldmk2.5dv3.0.3",
        "CP_hand_gesture_A13_static_back_IMS_RGB_static_v2.0_ldmk2.5dv3.0.3",
        "CP_bg_hand_gesture_A13_nds_IMS_IR_v1_static_v2.0_ldmk2.5dv3.0.3",
        "CP_hand_gesture_A18_static_front_IMS_RGB_static_v2.0_ldmk2.5dv3.0.3",
        "CP_hand_gesture_A18_static_back_IMS_RGB_static_v2.0_ldmk2.5dv3.0.3",
        "CP_hand_gesture_A20_static_front_IMS_IR_static_v2.0_ldmk2.5dv3.0.3",
        "CP_hand_gesture_A20_static_back_IMS_IR_static_v2.0_ldmk2.5dv3.0.3",
        "CP_hand_gesture_A20_static_back_IMS_RGB_static_v2.0_ldmk2.5dv3.0.3",
        "CP_hand_gesture_A20_static_front_IMS_RGB_static_v2.0_ldmk2.5dv3.0.3",
        "CP_hand_gesture_A29_static_front_IMS_IR_static_v2.0_ldmk2.5dv3.0.3",
        "CP_hand_gesture_A29_static_back_IMS_IR_static_v2.0_ldmk2.5dv3.0.3",
        "CP_hand_gesture_A29_static_front_IMS_RGB_static_v2.0_ldmk2.5dv3.0.3",
        "CP_hand_gesture_A29_static_back_IMS_RGB_static_v2.0_ldmk2.5dv3.0.3",
        "CP_hand_gesture_A55_static_front_IMS_IR_static_v2.0_ldmk2.5dv3.0.3",
        "CP_hand_gesture_A55_static_back_IMS_IR_static_v2.0_ldmk2.5dv3.0.3",
        "CP_hand_gesture_A55_static_back_IMS_RGB_static_v2.0_ldmk2.5dv3.0.3",
        "CP_hand_gesture_A55_static_front_IMS_RGB_static_v2.0_ldmk2.5dv3.0.3",
        "CP_hand_gesture_A57_static_back_IMS_RGB_static_v2.0_ldmk2.5dv3.0.3",
        "CP_hand_gesture_A57_static_front_IMS_RGB_static_v2.0_ldmk2.5dv3.0.3",
        "CP_hand_gesture_A60_static_front_IMS_IR_static_v2.0_ldmk2.5dv3.0.3",
        "CP_hand_gesture_A60_static_back_IMS_IR_static_v2.0_ldmk2.5dv3.0.3",
        "CP_hand_gesture_A60_static_front_IMS_RGB_static_v2.0_ldmk2.5dv3.0.3",
        "CP_hand_gesture_A60_static_back_IMS_RGB_static_v2.0_ldmk2.5dv3.0.3",
        # T18
        "T18_hand_gesture_IMS-IR_front_v5_static_v2.0_ldmk2.5dv3.0.3",
        "T18_hand_gesture_IMS-RGB_front_v5_static_v2.0_ldmk2.5dv3.0.3",
        "T18_hand_gesture_IMS-IR_back_v5_static_v2.0_ldmk2.5dv3.0.3",
        "T18_hand_gesture_IMS-RGB_back_v5_static_v2.0_ldmk2.5dv3.0.3",
        "T18_hand_gesture_ngtv_OMS-RGB_front_negtive_v2.0_ldmk2.5dv3.0.3",
        "T18_hand_gesture_ngtv_OMS-IR_front_negtive_v2.0_ldmk2.5dv3.0.3",
        # CD569 mute
        "CD569_hand_gesture_mute_back_IMS-RGB_v1_74507_static_v2.0_ldmk2.5dv3.0.3",
        "CD569_hand_gesture_mute_front_IMS-RGB_v1_74506_static_v2.0_ldmk2.5dv3.0.3",
        # E300 static
        "E300_hand_gesture_static_IMS-RGB_front_all_static_v2.0_ldmk2.5dv3.0.3",
        # S202实车badcase测试集
        "CP_bg_hand_gesture_S202_nds_rgb_IMS_RGB_v1_v0.11.1_static_gesture_static_v2.0_ldmk2.5dv3.0.3",
        # 算法测试集
        "CP_sta_hand_gesture_T18_220121_220124_220125_static_v2.2.1_ldmk2.5dv3.0.6",
        "GY_hand_gesture_MT_210630_210702_testset_cam3_static_v2.1_ldmk2.5dv3.0.3",
        "CD569_testset_hand_gesture_MT_220117_220223_dynamic_v2.0_ldmk2.5dv3.0.3",
        "C281_hand_gesture_211019_211022_static_v2.0_ldmk2.5dv3.0.3",
        "C281_hand_gesture_211104_211105_static_v2.0_ldmk2.5dv3.0.3",
        "CANJ_hand_gesture_210525_static_v2.1_ldmk2.5dv3.0.3",
        "CANJ_hand_gesture_210604_210607_210608_210609_210610_static_v2.1_ldmk2.5dv3.0.3",
        "x3tcl_hand_gesture_20210324_bg_nj_test_static_v2.9_ldmk2.5dv3.0.3",
        "GuangQiBg_handobj_hand_gesture_20211103_A11_static-gesture_static_v2.0_ldmk2.5dv3.0.3",
        "GuangQiBg_handobj_hand_gesture_20211026_A11_static-gesture_static_v2.0_ldmk2.5dv3.0.3",
    ]

    not_exist_data = [
        "CD569_hand_gesture_CD569_211102_211105_static_v2.0_ldmk2.5dv3.0.3",
        "GuangQi_hand_gesture_A18_A55_A57_211108_211116_static_v2.0_ldmk2.5dv3.0.3",
        "GuangQi_hand_gesture_another_a11_a20_a60_20211220_20211221_static_v2.0_ldmk2.5dv3.0.3",
    ]
    all_train_dataset = list(set(all_train_dataset) - set(not_exist_data))
    root_path = (
        "/horizon-bucket/MultiMode/mm_algorithms_data/hand_gesture/lmdb_data"
    )

    if small_data:
        train_dataset_list = white_box_train_datasets
        val_dataset_list = white_box_val_datasets
    else:
        train_dataset_list = all_train_dataset
        val_dataset_list = all_val_dataset

    train_path_list = [os.path.join(root_path, i) for i in train_dataset_list]
    val_path_list = [os.path.join(root_path, i) for i in val_dataset_list]

    return train_path_list, val_path_list


def get_lmdb_dataset(small_data=True):
    train_path_list, val_path_list = get_lmdb_train_data_list(small_data)
    train_transforms_list = get_train_transforms()

    repeat_times = [1] * len(train_path_list)
    train_dataset = dict(
        type="ConcatDataset",
        with_flag=True,
        record_index=False,
        datasets=[
            dict(
                type="RepeatDataset",
                times=repeat_times[i],
                dataset=dict(
                    type="GesDatasetLmdb",
                    lmdb_path=train_path_list[i],
                    transforms=train_transforms_list,
                    seq_len=16,
                    box_len=4,
                    feat_len=84,
                    time_stride=0.0156,
                    temporal_jitter=True,
                    temporal_jitter_type="seq",
                    temporal_jitter_range=0.1,
                    temporal_rand_scale=True,
                    temporal_scale_range=[0.5, 2.0],
                    use_rgb_branch=True,
                    to_rgb=False,
                    mode="Train",
                ),
            )
            for i in range(len(train_path_list))
        ],
    )

    val_transforms_list = get_test_transforms()
    repeat_times = [1] * len(val_path_list)
    val_dataset = dict(
        type="ConcatDataset",
        with_flag=True,
        record_index=False,
        datasets=[
            dict(
                type="RepeatDataset",
                times=repeat_times[i],
                dataset=dict(
                    type="GesDatasetLmdb",
                    lmdb_path=val_path_list[i],
                    transforms=val_transforms_list,
                    seq_len=16,
                    box_len=4,
                    feat_len=21 * 4,
                    time_stride=0.015625,
                    temporal_jitter=False,
                    temporal_jitter_type="seq",
                    temporal_jitter_range=0.1,
                    temporal_rand_scale=False,
                    temporal_scale_range=[0.5, 2.0],
                    use_rgb_branch=True,
                    to_rgb=False,
                    mode="validation",
                ),
            )
            for i in range(len(val_path_list))
        ],
    )

    return train_dataset, val_dataset
