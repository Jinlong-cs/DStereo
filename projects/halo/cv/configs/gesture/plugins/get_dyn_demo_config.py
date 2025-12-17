import os
import random

from aidisdk.utils import running_in_cluster

from projects.halo.cv.configs.gesture.plugins.dyn_2_5d_datahub import datasets

PATH_DATA = (
    "./tmp_orig_data/action/gesture"
    if running_in_cluster()
    # else "/horizon-bucket/HDLTAlgorithm/data/orig_data/action/gesture"
    else "/home/users/yangkun.zhu/tmp"
)


def get_dataset_by_keyname(keyname):
    path_data = os.path.join(PATH_DATA, "hat_dyn_demo")

    def _replace(_string, path_data=path_data):
        return os.path.join(path_data, os.path.basename(_string))

    imgrec = _replace(datasets[keyname]["imgrec"])
    roidb = _replace(datasets[keyname]["roidb"])
    act_roidb = _replace(datasets[keyname]["act_roidb"])

    return imgrec, roidb, act_roidb


def get_traindata_miniv2(data_small=False):
    "ref: https://horizonrobotics.feishu.cn/docs/doccnFM1TM3MIAUP0KYim4Qgh6I"

    train_datasets = [
        "Nj_hand_gesture_210428_210507_210508_210510_210511_dynamic_v2.1_ldmk2.5dv3.0.3",  # dyn_train_2 # noqa
        "GyWaveForwardBackward_hand_gesture_110042_110041_110021_109994_110013_110012_109995_109792_dynamic_v2.1_ldmk2.5dv3.0.3",  # dyn_train_7 # noqa
        "C281_hand_gesture_IFRC_IFARC_211019_211112_dynamic_v2.0_ldmk2.5dv3.0.3",  # dyn_train_8 # noqa
        # 'gaungqi_a11_a13_a20_a29_a55_a60_up1_down_left_right_hand_gesture_13351_13334_13335_13337_13338_13339_dynamic_v2.0_ldmk2.5dv3.0.3',  # dyn_train_10 # noqa
    ]

    train_rec_list = []
    train_roidb_list = []
    train_roidb_seq_list = []
    train_repeat_times = [1, 2, 3]
    for _dataset in train_datasets:
        _path = get_dataset_by_keyname(_dataset)

        train_rec_list.append(_path[0])
        train_roidb_list.append(_path[1])
        train_roidb_seq_list.append(_path[2])
    if data_small:
        choose_idx = 0
        return (
            [train_rec_list[choose_idx]],
            [train_roidb_list[choose_idx]],
            [train_roidb_seq_list[choose_idx]],
            [train_repeat_times[choose_idx]],
        )
    else:
        return (
            train_rec_list,
            train_roidb_list,
            train_roidb_seq_list,
            train_repeat_times,
        )


def get_testdata_miniv2(data_small=False):
    test_datasets = [
        "Nj_hand_gesture_210422_210423_dynamic_v2.1_ldmk2.5dv3.0.3",  # dyn_test_0 # noqa
        "Ca_hand_gesture_20201123_anno_dynamic_v2.1_ldmk2.5dv3.0.3",  # dyn_test_3 # noqa
        "GyWaveForward_hand_gesture_92134_dynamic_v2.1_ldmk2.5dv3.0.3",  # dyn_test_4 # noqa
        "GyWaveBackward_hand_gesture_91993_dynamic_v2.1_ldmk2.5dv3.0.3",  # dyn_test_5 # noqa
    ]

    test_rec_list = []
    test_roidb_list = []
    test_roidb_seq_list = []
    test_repeat_times = [1, 1, 1, 1]
    for _dataset in test_datasets:
        _path = get_dataset_by_keyname(_dataset)

        test_rec_list.append(_path[0])
        test_roidb_list.append(_path[1])
        test_roidb_seq_list.append(_path[2])
    if data_small:
        choose_idx = random.randint(0, len(test_repeat_times) - 1)
        return (
            [test_rec_list[choose_idx]],
            [test_roidb_list[choose_idx]],
            [test_roidb_seq_list[choose_idx]],
            [test_repeat_times[choose_idx]],
        )
    else:
        return (
            test_rec_list,
            test_roidb_list,
            test_roidb_seq_list,
            test_repeat_times,
        )


def get_train_transforms():
    # ref:
    # gluon - scripts/action/act_kps/examples/hgr_cockpit/config_cockpit_halo_hgr_v3_dyn_v1.0.6_kps.py # noqa
    num_classes = 59
    seq_len = 32
    num_kps = 21
    use_3d_kps = False  # gluon.config.aug_params.use_3d_ldmk
    use_kps_score = True
    feat_ch = 4 if use_3d_kps else 3  # xyzs or xys
    if not use_kps_score:
        feat_ch -= 1  # remove score

    # flip
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
            # from gluon.config.act_kps_roidb_dataset_params
            type="ActionGetMetaData",
            seq_len=seq_len,
            feat_len=num_kps * 4,  # roidb x,y,z,score
            box_len=4,
            time_stride=0.015625,
            temporal_jitter=True,
            temporal_jitter_type="seq",
            temporal_jitter_range=0.1,
            temporal_rand_scale=True,
            temporal_scale_range=[0.5, 2.0],
        ),
        dict(
            # from gluon.config.train.data_params.replenish
            type="ActionClipDataPreProcess",
            act_kps_num_classes=num_classes,
            act_kps_seq_len=seq_len,
            num_kps=num_kps,
            use_3d_kps=use_3d_kps,
            use_kps_score=use_kps_score,
            act_img_seq_len=None,
            use_rgb_branch=False,
            use_replenish=True,
            repenish_max_frame=10000,
        ),
        dict(
            # from gluon.config.data_params.aug_params
            type="ActionRotateKpsFrames",
            p_rotate=1,
            rotate_range=15,
            feat_ch=feat_ch,
            use_rgb_branch=False,
        ),
        dict(
            # from gluon.config.data_params.aug_params
            type="ActionFlipKpsFrames",
            p_flip=0.5,
            feat_ch=feat_ch,
            flip_label_dict=flip_label_dict,
        ),
        dict(
            # from gluon.config.data_params.aug_params
            type="ActionLabelMap",
            convert_label_map={"9": 8, "10": 8, "25": 24, "26": 58},
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
            multiscale_norm_ratio_list=[0.25, 0.5, 1.0, 2.0, 4.0],
            multi_ratio_channel="xy",
            kps_num_list=[num_kps],
            fix_norm_len=None,
        ),
        dict(
            type="ActionKpsJitter",
            p_point_jitter=0.5,
            jitter_std_value=0.03,
        ),
        dict(
            type="ActionKpsRandScale",
            p_rand_scale=1.0,
            rand_scale_mu=1.0,
            rand_scale_sigma=0.1,
        ),
        dict(
            type="ActionKpsAddFingerEncoding",
            use_finger_encoding=True,
            encoding_type="finger_root_onehot",
            as_separate_input=False,
            use_kps_score=use_kps_score,
        ),
        dict(
            type="ActionKpsToTensor",
            tensor_layout="chw",
            use_3d_kps=use_3d_kps,
        ),
        dict(
            type="ActionLabelToTensor",
            seq_len=seq_len,
            use_weight_box_vis_ratio=True,
        ),
        dict(
            type="ActionClipDataPostProcess",
            keep_keynames=[
                "act_label",
                # "seq_label",
                "label_weight",
                "clip_keypoints",
            ],
        ),
    ]


def get_test_transforms():
    # ref:
    # gluon - scripts/action/act_kps/examples/hgr_cockpit/config_cockpit_halo_hgr_v3_dyn_v1.0.6_kps.py # noqa
    num_classes = 59
    seq_len = 32
    num_kps = 21
    use_3d_kps = False  # gluon.config.aug_params.use_3d_ldmk
    use_kps_score = True
    feat_ch = 4 if use_3d_kps else 3  # xyzs or xys
    if not use_kps_score:
        feat_ch -= 1  # remove score

    return [
        dict(
            # from gluon.config.act_kps_roidb_dataset_params
            type="ActionGetMetaData",
            seq_len=seq_len,
            feat_len=num_kps * 4,  # roidb x,y,z,score
            box_len=4,
            time_stride=0.015625,
            temporal_jitter=False,  # test False
            temporal_jitter_type="seq",
            temporal_jitter_range=0.1,
            temporal_rand_scale=False,  # test False
            temporal_scale_range=[0.5, 2.0],
        ),
        dict(
            # from gluon.config.train.data_params.replenish
            type="ActionClipDataPreProcess",
            act_kps_num_classes=num_classes,
            act_kps_seq_len=seq_len,
            num_kps=num_kps,
            use_3d_kps=use_3d_kps,
            use_kps_score=use_kps_score,
            act_img_seq_len=None,
            use_rgb_branch=False,
            use_replenish=True,
            repenish_max_frame=10000,
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
            multiscale_norm_ratio_list=[0.25, 0.5, 1.0, 2.0, 4.0],
            multi_ratio_channel="xy",
            kps_num_list=[num_kps],
            fix_norm_len=None,
        ),
        dict(
            type="ActionKpsAddFingerEncoding",
            use_finger_encoding=True,
            encoding_type="finger_root_onehot",
            as_separate_input=False,
            use_kps_score=use_kps_score,
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
        dict(
            type="ActionClipDataPostProcess",
            keep_keynames=[
                "act_label",
                # "seq_label",
                "label_weight",
                "clip_keypoints",
            ],
        ),
    ]
