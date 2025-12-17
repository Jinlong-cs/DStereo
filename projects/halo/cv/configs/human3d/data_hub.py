import os

# dataset path
# dict{dataset name: dataset lmdb path}
dataset_paths = {
    "h9_train_01": "/horizon-bucket/interaction/active/human3d/data/human3d_lmdb/h9_train_01/lmdb/",  # noqa
    "changan_202003_RGB": "/horizon-bucket/interaction/active/human3d/data/human3d_lmdb/changan_202003_RGB/lmdb/",  # noqa
    "MMGestureBatch_01": "/horizon-bucket/interaction/active/human3d/data/human3d_lmdb/MMGestureBatch_01/lmdb/",  # noqa
    "MMGestureBatch_02": "/horizon-bucket/interaction/active/human3d/data/human3d_lmdb/MMGestureBatch_02/lmdb/",  # noqa
    "MMGestureBatch_03": "/horizon-bucket/interaction/active/human3d/data/human3d_lmdb/MMGestureBatch_03/lmdb/",  # noqa
    "MMGestureBatch_04": "/horizon-bucket/interaction/active/human3d/data/human3d_lmdb/MMGestureBatch_04/lmdb/",  # noqa
    "MMGestureBatch_val": "/horizon-bucket/interaction/active/human3d/data/human3d_lmdb/MMGestureBatch_val/lmdb/",  # noqa
    "CD569_nonproduct_4ways_valset": "/horizon-bucket/interaction/active/human3d/data/human3d_lmdb/CD569_nonproduct_4ways_valset/lmdb/",  # noqa
    "C281_gesture_cockpit_valset_01": "/horizon-bucket/interaction/active/human3d/data/human3d_lmdb/C281_gesture_cockpit_valset_01/lmdb/",  # noqa
    "C281_gesture_cockpit_valset_02": "/horizon-bucket/interaction/active/human3d/data/human3d_lmdb/C281_gesture_cockpit_valset_02/lmdb/",  # noqa
    "TrainValsetCD569_trainset": "/horizon-bucket/interaction/active/human3d/data/human3d_lmdb/TrainValsetCD569_trainset/lmdb/",  # noqa
    "A11_RGB_IMS_trainset_01": "/horizon-bucket/interaction/active/human3d/data/human3d_lmdb/A11_RGB_IMS_trainset_01/lmdb/",  # noqa
    "A11_RGB_IMS_trainset_02": "/horizon-bucket/interaction/active/human3d/data/human3d_lmdb/A11_RGB_IMS_trainset_02/lmdb/",  # noqa
    "A11_RGB_IMS_trainset_03": "/horizon-bucket/interaction/active/human3d/data/human3d_lmdb/A11_RGB_IMS_trainset_03/lmdb/",  # noqa
    "A11_RGB_IMS_trainset_04": "/horizon-bucket/interaction/active/human3d/data/human3d_lmdb/A11_RGB_IMS_trainset_04/lmdb/",  # noqa
    "A11_RGB_IMS_trainset_05": "/horizon-bucket/interaction/active/human3d/data/human3d_lmdb/A11_RGB_IMS_trainset_05/lmdb/",  # noqa
    "A11_RGB_IMS_trainset_06": "/horizon-bucket/interaction/active/human3d/data/human3d_lmdb/A11_RGB_IMS_trainset_06/lmdb/",  # noqa
    "A11_RGB_IMS_trainset_07": "/horizon-bucket/interaction/active/human3d/data/human3d_lmdb/A11_RGB_IMS_trainset_07/lmdb/",  # noqa
    "CD569_nonproduct_4ways_trainset_01": "/horizon-bucket/interaction/active/human3d/data/human3d_lmdb/CD569_nonproduct_4ways_trainset_01/lmdb/",  # noqa
    "CD569_nonproduct_4ways_trainset_02": "/horizon-bucket/interaction/active/human3d/data/human3d_lmdb/CD569_nonproduct_4ways_trainset_02/lmdb/",  # noqa
    "CD569_nonproduct_4ways_trainset_03": "/horizon-bucket/interaction/active/human3d/data/human3d_lmdb/CD569_nonproduct_4ways_trainset_03/lmdb/",  # noqa
    "CD569_nonproduct_4ways_trainset_04": "/horizon-bucket/interaction/active/human3d/data/human3d_lmdb/CD569_nonproduct_4ways_trainset_04/lmdb/",  # noqa
    "CD569_RGB_trainset_01": "/horizon-bucket/interaction/active/human3d/data/human3d_lmdb/CD569_RGB_trainset_01/lmdb/",  # noqa
    "CD569_hand_hardcase": "/horizon-bucket/interaction/active/human3d/data/human3d_lmdb/CD569_hand_hardcase/lmdb/",  # noqa
    "A11_RGB_RMS_trainset_01": "/horizon-bucket/interaction/active/human3d/data/human3d_lmdb/A11_RGB_RMS_trainset_01/lmdb/",  # noqa
    "A11_RGB_RMS_trainset_02": "/horizon-bucket/interaction/active/human3d/data/human3d_lmdb/A11_RGB_RMS_trainset_02/lmdb/",  # noqa
    "A11_RGB_RMS_trainset_03": "/horizon-bucket/interaction/active/human3d/data/human3d_lmdb/A11_RGB_RMS_trainset_03/lmdb/",  # noqa
    "A11_RGB_RMS_trainset_04": "/horizon-bucket/interaction/active/human3d/data/human3d_lmdb/A11_RGB_RMS_trainset_04/lmdb/",  # noqa
    "A11_RGB_RMS_trainset_05": "/horizon-bucket/interaction/active/human3d/data/human3d_lmdb/A11_RGB_RMS_trainset_05/lmdb/",  # noqa
    "A11_RGB_RMS_trainset_06": "/horizon-bucket/interaction/active/human3d/data/human3d_lmdb/A11_RGB_RMS_trainset_06/lmdb/",  # noqa
    "A11_RGB_RMS_trainset_07": "/horizon-bucket/interaction/active/human3d/data/human3d_lmdb/A11_RGB_RMS_trainset_07/lmdb/",  # noqa
    "C281_IMS_trainset_01": "/horizon-bucket/interaction/active/human3d/data/human3d_lmdb/C281_IMS_trainset_01/lmdb/",  # noqa
    "C281_gesture_cockpit_trainset_01": "/horizon-bucket/interaction/active/human3d/data/human3d_lmdb/C281_gesture_cockpit_trainset_01/lmdb/",  # noqa
    "C281_gesture_cockpit_trainset_02": "/horizon-bucket/interaction/active/human3d/data/human3d_lmdb/C281_gesture_cockpit_trainset_02/lmdb/",  # noqa
    "C281_gesture_cockpit_trainset_03": "/horizon-bucket/interaction/active/human3d/data/human3d_lmdb/C281_gesture_cockpit_trainset_03/lmdb/",  # noqa
    "T18_gesture_cockpit_trainset_01": "/horizon-bucket/interaction/active/human3d/data/human3d_lmdb/T18_gesture_cockpit_trainset_01/lmdb/",  # noqa
    "T18_gesture_cockpit_trainset_02": "/horizon-bucket/interaction/active/human3d/data/human3d_lmdb/T18_gesture_cockpit_trainset_02/lmdb/",  # noqa
    "S311_gesture_cockpit_trainset_01": "/horizon-bucket/interaction/active/human3d/data/human3d_lmdb/S311_gesture_cockpit_trainset_01/lmdb/",  # noqa
    "mpi-inf-3dhp": "/horizon-bucket/interaction/active/human3d/data/human3d_lmdb/mpi_inf_3dhp_train/",  # noqa
    "mpii": "/horizon-bucket/interaction/active/human3d/data/human3d_lmdb/mpii_train/",  # noqa
    "lsp-orig": "/horizon-bucket/interaction/active/human3d/data/human3d_lmdb/lsp_dataset_original_train/",  # noqa
    "lspet": "/horizon-bucket/interaction/active/human3d/data/human3d_lmdb/hr-lspet_train/",  # noqa
    "h36m": "/horizon-bucket/interaction/active/human3d/data/human3d_lmdb/h36m_train/",  # noqa
    "coco": "/horizon-bucket/interaction/active/human3d/data/human3d_lmdb/coco_2014_train/",  # noqa
    "h36m_valid_protocol1": "/horizon-bucket/interaction/active/human3d/data/human3d_lmdb/h36m_valid_protocol1/",  # noqa
    "h36m_valid_protocol2": "/horizon-bucket/interaction/active/human3d/data/human3d_lmdb/h36m_valid_protocol2/",  # noqa
    "mpi_inf_3dhp_valid": "/horizon-bucket/interaction/active/human3d/data/human3d_lmdb/mpi_inf_3dhp_valid/",  # noqa
    "3dpw_test": "/horizon-bucket/interaction/active/human3d/data/human3d_lmdb/3dpw_test/",  # noqa
    # children
    "chengdu_s311": "/horizon-bucket/interaction/active/human3d/anno/child_anno/chengdu_s311/",  # noqa
    # aiot
    "xiaodu": "/horizon-bucket/interaction/active/human3d/anno/aiot_anno/xiaodu",  # noqa
    "lingkang": "/horizon-bucket/interaction/active/human3d/anno/aiot_anno/lingkang",  # noqa
    # workshop
    "20230630": "/horizon-bucket/interaction/active/human3d/anno/workshop_anno_/20230630",  # noqa
    "20230629": "/horizon-bucket/interaction/active/human3d/anno/workshop_anno_/20230629",  # noqa
    "20230626": "/horizon-bucket/interaction/active/human3d/anno/workshop_anno_/20230626",  # noqa
    "20230621": "/horizon-bucket/interaction/active/human3d/anno/workshop_anno_/20230621",  # noqa
    "20230620": "/horizon-bucket/interaction/active/human3d/anno/workshop_anno_/20230620",  # noqa
    "20230619": "/horizon-bucket/interaction/active/human3d/anno/workshop_anno_/20230619",  # noqa
    "20230613": "/horizon-bucket/interaction/active/human3d/anno/workshop_anno_/20230613",  # noqa
    "20230609": "/horizon-bucket/interaction/active/human3d/anno/workshop_anno_/20230609",  # noqa
    "20230608": "/horizon-bucket/interaction/active/human3d/anno/workshop_anno_/20230608",  # noqa
    "20230531": "/horizon-bucket/interaction/active/human3d/anno/workshop_anno_/20230531",  # noqa
    "20230530": "/horizon-bucket/interaction/active/human3d/anno/workshop_anno_/20230530",  # noqa
    "20230522": "/horizon-bucket/interaction/active/human3d/anno/workshop_anno_/20230522",  # noqa
    "20230518": "/horizon-bucket/interaction/active/human3d/anno/workshop_anno_/20230518",  # noqa
    "20230516": "/horizon-bucket/interaction/active/human3d/anno/workshop_anno_/20230516",  # noqa
    "20230515": "/horizon-bucket/interaction/active/human3d/anno/workshop_anno_/20230515",  # noqa
    "20230512": "/horizon-bucket/interaction/active/human3d/anno/workshop_anno_/20230512",  # noqa
    "20230511": "/horizon-bucket/interaction/active/human3d/anno/workshop_anno_/20230511",  # noqa
    "20230510": "/horizon-bucket/interaction/active/human3d/anno/workshop_anno_/20230510",  # noqa
    "20230509": "/horizon-bucket/interaction/active/human3d/anno/workshop_anno_/20230509",  # noqa
    "20230508": "/horizon-bucket/interaction/active/human3d/anno/workshop_anno_/20230508",  # noqa
    "20230506": "/horizon-bucket/interaction/active/human3d/anno/workshop_anno_/20230506",  # noqa
    "20230505": "/horizon-bucket/interaction/active/human3d/anno/workshop_anno_/20230505",  # noqa
    "20230504": "/horizon-bucket/interaction/active/human3d/anno/workshop_anno_/20230504",  # noqa
    # valset
    "C281_gesture_cockpit_valset_01": "/horizon-bucket/interaction/active/human3d/data/human3d_lmdb/C281_gesture_cockpit_valset_01/lmdb/",  # noqa
    "C281_gesture_cockpit_valset_02": "/horizon-bucket/interaction/active/human3d/data/human3d_lmdb/C281_gesture_cockpit_valset_02/lmdb/",  # noqa
    "CD569_nonproduct_4ways_valset": "/horizon-bucket/interaction/active/human3d/data/human3d_lmdb/CD569_nonproduct_4ways_valset/lmdb/",  # noqa
    "MMGestureBatch_val": "/horizon-bucket/interaction/active/human3d/data/human3d_lmdb/MMGestureBatch_val/lmdb/",  # noqa
    "chengdu_s311_testset_01": "/horizon-bucket/interaction/active/human3d/data/human3d_lmdb/chengdu_s311_testset_01/lmdb/",  # noqa
    "chengdu_s311_testset_02": "/horizon-bucket/interaction/active/human3d/data/human3d_lmdb/chengdu_s311_testset_02/lmdb/",  # noqa
    "chengdu_s311_testset_03": "/horizon-bucket/interaction/active/human3d/data/human3d_lmdb/chengdu_s311_testset_03/lmdb/",  # noqa
    "chengdu_s311_testset_04": "/horizon-bucket/interaction/active/human3d/data/human3d_lmdb/chengdu_s311_testset_04/lmdb/",  # noqa
    "chengdu_s311_testset_05": "/horizon-bucket/interaction/active/human3d/data/human3d_lmdb/chengdu_s311_testset_05/lmdb/",  # noqa
    "chengdu_s311_testset_06": "/horizon-bucket/interaction/active/human3d/data/human3d_lmdb/chengdu_s311_testset_06/lmdb/",  # noqa
}


def get_image_path_list(name_list):
    image_path_list = []
    for name in name_list:
        image_path_list.append(os.path.join(dataset_paths[name], "image_lmdb"))
    return image_path_list


def get_anno_path_list(name_list, prefix):
    anno_path_list = []
    for name in name_list:
        anno_path_list.append(os.path.join(dataset_paths[name], prefix))
    return anno_path_list


def get_image_path(name):
    return os.path.join(dataset_paths[name], "image_lmdb")


def get_anno_path(name, prefix):
    return os.path.join(dataset_paths[name], prefix)
