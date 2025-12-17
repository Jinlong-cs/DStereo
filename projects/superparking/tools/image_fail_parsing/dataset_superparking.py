import os

from easydict import EasyDict
from hatbc.filestream.bucket import BucketClient

bkt_clt = BucketClient()
root = bkt_clt.url_to_local("dmpv2://SuperParking/")
# ----- Required -----

datapaths = dict(
    image_fail_parsing=dict(
        train_data_paths=[
            # v0.6.0
            dict(
                rec_path=f"{root}/LastVersion/Dataset/ImageFail/train/glare_cmp/train_fisheye_sp_train_ego_as_normal_True_is_ground_glare_as_normal_True_20220812_232739.rec",  # noqa
                anno_path=f"{root}/LastVersion/Dataset/ImageFail/train/glare_cmp/train_fisheye_sp_train_ego_as_normal_True_is_ground_glare_as_normal_True_20220812_232739.rec.idx",  # noqa
                sample_weight=4,
            ),
            dict(
                rec_path=f"{root}/LastVersion/Dataset/ImageFail/train/20221014_223330/train_fisheye_sp_train_ego_as_normal_True_is_ground_glare_as_normal_False_20221014_223330.rec",  # noqa
                anno_path=f"{root}/LastVersion/Dataset/ImageFail/train/20221014_223330/train_fisheye_sp_train_ego_as_normal_True_is_ground_glare_as_normal_False_20221014_223330.rec.idx",  # noqa
                sample_weight=4,
            ),
            dict(
                rec_path=f"{root}/LastVersion/Dataset/ImageFail/train/20221014_230100/train_fisheye_sp_train_ego_as_normal_True_is_ground_glare_as_normal_False_20221014_230100.rec",  # noqa
                anno_path=f"{root}/LastVersion/Dataset/ImageFail/train/20221014_230100/train_fisheye_sp_train_ego_as_normal_True_is_ground_glare_as_normal_False_20221014_230100.rec.idx",  # noqa
                sample_weight=4,
            ),
            dict(
                rec_path=f"{root}/LastVersion/Dataset/ImageFail/train/20221015_160135/train_fisheye_sp_train_ego_as_normal_True_is_ground_glare_as_normal_False_20221015_160135.rec",  # noqa
                anno_path=f"{root}/LastVersion/Dataset/ImageFail/train/20221015_160135/train_fisheye_sp_train_ego_as_normal_True_is_ground_glare_as_normal_False_20221015_160135.rec.idx",  # noqa
                sample_weight=4,
            ),
        ],
        val_data_paths=[
            dict(
                rec_path=f"{root}/LastVersion/Dataset/ImageFail/val/20220711_131327/val_None_sp_version_0.0.1_part3_20220711_131327.rec",  # noqa
                anno_path=f"{root}/LastVersion/Dataset/ImageFail/val/20220711_131327/val_None_sp_version_0.0.1_part3_20220711_131327.rec.idx",  # noqa
                sample_weight=4,
            ),
            dict(
                rec_path=f"{root}/LastVersion/Dataset/ImageFail/val/20221014_223753/val_fisheye_sp_train_ego_as_normal_True_is_ground_glare_as_normal_False_20221014_223753.rec",  # noqa
                anno_path=f"{root}/LastVersion/Dataset/ImageFail/val/20221014_223753/val_fisheye_sp_train_ego_as_normal_True_is_ground_glare_as_normal_False_20221014_223753.rec.idx",  # noqa
                sample_weight=4,
            ),
            dict(
                rec_path=f"{root}/LastVersion/Dataset/ImageFail/val/20221014_230916/val_fisheye_sp_train_ego_as_normal_True_is_ground_glare_as_normal_False_20221014_230916.rec",  # noqa
                anno_path=f"{root}/LastVersion/Dataset/ImageFail/val/20221014_230916/val_fisheye_sp_train_ego_as_normal_True_is_ground_glare_as_normal_False_20221014_230916.rec.idx",  # noqa.idx",  # noqa
                sample_weight=4,
            ),
            dict(
                rec_path=f"{root}/LastVersion/Dataset/ImageFail/val/20221015_155317/val_fisheye_sp_train_ego_as_normal_True_is_ground_glare_as_normal_False_20221015_155317.rec",  # noqa
                anno_path=f"{root}/LastVersion/Dataset/ImageFail/val/20221015_155317/val_fisheye_sp_train_ego_as_normal_True_is_ground_glare_as_normal_False_20221015_155317.rec.idx",  # noqa.idx",  # noqa
                sample_weight=4,
            ),
        ],
    ),
)

datapaths_val = EasyDict(datapaths)
paths_val_sp = [
    os.path.splitext(path_list["rec_path"])[0]
    for path_list in datapaths_val["image_fail_parsing"]["val_data_paths"]
]

datapaths_train = EasyDict(datapaths)
paths_train_sp = [
    os.path.splitext(path_list["rec_path"])[0]
    for path_list in datapaths_train["image_fail_parsing"]["train_data_paths"]
]
# from hatbc.filestream.bucket import BucketClient
# bkt_clt = BucketClient()
# for i in paths_train_sp:
#     src_dir = os.path.dirname(i)
#     dst_dir = os.path.basename(src_dir)
#     train_dir = "/horizon-bucket/SuperParking/LastVersion/Dataset/ImageFail/train/"
#     src_url = bkt_clt.local_to_url(src_dir)
#     dst_url = bkt_clt.local_to_url(train_dir)
#     os.system(f"hitc bkt-file mv {src_url} {dst_url} --recursive -y")
# path = "/horizon-bucket/SuperParking/LastVersion/Dataset/ImageFail/val/20220711_131327/*"
# import glob
# paths = glob.glob(path)
# print(paths)
# for i in paths:
#     if os.path.basename(i).startswith("train"):
#         new_name = os.path.basename(i).replace("train", "val")
#         src_dir = os.path.dirname(i)
#         train_dir = src_dir + "/" + new_name
#         src_url = bkt_clt.local_to_url(i)
#         dst_url = bkt_clt.local_to_url(train_dir)
#         print(f"hitc bkt-file mv {src_url} {dst_url} -y")
#         os.system(f"hitc bkt-file mv {src_url} {dst_url} -y")
