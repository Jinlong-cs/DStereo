from easydict import EasyDict
from hatbc.filestream.bucket.client import BucketClient

root = BucketClient().get_mount_root("MultiMode_2")
# if root is None:
#     raise_find_gpfs_bucket_mount_root_error("MultiMode_2")

# ----- Required -----

datapaths = dict(
    image_fail_parsing=dict(
        train_batch_size_per_ctx=16,  # for vargnet
        train_data_paths=[
            # 2023.02PDT采集数据转向管柱 normal 误检优化
            dict(
                rec_path=f"{root}/qi.xiang/IQA_Dataset/IMGQ_all_info/train/blockage/20230324_144022/train_None_blockage_cls5_20230324_num1232_144022.rec",  # noqa
                anno_path=f"{root}/qi.xiang/IQA_Dataset/IMGQ_all_info/train/blockage/20230324_144022/train_None_blockage_cls5_20230324_num1232_144022.json",  # noqa
                sample_weight=2,
            ),
            # 2023.02PDT采集数据转向管柱 normal
            dict(
                rec_path=f"{root}/qi.xiang/IQA_Dataset/IMGQ_all_info/train/blockage/20230324_144425/train_None_blockage_cls5_20230324_num1071_144425.rec",  # noqa
                anno_path=f"{root}/qi.xiang/IQA_Dataset/IMGQ_all_info/train/blockage/20230324_144425/train_None_blockage_cls5_20230324_num1071_144425.json",  # noqa
                sample_weight=2,
            ),
            # 2023.02PDT采集数据转向管柱 abnormal
            dict(
                rec_path=f"{root}/qi.xiang/IQA_Dataset/IMGQ_all_info/train/blockage/20230324_145229/train_None_blockage_cls5_20230324_num2812_145229.rec",  # noqa
                anno_path=f"{root}/qi.xiang/IQA_Dataset/IMGQ_all_info/train/blockage/20230324_145229/train_None_blockage_cls5_20230324_num2812_145229.json",  # noqa
                sample_weight=4,
            ),
            # 2023.02PDT采集数据A柱 normal
            dict(
                rec_path=f"{root}/qi.xiang/IQA_Dataset/IMGQ_all_info/train/blockage/20230324_150007/train_None_blockage_cls5_20230324_num2377_150007.rec",  # noqa
                anno_path=f"{root}/qi.xiang/IQA_Dataset/IMGQ_all_info/train/blockage/20230324_150007/train_None_blockage_cls5_20230324_num2377_150007.json",  # noqa
                sample_weight=4,
            ),
            # 2023.02PDT采集数据A柱 abnormal
            dict(
                rec_path=f"{root}/qi.xiang/IQA_Dataset/IMGQ_all_info/train/blockage/20230324_150503/train_None_blockage_cls5_20230324_num1679_150503.rec",  # noqa
                anno_path=f"{root}/qi.xiang/IQA_Dataset/IMGQ_all_info/train/blockage/20230324_150503/train_None_blockage_cls5_20230324_num1679_150503.json",  # noqa
                sample_weight=3,
            ),
            # 2022.08采集遮挡数据不需要清洗的部分
            dict(
                rec_path=f"{root}/qi.xiang/IQA_Dataset/IMGQ_all_info/train/blockage/20230309_111530/train_None_blockage_cls5_20230309_num8637_111530.rec",  # noqa
                anno_path=f"{root}/qi.xiang/IQA_Dataset/IMGQ_all_info/train/blockage/20230309_111530/train_None_blockage_cls5_20230309_num8637_111530.json",  # noqa
                sample_weight=14,
            ),
            # 2022.08采集模糊数据不需要清洗的部分
            dict(
                rec_path=f"{root}/qi.xiang/IQA_Dataset/IMGQ_all_info/train/blur/20230309_112341/train_None_blur_cls5_20230309_num9037_112341.rec",  # noqa
                anno_path=f"{root}/qi.xiang/IQA_Dataset/IMGQ_all_info/train/blur/20230309_112341/train_None_blur_cls5_20230309_num9037_112341.json",  # noqa
                sample_weight=14,
            ),
            # 2023.01采集遮挡数据中不需要清洗的部分
            dict(
                rec_path=f"{root}/qi.xiang/IQA_Dataset/IMGQ_all_info/train/blockage/20230309_113717/train_None_blockage_cls5_20230309_num10523_113717.rec",  # noqa
                anno_path=f"{root}/qi.xiang/IQA_Dataset/IMGQ_all_info/train/blockage/20230309_113717/train_None_blockage_cls5_20230309_num10523_113717.json",  # noqa
                sample_weight=17,
            ),
            # 2022.12采集模糊数据不需要清洗的部分
            dict(
                rec_path=f"{root}/qi.xiang/IQA_Dataset/IMGQ_all_info/train/blur/20230309_115226/train_None_blur_cls5_20230309_num1778_115226.rec",  # noqa
                anno_path=f"{root}/qi.xiang/IQA_Dataset/IMGQ_all_info/train/blur/20230309_115226/train_None_blur_cls5_20230309_num1778_115226.json",  # noqa
                sample_weight=3,
            ),
            # 全图正常的数据
            dict(
                rec_path=f"{root}/qi.xiang/IQA_Dataset/IMGQ_all_info/train/blockage/20230309_115801/train_None_blockage_cls5_20230309_num3973_115801.rec",  # noqa
                anno_path=f"{root}/qi.xiang/IQA_Dataset/IMGQ_all_info/train/blockage/20230309_115801/train_None_blockage_cls5_20230309_num3973_115801.json",  # noqa
                sample_weight=6,
            ),
            # 2022.08采集遮挡数据中需要清洗的部分
            dict(
                rec_path=f"{root}/qi.xiang/IQA_Dataset/IMGQ_all_info/train/blockage/20230309_143640/train_None_blockage_cls5_20230309_num3281_143640.rec",  # noqa
                anno_path=f"{root}/qi.xiang/IQA_Dataset/IMGQ_all_info/train/blockage/20230309_143640/train_None_blockage_cls5_20230309_num3281_143640.json",  # noqa
                sample_weight=5,
            ),
            # 2023.01采集遮挡数据中需要清洗的部分
            dict(
                rec_path=f"{root}/qi.xiang/IQA_Dataset/IMGQ_all_info/train/blockage/20230309_150431/train_None_blockage_cls5_20230309_num9173_150431.rec",  # noqa
                anno_path=f"{root}/qi.xiang/IQA_Dataset/IMGQ_all_info/train/blockage/20230309_150431/train_None_blockage_cls5_20230309_num9173_150431.json",  # noqa
                sample_weight=15,
            ),
            # 2022.08采集模糊数据中需要清洗的部分
            dict(
                rec_path=f"{root}/qi.xiang/IQA_Dataset/IMGQ_all_info/train/blur/20230309_152548/train_None_blur_cls5_20230309_num4882_152548.rec",  # noqa
                anno_path=f"{root}/qi.xiang/IQA_Dataset/IMGQ_all_info/train/blur/20230309_152548/train_None_blur_cls5_20230309_num4882_152548.json",  # noqa
                sample_weight=8,
            ),
            # 2022.12采集模糊数据需要清洗的部分
            dict(
                rec_path=f"{root}/qi.xiang/IQA_Dataset/IMGQ_all_info/train/blur/20230309_154026/train_None_blur_cls5_20230309_num3590_154026.rec",  # noqa
                anno_path=f"{root}/qi.xiang/IQA_Dataset/IMGQ_all_info/train/blur/20230309_154026/train_None_blur_cls5_20230309_num3590_154026.json",  # noqa
                sample_weight=6,
            ),
            # finetune
            dict(
                rec_path=f"{root}/qi.xiang/IQA_Dataset/IMGQ_all_info/train/blockage/20230313_173935/train_None_blockage_cls5_20230313_num334_173935.rec",  # noqa
                anno_path=f"{root}/qi.xiang/IQA_Dataset/IMGQ_all_info/train/blockage/20230313_173935/train_None_blockage_cls5_20230313_num334_173935.json",  # noqa
                sample_weight=1,
            ),
            dict(
                rec_path=f"{root}/qi.xiang/IQA_Dataset/IMGQ_all_info/train/blockage/20230313_173935/train_None_blockage_cls5_20230313_num334_173935.rec",  # noqa
                anno_path=f"{root}/qi.xiang/IQA_Dataset/IMGQ_all_info/train/blockage/20230313_173935/train_None_blockage_cls5_20230313_num334_173935.json",  # noqa
                sample_weight=1,
            ),
            dict(
                rec_path=f"{root}/qi.xiang/IQA_Dataset/IMGQ_all_info/train/blockage/20230313_173935/train_None_blockage_cls5_20230313_num334_173935.rec",  # noqa
                anno_path=f"{root}/qi.xiang/IQA_Dataset/IMGQ_all_info/train/blockage/20230313_173935/train_None_blockage_cls5_20230313_num334_173935.json",  # noqa
                sample_weight=1,
            ),
            dict(
                rec_path=f"{root}/qi.xiang/IQA_Dataset/IMGQ_all_info/train/blockage/20230313_185926/train_None_blockage_cls5_20230313_num1676_185926.rec",  # noqa
                anno_path=f"{root}/qi.xiang/IQA_Dataset/IMGQ_all_info/train/blockage/20230313_185926/train_None_blockage_cls5_20230313_num1676_185926.json",  # noqa
                sample_weight=3,
            ),
        ],
        val_batch_size_per_ctx=4,
        val_data_paths=None,
    ),
)


datapaths = EasyDict(datapaths)
buckets = [
    "MultiMode_2",
]
