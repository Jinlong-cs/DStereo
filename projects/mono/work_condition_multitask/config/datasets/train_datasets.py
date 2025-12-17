from easydict import EasyDict
from hatbc.filestream.bucket.client import get_gpfs_bucket_mount_root

bucket2mount_root = get_gpfs_bucket_mount_root()
hdfs_root_bucket = bucket2mount_root.get("mono", None)
root_bucket = bucket2mount_root.get("adas", None)
hdfs_root_rec = "%s/data/workcondition" % hdfs_root_bucket
hdfs_root_anno = "%s/data/workcondition" % hdfs_root_bucket
root_rec = "%s/big_model/train_dataset" % root_bucket
root_anno = "%s/big_model/train_dataset" % root_bucket

datapaths = dict(
    scene_classification=dict(
        train_batch_size_per_ctx=48,
        train_data_paths=[
            dict(
                rec_path=f"{root_rec}/workcondition_rec/sd_wk_scene_total_1110_v7/train.rec",  # noqa
                anno_path=f"{root_anno}/workcondition_rec/sd_wk_scene_total_1110_v7/train_attrs.anno.pb_rec",  # noqa
                sample_weight=5,
            ),
            dict(
                rec_path=f"{root_rec}/scene_classification/wk_scene_14cls_96577_97207_97209/20221223-185310/data.rec",  # noqa
                anno_path=f"{root_anno}/scene_classification/wk_scene_14cls_96577_97207_97209/20221223-185310/data.anno.pb_rec",  # noqa
                sample_weight=1,
            ),
            dict(
                rec_path=f"{hdfs_root_bucket}/data/sd_scene_classification/wk_scene_20230711_badcase_cls/20230711-112924/data.rec",  # noqa
                anno_path=f"{hdfs_root_bucket}/data/sd_scene_classification/wk_scene_20230711_badcase_cls/20230711-112924/data.anno.pb_rec",  # noqa
                sample_weight=1,
            ),
        ],
    ),
    weather_classification=dict(
        train_batch_size_per_ctx=52,
        train_data_paths=[
            dict(
                rec_path=f"{root_rec}/workcondition_rec/sd_wk_weather_total_1110_v7/train.rec",  # noqa
                anno_path=f"{root_anno}/workcondition_rec/sd_wk_weather_total_1110_v7/train_attrs.anno.pb_rec",  # noqa
                sample_weight=383,
            ),
            dict(
                rec_path=f"{hdfs_root_rec}/weather/data/rec/2pe_wk_weather_snowy_not_in_total/train.rec",  # noqa
                anno_path=f"{hdfs_root_anno}/weather/data/rec/2pe_wk_weather_snowy_not_in_total/train_attrs.anno.pb_rec",  # noqa
                sample_weight=32,
            ),
            dict(
                rec_path=f"{hdfs_root_rec}/weather/data/rec/2pe_wk_weather_total_v20_heavyrain_part/train.rec",  # noqa
                anno_path=f"{hdfs_root_anno}/weather/data/rec/2pe_wk_weather_total_v20_heavyrain_part/train_attrs.anno.pb_rec",  # noqa
                sample_weight=1,
            ),
            dict(
                rec_path=f"{root_rec}/weather_classification/wk_weather_12cls_125848_125850_138001/20221223-172249/data.rec",  # noqa
                anno_path=f"{root_anno}/weather_classification/wk_weather_12cls_125848_125850_138001/20221223-172249/data.anno.pb_rec",  # noqa
                sample_weight=6,
            ),
            dict(
                rec_path=f"{hdfs_root_bucket}/data/sd_weather_classification/wk_weather_20230711_badcase_cls/20230711-113324/data.rec",  # noqa
                anno_path=f"{hdfs_root_bucket}/data/sd_weather_classification/wk_weather_20230711_badcase_cls/20230711-113324/data.anno.pb_rec",  # noqa
                sample_weight=1,
            ),
        ],
    ),
    illumination_classification=dict(
        train_batch_size_per_ctx=32,
        train_data_paths=[
            dict(
                rec_path=f"{hdfs_root_rec}/light_status/data/rec/2pe_wk_lightstatus_total_v9/2pe_wk_lightstatus_total_v9/train.rec",  # noqa
                anno_path=f"{hdfs_root_anno}/light_status/data/rec/2pe_wk_lightstatus_total_v9/2pe_wk_lightstatus_total_v9/train_attrs.anno.pb_rec",  # noqa
                sample_weight=1,
            ),
            dict(
                rec_path=f"{hdfs_root_bucket}/data/sd_lightstatus_classification/wk_light_230711_badcase_cls/20230711-114241/data.rec",  # noqa
                anno_path=f"{hdfs_root_bucket}/data/sd_lightstatus_classification/wk_light_230711_badcase_cls/20230711-114241/data.anno.pb_rec",  # noqa
                sample_weight=1,
            ),
        ],
    ),
    time_classification=dict(
        train_batch_size_per_ctx=32,
        train_data_paths=[
            dict(
                rec_path=f"{hdfs_root_rec}/Time/data/rec/2pe_wk_time_total_v8/2pe_wk_time_total_v8/train.rec",  # noqa
                anno_path=f"{hdfs_root_anno}/Time/data/rec/2pe_wk_time_total_v8/2pe_wk_time_total_v8/train_attrs.anno.pb_rec",  # noqa
                sample_weight=1,
            ),
            dict(
                rec_path=f"{hdfs_root_bucket}/data/sd_time_classification/wk_time_230711_badcase_cls/20230711-113744/data.rec",  # noqa
                anno_path=f"{hdfs_root_bucket}/data/sd_time_classification/wk_time_230711_badcase_cls/20230711-113744/data.anno.pb_rec",  # noqa
                sample_weight=1,
            ),
        ],
    ),
)

datapaths = EasyDict(datapaths)
