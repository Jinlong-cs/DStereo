from easydict import EasyDict
from hatbc.filestream.bucket.client import get_gpfs_bucket_mount_root

bucket2mount_root = get_gpfs_bucket_mount_root()


root_bucket = bucket2mount_root.get("adas", None)
root = "%s/big_model" % root_bucket

root_bucket_mono = bucket2mount_root.get("mono", None)


train_data_rec = {
    "traffic_sign_US_medium_attribute_cls": [
        f"{root}/train_dataset/traffic_sign_US_medium_attribute_cls/traffic_sign_US_medium_attribute_1/20221006-073925/data.rec",
    ],
    "light_attribute_cls": [
        f"{root}/train_dataset/light_attribute_cls/light_attribute_1/20221006-023011/data.rec"
    ],
    "road_arrow_attribute_cls": [
        f"{root}/train_dataset/road_arrow_attribute_cls/road_arrow_attribute_before_202212_1/20230318-034524/data.rec",
        f"{root}/train_dataset/road_arrow_attribute_cls/road_arrow_attribute_before_202212_2/20230317-104648/data.rec",
        f"{root}/train_dataset/road_arrow_attribute_cls/road_arrow_attribute_before_202212_3/20230317-164727/data.rec",
        f"{root}/train_dataset/road_arrow_attribute_cls/road_arrow_attribute_before_202212_4/20230317-014356/data.rec",
        f"{root}/train_dataset/road_arrow_attribute_cls/road_arrow_attribute_before_202212_5/20230317-044131/data.rec",
        # f"{root}/train_dataset/road_arrow_attribute_cls/road_arrow_attribute_before_202212_day_test/20230317-110352/data.rec",
        # f"{root}/train_dataset/road_arrow_attribute_cls/road_arrow_attribute_before_202212_night_test/20230317-080613/data.rec",
        # f"{root}/train_dataset/road_arrow_attribute_cls/road_arrow_attribute_before_202212_other_test/20230317-120127/data.rec"
    ],
    "traffic_sign_sub_attribute_cls": [
        f"{root}/train_dataset/traffic_sign_sub_attribute_cls/traffic_sign_sub_attribute_1/20221005-213430/data.rec",
        f"{root}/train_dataset/traffic_sign_sub_attribute_cls/traffic_sign_sub_attribute_2/20221006-063254/data.rec",
        f"{root}/train_dataset/traffic_sign_sub_attribute_cls/traffic_sign_sub_attribute_3/20221006-142449/data.rec",
    ],
    "traffic_light_secondary_attribute_cls": [
        f"{root}/train_dataset/traffic_light_secondary_attribute_cls/traffic_light_secondary_attribute_day_before_202212_1/20230315-141635/data.rec",
        f"{root}/train_dataset/traffic_light_secondary_attribute_cls/traffic_light_secondary_attribute_day_before_202212_2/20230315-144501/data.rec",
        f"{root}/train_dataset/traffic_light_secondary_attribute_cls/traffic_light_secondary_attribute_day_before_202212_3/20230315-145546/data.rec",
        # f"{root}/train_dataset/traffic_light_secondary_attribute_cls/traffic_light_secondary_attribute_day_before_202212_4/20230315-155439/data.rec",
        f"{root}/train_dataset/traffic_light_secondary_attribute_cls/traffic_light_secondary_attribute_night_before_202212_1/20230315-151546/data.rec",
        # f"{root}/train_dataset/traffic_light_secondary_attribute_cls/traffic_light_secondary_attribute_night_before_202212_2/20230315-153959/data.rec",
        f"{root}/train_dataset/traffic_light_secondary_attribute_cls/traffic_light_secondary_attribute_others_before_202212_1/20230317-203246/data.rec",
        f"{root}/train_dataset/traffic_light_secondary_attribute_cls/traffic_light_secondary_attribute_others_before_202212_2/20230317-201439/data.rec",
        f"{root}/train_dataset/traffic_light_secondary_attribute_cls/traffic_light_secondary_attribute_others_before_202212_3/20230315-175931/data.rec",
        f"{root}/train_dataset/traffic_light_secondary_attribute_cls/traffic_light_secondary_attribute_others_before_202212_4/20230316-001747/data.rec",
        # f"{root}/train_dataset/traffic_light_secondary_attribute_cls/traffic_light_secondary_attribute_others_before_202212_5/20230316-145701/data.rec"
    ],
    "traffic_cone_attribute_cls": [
        f"{root}/train_dataset/traffic_cone_attribute_cls/traffic_cone_attribute_before_202212_ag_1/20230318-112400/data.rec",
        f"{root}/train_dataset/traffic_cone_attribute_cls/traffic_cone_attribute_before_202212_ag_10/20230318-131603/data.rec",
        f"{root}/train_dataset/traffic_cone_attribute_cls/traffic_cone_attribute_before_202212_ag_11/20230318-131303/data.rec",
        f"{root}/train_dataset/traffic_cone_attribute_cls/traffic_cone_attribute_before_202212_ag_12/20230318-140923/data.rec",
        f"{root}/train_dataset/traffic_cone_attribute_cls/traffic_cone_attribute_before_202212_ag_13/20230318-135158/data.rec",
        f"{root}/train_dataset/traffic_cone_attribute_cls/traffic_cone_attribute_before_202212_ag_14/20230318-135926/data.rec",
        f"{root}/train_dataset/traffic_cone_attribute_cls/traffic_cone_attribute_before_202212_ag_15/20230318-143751/data.rec",
        f"{root}/train_dataset/traffic_cone_attribute_cls/traffic_cone_attribute_before_202212_ag_2/20230318-115639/data.rec",
        f"{root}/train_dataset/traffic_cone_attribute_cls/traffic_cone_attribute_before_202212_ag_3/20230318-120949/data.rec",
        f"{root}/train_dataset/traffic_cone_attribute_cls/traffic_cone_attribute_before_202212_ag_4/20230318-120746/data.rec",
        f"{root}/train_dataset/traffic_cone_attribute_cls/traffic_cone_attribute_before_202212_ag_5/20230318-115709/data.rec",
        f"{root}/train_dataset/traffic_cone_attribute_cls/traffic_cone_attribute_before_202212_ag_6/20230318-120908/data.rec",
        f"{root}/train_dataset/traffic_cone_attribute_cls/traffic_cone_attribute_before_202212_ag_7/20230318-124357/data.rec",
        f"{root}/train_dataset/traffic_cone_attribute_cls/traffic_cone_attribute_before_202212_ag_8/20230318-125842/data.rec",
        f"{root}/train_dataset/traffic_cone_attribute_cls/traffic_cone_attribute_before_202212_ag_9/20230318-133922/data.rec",
        # f"{root}/train_dataset/traffic_cone_attribute_cls/traffic_cone_attribute_before_202212_day_test/20230317-073338/data.rec",
        # f"{root}/train_dataset/traffic_cone_attribute_cls/traffic_cone_attribute_before_202212_night_test/20230317-140310/data.rec",
        # f"{root}/train_dataset/traffic_cone_attribute_cls/traffic_cone_attribute_before_202212_other_test/20230317-125953/data.rec"
    ],
    "traffic_light_primary_attribute_cls": [
        f"{root}/train_dataset/traffic_light_primary_attribute_cls/traffic_light_primary_attribute_day_before_202212_1/20230313-214931/data.rec",
        f"{root}/train_dataset/traffic_light_primary_attribute_cls/traffic_light_primary_attribute_day_before_202212_2/20230313-223153/data.rec",
        f"{root}/train_dataset/traffic_light_primary_attribute_cls/traffic_light_primary_attribute_day_before_202212_3/20230313-225149/data.rec",
        f"{root}/train_dataset/traffic_light_primary_attribute_cls/traffic_light_primary_attribute_day_before_202212_4/20230314-000353/data.rec",
        # f"{root}/train_dataset/traffic_light_primary_attribute_cls/traffic_light_primary_attribute_day_before_202212_5/20230314-003631/data.rec",
        f"{root}/train_dataset/traffic_light_primary_attribute_cls/traffic_light_primary_attribute_night_before_202212_1/20230314-005552/data.rec",
        # f"{root}/train_dataset/traffic_light_primary_attribute_cls/traffic_light_primary_attribute_night_before_202212_2/20230314-003019/data.rec",
        f"{root}/train_dataset/traffic_light_primary_attribute_cls/traffic_light_primary_attribute_others_before_202212_1/20230314-024037/data.rec",
        f"{root}/train_dataset/traffic_light_primary_attribute_cls/traffic_light_primary_attribute_others_before_202212_2/20230314-045903/data.rec",
        f"{root}/train_dataset/traffic_light_primary_attribute_cls/traffic_light_primary_attribute_others_before_202212_3/20230314-030407/data.rec",
        f"{root}/train_dataset/traffic_light_primary_attribute_cls/traffic_light_primary_attribute_others_before_202212_4/20230314-031124/data.rec",
        f"{root}/train_dataset/traffic_light_primary_attribute_cls/traffic_light_primary_attribute_others_before_202212_5/20230314-032037/data.rec",
        # f"{root}/train_dataset/traffic_light_primary_attribute_cls/traffic_light_primary_attribute_others_before_202212_6/20230314-032616/data.rec"
    ],
    "person_attribute_cls": [
        f"{root}/train_dataset/person_attribute_cls/person_SuperParking3_PDT2021002_before_202208_attribute_1/20230105-224652/data.rec",
        f"{root}/train_dataset/person_attribute_cls/person_SuperParking3_PDT2021002_before_202208_attribute_2/20230106-050828/data.rec",
        f"{root}/train_dataset/person_attribute_cls/person_mono3_PDT2020005_before_202208_attribute_1/20230105-193527/data.rec",
        f"{root}/train_dataset/person_attribute_cls/person_mono3_PDT2020005_before_202208_attribute_10/20230106-000335/data.rec",
        f"{root}/train_dataset/person_attribute_cls/person_mono3_PDT2020005_before_202208_attribute_11/20230106-045712/data.rec",
        f"{root}/train_dataset/person_attribute_cls/person_mono3_PDT2020005_before_202208_attribute_12/20230109-030719/data.rec",
        f"{root}/train_dataset/person_attribute_cls/person_mono3_PDT2020005_before_202208_attribute_13/20230109-064746/data.rec",
        f"{root}/train_dataset/person_attribute_cls/person_mono3_PDT2020005_before_202208_attribute_14/20230109-081327/data.rec",
        f"{root}/train_dataset/person_attribute_cls/person_mono3_PDT2020005_before_202208_attribute_15/20230105-205437/data.rec",
        f"{root}/train_dataset/person_attribute_cls/person_mono3_PDT2020005_before_202208_attribute_16/20230106-003424/data.rec",
        f"{root}/train_dataset/person_attribute_cls/person_mono3_PDT2020005_before_202208_attribute_2/20230109-085729/data.rec",
        f"{root}/train_dataset/person_attribute_cls/person_mono3_PDT2020005_before_202208_attribute_3/20230109-071846/data.rec",
        f"{root}/train_dataset/person_attribute_cls/person_mono3_PDT2020005_before_202208_attribute_4/20230106-033904/data.rec",
        f"{root}/train_dataset/person_attribute_cls/person_mono3_PDT2020005_before_202208_attribute_5/20230109-081836/data.rec",
        f"{root}/train_dataset/person_attribute_cls/person_mono3_PDT2020005_before_202208_attribute_6/20230105-212453/data.rec",
        f"{root}/train_dataset/person_attribute_cls/person_mono3_PDT2020005_before_202208_attribute_7/20230109-001800/data.rec",
        f"{root}/train_dataset/person_attribute_cls/person_mono3_PDT2020005_before_202208_attribute_8/20230109-080921/data.rec",
        f"{root}/train_dataset/person_attribute_cls/person_mono3_PDT2020005_before_202208_attribute_9/20230109-115238/data.rec",
        f"{root}/train_dataset/person_attribute_cls/person_mono3_PDT2020005_day_202208_202212_attribute_1/20230104-185222/data.rec",
        f"{root}/train_dataset/person_attribute_cls/person_mono3_PDT2020005_day_202208_202212_attribute_2/20230104-190301/data.rec",
        # f"{root}/train_dataset/person_attribute_cls/person_mono3_PDT2020005_day_202208_202212_attribute_3/20230104-192254/data.rec",
        f"{root}/train_dataset/person_attribute_cls/person_mono3_PDT2020005_night_202208_202212_attribute_1/20230104-190905/data.rec",
        # f"{root}/train_dataset/person_attribute_cls/person_mono3_PDT2020005_night_202208_202212_attribute_2/20230104-193638/data.rec",
        f"{root}/train_dataset/person_attribute_cls/person_mono3_PDT2020005_others_202208_202212_attribute_1/20230104-195047/data.rec",
        f"{root}/train_dataset/person_attribute_cls/person_pilot3_PDT2020008_before_202208_attribute_1/20230107-005733/data.rec",
        f"{root}/train_dataset/person_attribute_cls/person_pilot3_PDT2020008_before_202208_attribute_10/20230109-042933/data.rec",
        f"{root}/train_dataset/person_attribute_cls/person_pilot3_PDT2020008_before_202208_attribute_11/20230109-064040/data.rec",
        f"{root}/train_dataset/person_attribute_cls/person_pilot3_PDT2020008_before_202208_attribute_12/20230109-085901/data.rec",
        f"{root}/train_dataset/person_attribute_cls/person_pilot3_PDT2020008_before_202208_attribute_13/20230109-104805/data.rec",
        f"{root}/train_dataset/person_attribute_cls/person_pilot3_PDT2020008_before_202208_attribute_14/20230105-221203/data.rec",
        f"{root}/train_dataset/person_attribute_cls/person_pilot3_PDT2020008_before_202208_attribute_15/20230106-010504/data.rec",
        f"{root}/train_dataset/person_attribute_cls/person_pilot3_PDT2020008_before_202208_attribute_16/20230105-220939/data.rec",
        f"{root}/train_dataset/person_attribute_cls/person_pilot3_PDT2020008_before_202208_attribute_17/20230109-024628/data.rec",
        f"{root}/train_dataset/person_attribute_cls/person_pilot3_PDT2020008_before_202208_attribute_18/20230109-033025/data.rec",
        f"{root}/train_dataset/person_attribute_cls/person_pilot3_PDT2020008_before_202208_attribute_19/20230109-093125/data.rec",
        f"{root}/train_dataset/person_attribute_cls/person_pilot3_PDT2020008_before_202208_attribute_2/20230109-101051/data.rec",
        f"{root}/train_dataset/person_attribute_cls/person_pilot3_PDT2020008_before_202208_attribute_20/20230109-052039/data.rec",
        f"{root}/train_dataset/person_attribute_cls/person_pilot3_PDT2020008_before_202208_attribute_21/20230105-223622/data.rec",
        f"{root}/train_dataset/person_attribute_cls/person_pilot3_PDT2020008_before_202208_attribute_22/20230109-024232/data.rec",
        f"{root}/train_dataset/person_attribute_cls/person_pilot3_PDT2020008_before_202208_attribute_23/20230106-002038/data.rec",
        f"{root}/train_dataset/person_attribute_cls/person_pilot3_PDT2020008_before_202208_attribute_3/20230109-031742/data.rec",
        f"{root}/train_dataset/person_attribute_cls/person_pilot3_PDT2020008_before_202208_attribute_4/20230109-093520/data.rec",
        f"{root}/train_dataset/person_attribute_cls/person_pilot3_PDT2020008_before_202208_attribute_5/20230105-213703/data.rec",
        f"{root}/train_dataset/person_attribute_cls/person_pilot3_PDT2020008_before_202208_attribute_6/20230106-010257/data.rec",
        f"{root}/train_dataset/person_attribute_cls/person_pilot3_PDT2020008_before_202208_attribute_7/20230106-041549/data.rec",
        f"{root}/train_dataset/person_attribute_cls/person_pilot3_PDT2020008_before_202208_attribute_8/20230109-020435/data.rec",
        f"{root}/train_dataset/person_attribute_cls/person_pilot3_PDT2020008_before_202208_attribute_9/20230109-042434/data.rec",
        f"{root}/train_dataset/person_attribute_cls/person_pilot3_pilot5_day_202208_202212_attribute_1/20230104-165151/data.rec",
        f"{root}/train_dataset/person_attribute_cls/person_pilot3_pilot5_day_202208_202212_attribute_2/20230104-215026/data.rec",
        f"{root}/train_dataset/person_attribute_cls/person_pilot3_pilot5_day_202208_202212_attribute_3/20230104-165450/data.rec",
        f"{root}/train_dataset/person_attribute_cls/person_pilot3_pilot5_day_202208_202212_attribute_4/20230104-165524/data.rec",
        f"{root}/train_dataset/person_attribute_cls/person_pilot3_pilot5_day_202208_202212_attribute_5/20230104-170330/data.rec",
        f"{root}/train_dataset/person_attribute_cls/person_pilot3_pilot5_day_202208_202212_attribute_6/20230104-171710/data.rec",
        f"{root}/train_dataset/person_attribute_cls/person_pilot3_pilot5_day_202208_202212_attribute_7/20230104-171933/data.rec",
        f"{root}/train_dataset/person_attribute_cls/person_pilot3_pilot5_day_202208_202212_attribute_8/20230104-172648/data.rec",
        # f"{root}/train_dataset/person_attribute_cls/person_pilot3_pilot5_day_202208_202212_attribute_9/20230104-172844/data.rec",
        f"{root}/train_dataset/person_attribute_cls/person_pilot3_pilot5_night_202208_202212_attribute_1/20230104-205639/data.rec",
        f"{root}/train_dataset/person_attribute_cls/person_pilot3_pilot5_night_202208_202212_attribute_2/20230104-205547/data.rec",
        # f"{root}/train_dataset/person_attribute_cls/person_pilot3_pilot5_night_202208_202212_attribute_3/20230104-205534/data.rec",
        f"{root}/train_dataset/person_attribute_cls/person_pilot3_pilot5_night_202208_202212_attribute_4/20230104-213017/data.rec",
        f"{root}/train_dataset/person_attribute_cls/person_pilot3_pilot5_night_202208_202212_attribute_5/20230104-210946/data.rec",
        f"{root}/train_dataset/person_attribute_cls/person_pilot3_pilot5_others_202208_202212_attribute_1/20230104-184756/data.rec",
        f"{root}/train_dataset/person_attribute_cls/person_pilot3_pilot5_others_202208_202212_attribute_2/20230104-183121/data.rec",
    ],
    "traffic_sign_medium_attribute_cls": [
        f"{root}/train_dataset/traffic_sign_medium_attribute_cls/traffic_sign_medium_attribute_before_202212_1/20230317-064748/data.rec",
        f"{root}/train_dataset/traffic_sign_medium_attribute_cls/traffic_sign_medium_attribute_before_202212_10/20230317-142604/data.rec",
        f"{root}/train_dataset/traffic_sign_medium_attribute_cls/traffic_sign_medium_attribute_before_202212_11/20230317-145111/data.rec",
        f"{root}/train_dataset/traffic_sign_medium_attribute_cls/traffic_sign_medium_attribute_before_202212_12/20230317-140803/data.rec",
        f"{root}/train_dataset/traffic_sign_medium_attribute_cls/traffic_sign_medium_attribute_before_202212_13/20230317-151612/data.rec",
        f"{root}/train_dataset/traffic_sign_medium_attribute_cls/traffic_sign_medium_attribute_before_202212_14/20230318-030913/data.rec",
        f"{root}/train_dataset/traffic_sign_medium_attribute_cls/traffic_sign_medium_attribute_before_202212_15/20230317-153258/data.rec",
        f"{root}/train_dataset/traffic_sign_medium_attribute_cls/traffic_sign_medium_attribute_before_202212_16/20230317-121345/data.rec",
        f"{root}/train_dataset/traffic_sign_medium_attribute_cls/traffic_sign_medium_attribute_before_202212_17/20230317-133648/data.rec",
        f"{root}/train_dataset/traffic_sign_medium_attribute_cls/traffic_sign_medium_attribute_before_202212_18/20230317-152640/data.rec",
        f"{root}/train_dataset/traffic_sign_medium_attribute_cls/traffic_sign_medium_attribute_before_202212_19/20230317-153506/data.rec",
        f"{root}/train_dataset/traffic_sign_medium_attribute_cls/traffic_sign_medium_attribute_before_202212_2/20230318-025536/data.rec",
        f"{root}/train_dataset/traffic_sign_medium_attribute_cls/traffic_sign_medium_attribute_before_202212_3/20230317-142223/data.rec",
        f"{root}/train_dataset/traffic_sign_medium_attribute_cls/traffic_sign_medium_attribute_before_202212_4/20230317-135923/data.rec",
        f"{root}/train_dataset/traffic_sign_medium_attribute_cls/traffic_sign_medium_attribute_before_202212_5/20230317-140710/data.rec",
        f"{root}/train_dataset/traffic_sign_medium_attribute_cls/traffic_sign_medium_attribute_before_202212_6/20230317-144108/data.rec",
        f"{root}/train_dataset/traffic_sign_medium_attribute_cls/traffic_sign_medium_attribute_before_202212_7/20230317-134922/data.rec",
        f"{root}/train_dataset/traffic_sign_medium_attribute_cls/traffic_sign_medium_attribute_before_202212_8/20230317-144659/data.rec",
        f"{root}/train_dataset/traffic_sign_medium_attribute_cls/traffic_sign_medium_attribute_before_202212_9/20230317-123632/data.rec",
        # f"{root}/train_dataset/traffic_sign_medium_attribute_cls/traffic_sign_medium_attribute_before_202212_day_test/20230317-161422/data.rec",
        # f"{root}/train_dataset/traffic_sign_medium_attribute_cls/traffic_sign_medium_attribute_before_202212_night_test/20230317-144827/data.rec",
        # f"{root}/train_dataset/traffic_sign_medium_attribute_cls/traffic_sign_medium_attribute_before_202212_other_test/20230317-150221/data.rec"
    ],
    "vehicle_rear_attribute_cls": [
        f"{root}/train_dataset/vehicle_rear_attribute_cls/vehiclerear_mono3_PDT2020005_before_202208_attribute_1/20230105-145641/data.rec",
        f"{root}/train_dataset/vehicle_rear_attribute_cls/vehiclerear_mono3_PDT2020005_before_202208_attribute_10/20230109-050609/data.rec",
        f"{root}/train_dataset/vehicle_rear_attribute_cls/vehiclerear_mono3_PDT2020005_before_202208_attribute_11/20230105-175553/data.rec",
        f"{root}/train_dataset/vehicle_rear_attribute_cls/vehiclerear_mono3_PDT2020005_before_202208_attribute_12/20230105-161338/data.rec",
        f"{root}/train_dataset/vehicle_rear_attribute_cls/vehiclerear_mono3_PDT2020005_before_202208_attribute_13/20230105-193302/data.rec",
        f"{root}/train_dataset/vehicle_rear_attribute_cls/vehiclerear_mono3_PDT2020005_before_202208_attribute_14/20230109-101720/data.rec",
        f"{root}/train_dataset/vehicle_rear_attribute_cls/vehiclerear_mono3_PDT2020005_before_202208_attribute_15/20230109-010907/data.rec",
        f"{root}/train_dataset/vehicle_rear_attribute_cls/vehiclerear_mono3_PDT2020005_before_202208_attribute_16/20230109-021550/data.rec",
        f"{root}/train_dataset/vehicle_rear_attribute_cls/vehiclerear_mono3_PDT2020005_before_202208_attribute_2/20230105-143348/data.rec",
        f"{root}/train_dataset/vehicle_rear_attribute_cls/vehiclerear_mono3_PDT2020005_before_202208_attribute_3/20230109-005122/data.rec",
        f"{root}/train_dataset/vehicle_rear_attribute_cls/vehiclerear_mono3_PDT2020005_before_202208_attribute_5/20230109-082545/data.rec",
        f"{root}/train_dataset/vehicle_rear_attribute_cls/vehiclerear_mono3_PDT2020005_before_202208_attribute_6/20230109-102853/data.rec",
        f"{root}/train_dataset/vehicle_rear_attribute_cls/vehiclerear_mono3_PDT2020005_before_202208_attribute_7/20230105-160751/data.rec",
        f"{root}/train_dataset/vehicle_rear_attribute_cls/vehiclerear_mono3_PDT2020005_before_202208_attribute_8/20230106-032342/data.rec",
        f"{root}/train_dataset/vehicle_rear_attribute_cls/vehiclerear_mono3_PDT2020005_before_202208_attribute_9/20230108-235331/data.rec",
        f"{root}/train_dataset/vehicle_rear_attribute_cls/vehiclerear_mono3_PDT2020005_day_202208_202212_attribute_1/20230104-151044/data.rec",
        f"{root}/train_dataset/vehicle_rear_attribute_cls/vehiclerear_mono3_PDT2020005_day_202208_202212_attribute_2/20230104-151253/data.rec",
        f"{root}/train_dataset/vehicle_rear_attribute_cls/vehiclerear_mono3_PDT2020005_day_202208_202212_attribute_3/20230104-152346/data.rec",
        # f"{root}/train_dataset/vehicle_rear_attribute_cls/vehiclerear_mono3_PDT2020005_day_202208_202212_attribute_4/20230104-151955/data.rec",
        f"{root}/train_dataset/vehicle_rear_attribute_cls/vehiclerear_mono3_PDT2020005_night_202208_202212_attribute_1/20230104-152607/data.rec",
        f"{root}/train_dataset/vehicle_rear_attribute_cls/vehiclerear_mono3_PDT2020005_night_202208_202212_attribute_2/20230104-154131/data.rec",
        # f"{root}/train_dataset/vehicle_rear_attribute_cls/vehiclerear_mono3_PDT2020005_night_202208_202212_attribute_3/20230104-154403/data.rec",
        f"{root}/train_dataset/vehicle_rear_attribute_cls/vehiclerear_pilot3_PDT2020008_before_202208_attribute_1/20230109-053014/data.rec",
        f"{root}/train_dataset/vehicle_rear_attribute_cls/vehiclerear_pilot3_PDT2020008_before_202208_attribute_11/20230109-091627/data.rec",
        f"{root}/train_dataset/vehicle_rear_attribute_cls/vehiclerear_pilot3_PDT2020008_before_202208_attribute_2/20230109-091853/data.rec",
        f"{root}/train_dataset/vehicle_rear_attribute_cls/vehiclerear_pilot3_PDT2020008_before_202208_attribute_3/20230105-165908/data.rec",
        f"{root}/train_dataset/vehicle_rear_attribute_cls/vehiclerear_pilot3_PDT2020008_before_202208_attribute_4/20230109-010349/data.rec",
        f"{root}/train_dataset/vehicle_rear_attribute_cls/vehiclerear_pilot3_PDT2020008_before_202208_attribute_5/20230109-041608/data.rec",
        f"{root}/train_dataset/vehicle_rear_attribute_cls/vehiclerear_pilot3_PDT2020008_before_202208_attribute_7/20230109-011032/data.rec",
        f"{root}/train_dataset/vehicle_rear_attribute_cls/vehiclerear_pilot3_PDT2020008_before_202208_attribute_9/20230109-060043/data.rec",
        f"{root}/train_dataset/vehicle_rear_attribute_cls/vehiclerear_pilot3_PDT2020008_day_202208_202212_attribute_1/20230104-123910/data.rec",
        # f"{root}/train_dataset/vehicle_rear_attribute_cls/vehiclerear_pilot3_PDT2020008_day_202208_202212_attribute_2/20230104-124722/data.rec",
        f"{root}/train_dataset/vehicle_rear_attribute_cls/vehiclerear_pilot3_PDT2020008_day_202208_202212_attribute_3/20230104-125531/data.rec",
        f"{root}/train_dataset/vehicle_rear_attribute_cls/vehiclerear_pilot3_PDT2020008_day_202208_202212_attribute_4/20230104-130720/data.rec",
        f"{root}/train_dataset/vehicle_rear_attribute_cls/vehiclerear_pilot3_PDT2020008_night_202208_202212_attribute_1/20230104-131335/data.rec",
        # f"{root}/train_dataset/vehicle_rear_attribute_cls/vehiclerear_pilot3_PDT2020008_night_202208_202212_attribute_2/20230104-132053/data.rec",
        f"{root}/train_dataset/vehicle_rear_attribute_cls/vehiclerear_pilot3_PDT2020008_night_202208_202212_attribute_3/20230104-132818/data.rec",
        f"{root}/train_dataset/vehicle_rear_attribute_cls/vehiclerear_pilot3_PDT2020008_others_202208_202212_attribute_1/20230104-134726/data.rec",
        f"{root}/train_dataset/vehicle_rear_attribute_cls/vehiclerear_pilot5_PDT20220001_day_202208_202212_attribute_1/20230104-141250/data.rec",
        f"{root}/train_dataset/vehicle_rear_attribute_cls/vehiclerear_pilot5_PDT20220001_day_202208_202212_attribute_2/20230104-144001/data.rec",
        f"{root}/train_dataset/vehicle_rear_attribute_cls/vehiclerear_pilot5_PDT20220001_day_202208_202212_attribute_3/20230104-142703/data.rec",
        f"{root}/train_dataset/vehicle_rear_attribute_cls/vehiclerear_pilot5_PDT20220001_day_202208_202212_attribute_4/20230104-144248/data.rec",
        f"{root}/train_dataset/vehicle_rear_attribute_cls/vehiclerear_pilot5_PDT20220001_day_202208_202212_attribute_5/20230104-145408/data.rec",
        f"{root}/train_dataset/vehicle_rear_attribute_cls/vehiclerear_pilot5_PDT20220001_night_202208_202212_attribute_1/20230104-151357/data.rec",
        f"{root}/train_dataset/vehicle_rear_attribute_cls/vehiclerear_pilot5_PDT20220001_night_202208_202212_attribute_2/20230104-152043/data.rec",
        f"{root}/train_dataset/vehicle_rear_attribute_cls/vehiclerear_pilot5_PDT20220001_night_202208_202212_attribute_3/20230104-144545/data.rec",
    ],
    "vehicle_attribute_cls": [
        f"{root}/train_dataset/vehicle_again/vehicle_attribute_cls/vehicle_mono3_PDT2020005_before_202208_attribute_44cls_10/20230226-213125/data.rec",
        f"{root}/train_dataset/vehicle_again/vehicle_attribute_cls/vehicle_mono3_PDT2020005_before_202208_attribute_44cls_11/20230226-220322/data.rec",
        f"{root}/train_dataset/vehicle_again/vehicle_attribute_cls/vehicle_mono3_PDT2020005_before_202208_attribute_44cls_12/20230226-234943/data.rec",
        f"{root}/train_dataset/vehicle_again/vehicle_attribute_cls/vehicle_mono3_PDT2020005_before_202208_attribute_44cls_13/20230227-081832/data.rec",
        f"{root}/train_dataset/vehicle_again/vehicle_attribute_cls/vehicle_mono3_PDT2020005_before_202208_attribute_44cls_14/20230226-185326/data.rec",
        f"{root}/train_dataset/vehicle_again/vehicle_attribute_cls/vehicle_mono3_PDT2020005_before_202208_attribute_44cls_2/20230226-165525/data.rec",
        f"{root}/train_dataset/vehicle_again/vehicle_attribute_cls/vehicle_mono3_PDT2020005_before_202208_attribute_44cls_3/20230226-163651/data.rec",
        f"{root}/train_dataset/vehicle_again/vehicle_attribute_cls/vehicle_mono3_PDT2020005_before_202208_attribute_44cls_4/20230226-172712/data.rec",
        f"{root}/train_dataset/vehicle_again/vehicle_attribute_cls/vehicle_mono3_PDT2020005_before_202208_attribute_44cls_7/20230226-201517/data.rec",
        f"{root}/train_dataset/vehicle_again/vehicle_attribute_cls/vehicle_mono3_PDT2020005_before_202208_attribute_44cls_8/20230226-211033/data.rec",
        f"{root}/train_dataset/vehicle_again/vehicle_attribute_cls/vehicle_mono3_PDT2020005_before_202208_attribute_44cls_9/20230227-000910/data.rec",
        f"{root}/train_dataset/vehicle_attribute_cls/vehicle_mono3_PDT2020005_day_202208_202212_attribute_44cls_1/20230104-064152/data.rec",
        f"{root}/train_dataset/vehicle_attribute_cls/vehicle_mono3_PDT2020005_day_202208_202212_attribute_44cls_2/20230104-064954/data.rec",
        f"{root}/train_dataset/vehicle_attribute_cls/vehicle_mono3_PDT2020005_day_202208_202212_attribute_44cls_3/20230104-065720/data.rec",
        f"{root}/train_dataset/vehicle_attribute_cls/vehicle_mono3_PDT2020005_day_202208_202212_attribute_44cls_4/20230104-213637/data.rec",
        f"{root}/train_dataset/vehicle_attribute_cls/vehicle_mono3_PDT2020005_day_202208_202212_attribute_44cls_5/20230104-071925/data.rec",
        f"{root}/train_dataset/vehicle_attribute_cls/vehicle_mono3_PDT2020005_day_202208_202212_attribute_44cls_6/20230104-072325/data.rec",
        f"{root}/train_dataset/vehicle_attribute_cls/vehicle_mono3_PDT2020005_day_202208_202212_attribute_44cls_7/20230104-073154/data.rec",
        f"{root}/train_dataset/vehicle_attribute_cls/vehicle_mono3_PDT2020005_day_202208_202212_attribute_44cls_8/20230104-074243/data.rec",
        # f"{root}/train_dataset/vehicle_attribute_cls/vehicle_mono3_PDT2020005_day_202208_202212_attribute_44cls_9/20230104-074656/data.rec",
        f"{root}/train_dataset/vehicle_attribute_cls/vehicle_mono3_PDT2020005_night_202208_202212_attribute_44cls_1/20230104-075109/data.rec",
        f"{root}/train_dataset/vehicle_attribute_cls/vehicle_mono3_PDT2020005_night_202208_202212_attribute_44cls_2/20230104-075952/data.rec",
        f"{root}/train_dataset/vehicle_attribute_cls/vehicle_mono3_PDT2020005_night_202208_202212_attribute_44cls_3/20230104-080634/data.rec",
        # f"{root}/train_dataset/vehicle_attribute_cls/vehicle_mono3_PDT2020005_night_202208_202212_attribute_44cls_4/20230104-081523/data.rec",
        f"{root}/train_dataset/vehicle_again/vehicle_attribute_cls/vehicle_pilot3_PDT2020008_before_202208_attribute_44cls_10/20230227-021453/data.rec",
        f"{root}/train_dataset/vehicle_again/vehicle_attribute_cls/vehicle_pilot3_PDT2020008_before_202208_attribute_44cls_11/20230227-030513/data.rec",
        f"{root}/train_dataset/vehicle_again/vehicle_attribute_cls/vehicle_pilot3_PDT2020008_before_202208_attribute_44cls_12/20230227-033919/data.rec",
        f"{root}/train_dataset/vehicle_again/vehicle_attribute_cls/vehicle_pilot3_PDT2020008_before_202208_attribute_44cls_13/20230227-034014/data.rec",
        f"{root}/train_dataset/vehicle_again/vehicle_attribute_cls/vehicle_pilot3_PDT2020008_before_202208_attribute_44cls_14/20230227-045444/data.rec",
        f"{root}/train_dataset/vehicle_again/vehicle_attribute_cls/vehicle_pilot3_PDT2020008_before_202208_attribute_44cls_15/20230227-055702/data.rec",
        f"{root}/train_dataset/vehicle_again/vehicle_attribute_cls/vehicle_pilot3_PDT2020008_before_202208_attribute_44cls_17/20230227-021356/data.rec",
        f"{root}/train_dataset/vehicle_again/vehicle_attribute_cls/vehicle_pilot3_PDT2020008_before_202208_attribute_44cls_19/20230227-051206/data.rec",
        f"{root}/train_dataset/vehicle_again/vehicle_attribute_cls/vehicle_pilot3_PDT2020008_before_202208_attribute_44cls_2/20230227-054018/data.rec",
        f"{root}/train_dataset/vehicle_again/vehicle_attribute_cls/vehicle_pilot3_PDT2020008_before_202208_attribute_44cls_21/20230227-071102/data.rec",
        f"{root}/train_dataset/vehicle_again/vehicle_attribute_cls/vehicle_pilot3_PDT2020008_before_202208_attribute_44cls_22/20230227-022856/data.rec",
        f"{root}/train_dataset/vehicle_again/vehicle_attribute_cls/vehicle_pilot3_PDT2020008_before_202208_attribute_44cls_4/20230227-071033/data.rec",
        f"{root}/train_dataset/vehicle_again/vehicle_attribute_cls/vehicle_pilot3_PDT2020008_before_202208_attribute_44cls_5/20230227-042159/data.rec",
        f"{root}/train_dataset/vehicle_again/vehicle_attribute_cls/vehicle_pilot3_PDT2020008_before_202208_attribute_44cls_6/20230227-020607/data.rec",
        f"{root}/train_dataset/vehicle_again/vehicle_attribute_cls/vehicle_pilot3_PDT2020008_before_202208_attribute_44cls_8/20230227-074939/data.rec",
        f"{root}/train_dataset/vehicle_again/vehicle_attribute_cls/vehicle_pilot3_PDT2020008_before_202208_attribute_44cls_9/20230227-013133/data.rec",
        f"{root}/train_dataset/vehicle_again/vehicle_attribute_cls/vehicle_pilot3_PDT2020008_day_202208_202212_attribute_44cls_1/20230227-110240/data.rec",
        f"{root}/train_dataset/vehicle_again/vehicle_attribute_cls/vehicle_pilot3_PDT2020008_day_202208_202212_attribute_44cls_2/20230227-110831/data.rec",
        f"{root}/train_dataset/vehicle_again/vehicle_attribute_cls/vehicle_pilot3_PDT2020008_day_202208_202212_attribute_44cls_3/20230227-111925/data.rec",
        f"{root}/train_dataset/vehicle_again/vehicle_attribute_cls/vehicle_pilot3_PDT2020008_day_202208_202212_attribute_44cls_4/20230227-113246/data.rec",
        f"{root}/train_dataset/vehicle_again/vehicle_attribute_cls/vehicle_pilot3_PDT2020008_day_202208_202212_attribute_44cls_5/20230227-115045/data.rec",
        f"{root}/train_dataset/vehicle_again/vehicle_attribute_cls/vehicle_pilot3_PDT2020008_night_202208_202212_attribute_44cls_1/20230227-115735/data.rec",
        f"{root}/train_dataset/vehicle_again/vehicle_attribute_cls/vehicle_pilot3_PDT2020008_night_202208_202212_attribute_44cls_2/20230227-120136/data.rec",
        f"{root}/train_dataset/vehicle_again/vehicle_attribute_cls/vehicle_pilot3_PDT2020008_night_202208_202212_attribute_44cls_3/20230227-121201/data.rec",
        f"{root}/train_dataset/vehicle_again/vehicle_attribute_cls/vehicle_pilot3_PDT2020008_night_202208_202212_attribute_44cls_4/20230227-122137/data.rec",
        f"{root}/train_dataset/vehicle_again/vehicle_attribute_cls/vehicle_pilot3_PDT2020008_night_202208_202212_attribute_44cls_5/20230227-122657/data.rec",
        f"{root}/train_dataset/vehicle_attribute_cls/vehicle_pilot5_PDT20220001_day_202208_202212_attribute_44cls_1/20230104-045030/data.rec",
        f"{root}/train_dataset/vehicle_attribute_cls/vehicle_pilot5_PDT20220001_day_202208_202212_attribute_44cls_10/20230104-055443/data.rec",
        f"{root}/train_dataset/vehicle_attribute_cls/vehicle_pilot5_PDT20220001_day_202208_202212_attribute_44cls_11/20230104-061155/data.rec",
        f"{root}/train_dataset/vehicle_attribute_cls/vehicle_pilot5_PDT20220001_day_202208_202212_attribute_44cls_2/20230104-045350/data.rec",
        # f"{root}/train_dataset/vehicle_attribute_cls/vehicle_pilot5_PDT20220001_day_202208_202212_attribute_44cls_3/20230104-045228/data.rec",
        f"{root}/train_dataset/vehicle_attribute_cls/vehicle_pilot5_PDT20220001_day_202208_202212_attribute_44cls_4/20230104-051745/data.rec",
        f"{root}/train_dataset/vehicle_attribute_cls/vehicle_pilot5_PDT20220001_day_202208_202212_attribute_44cls_5/20230104-052318/data.rec",
        f"{root}/train_dataset/vehicle_attribute_cls/vehicle_pilot5_PDT20220001_day_202208_202212_attribute_44cls_6/20230104-053209/data.rec",
        f"{root}/train_dataset/vehicle_attribute_cls/vehicle_pilot5_PDT20220001_day_202208_202212_attribute_44cls_7/20230104-053849/data.rec",
        f"{root}/train_dataset/vehicle_attribute_cls/vehicle_pilot5_PDT20220001_day_202208_202212_attribute_44cls_8/20230104-054649/data.rec",
        f"{root}/train_dataset/vehicle_attribute_cls/vehicle_pilot5_PDT20220001_day_202208_202212_attribute_44cls_9/20230104-054248/data.rec",
        f"{root}/train_dataset/vehicle_attribute_cls/vehicle_pilot5_PDT20220001_night_202208_202212_attribute_44cls_1/20230104-060948/data.rec",
        f"{root}/train_dataset/vehicle_attribute_cls/vehicle_pilot5_PDT20220001_night_202208_202212_attribute_44cls_2/20230104-061850/data.rec",
        f"{root}/train_dataset/vehicle_attribute_cls/vehicle_pilot5_PDT20220001_night_202208_202212_attribute_44cls_3/20230104-061931/data.rec",
        # f"{root}/train_dataset/vehicle_attribute_cls/vehicle_pilot5_PDT20220001_night_202208_202212_attribute_44cls_4/20230104-063136/data.rec",
    ],
    "vertical_pole_attribute_cls": [
        f"{root}/train_dataset/vertical_pole_attribute_cls/vertical_pole_attribute_1/20221214-234552/data.rec",
        # f"{root}/train_dataset/vertical_pole_attribute_cls/vertical_pole_attribute_2/20221214-232059/data.rec"
    ],
    "face_attribute_cls": [
        f"{root}/train_dataset/face_attribute_cls/face_attribute_before_202212_1/20230321-153457/data.rec",
        f"{root}/train_dataset/face_attribute_cls/face_attribute_before_202212_10/20230321-170154/data.rec",
        f"{root}/train_dataset/face_attribute_cls/face_attribute_before_202212_11/20230321-171243/data.rec",
        f"{root}/train_dataset/face_attribute_cls/face_attribute_before_202212_12/20230321-172649/data.rec",
        f"{root}/train_dataset/face_attribute_cls/face_attribute_before_202212_13/20230321-172910/data.rec",
        f"{root}/train_dataset/face_attribute_cls/face_attribute_before_202212_14/20230321-173959/data.rec",
        f"{root}/train_dataset/face_attribute_cls/face_attribute_before_202212_2/20230321-154757/data.rec",
        f"{root}/train_dataset/face_attribute_cls/face_attribute_before_202212_3/20230321-155056/data.rec",
        f"{root}/train_dataset/face_attribute_cls/face_attribute_before_202212_4/20230321-161002/data.rec",
        f"{root}/train_dataset/face_attribute_cls/face_attribute_before_202212_5/20230321-161429/data.rec",
        f"{root}/train_dataset/face_attribute_cls/face_attribute_before_202212_6/20230321-163035/data.rec",
        f"{root}/train_dataset/face_attribute_cls/face_attribute_before_202212_7/20230321-163703/data.rec",
        f"{root}/train_dataset/face_attribute_cls/face_attribute_before_202212_8/20230321-163857/data.rec",
        f"{root}/train_dataset/face_attribute_cls/face_attribute_before_202212_9/20230321-165146/data.rec",
        # f"{root}/train_dataset/face_attribute_cls/face_attribute_before_202212_day_test/20230321-172321/data.rec",
        # f"{root}/train_dataset/face_attribute_cls/face_attribute_before_202212_night_test/20230321-172904/data.rec",
        # f"{root}/train_dataset/face_attribute_cls/face_attribute_before_202212_other_test/20230321-174314/data.rec"
    ],
    "person_head_attribute_cls": [
        f"{root}/train_dataset/person_head_attribute_cls/person_head_attribute_before_202212_1/20230321-183119/data.rec",
        f"{root}/train_dataset/person_head_attribute_cls/person_head_attribute_before_202212_2/20230321-185930/data.rec",
        f"{root}/train_dataset/person_head_attribute_cls/person_head_attribute_before_202212_3/20230321-190055/data.rec",
        f"{root}/train_dataset/person_head_attribute_cls/person_head_attribute_before_202212_4/20230321-191737/data.rec",
        f"{root}/train_dataset/person_head_attribute_cls/person_head_attribute_before_202212_5/20230321-192555/data.rec",
        f"{root}/train_dataset/person_head_attribute_cls/person_head_attribute_before_202212_6/20230321-193357/data.rec",
        f"{root}/train_dataset/person_head_attribute_cls/person_head_attribute_before_202212_7/20230321-194206/data.rec",
        f"{root}/train_dataset/person_head_attribute_cls/person_head_attribute_before_202212_8/20230321-195125/data.rec",
        f"{root}/train_dataset/person_head_attribute_cls/person_head_attribute_before_202212_9/20230321-200506/data.rec",
        # f"{root}/train_dataset/person_head_attribute_cls/person_head_attribute_before_202212_other_test/20230321-193911/data.rec"
    ],
    "wheel_attribute_cls": [
        f"{root}/train_dataset/wheel_attribute_cls/wheel_attribute_before_202212_noSP_1/20230321-125034/data.rec",
        f"{root}/train_dataset/wheel_attribute_cls/wheel_attribute_before_202212_noSP_2/20230321-125032/data.rec",
        f"{root}/train_dataset/wheel_attribute_cls/wheel_attribute_before_202212_noSP_3/20230321-131919/data.rec",
        f"{root}/train_dataset/wheel_attribute_cls/wheel_attribute_before_202212_noSP_4/20230321-131744/data.rec",
        f"{root}/train_dataset/wheel_attribute_cls/wheel_attribute_before_202212_noSP_5/20230321-134413/data.rec",
        f"{root}/train_dataset/wheel_attribute_cls/wheel_attribute_before_202212_noSP_6/20230321-133920/data.rec",
        # f"{root}/train_dataset/wheel_attribute_cls/wheel_attribute_before_202212_noSP_other_test/20230321-133358/data.rec"
    ],
    "cyclist_wheel_attribute_cls": [
        f"{root}/train_dataset/cyclist_wheel_attribute_cls/cyclist_wheel_attribute_before_202212_1/20230322-160453/data.rec",
        # f"{root}/train_dataset/cyclist_wheel_attribute_cls/cyclist_wheel_attribute_before_202212_day_test/20230322-153913/data.rec",
        # f"{root}/train_dataset/cyclist_wheel_attribute_cls/cyclist_wheel_attribute_before_202212_night_test/20230322-155252/data.rec"
    ],
    "vehicle_light_detection_attribute_cls": [
        f"{root}/train_dataset/vehicle_light_detection_attribute_cls/vehicle_light_detection_attribute_before_202212_1/20230322-170257/data.rec",
        f"{root}/train_dataset/vehicle_light_detection_attribute_cls/vehicle_light_detection_attribute_before_202212_2/20230322-171743/data.rec",
        f"{root}/train_dataset/vehicle_light_detection_attribute_cls/vehicle_light_detection_attribute_before_202212_3/20230322-172846/data.rec",
        # f"{root}/train_dataset/vehicle_light_detection_attribute_cls/vehicle_light_detection_attribute_before_202212_other_test/20230322-165354/data.rec"
    ],
    "vehicle_rear_recognition": [
        f"{root}/train_dataset/vehicle_rear_recognition/vehicle_rear_recognition_before_202212_1/20230323-124945/data.rec",
        f"{root}/train_dataset/vehicle_rear_recognition/vehicle_rear_recognition_before_202212_10/20230323-145007/data.rec",
        f"{root}/train_dataset/vehicle_rear_recognition/vehicle_rear_recognition_before_202212_2/20230323-130529/data.rec",
        f"{root}/train_dataset/vehicle_rear_recognition/vehicle_rear_recognition_before_202212_3/20230323-132234/data.rec",
        f"{root}/train_dataset/vehicle_rear_recognition/vehicle_rear_recognition_before_202212_4/20230323-133603/data.rec",
        f"{root}/train_dataset/vehicle_rear_recognition/vehicle_rear_recognition_before_202212_5/20230323-134229/data.rec",
        f"{root}/train_dataset/vehicle_rear_recognition/vehicle_rear_recognition_before_202212_6/20230323-135949/data.rec",
        f"{root}/train_dataset/vehicle_rear_recognition/vehicle_rear_recognition_before_202212_7/20230323-140244/data.rec",
        f"{root}/train_dataset/vehicle_rear_recognition/vehicle_rear_recognition_before_202212_8/20230323-140920/data.rec",
        f"{root}/train_dataset/vehicle_rear_recognition/vehicle_rear_recognition_before_202212_9/20230323-143107/data.rec",
        # f"{root}/train_dataset/vehicle_rear_recognition/vehicle_rear_recognition_before_202212_day_test/20230323-142820/data.rec",
        # f"{root}/train_dataset/vehicle_rear_recognition/vehicle_rear_recognition_before_202212_other_test/20230323-144519/data.rec"
    ],
    "vehicle_light_recognition": [
        f"{root}/train_dataset/vehicle_light_recognition/vehicle_light_recognition_before_202212_1/20230323-162655/data.rec",
        f"{root}/train_dataset/vehicle_light_recognition/vehicle_light_recognition_before_202212_10/20230323-171832/data.rec",
        f"{root}/train_dataset/vehicle_light_recognition/vehicle_light_recognition_before_202212_11/20230323-174710/data.rec",
        f"{root}/train_dataset/vehicle_light_recognition/vehicle_light_recognition_before_202212_12/20230323-175803/data.rec",
        f"{root}/train_dataset/vehicle_light_recognition/vehicle_light_recognition_before_202212_13/20230323-181733/data.rec",
        f"{root}/train_dataset/vehicle_light_recognition/vehicle_light_recognition_before_202212_14/20230323-184116/data.rec",
        f"{root}/train_dataset/vehicle_light_recognition/vehicle_light_recognition_before_202212_15/20230323-182440/data.rec",
        f"{root}/train_dataset/vehicle_light_recognition/vehicle_light_recognition_before_202212_16/20230323-183424/data.rec",
        f"{root}/train_dataset/vehicle_light_recognition/vehicle_light_recognition_before_202212_17/20230323-185250/data.rec",
        f"{root}/train_dataset/vehicle_light_recognition/vehicle_light_recognition_before_202212_18/20230323-190157/data.rec",
        f"{root}/train_dataset/vehicle_light_recognition/vehicle_light_recognition_before_202212_2/20230323-161320/data.rec",
        f"{root}/train_dataset/vehicle_light_recognition/vehicle_light_recognition_before_202212_3/20230323-160000/data.rec",
        f"{root}/train_dataset/vehicle_light_recognition/vehicle_light_recognition_before_202212_4/20230323-162545/data.rec",
        f"{root}/train_dataset/vehicle_light_recognition/vehicle_light_recognition_before_202212_5/20230323-162910/data.rec",
        f"{root}/train_dataset/vehicle_light_recognition/vehicle_light_recognition_before_202212_6/20230323-165619/data.rec",
        f"{root}/train_dataset/vehicle_light_recognition/vehicle_light_recognition_before_202212_7/20230323-165721/data.rec",
        f"{root}/train_dataset/vehicle_light_recognition/vehicle_light_recognition_before_202212_8/20230323-172926/data.rec",
        f"{root}/train_dataset/vehicle_light_recognition/vehicle_light_recognition_before_202212_9/20230323-173029/data.rec",
        # f"{root}/train_dataset/vehicle_light_recognition/vehicle_light_recognition_before_202212_day_test/20230323-175351/data.rec",
        # f"{root}/train_dataset/vehicle_light_recognition/vehicle_light_recognition_before_202212_night_test/20230323-182144/data.rec",
        # f"{root}/train_dataset/vehicle_light_recognition/vehicle_light_recognition_before_202212_other_test/20230323-181856/data.rec"
    ],
    "traffic_sign_recognition": [
        f"{root}/train_dataset/traffic_sign_recognition/traffic_sign_recognition_before_202212_1/20230323-202249/data.rec",
        f"{root}/train_dataset/traffic_sign_recognition/traffic_sign_recognition_before_202212_10/20230323-220650/data.rec",
        f"{root}/train_dataset/traffic_sign_recognition/traffic_sign_recognition_before_202212_11/20230323-221210/data.rec",
        f"{root}/train_dataset/traffic_sign_recognition/traffic_sign_recognition_before_202212_12/20230323-224004/data.rec",
        f"{root}/train_dataset/traffic_sign_recognition/traffic_sign_recognition_before_202212_13/20230323-222330/data.rec",
        f"{root}/train_dataset/traffic_sign_recognition/traffic_sign_recognition_before_202212_14/20230323-225452/data.rec",
        f"{root}/train_dataset/traffic_sign_recognition/traffic_sign_recognition_before_202212_15/20230323-224915/data.rec",
        f"{root}/train_dataset/traffic_sign_recognition/traffic_sign_recognition_before_202212_16/20230323-222615/data.rec",
        f"{root}/train_dataset/traffic_sign_recognition/traffic_sign_recognition_before_202212_2/20230323-202937/data.rec",
        f"{root}/train_dataset/traffic_sign_recognition/traffic_sign_recognition_before_202212_3/20230323-204724/data.rec",
        f"{root}/train_dataset/traffic_sign_recognition/traffic_sign_recognition_before_202212_4/20230323-205822/data.rec",
        f"{root}/train_dataset/traffic_sign_recognition/traffic_sign_recognition_before_202212_5/20230323-212823/data.rec",
        f"{root}/train_dataset/traffic_sign_recognition/traffic_sign_recognition_before_202212_6/20230323-210038/data.rec",
        f"{root}/train_dataset/traffic_sign_recognition/traffic_sign_recognition_before_202212_7/20230323-213817/data.rec",
        f"{root}/train_dataset/traffic_sign_recognition/traffic_sign_recognition_before_202212_8/20230323-213710/data.rec",
        f"{root}/train_dataset/traffic_sign_recognition/traffic_sign_recognition_before_202212_9/20230323-221440/data.rec",
        # f"{root}/train_dataset/traffic_sign_recognition/traffic_sign_recognition_before_202212_day_test/20230323-213617/data.rec",
        # f"{root}/train_dataset/traffic_sign_recognition/traffic_sign_recognition_before_202212_night_test/20230323-214551/data.rec",
        # f"{root}/train_dataset/traffic_sign_recognition/traffic_sign_recognition_before_202212_other_test/20230323-220346/data.rec"
    ],
    "vehicle_recognition": [
        f"{root}/train_dataset/vehicle_recognition/vehicle_recognition_before_202212_1/20230324-002432/data.rec",
        f"{root}/train_dataset/vehicle_recognition/vehicle_recognition_before_202212_2/20230323-235328/data.rec",
        f"{root}/train_dataset/vehicle_recognition/vehicle_recognition_before_202212_3/20230324-001558/data.rec",
        f"{root}/train_dataset/vehicle_recognition/vehicle_recognition_before_202212_4/20230324-005552/data.rec",
        # f"{root}/train_dataset/vehicle_recognition/vehicle_recognition_before_202212_day_test/20230323-230547/data.rec",
        # f"{root}/train_dataset/vehicle_recognition/vehicle_recognition_before_202212_night_test/20230323-225953/data.rec",
        # f"{root}/train_dataset/vehicle_recognition/vehicle_recognition_before_202212_other_test/20230323-230612/data.rec"
    ],
    "person_negative_recognition": [
        f"{root}/train_dataset/person_negative_recognition/person_negative_recognition_before_202212_1/20230324-123345/data.rec",
        f"{root}/train_dataset/person_negative_recognition/person_negative_recognition_before_202212_2/20230324-134305/data.rec",
        f"{root}/train_dataset/person_negative_recognition/person_negative_recognition_before_202212_3/20230324-131441/data.rec",
        # f"{root}/train_dataset/person_negative_recognition/person_negative_recognition_before_202212_day_test/20230324-113101/data.rec",
        # f"{root}/train_dataset/person_negative_recognition/person_negative_recognition_before_202212_other_test/20230324-120911/data.rec"
    ],
    "vehicle_negative_recognition": [
        f"{root}/train_dataset/vehicle_negative_recognition/vehicle_negative_recognition_before_202212_1/20230324-125118/data.rec",
        f"{root}/train_dataset/vehicle_negative_recognition/vehicle_negative_recognition_before_202212_10/20230324-143005/data.rec",
        f"{root}/train_dataset/vehicle_negative_recognition/vehicle_negative_recognition_before_202212_11/20230324-143328/data.rec",
        f"{root}/train_dataset/vehicle_negative_recognition/vehicle_negative_recognition_before_202212_12/20230324-145750/data.rec",
        f"{root}/train_dataset/vehicle_negative_recognition/vehicle_negative_recognition_before_202212_2/20230324-130417/data.rec",
        f"{root}/train_dataset/vehicle_negative_recognition/vehicle_negative_recognition_before_202212_3/20230324-133214/data.rec",
        f"{root}/train_dataset/vehicle_negative_recognition/vehicle_negative_recognition_before_202212_4/20230324-132817/data.rec",
        f"{root}/train_dataset/vehicle_negative_recognition/vehicle_negative_recognition_before_202212_5/20230324-134606/data.rec",
        f"{root}/train_dataset/vehicle_negative_recognition/vehicle_negative_recognition_before_202212_6/20230324-143959/data.rec",
        f"{root}/train_dataset/vehicle_negative_recognition/vehicle_negative_recognition_before_202212_7/20230324-135029/data.rec",
        f"{root}/train_dataset/vehicle_negative_recognition/vehicle_negative_recognition_before_202212_8/20230324-134426/data.rec",
        f"{root}/train_dataset/vehicle_negative_recognition/vehicle_negative_recognition_before_202212_9/20230324-140758/data.rec",
        # f"{root}/train_dataset/vehicle_negative_recognition/vehicle_negative_recognition_before_202212_day_test/20230324-135100/data.rec",
        # f"{root}/train_dataset/vehicle_negative_recognition/vehicle_negative_recognition_before_202212_night_test/20230324-145017/data.rec",
        # f"{root}/train_dataset/vehicle_negative_recognition/vehicle_negative_recognition_before_202212_other_test/20230324-135901/data.rec"
    ],
    "person_orientation_recognition": [
        f"{root}/train_dataset/person_orientation_recognition/person_orientation_recognition_before_202212_1/20230324-143356/data.rec",
        f"{root}/train_dataset/person_orientation_recognition/person_orientation_recognition_before_202212_10/20230324-160437/data.rec",
        f"{root}/train_dataset/person_orientation_recognition/person_orientation_recognition_before_202212_11/20230324-160749/data.rec",
        f"{root}/train_dataset/person_orientation_recognition/person_orientation_recognition_before_202212_12/20230324-162406/data.rec",
        f"{root}/train_dataset/person_orientation_recognition/person_orientation_recognition_before_202212_2/20230324-145252/data.rec",
        f"{root}/train_dataset/person_orientation_recognition/person_orientation_recognition_before_202212_3/20230324-145253/data.rec",
        f"{root}/train_dataset/person_orientation_recognition/person_orientation_recognition_before_202212_4/20230324-150241/data.rec",
        f"{root}/train_dataset/person_orientation_recognition/person_orientation_recognition_before_202212_5/20230324-150930/data.rec",
        f"{root}/train_dataset/person_orientation_recognition/person_orientation_recognition_before_202212_6/20230324-152346/data.rec",
        f"{root}/train_dataset/person_orientation_recognition/person_orientation_recognition_before_202212_7/20230324-154016/data.rec",
        f"{root}/train_dataset/person_orientation_recognition/person_orientation_recognition_before_202212_8/20230324-154629/data.rec",
        f"{root}/train_dataset/person_orientation_recognition/person_orientation_recognition_before_202212_9/20230324-154923/data.rec",
        # f"{root}/train_dataset/person_orientation_recognition/person_orientation_recognition_before_202212_other_test/20230324-161244/data.rec"
    ],
}

train_anno_rec = {
    "traffic_sign_US_medium_attribute_cls": [
        f"{root}/train_dataset/traffic_sign_US_medium_attribute_cls/traffic_sign_US_medium_attribute_1/20221006-073925/data.anno.pb_rec",
    ],
    "light_attribute_cls": [
        f"{root}/train_dataset/light_attribute_cls/light_attribute_1/20221006-023011/data.anno.pb_rec"
    ],
    "road_arrow_attribute_cls": [
        f"{root}/train_dataset/road_arrow_attribute_cls/road_arrow_attribute_before_202212_1/20230318-034524/data.anno.pb_rec",
        f"{root}/train_dataset/road_arrow_attribute_cls/road_arrow_attribute_before_202212_2/20230317-104648/data.anno.pb_rec",
        f"{root}/train_dataset/road_arrow_attribute_cls/road_arrow_attribute_before_202212_3/20230317-164727/data.anno.pb_rec",
        f"{root}/train_dataset/road_arrow_attribute_cls/road_arrow_attribute_before_202212_4/20230317-014356/data.anno.pb_rec",
        f"{root}/train_dataset/road_arrow_attribute_cls/road_arrow_attribute_before_202212_5/20230317-044131/data.anno.pb_rec",
        # f"{root}/train_dataset/road_arrow_attribute_cls/road_arrow_attribute_before_202212_day_test/20230317-110352/data.anno.pb_rec",
        # f"{root}/train_dataset/road_arrow_attribute_cls/road_arrow_attribute_before_202212_night_test/20230317-080613/data.anno.pb_rec",
        # f"{root}/train_dataset/road_arrow_attribute_cls/road_arrow_attribute_before_202212_other_test/20230317-120127/data.anno.pb_rec"
    ],
    "traffic_sign_sub_attribute_cls": [
        f"{root}/train_dataset/traffic_sign_sub_attribute_cls/traffic_sign_sub_attribute_1/20221005-213430/data.anno.pb_rec",
        f"{root}/train_dataset/traffic_sign_sub_attribute_cls/traffic_sign_sub_attribute_2/20221006-063254/data.anno.pb_rec",
        f"{root}/train_dataset/traffic_sign_sub_attribute_cls/traffic_sign_sub_attribute_3/20221006-142449/data.anno.pb_rec",
    ],
    "traffic_light_secondary_attribute_cls": [
        f"{root}/train_dataset/traffic_light_secondary_attribute_cls/traffic_light_secondary_attribute_day_before_202212_1/20230315-141635/data.anno.pb_rec",
        f"{root}/train_dataset/traffic_light_secondary_attribute_cls/traffic_light_secondary_attribute_day_before_202212_2/20230315-144501/data.anno.pb_rec",
        f"{root}/train_dataset/traffic_light_secondary_attribute_cls/traffic_light_secondary_attribute_day_before_202212_3/20230315-145546/data.anno.pb_rec",
        # f"{root}/train_dataset/traffic_light_secondary_attribute_cls/traffic_light_secondary_attribute_day_before_202212_4/20230315-155439/data.anno.pb_rec",
        f"{root}/train_dataset/traffic_light_secondary_attribute_cls/traffic_light_secondary_attribute_night_before_202212_1/20230315-151546/data.anno.pb_rec",
        # f"{root}/train_dataset/traffic_light_secondary_attribute_cls/traffic_light_secondary_attribute_night_before_202212_2/20230315-153959/data.anno.pb_rec",
        f"{root}/train_dataset/traffic_light_secondary_attribute_cls/traffic_light_secondary_attribute_others_before_202212_1/20230317-203246/data.anno.pb_rec",
        f"{root}/train_dataset/traffic_light_secondary_attribute_cls/traffic_light_secondary_attribute_others_before_202212_2/20230317-201439/data.anno.pb_rec",
        f"{root}/train_dataset/traffic_light_secondary_attribute_cls/traffic_light_secondary_attribute_others_before_202212_3/20230315-175931/data.anno.pb_rec",
        f"{root}/train_dataset/traffic_light_secondary_attribute_cls/traffic_light_secondary_attribute_others_before_202212_4/20230316-001747/data.anno.pb_rec",
        # f"{root}/train_dataset/traffic_light_secondary_attribute_cls/traffic_light_secondary_attribute_others_before_202212_5/20230316-145701/data.anno.pb_rec"
    ],
    "traffic_cone_attribute_cls": [
        f"{root}/train_dataset/traffic_cone_attribute_cls/traffic_cone_attribute_before_202212_ag_1/20230318-112400/data.anno.pb_rec",
        f"{root}/train_dataset/traffic_cone_attribute_cls/traffic_cone_attribute_before_202212_ag_10/20230318-131603/data.anno.pb_rec",
        f"{root}/train_dataset/traffic_cone_attribute_cls/traffic_cone_attribute_before_202212_ag_11/20230318-131303/data.anno.pb_rec",
        f"{root}/train_dataset/traffic_cone_attribute_cls/traffic_cone_attribute_before_202212_ag_12/20230318-140923/data.anno.pb_rec",
        f"{root}/train_dataset/traffic_cone_attribute_cls/traffic_cone_attribute_before_202212_ag_13/20230318-135158/data.anno.pb_rec",
        f"{root}/train_dataset/traffic_cone_attribute_cls/traffic_cone_attribute_before_202212_ag_14/20230318-135926/data.anno.pb_rec",
        f"{root}/train_dataset/traffic_cone_attribute_cls/traffic_cone_attribute_before_202212_ag_15/20230318-143751/data.anno.pb_rec",
        f"{root}/train_dataset/traffic_cone_attribute_cls/traffic_cone_attribute_before_202212_ag_2/20230318-115639/data.anno.pb_rec",
        f"{root}/train_dataset/traffic_cone_attribute_cls/traffic_cone_attribute_before_202212_ag_3/20230318-120949/data.anno.pb_rec",
        f"{root}/train_dataset/traffic_cone_attribute_cls/traffic_cone_attribute_before_202212_ag_4/20230318-120746/data.anno.pb_rec",
        f"{root}/train_dataset/traffic_cone_attribute_cls/traffic_cone_attribute_before_202212_ag_5/20230318-115709/data.anno.pb_rec",
        f"{root}/train_dataset/traffic_cone_attribute_cls/traffic_cone_attribute_before_202212_ag_6/20230318-120908/data.anno.pb_rec",
        f"{root}/train_dataset/traffic_cone_attribute_cls/traffic_cone_attribute_before_202212_ag_7/20230318-124357/data.anno.pb_rec",
        f"{root}/train_dataset/traffic_cone_attribute_cls/traffic_cone_attribute_before_202212_ag_8/20230318-125842/data.anno.pb_rec",
        f"{root}/train_dataset/traffic_cone_attribute_cls/traffic_cone_attribute_before_202212_ag_9/20230318-133922/data.anno.pb_rec",
        # f"{root}/train_dataset/traffic_cone_attribute_cls/traffic_cone_attribute_before_202212_day_test/20230317-073338/data.anno.pb_rec",
        # f"{root}/train_dataset/traffic_cone_attribute_cls/traffic_cone_attribute_before_202212_night_test/20230317-140310/data.anno.pb_rec",
        # f"{root}/train_dataset/traffic_cone_attribute_cls/traffic_cone_attribute_before_202212_other_test/20230317-125953/data.anno.pb_rec"
    ],
    "traffic_light_primary_attribute_cls": [
        f"{root}/train_dataset/traffic_light_primary_attribute_cls/traffic_light_primary_attribute_day_before_202212_1/20230313-214931/data.anno.pb_rec",
        f"{root}/train_dataset/traffic_light_primary_attribute_cls/traffic_light_primary_attribute_day_before_202212_2/20230313-223153/data.anno.pb_rec",
        f"{root}/train_dataset/traffic_light_primary_attribute_cls/traffic_light_primary_attribute_day_before_202212_3/20230313-225149/data.anno.pb_rec",
        f"{root}/train_dataset/traffic_light_primary_attribute_cls/traffic_light_primary_attribute_day_before_202212_4/20230314-000353/data.anno.pb_rec",
        # f"{root}/train_dataset/traffic_light_primary_attribute_cls/traffic_light_primary_attribute_day_before_202212_5/20230314-003631/data.anno.pb_rec",
        f"{root}/train_dataset/traffic_light_primary_attribute_cls/traffic_light_primary_attribute_night_before_202212_1/20230314-005552/data.anno.pb_rec",
        # f"{root}/train_dataset/traffic_light_primary_attribute_cls/traffic_light_primary_attribute_night_before_202212_2/20230314-003019/data.anno.pb_rec",
        f"{root}/train_dataset/traffic_light_primary_attribute_cls/traffic_light_primary_attribute_others_before_202212_1/20230314-024037/data.anno.pb_rec",
        f"{root}/train_dataset/traffic_light_primary_attribute_cls/traffic_light_primary_attribute_others_before_202212_2/20230314-045903/data.anno.pb_rec",
        f"{root}/train_dataset/traffic_light_primary_attribute_cls/traffic_light_primary_attribute_others_before_202212_3/20230314-030407/data.anno.pb_rec",
        f"{root}/train_dataset/traffic_light_primary_attribute_cls/traffic_light_primary_attribute_others_before_202212_4/20230314-031124/data.anno.pb_rec",
        f"{root}/train_dataset/traffic_light_primary_attribute_cls/traffic_light_primary_attribute_others_before_202212_5/20230314-032037/data.anno.pb_rec",
        # f"{root}/train_dataset/traffic_light_primary_attribute_cls/traffic_light_primary_attribute_others_before_202212_6/20230314-032616/data.anno.pb_rec"
    ],
    "person_attribute_cls": [
        f"{root}/train_dataset/person_attribute_cls/person_SuperParking3_PDT2021002_before_202208_attribute_1/20230105-224652/data.anno.pb_rec",
        f"{root}/train_dataset/person_attribute_cls/person_SuperParking3_PDT2021002_before_202208_attribute_2/20230106-050828/data.anno.pb_rec",
        f"{root}/train_dataset/person_attribute_cls/person_mono3_PDT2020005_before_202208_attribute_1/20230105-193527/data.anno.pb_rec",
        f"{root}/train_dataset/person_attribute_cls/person_mono3_PDT2020005_before_202208_attribute_10/20230106-000335/data.anno.pb_rec",
        f"{root}/train_dataset/person_attribute_cls/person_mono3_PDT2020005_before_202208_attribute_11/20230106-045712/data.anno.pb_rec",
        f"{root}/train_dataset/person_attribute_cls/person_mono3_PDT2020005_before_202208_attribute_12/20230109-030719/data.anno.pb_rec",
        f"{root}/train_dataset/person_attribute_cls/person_mono3_PDT2020005_before_202208_attribute_13/20230109-064746/data.anno.pb_rec",
        f"{root}/train_dataset/person_attribute_cls/person_mono3_PDT2020005_before_202208_attribute_14/20230109-081327/data.anno.pb_rec",
        f"{root}/train_dataset/person_attribute_cls/person_mono3_PDT2020005_before_202208_attribute_15/20230105-205437/data.anno.pb_rec",
        f"{root}/train_dataset/person_attribute_cls/person_mono3_PDT2020005_before_202208_attribute_16/20230106-003424/data.anno.pb_rec",
        f"{root}/train_dataset/person_attribute_cls/person_mono3_PDT2020005_before_202208_attribute_2/20230109-085729/data.anno.pb_rec",
        f"{root}/train_dataset/person_attribute_cls/person_mono3_PDT2020005_before_202208_attribute_3/20230109-071846/data.anno.pb_rec",
        f"{root}/train_dataset/person_attribute_cls/person_mono3_PDT2020005_before_202208_attribute_4/20230106-033904/data.anno.pb_rec",
        f"{root}/train_dataset/person_attribute_cls/person_mono3_PDT2020005_before_202208_attribute_5/20230109-081836/data.anno.pb_rec",
        f"{root}/train_dataset/person_attribute_cls/person_mono3_PDT2020005_before_202208_attribute_6/20230105-212453/data.anno.pb_rec",
        f"{root}/train_dataset/person_attribute_cls/person_mono3_PDT2020005_before_202208_attribute_7/20230109-001800/data.anno.pb_rec",
        f"{root}/train_dataset/person_attribute_cls/person_mono3_PDT2020005_before_202208_attribute_8/20230109-080921/data.anno.pb_rec",
        f"{root}/train_dataset/person_attribute_cls/person_mono3_PDT2020005_before_202208_attribute_9/20230109-115238/data.anno.pb_rec",
        f"{root}/train_dataset/person_attribute_cls/person_mono3_PDT2020005_day_202208_202212_attribute_1/20230104-185222/data.anno.pb_rec",
        f"{root}/train_dataset/person_attribute_cls/person_mono3_PDT2020005_day_202208_202212_attribute_2/20230104-190301/data.anno.pb_rec",
        # f"{root}/train_dataset/person_attribute_cls/person_mono3_PDT2020005_day_202208_202212_attribute_3/20230104-192254/data.anno.pb_rec",
        f"{root}/train_dataset/person_attribute_cls/person_mono3_PDT2020005_night_202208_202212_attribute_1/20230104-190905/data.anno.pb_rec",
        # f"{root}/train_dataset/person_attribute_cls/person_mono3_PDT2020005_night_202208_202212_attribute_2/20230104-193638/data.anno.pb_rec",
        f"{root}/train_dataset/person_attribute_cls/person_mono3_PDT2020005_others_202208_202212_attribute_1/20230104-195047/data.anno.pb_rec",
        f"{root}/train_dataset/person_attribute_cls/person_pilot3_PDT2020008_before_202208_attribute_1/20230107-005733/data.anno.pb_rec",
        f"{root}/train_dataset/person_attribute_cls/person_pilot3_PDT2020008_before_202208_attribute_10/20230109-042933/data.anno.pb_rec",
        f"{root}/train_dataset/person_attribute_cls/person_pilot3_PDT2020008_before_202208_attribute_11/20230109-064040/data.anno.pb_rec",
        f"{root}/train_dataset/person_attribute_cls/person_pilot3_PDT2020008_before_202208_attribute_12/20230109-085901/data.anno.pb_rec",
        f"{root}/train_dataset/person_attribute_cls/person_pilot3_PDT2020008_before_202208_attribute_13/20230109-104805/data.anno.pb_rec",
        f"{root}/train_dataset/person_attribute_cls/person_pilot3_PDT2020008_before_202208_attribute_14/20230105-221203/data.anno.pb_rec",
        f"{root}/train_dataset/person_attribute_cls/person_pilot3_PDT2020008_before_202208_attribute_15/20230106-010504/data.anno.pb_rec",
        f"{root}/train_dataset/person_attribute_cls/person_pilot3_PDT2020008_before_202208_attribute_16/20230105-220939/data.anno.pb_rec",
        f"{root}/train_dataset/person_attribute_cls/person_pilot3_PDT2020008_before_202208_attribute_17/20230109-024628/data.anno.pb_rec",
        f"{root}/train_dataset/person_attribute_cls/person_pilot3_PDT2020008_before_202208_attribute_18/20230109-033025/data.anno.pb_rec",
        f"{root}/train_dataset/person_attribute_cls/person_pilot3_PDT2020008_before_202208_attribute_19/20230109-093125/data.anno.pb_rec",
        f"{root}/train_dataset/person_attribute_cls/person_pilot3_PDT2020008_before_202208_attribute_2/20230109-101051/data.anno.pb_rec",
        f"{root}/train_dataset/person_attribute_cls/person_pilot3_PDT2020008_before_202208_attribute_20/20230109-052039/data.anno.pb_rec",
        f"{root}/train_dataset/person_attribute_cls/person_pilot3_PDT2020008_before_202208_attribute_21/20230105-223622/data.anno.pb_rec",
        f"{root}/train_dataset/person_attribute_cls/person_pilot3_PDT2020008_before_202208_attribute_22/20230109-024232/data.anno.pb_rec",
        f"{root}/train_dataset/person_attribute_cls/person_pilot3_PDT2020008_before_202208_attribute_23/20230106-002038/data.anno.pb_rec",
        f"{root}/train_dataset/person_attribute_cls/person_pilot3_PDT2020008_before_202208_attribute_3/20230109-031742/data.anno.pb_rec",
        f"{root}/train_dataset/person_attribute_cls/person_pilot3_PDT2020008_before_202208_attribute_4/20230109-093520/data.anno.pb_rec",
        f"{root}/train_dataset/person_attribute_cls/person_pilot3_PDT2020008_before_202208_attribute_5/20230105-213703/data.anno.pb_rec",
        f"{root}/train_dataset/person_attribute_cls/person_pilot3_PDT2020008_before_202208_attribute_6/20230106-010257/data.anno.pb_rec",
        f"{root}/train_dataset/person_attribute_cls/person_pilot3_PDT2020008_before_202208_attribute_7/20230106-041549/data.anno.pb_rec",
        f"{root}/train_dataset/person_attribute_cls/person_pilot3_PDT2020008_before_202208_attribute_8/20230109-020435/data.anno.pb_rec",
        f"{root}/train_dataset/person_attribute_cls/person_pilot3_PDT2020008_before_202208_attribute_9/20230109-042434/data.anno.pb_rec",
        f"{root}/train_dataset/person_attribute_cls/person_pilot3_pilot5_day_202208_202212_attribute_1/20230104-165151/data.anno.pb_rec",
        f"{root}/train_dataset/person_attribute_cls/person_pilot3_pilot5_day_202208_202212_attribute_2/20230104-215026/data.anno.pb_rec",
        f"{root}/train_dataset/person_attribute_cls/person_pilot3_pilot5_day_202208_202212_attribute_3/20230104-165450/data.anno.pb_rec",
        f"{root}/train_dataset/person_attribute_cls/person_pilot3_pilot5_day_202208_202212_attribute_4/20230104-165524/data.anno.pb_rec",
        f"{root}/train_dataset/person_attribute_cls/person_pilot3_pilot5_day_202208_202212_attribute_5/20230104-170330/data.anno.pb_rec",
        f"{root}/train_dataset/person_attribute_cls/person_pilot3_pilot5_day_202208_202212_attribute_6/20230104-171710/data.anno.pb_rec",
        f"{root}/train_dataset/person_attribute_cls/person_pilot3_pilot5_day_202208_202212_attribute_7/20230104-171933/data.anno.pb_rec",
        f"{root}/train_dataset/person_attribute_cls/person_pilot3_pilot5_day_202208_202212_attribute_8/20230104-172648/data.anno.pb_rec",
        # f"{root}/train_dataset/person_attribute_cls/person_pilot3_pilot5_day_202208_202212_attribute_9/20230104-172844/data.anno.pb_rec",
        f"{root}/train_dataset/person_attribute_cls/person_pilot3_pilot5_night_202208_202212_attribute_1/20230104-205639/data.anno.pb_rec",
        f"{root}/train_dataset/person_attribute_cls/person_pilot3_pilot5_night_202208_202212_attribute_2/20230104-205547/data.anno.pb_rec",
        # f"{root}/train_dataset/person_attribute_cls/person_pilot3_pilot5_night_202208_202212_attribute_3/20230104-205534/data.anno.pb_rec",
        f"{root}/train_dataset/person_attribute_cls/person_pilot3_pilot5_night_202208_202212_attribute_4/20230104-213017/data.anno.pb_rec",
        f"{root}/train_dataset/person_attribute_cls/person_pilot3_pilot5_night_202208_202212_attribute_5/20230104-210946/data.anno.pb_rec",
        f"{root}/train_dataset/person_attribute_cls/person_pilot3_pilot5_others_202208_202212_attribute_1/20230104-184756/data.anno.pb_rec",
        f"{root}/train_dataset/person_attribute_cls/person_pilot3_pilot5_others_202208_202212_attribute_2/20230104-183121/data.anno.pb_rec",
    ],
    "traffic_sign_medium_attribute_cls": [
        f"{root}/train_dataset/traffic_sign_medium_attribute_cls/traffic_sign_medium_attribute_before_202212_1/20230317-064748/data.anno.pb_rec",
        f"{root}/train_dataset/traffic_sign_medium_attribute_cls/traffic_sign_medium_attribute_before_202212_10/20230317-142604/data.anno.pb_rec",
        f"{root}/train_dataset/traffic_sign_medium_attribute_cls/traffic_sign_medium_attribute_before_202212_11/20230317-145111/data.anno.pb_rec",
        f"{root}/train_dataset/traffic_sign_medium_attribute_cls/traffic_sign_medium_attribute_before_202212_12/20230317-140803/data.anno.pb_rec",
        f"{root}/train_dataset/traffic_sign_medium_attribute_cls/traffic_sign_medium_attribute_before_202212_13/20230317-151612/data.anno.pb_rec",
        f"{root}/train_dataset/traffic_sign_medium_attribute_cls/traffic_sign_medium_attribute_before_202212_14/20230318-030913/data.anno.pb_rec",
        f"{root}/train_dataset/traffic_sign_medium_attribute_cls/traffic_sign_medium_attribute_before_202212_15/20230317-153258/data.anno.pb_rec",
        f"{root}/train_dataset/traffic_sign_medium_attribute_cls/traffic_sign_medium_attribute_before_202212_16/20230317-121345/data.anno.pb_rec",
        f"{root}/train_dataset/traffic_sign_medium_attribute_cls/traffic_sign_medium_attribute_before_202212_17/20230317-133648/data.anno.pb_rec",
        f"{root}/train_dataset/traffic_sign_medium_attribute_cls/traffic_sign_medium_attribute_before_202212_18/20230317-152640/data.anno.pb_rec",
        f"{root}/train_dataset/traffic_sign_medium_attribute_cls/traffic_sign_medium_attribute_before_202212_19/20230317-153506/data.anno.pb_rec",
        f"{root}/train_dataset/traffic_sign_medium_attribute_cls/traffic_sign_medium_attribute_before_202212_2/20230318-025536/data.anno.pb_rec",
        f"{root}/train_dataset/traffic_sign_medium_attribute_cls/traffic_sign_medium_attribute_before_202212_3/20230317-142223/data.anno.pb_rec",
        f"{root}/train_dataset/traffic_sign_medium_attribute_cls/traffic_sign_medium_attribute_before_202212_4/20230317-135923/data.anno.pb_rec",
        f"{root}/train_dataset/traffic_sign_medium_attribute_cls/traffic_sign_medium_attribute_before_202212_5/20230317-140710/data.anno.pb_rec",
        f"{root}/train_dataset/traffic_sign_medium_attribute_cls/traffic_sign_medium_attribute_before_202212_6/20230317-144108/data.anno.pb_rec",
        f"{root}/train_dataset/traffic_sign_medium_attribute_cls/traffic_sign_medium_attribute_before_202212_7/20230317-134922/data.anno.pb_rec",
        f"{root}/train_dataset/traffic_sign_medium_attribute_cls/traffic_sign_medium_attribute_before_202212_8/20230317-144659/data.anno.pb_rec",
        f"{root}/train_dataset/traffic_sign_medium_attribute_cls/traffic_sign_medium_attribute_before_202212_9/20230317-123632/data.anno.pb_rec",
        # f"{root}/train_dataset/traffic_sign_medium_attribute_cls/traffic_sign_medium_attribute_before_202212_day_test/20230317-161422/data.anno.pb_rec",
        # f"{root}/train_dataset/traffic_sign_medium_attribute_cls/traffic_sign_medium_attribute_before_202212_night_test/20230317-144827/data.anno.pb_rec",
        # f"{root}/train_dataset/traffic_sign_medium_attribute_cls/traffic_sign_medium_attribute_before_202212_other_test/20230317-150221/data.anno.pb_rec"
    ],
    "vehicle_rear_attribute_cls": [
        f"{root}/train_dataset/vehicle_rear_attribute_cls/vehiclerear_mono3_PDT2020005_before_202208_attribute_1/20230105-145641/data.anno.pb_rec",
        f"{root}/train_dataset/vehicle_rear_attribute_cls/vehiclerear_mono3_PDT2020005_before_202208_attribute_10/20230109-050609/data.anno.pb_rec",
        f"{root}/train_dataset/vehicle_rear_attribute_cls/vehiclerear_mono3_PDT2020005_before_202208_attribute_11/20230105-175553/data.anno.pb_rec",
        f"{root}/train_dataset/vehicle_rear_attribute_cls/vehiclerear_mono3_PDT2020005_before_202208_attribute_12/20230105-161338/data.anno.pb_rec",
        f"{root}/train_dataset/vehicle_rear_attribute_cls/vehiclerear_mono3_PDT2020005_before_202208_attribute_13/20230105-193302/data.anno.pb_rec",
        f"{root}/train_dataset/vehicle_rear_attribute_cls/vehiclerear_mono3_PDT2020005_before_202208_attribute_14/20230109-101720/data.anno.pb_rec",
        f"{root}/train_dataset/vehicle_rear_attribute_cls/vehiclerear_mono3_PDT2020005_before_202208_attribute_15/20230109-010907/data.anno.pb_rec",
        f"{root}/train_dataset/vehicle_rear_attribute_cls/vehiclerear_mono3_PDT2020005_before_202208_attribute_16/20230109-021550/data.anno.pb_rec",
        f"{root}/train_dataset/vehicle_rear_attribute_cls/vehiclerear_mono3_PDT2020005_before_202208_attribute_2/20230105-143348/data.anno.pb_rec",
        f"{root}/train_dataset/vehicle_rear_attribute_cls/vehiclerear_mono3_PDT2020005_before_202208_attribute_3/20230109-005122/data.anno.pb_rec",
        f"{root}/train_dataset/vehicle_rear_attribute_cls/vehiclerear_mono3_PDT2020005_before_202208_attribute_5/20230109-082545/data.anno.pb_rec",
        f"{root}/train_dataset/vehicle_rear_attribute_cls/vehiclerear_mono3_PDT2020005_before_202208_attribute_6/20230109-102853/data.anno.pb_rec",
        f"{root}/train_dataset/vehicle_rear_attribute_cls/vehiclerear_mono3_PDT2020005_before_202208_attribute_7/20230105-160751/data.anno.pb_rec",
        f"{root}/train_dataset/vehicle_rear_attribute_cls/vehiclerear_mono3_PDT2020005_before_202208_attribute_8/20230106-032342/data.anno.pb_rec",
        f"{root}/train_dataset/vehicle_rear_attribute_cls/vehiclerear_mono3_PDT2020005_before_202208_attribute_9/20230108-235331/data.anno.pb_rec",
        f"{root}/train_dataset/vehicle_rear_attribute_cls/vehiclerear_mono3_PDT2020005_day_202208_202212_attribute_1/20230104-151044/data.anno.pb_rec",
        f"{root}/train_dataset/vehicle_rear_attribute_cls/vehiclerear_mono3_PDT2020005_day_202208_202212_attribute_2/20230104-151253/data.anno.pb_rec",
        f"{root}/train_dataset/vehicle_rear_attribute_cls/vehiclerear_mono3_PDT2020005_day_202208_202212_attribute_3/20230104-152346/data.anno.pb_rec",
        # f"{root}/train_dataset/vehicle_rear_attribute_cls/vehiclerear_mono3_PDT2020005_day_202208_202212_attribute_4/20230104-151955/data.anno.pb_rec",
        f"{root}/train_dataset/vehicle_rear_attribute_cls/vehiclerear_mono3_PDT2020005_night_202208_202212_attribute_1/20230104-152607/data.anno.pb_rec",
        f"{root}/train_dataset/vehicle_rear_attribute_cls/vehiclerear_mono3_PDT2020005_night_202208_202212_attribute_2/20230104-154131/data.anno.pb_rec",
        # f"{root}/train_dataset/vehicle_rear_attribute_cls/vehiclerear_mono3_PDT2020005_night_202208_202212_attribute_3/20230104-154403/data.anno.pb_rec",
        f"{root}/train_dataset/vehicle_rear_attribute_cls/vehiclerear_pilot3_PDT2020008_before_202208_attribute_1/20230109-053014/data.anno.pb_rec",
        f"{root}/train_dataset/vehicle_rear_attribute_cls/vehiclerear_pilot3_PDT2020008_before_202208_attribute_11/20230109-091627/data.anno.pb_rec",
        f"{root}/train_dataset/vehicle_rear_attribute_cls/vehiclerear_pilot3_PDT2020008_before_202208_attribute_2/20230109-091853/data.anno.pb_rec",
        f"{root}/train_dataset/vehicle_rear_attribute_cls/vehiclerear_pilot3_PDT2020008_before_202208_attribute_3/20230105-165908/data.anno.pb_rec",
        f"{root}/train_dataset/vehicle_rear_attribute_cls/vehiclerear_pilot3_PDT2020008_before_202208_attribute_4/20230109-010349/data.anno.pb_rec",
        f"{root}/train_dataset/vehicle_rear_attribute_cls/vehiclerear_pilot3_PDT2020008_before_202208_attribute_5/20230109-041608/data.anno.pb_rec",
        f"{root}/train_dataset/vehicle_rear_attribute_cls/vehiclerear_pilot3_PDT2020008_before_202208_attribute_7/20230109-011032/data.anno.pb_rec",
        f"{root}/train_dataset/vehicle_rear_attribute_cls/vehiclerear_pilot3_PDT2020008_before_202208_attribute_9/20230109-060043/data.anno.pb_rec",
        f"{root}/train_dataset/vehicle_rear_attribute_cls/vehiclerear_pilot3_PDT2020008_day_202208_202212_attribute_1/20230104-123910/data.anno.pb_rec",
        # f"{root}/train_dataset/vehicle_rear_attribute_cls/vehiclerear_pilot3_PDT2020008_day_202208_202212_attribute_2/20230104-124722/data.anno.pb_rec",
        f"{root}/train_dataset/vehicle_rear_attribute_cls/vehiclerear_pilot3_PDT2020008_day_202208_202212_attribute_3/20230104-125531/data.anno.pb_rec",
        f"{root}/train_dataset/vehicle_rear_attribute_cls/vehiclerear_pilot3_PDT2020008_day_202208_202212_attribute_4/20230104-130720/data.anno.pb_rec",
        f"{root}/train_dataset/vehicle_rear_attribute_cls/vehiclerear_pilot3_PDT2020008_night_202208_202212_attribute_1/20230104-131335/data.anno.pb_rec",
        # f"{root}/train_dataset/vehicle_rear_attribute_cls/vehiclerear_pilot3_PDT2020008_night_202208_202212_attribute_2/20230104-132053/data.anno.pb_rec",
        f"{root}/train_dataset/vehicle_rear_attribute_cls/vehiclerear_pilot3_PDT2020008_night_202208_202212_attribute_3/20230104-132818/data.anno.pb_rec",
        f"{root}/train_dataset/vehicle_rear_attribute_cls/vehiclerear_pilot3_PDT2020008_others_202208_202212_attribute_1/20230104-134726/data.anno.pb_rec",
        f"{root}/train_dataset/vehicle_rear_attribute_cls/vehiclerear_pilot5_PDT20220001_day_202208_202212_attribute_1/20230104-141250/data.anno.pb_rec",
        f"{root}/train_dataset/vehicle_rear_attribute_cls/vehiclerear_pilot5_PDT20220001_day_202208_202212_attribute_2/20230104-144001/data.anno.pb_rec",
        f"{root}/train_dataset/vehicle_rear_attribute_cls/vehiclerear_pilot5_PDT20220001_day_202208_202212_attribute_3/20230104-142703/data.anno.pb_rec",
        f"{root}/train_dataset/vehicle_rear_attribute_cls/vehiclerear_pilot5_PDT20220001_day_202208_202212_attribute_4/20230104-144248/data.anno.pb_rec",
        f"{root}/train_dataset/vehicle_rear_attribute_cls/vehiclerear_pilot5_PDT20220001_day_202208_202212_attribute_5/20230104-145408/data.anno.pb_rec",
        f"{root}/train_dataset/vehicle_rear_attribute_cls/vehiclerear_pilot5_PDT20220001_night_202208_202212_attribute_1/20230104-151357/data.anno.pb_rec",
        f"{root}/train_dataset/vehicle_rear_attribute_cls/vehiclerear_pilot5_PDT20220001_night_202208_202212_attribute_2/20230104-152043/data.anno.pb_rec",
        f"{root}/train_dataset/vehicle_rear_attribute_cls/vehiclerear_pilot5_PDT20220001_night_202208_202212_attribute_3/20230104-144545/data.anno.pb_rec",
    ],
    "vehicle_attribute_cls": [
        f"{root}/train_dataset/vehicle_again/vehicle_attribute_cls/vehicle_mono3_PDT2020005_before_202208_attribute_44cls_10/20230226-213125/data.anno.pb_rec",
        f"{root}/train_dataset/vehicle_again/vehicle_attribute_cls/vehicle_mono3_PDT2020005_before_202208_attribute_44cls_11/20230226-220322/data.anno.pb_rec",
        f"{root}/train_dataset/vehicle_again/vehicle_attribute_cls/vehicle_mono3_PDT2020005_before_202208_attribute_44cls_12/20230226-234943/data.anno.pb_rec",
        f"{root}/train_dataset/vehicle_again/vehicle_attribute_cls/vehicle_mono3_PDT2020005_before_202208_attribute_44cls_13/20230227-081832/data.anno.pb_rec",
        f"{root}/train_dataset/vehicle_again/vehicle_attribute_cls/vehicle_mono3_PDT2020005_before_202208_attribute_44cls_14/20230226-185326/data.anno.pb_rec",
        f"{root}/train_dataset/vehicle_again/vehicle_attribute_cls/vehicle_mono3_PDT2020005_before_202208_attribute_44cls_2/20230226-165525/data.anno.pb_rec",
        f"{root}/train_dataset/vehicle_again/vehicle_attribute_cls/vehicle_mono3_PDT2020005_before_202208_attribute_44cls_3/20230226-163651/data.anno.pb_rec",
        f"{root}/train_dataset/vehicle_again/vehicle_attribute_cls/vehicle_mono3_PDT2020005_before_202208_attribute_44cls_4/20230226-172712/data.anno.pb_rec",
        f"{root}/train_dataset/vehicle_again/vehicle_attribute_cls/vehicle_mono3_PDT2020005_before_202208_attribute_44cls_7/20230226-201517/data.anno.pb_rec",
        f"{root}/train_dataset/vehicle_again/vehicle_attribute_cls/vehicle_mono3_PDT2020005_before_202208_attribute_44cls_8/20230226-211033/data.anno.pb_rec",
        f"{root}/train_dataset/vehicle_again/vehicle_attribute_cls/vehicle_mono3_PDT2020005_before_202208_attribute_44cls_9/20230227-000910/data.anno.pb_rec",
        f"{root}/train_dataset/vehicle_attribute_cls/vehicle_mono3_PDT2020005_day_202208_202212_attribute_44cls_1/20230104-064152/data.anno.pb_rec",
        f"{root}/train_dataset/vehicle_attribute_cls/vehicle_mono3_PDT2020005_day_202208_202212_attribute_44cls_2/20230104-064954/data.anno.pb_rec",
        f"{root}/train_dataset/vehicle_attribute_cls/vehicle_mono3_PDT2020005_day_202208_202212_attribute_44cls_3/20230104-065720/data.anno.pb_rec",
        f"{root}/train_dataset/vehicle_attribute_cls/vehicle_mono3_PDT2020005_day_202208_202212_attribute_44cls_4/20230104-213637/data.anno.pb_rec",
        f"{root}/train_dataset/vehicle_attribute_cls/vehicle_mono3_PDT2020005_day_202208_202212_attribute_44cls_5/20230104-071925/data.anno.pb_rec",
        f"{root}/train_dataset/vehicle_attribute_cls/vehicle_mono3_PDT2020005_day_202208_202212_attribute_44cls_6/20230104-072325/data.anno.pb_rec",
        f"{root}/train_dataset/vehicle_attribute_cls/vehicle_mono3_PDT2020005_day_202208_202212_attribute_44cls_7/20230104-073154/data.anno.pb_rec",
        f"{root}/train_dataset/vehicle_attribute_cls/vehicle_mono3_PDT2020005_day_202208_202212_attribute_44cls_8/20230104-074243/data.anno.pb_rec",
        # f"{root}/train_dataset/vehicle_attribute_cls/vehicle_mono3_PDT2020005_day_202208_202212_attribute_44cls_9/20230104-074656/data.anno.pb_rec",
        f"{root}/train_dataset/vehicle_attribute_cls/vehicle_mono3_PDT2020005_night_202208_202212_attribute_44cls_1/20230104-075109/data.anno.pb_rec",
        f"{root}/train_dataset/vehicle_attribute_cls/vehicle_mono3_PDT2020005_night_202208_202212_attribute_44cls_2/20230104-075952/data.anno.pb_rec",
        f"{root}/train_dataset/vehicle_attribute_cls/vehicle_mono3_PDT2020005_night_202208_202212_attribute_44cls_3/20230104-080634/data.anno.pb_rec",
        # f"{root}/train_dataset/vehicle_attribute_cls/vehicle_mono3_PDT2020005_night_202208_202212_attribute_44cls_4/20230104-081523/data.anno.pb_rec",
        f"{root}/train_dataset/vehicle_again/vehicle_attribute_cls/vehicle_pilot3_PDT2020008_before_202208_attribute_44cls_10/20230227-021453/data.anno.pb_rec",
        f"{root}/train_dataset/vehicle_again/vehicle_attribute_cls/vehicle_pilot3_PDT2020008_before_202208_attribute_44cls_11/20230227-030513/data.anno.pb_rec",
        f"{root}/train_dataset/vehicle_again/vehicle_attribute_cls/vehicle_pilot3_PDT2020008_before_202208_attribute_44cls_12/20230227-033919/data.anno.pb_rec",
        f"{root}/train_dataset/vehicle_again/vehicle_attribute_cls/vehicle_pilot3_PDT2020008_before_202208_attribute_44cls_13/20230227-034014/data.anno.pb_rec",
        f"{root}/train_dataset/vehicle_again/vehicle_attribute_cls/vehicle_pilot3_PDT2020008_before_202208_attribute_44cls_14/20230227-045444/data.anno.pb_rec",
        f"{root}/train_dataset/vehicle_again/vehicle_attribute_cls/vehicle_pilot3_PDT2020008_before_202208_attribute_44cls_15/20230227-055702/data.anno.pb_rec",
        f"{root}/train_dataset/vehicle_again/vehicle_attribute_cls/vehicle_pilot3_PDT2020008_before_202208_attribute_44cls_17/20230227-021356/data.anno.pb_rec",
        f"{root}/train_dataset/vehicle_again/vehicle_attribute_cls/vehicle_pilot3_PDT2020008_before_202208_attribute_44cls_19/20230227-051206/data.anno.pb_rec",
        f"{root}/train_dataset/vehicle_again/vehicle_attribute_cls/vehicle_pilot3_PDT2020008_before_202208_attribute_44cls_2/20230227-054018/data.anno.pb_rec",
        f"{root}/train_dataset/vehicle_again/vehicle_attribute_cls/vehicle_pilot3_PDT2020008_before_202208_attribute_44cls_21/20230227-071102/data.anno.pb_rec",
        f"{root}/train_dataset/vehicle_again/vehicle_attribute_cls/vehicle_pilot3_PDT2020008_before_202208_attribute_44cls_22/20230227-022856/data.anno.pb_rec",
        f"{root}/train_dataset/vehicle_again/vehicle_attribute_cls/vehicle_pilot3_PDT2020008_before_202208_attribute_44cls_4/20230227-071033/data.anno.pb_rec",
        f"{root}/train_dataset/vehicle_again/vehicle_attribute_cls/vehicle_pilot3_PDT2020008_before_202208_attribute_44cls_5/20230227-042159/data.anno.pb_rec",
        f"{root}/train_dataset/vehicle_again/vehicle_attribute_cls/vehicle_pilot3_PDT2020008_before_202208_attribute_44cls_6/20230227-020607/data.anno.pb_rec",
        f"{root}/train_dataset/vehicle_again/vehicle_attribute_cls/vehicle_pilot3_PDT2020008_before_202208_attribute_44cls_8/20230227-074939/data.anno.pb_rec",
        f"{root}/train_dataset/vehicle_again/vehicle_attribute_cls/vehicle_pilot3_PDT2020008_before_202208_attribute_44cls_9/20230227-013133/data.anno.pb_rec",
        f"{root}/train_dataset/vehicle_again/vehicle_attribute_cls/vehicle_pilot3_PDT2020008_day_202208_202212_attribute_44cls_1/20230227-110240/data.anno.pb_rec",
        f"{root}/train_dataset/vehicle_again/vehicle_attribute_cls/vehicle_pilot3_PDT2020008_day_202208_202212_attribute_44cls_2/20230227-110831/data.anno.pb_rec",
        f"{root}/train_dataset/vehicle_again/vehicle_attribute_cls/vehicle_pilot3_PDT2020008_day_202208_202212_attribute_44cls_3/20230227-111925/data.anno.pb_rec",
        f"{root}/train_dataset/vehicle_again/vehicle_attribute_cls/vehicle_pilot3_PDT2020008_day_202208_202212_attribute_44cls_4/20230227-113246/data.anno.pb_rec",
        f"{root}/train_dataset/vehicle_again/vehicle_attribute_cls/vehicle_pilot3_PDT2020008_day_202208_202212_attribute_44cls_5/20230227-115045/data.anno.pb_rec",
        f"{root}/train_dataset/vehicle_again/vehicle_attribute_cls/vehicle_pilot3_PDT2020008_night_202208_202212_attribute_44cls_1/20230227-115735/data.anno.pb_rec",
        f"{root}/train_dataset/vehicle_again/vehicle_attribute_cls/vehicle_pilot3_PDT2020008_night_202208_202212_attribute_44cls_2/20230227-120136/data.anno.pb_rec",
        f"{root}/train_dataset/vehicle_again/vehicle_attribute_cls/vehicle_pilot3_PDT2020008_night_202208_202212_attribute_44cls_3/20230227-121201/data.anno.pb_rec",
        f"{root}/train_dataset/vehicle_again/vehicle_attribute_cls/vehicle_pilot3_PDT2020008_night_202208_202212_attribute_44cls_4/20230227-122137/data.anno.pb_rec",
        f"{root}/train_dataset/vehicle_again/vehicle_attribute_cls/vehicle_pilot3_PDT2020008_night_202208_202212_attribute_44cls_5/20230227-122657/data.anno.pb_rec",
        f"{root}/train_dataset/vehicle_attribute_cls/vehicle_pilot5_PDT20220001_day_202208_202212_attribute_44cls_1/20230104-045030/data.anno.pb_rec",
        f"{root}/train_dataset/vehicle_attribute_cls/vehicle_pilot5_PDT20220001_day_202208_202212_attribute_44cls_10/20230104-055443/data.anno.pb_rec",
        f"{root}/train_dataset/vehicle_attribute_cls/vehicle_pilot5_PDT20220001_day_202208_202212_attribute_44cls_11/20230104-061155/data.anno.pb_rec",
        f"{root}/train_dataset/vehicle_attribute_cls/vehicle_pilot5_PDT20220001_day_202208_202212_attribute_44cls_2/20230104-045350/data.anno.pb_rec",
        # f"{root}/train_dataset/vehicle_attribute_cls/vehicle_pilot5_PDT20220001_day_202208_202212_attribute_44cls_3/20230104-045228/data.anno.pb_rec",
        f"{root}/train_dataset/vehicle_attribute_cls/vehicle_pilot5_PDT20220001_day_202208_202212_attribute_44cls_4/20230104-051745/data.anno.pb_rec",
        f"{root}/train_dataset/vehicle_attribute_cls/vehicle_pilot5_PDT20220001_day_202208_202212_attribute_44cls_5/20230104-052318/data.anno.pb_rec",
        f"{root}/train_dataset/vehicle_attribute_cls/vehicle_pilot5_PDT20220001_day_202208_202212_attribute_44cls_6/20230104-053209/data.anno.pb_rec",
        f"{root}/train_dataset/vehicle_attribute_cls/vehicle_pilot5_PDT20220001_day_202208_202212_attribute_44cls_7/20230104-053849/data.anno.pb_rec",
        f"{root}/train_dataset/vehicle_attribute_cls/vehicle_pilot5_PDT20220001_day_202208_202212_attribute_44cls_8/20230104-054649/data.anno.pb_rec",
        f"{root}/train_dataset/vehicle_attribute_cls/vehicle_pilot5_PDT20220001_day_202208_202212_attribute_44cls_9/20230104-054248/data.anno.pb_rec",
        f"{root}/train_dataset/vehicle_attribute_cls/vehicle_pilot5_PDT20220001_night_202208_202212_attribute_44cls_1/20230104-060948/data.anno.pb_rec",
        f"{root}/train_dataset/vehicle_attribute_cls/vehicle_pilot5_PDT20220001_night_202208_202212_attribute_44cls_2/20230104-061850/data.anno.pb_rec",
        f"{root}/train_dataset/vehicle_attribute_cls/vehicle_pilot5_PDT20220001_night_202208_202212_attribute_44cls_3/20230104-061931/data.anno.pb_rec",
        # f"{root}/train_dataset/vehicle_attribute_cls/vehicle_pilot5_PDT20220001_night_202208_202212_attribute_44cls_4/20230104-063136/data.anno.pb_rec",
    ],
    "vertical_pole_attribute_cls": [
        f"{root}/train_dataset/vertical_pole_attribute_cls/vertical_pole_attribute_1/20221214-234552/data.anno.pb_rec",
        # f"{root}/train_dataset/vertical_pole_attribute_cls/vertical_pole_attribute_2/20221214-232059/data.anno.pb_rec",
    ],
    "face_attribute_cls": [
        f"{root}/train_dataset/face_attribute_cls/face_attribute_before_202212_1/20230321-153457/data.anno.pb_rec",
        f"{root}/train_dataset/face_attribute_cls/face_attribute_before_202212_10/20230321-170154/data.anno.pb_rec",
        f"{root}/train_dataset/face_attribute_cls/face_attribute_before_202212_11/20230321-171243/data.anno.pb_rec",
        f"{root}/train_dataset/face_attribute_cls/face_attribute_before_202212_12/20230321-172649/data.anno.pb_rec",
        f"{root}/train_dataset/face_attribute_cls/face_attribute_before_202212_13/20230321-172910/data.anno.pb_rec",
        f"{root}/train_dataset/face_attribute_cls/face_attribute_before_202212_14/20230321-173959/data.anno.pb_rec",
        f"{root}/train_dataset/face_attribute_cls/face_attribute_before_202212_2/20230321-154757/data.anno.pb_rec",
        f"{root}/train_dataset/face_attribute_cls/face_attribute_before_202212_3/20230321-155056/data.anno.pb_rec",
        f"{root}/train_dataset/face_attribute_cls/face_attribute_before_202212_4/20230321-161002/data.anno.pb_rec",
        f"{root}/train_dataset/face_attribute_cls/face_attribute_before_202212_5/20230321-161429/data.anno.pb_rec",
        f"{root}/train_dataset/face_attribute_cls/face_attribute_before_202212_6/20230321-163035/data.anno.pb_rec",
        f"{root}/train_dataset/face_attribute_cls/face_attribute_before_202212_7/20230321-163703/data.anno.pb_rec",
        f"{root}/train_dataset/face_attribute_cls/face_attribute_before_202212_8/20230321-163857/data.anno.pb_rec",
        f"{root}/train_dataset/face_attribute_cls/face_attribute_before_202212_9/20230321-165146/data.anno.pb_rec",
        # f"{root}/train_dataset/face_attribute_cls/face_attribute_before_202212_day_test/20230321-172321/data.anno.pb_rec",
        # f"{root}/train_dataset/face_attribute_cls/face_attribute_before_202212_night_test/20230321-172904/data.anno.pb_rec",
        # f"{root}/train_dataset/face_attribute_cls/face_attribute_before_202212_other_test/20230321-174314/data.anno.pb_rec"
    ],
    "person_head_attribute_cls": [
        f"{root}/train_dataset/person_head_attribute_cls/person_head_attribute_before_202212_1/20230321-183119/data.anno.pb_rec",
        f"{root}/train_dataset/person_head_attribute_cls/person_head_attribute_before_202212_2/20230321-185930/data.anno.pb_rec",
        f"{root}/train_dataset/person_head_attribute_cls/person_head_attribute_before_202212_3/20230321-190055/data.anno.pb_rec",
        f"{root}/train_dataset/person_head_attribute_cls/person_head_attribute_before_202212_4/20230321-191737/data.anno.pb_rec",
        f"{root}/train_dataset/person_head_attribute_cls/person_head_attribute_before_202212_5/20230321-192555/data.anno.pb_rec",
        f"{root}/train_dataset/person_head_attribute_cls/person_head_attribute_before_202212_6/20230321-193357/data.anno.pb_rec",
        f"{root}/train_dataset/person_head_attribute_cls/person_head_attribute_before_202212_7/20230321-194206/data.anno.pb_rec",
        f"{root}/train_dataset/person_head_attribute_cls/person_head_attribute_before_202212_8/20230321-195125/data.anno.pb_rec",
        f"{root}/train_dataset/person_head_attribute_cls/person_head_attribute_before_202212_9/20230321-200506/data.anno.pb_rec",
        # f"{root}/train_dataset/person_head_attribute_cls/person_head_attribute_before_202212_other_test/20230321-193911/data.anno.pb_rec"
    ],
    "wheel_attribute_cls": [
        f"{root}/train_dataset/wheel_attribute_cls/wheel_attribute_before_202212_noSP_1/20230321-125034/data.anno.pb_rec",
        f"{root}/train_dataset/wheel_attribute_cls/wheel_attribute_before_202212_noSP_2/20230321-125032/data.anno.pb_rec",
        f"{root}/train_dataset/wheel_attribute_cls/wheel_attribute_before_202212_noSP_3/20230321-131919/data.anno.pb_rec",
        f"{root}/train_dataset/wheel_attribute_cls/wheel_attribute_before_202212_noSP_4/20230321-131744/data.anno.pb_rec",
        f"{root}/train_dataset/wheel_attribute_cls/wheel_attribute_before_202212_noSP_5/20230321-134413/data.anno.pb_rec",
        f"{root}/train_dataset/wheel_attribute_cls/wheel_attribute_before_202212_noSP_6/20230321-133920/data.anno.pb_rec",
        # f"{root}/train_dataset/wheel_attribute_cls/wheel_attribute_before_202212_noSP_other_test/20230321-133358/data.anno.pb_rec"
    ],
    "cyclist_wheel_attribute_cls": [
        f"{root}/train_dataset/cyclist_wheel_attribute_cls/cyclist_wheel_attribute_before_202212_1/20230322-160453/data.anno.pb_rec",
        # f"{root}/train_dataset/cyclist_wheel_attribute_cls/cyclist_wheel_attribute_before_202212_day_test/20230322-153913/data.anno.pb_rec",
        # f"{root}/train_dataset/cyclist_wheel_attribute_cls/cyclist_wheel_attribute_before_202212_night_test/20230322-155252/data.anno.pb_rec"
    ],
    "vehicle_light_detection_attribute_cls": [
        f"{root}/train_dataset/vehicle_light_detection_attribute_cls/vehicle_light_detection_attribute_before_202212_1/20230322-170257/data.anno.pb_rec",
        f"{root}/train_dataset/vehicle_light_detection_attribute_cls/vehicle_light_detection_attribute_before_202212_2/20230322-171743/data.anno.pb_rec",
        f"{root}/train_dataset/vehicle_light_detection_attribute_cls/vehicle_light_detection_attribute_before_202212_3/20230322-172846/data.anno.pb_rec",
        # f"{root}/train_dataset/vehicle_light_detection_attribute_cls/vehicle_light_detection_attribute_before_202212_other_test/20230322-165354/data.anno.pb_rec"
    ],
    "vehicle_rear_recognition": [
        f"{root}/train_dataset/vehicle_rear_recognition/vehicle_rear_recognition_before_202212_1/20230323-124945/data.anno.pb_rec",
        f"{root}/train_dataset/vehicle_rear_recognition/vehicle_rear_recognition_before_202212_10/20230323-145007/data.anno.pb_rec",
        f"{root}/train_dataset/vehicle_rear_recognition/vehicle_rear_recognition_before_202212_2/20230323-130529/data.anno.pb_rec",
        f"{root}/train_dataset/vehicle_rear_recognition/vehicle_rear_recognition_before_202212_3/20230323-132234/data.anno.pb_rec",
        f"{root}/train_dataset/vehicle_rear_recognition/vehicle_rear_recognition_before_202212_4/20230323-133603/data.anno.pb_rec",
        f"{root}/train_dataset/vehicle_rear_recognition/vehicle_rear_recognition_before_202212_5/20230323-134229/data.anno.pb_rec",
        f"{root}/train_dataset/vehicle_rear_recognition/vehicle_rear_recognition_before_202212_6/20230323-135949/data.anno.pb_rec",
        f"{root}/train_dataset/vehicle_rear_recognition/vehicle_rear_recognition_before_202212_7/20230323-140244/data.anno.pb_rec",
        f"{root}/train_dataset/vehicle_rear_recognition/vehicle_rear_recognition_before_202212_8/20230323-140920/data.anno.pb_rec",
        f"{root}/train_dataset/vehicle_rear_recognition/vehicle_rear_recognition_before_202212_9/20230323-143107/data.anno.pb_rec",
        # f"{root}/train_dataset/vehicle_rear_recognition/vehicle_rear_recognition_before_202212_day_test/20230323-142820/data.anno.pb_rec",
        # f"{root}/train_dataset/vehicle_rear_recognition/vehicle_rear_recognition_before_202212_other_test/20230323-144519/data.anno.pb_rec"
    ],
    "vehicle_light_recognition": [
        f"{root}/train_dataset/vehicle_light_recognition/vehicle_light_recognition_before_202212_1/20230323-162655/data.anno.pb_rec",
        f"{root}/train_dataset/vehicle_light_recognition/vehicle_light_recognition_before_202212_10/20230323-171832/data.anno.pb_rec",
        f"{root}/train_dataset/vehicle_light_recognition/vehicle_light_recognition_before_202212_11/20230323-174710/data.anno.pb_rec",
        f"{root}/train_dataset/vehicle_light_recognition/vehicle_light_recognition_before_202212_12/20230323-175803/data.anno.pb_rec",
        f"{root}/train_dataset/vehicle_light_recognition/vehicle_light_recognition_before_202212_13/20230323-181733/data.anno.pb_rec",
        f"{root}/train_dataset/vehicle_light_recognition/vehicle_light_recognition_before_202212_14/20230323-184116/data.anno.pb_rec",
        f"{root}/train_dataset/vehicle_light_recognition/vehicle_light_recognition_before_202212_15/20230323-182440/data.anno.pb_rec",
        f"{root}/train_dataset/vehicle_light_recognition/vehicle_light_recognition_before_202212_16/20230323-183424/data.anno.pb_rec",
        f"{root}/train_dataset/vehicle_light_recognition/vehicle_light_recognition_before_202212_17/20230323-185250/data.anno.pb_rec",
        f"{root}/train_dataset/vehicle_light_recognition/vehicle_light_recognition_before_202212_18/20230323-190157/data.anno.pb_rec",
        f"{root}/train_dataset/vehicle_light_recognition/vehicle_light_recognition_before_202212_2/20230323-161320/data.anno.pb_rec",
        f"{root}/train_dataset/vehicle_light_recognition/vehicle_light_recognition_before_202212_3/20230323-160000/data.anno.pb_rec",
        f"{root}/train_dataset/vehicle_light_recognition/vehicle_light_recognition_before_202212_4/20230323-162545/data.anno.pb_rec",
        f"{root}/train_dataset/vehicle_light_recognition/vehicle_light_recognition_before_202212_5/20230323-162910/data.anno.pb_rec",
        f"{root}/train_dataset/vehicle_light_recognition/vehicle_light_recognition_before_202212_6/20230323-165619/data.anno.pb_rec",
        f"{root}/train_dataset/vehicle_light_recognition/vehicle_light_recognition_before_202212_7/20230323-165721/data.anno.pb_rec",
        f"{root}/train_dataset/vehicle_light_recognition/vehicle_light_recognition_before_202212_8/20230323-172926/data.anno.pb_rec",
        f"{root}/train_dataset/vehicle_light_recognition/vehicle_light_recognition_before_202212_9/20230323-173029/data.anno.pb_rec",
        # f"{root}/train_dataset/vehicle_light_recognition/vehicle_light_recognition_before_202212_day_test/20230323-175351/data.anno.pb_rec",
        # f"{root}/train_dataset/vehicle_light_recognition/vehicle_light_recognition_before_202212_night_test/20230323-182144/data.anno.pb_rec",
        # f"{root}/train_dataset/vehicle_light_recognition/vehicle_light_recognition_before_202212_other_test/20230323-181856/data.anno.pb_rec"
    ],
    "traffic_sign_recognition": [
        f"{root}/train_dataset/traffic_sign_recognition/traffic_sign_recognition_before_202212_1/20230323-202249/data.anno.pb_rec",
        f"{root}/train_dataset/traffic_sign_recognition/traffic_sign_recognition_before_202212_10/20230323-220650/data.anno.pb_rec",
        f"{root}/train_dataset/traffic_sign_recognition/traffic_sign_recognition_before_202212_11/20230323-221210/data.anno.pb_rec",
        f"{root}/train_dataset/traffic_sign_recognition/traffic_sign_recognition_before_202212_12/20230323-224004/data.anno.pb_rec",
        f"{root}/train_dataset/traffic_sign_recognition/traffic_sign_recognition_before_202212_13/20230323-222330/data.anno.pb_rec",
        f"{root}/train_dataset/traffic_sign_recognition/traffic_sign_recognition_before_202212_14/20230323-225452/data.anno.pb_rec",
        f"{root}/train_dataset/traffic_sign_recognition/traffic_sign_recognition_before_202212_15/20230323-224915/data.anno.pb_rec",
        f"{root}/train_dataset/traffic_sign_recognition/traffic_sign_recognition_before_202212_16/20230323-222615/data.anno.pb_rec",
        f"{root}/train_dataset/traffic_sign_recognition/traffic_sign_recognition_before_202212_2/20230323-202937/data.anno.pb_rec",
        f"{root}/train_dataset/traffic_sign_recognition/traffic_sign_recognition_before_202212_3/20230323-204724/data.anno.pb_rec",
        f"{root}/train_dataset/traffic_sign_recognition/traffic_sign_recognition_before_202212_4/20230323-205822/data.anno.pb_rec",
        f"{root}/train_dataset/traffic_sign_recognition/traffic_sign_recognition_before_202212_5/20230323-212823/data.anno.pb_rec",
        f"{root}/train_dataset/traffic_sign_recognition/traffic_sign_recognition_before_202212_6/20230323-210038/data.anno.pb_rec",
        f"{root}/train_dataset/traffic_sign_recognition/traffic_sign_recognition_before_202212_7/20230323-213817/data.anno.pb_rec",
        f"{root}/train_dataset/traffic_sign_recognition/traffic_sign_recognition_before_202212_8/20230323-213710/data.anno.pb_rec",
        f"{root}/train_dataset/traffic_sign_recognition/traffic_sign_recognition_before_202212_9/20230323-221440/data.anno.pb_rec",
        # f"{root}/train_dataset/traffic_sign_recognition/traffic_sign_recognition_before_202212_day_test/20230323-213617/data.anno.pb_rec",
        # f"{root}/train_dataset/traffic_sign_recognition/traffic_sign_recognition_before_202212_night_test/20230323-214551/data.anno.pb_rec",
        # f"{root}/train_dataset/traffic_sign_recognition/traffic_sign_recognition_before_202212_other_test/20230323-220346/data.anno.pb_rec"
    ],
    "vehicle_recognition": [
        f"{root}/train_dataset/vehicle_recognition/vehicle_recognition_before_202212_1/20230324-002432/data.anno.pb_rec",
        f"{root}/train_dataset/vehicle_recognition/vehicle_recognition_before_202212_2/20230323-235328/data.anno.pb_rec",
        f"{root}/train_dataset/vehicle_recognition/vehicle_recognition_before_202212_3/20230324-001558/data.anno.pb_rec",
        f"{root}/train_dataset/vehicle_recognition/vehicle_recognition_before_202212_4/20230324-005552/data.anno.pb_rec",
        # f"{root}/train_dataset/vehicle_recognition/vehicle_recognition_before_202212_day_test/20230323-230547/data.anno.pb_rec",
        # f"{root}/train_dataset/vehicle_recognition/vehicle_recognition_before_202212_night_test/20230323-225953/data.anno.pb_rec",
        # f"{root}/train_dataset/vehicle_recognition/vehicle_recognition_before_202212_other_test/20230323-230612/data.anno.pb_rec"
    ],
    "person_negative_recognition": [
        f"{root}/train_dataset/person_negative_recognition/person_negative_recognition_before_202212_1/20230324-123345/data.anno.pb_rec",
        f"{root}/train_dataset/person_negative_recognition/person_negative_recognition_before_202212_2/20230324-134305/data.anno.pb_rec",
        f"{root}/train_dataset/person_negative_recognition/person_negative_recognition_before_202212_3/20230324-131441/data.anno.pb_rec",
        # f"{root}/train_dataset/person_negative_recognition/person_negative_recognition_before_202212_day_test/20230324-113101/data.anno.pb_rec",
        # f"{root}/train_dataset/person_negative_recognition/person_negative_recognition_before_202212_other_test/20230324-120911/data.anno.pb_rec"
    ],
    "vehicle_negative_recognition": [
        f"{root}/train_dataset/vehicle_negative_recognition/vehicle_negative_recognition_before_202212_1/20230324-125118/data.anno.pb_rec",
        f"{root}/train_dataset/vehicle_negative_recognition/vehicle_negative_recognition_before_202212_10/20230324-143005/data.anno.pb_rec",
        f"{root}/train_dataset/vehicle_negative_recognition/vehicle_negative_recognition_before_202212_11/20230324-143328/data.anno.pb_rec",
        f"{root}/train_dataset/vehicle_negative_recognition/vehicle_negative_recognition_before_202212_12/20230324-145750/data.anno.pb_rec",
        f"{root}/train_dataset/vehicle_negative_recognition/vehicle_negative_recognition_before_202212_2/20230324-130417/data.anno.pb_rec",
        f"{root}/train_dataset/vehicle_negative_recognition/vehicle_negative_recognition_before_202212_3/20230324-133214/data.anno.pb_rec",
        f"{root}/train_dataset/vehicle_negative_recognition/vehicle_negative_recognition_before_202212_4/20230324-132817/data.anno.pb_rec",
        f"{root}/train_dataset/vehicle_negative_recognition/vehicle_negative_recognition_before_202212_5/20230324-134606/data.anno.pb_rec",
        f"{root}/train_dataset/vehicle_negative_recognition/vehicle_negative_recognition_before_202212_6/20230324-143959/data.anno.pb_rec",
        f"{root}/train_dataset/vehicle_negative_recognition/vehicle_negative_recognition_before_202212_7/20230324-135029/data.anno.pb_rec",
        f"{root}/train_dataset/vehicle_negative_recognition/vehicle_negative_recognition_before_202212_8/20230324-134426/data.anno.pb_rec",
        f"{root}/train_dataset/vehicle_negative_recognition/vehicle_negative_recognition_before_202212_9/20230324-140758/data.anno.pb_rec",
        # f"{root}/train_dataset/vehicle_negative_recognition/vehicle_negative_recognition_before_202212_day_test/20230324-135100/data.anno.pb_rec",
        # f"{root}/train_dataset/vehicle_negative_recognition/vehicle_negative_recognition_before_202212_night_test/20230324-145017/data.anno.pb_rec",
        # f"{root}/train_dataset/vehicle_negative_recognition/vehicle_negative_recognition_before_202212_other_test/20230324-135901/data.anno.pb_rec"
    ],
    "person_orientation_recognition": [
        f"{root}/train_dataset/person_orientation_recognition/person_orientation_recognition_before_202212_1/20230324-143356/data.anno.pb_rec",
        f"{root}/train_dataset/person_orientation_recognition/person_orientation_recognition_before_202212_10/20230324-160437/data.anno.pb_rec",
        f"{root}/train_dataset/person_orientation_recognition/person_orientation_recognition_before_202212_11/20230324-160749/data.anno.pb_rec",
        f"{root}/train_dataset/person_orientation_recognition/person_orientation_recognition_before_202212_12/20230324-162406/data.anno.pb_rec",
        f"{root}/train_dataset/person_orientation_recognition/person_orientation_recognition_before_202212_2/20230324-145252/data.anno.pb_rec",
        f"{root}/train_dataset/person_orientation_recognition/person_orientation_recognition_before_202212_3/20230324-145253/data.anno.pb_rec",
        f"{root}/train_dataset/person_orientation_recognition/person_orientation_recognition_before_202212_4/20230324-150241/data.anno.pb_rec",
        f"{root}/train_dataset/person_orientation_recognition/person_orientation_recognition_before_202212_5/20230324-150930/data.anno.pb_rec",
        f"{root}/train_dataset/person_orientation_recognition/person_orientation_recognition_before_202212_6/20230324-152346/data.anno.pb_rec",
        f"{root}/train_dataset/person_orientation_recognition/person_orientation_recognition_before_202212_7/20230324-154016/data.anno.pb_rec",
        f"{root}/train_dataset/person_orientation_recognition/person_orientation_recognition_before_202212_8/20230324-154629/data.anno.pb_rec",
        f"{root}/train_dataset/person_orientation_recognition/person_orientation_recognition_before_202212_9/20230324-154923/data.anno.pb_rec",
        # f"{root}/train_dataset/person_orientation_recognition/person_orientation_recognition_before_202212_other_test/20230324-161244/data.anno.pb_rec"
    ],
}


val_data_rec = {
    "traffic_sign_US_medium_attribute_cls": [
        f"{root}/train_dataset/traffic_sign_US_medium_attribute_cls/traffic_sign_US_medium_attribute_2/20221006-095717/data.rec"
    ],
    "light_attribute_cls": [
        f"{root}/train_dataset/light_attribute_cls/light_attribute_2/20221005-222140/data.rec",
    ],
    "road_arrow_attribute_cls": [
        f"{root}/train_dataset/road_arrow_attribute_cls/road_arrow_attribute_before_202212_day_test/20230317-110352/data.rec",
        f"{root}/train_dataset/road_arrow_attribute_cls/road_arrow_attribute_before_202212_night_test/20230317-080613/data.rec",
        f"{root}/train_dataset/road_arrow_attribute_cls/road_arrow_attribute_before_202212_other_test/20230317-120127/data.rec",
    ],
    "traffic_sign_sub_attribute_cls": [
        f"{root}/train_dataset/traffic_sign_sub_attribute_cls/traffic_sign_sub_attribute_4/20221006-164347/data.rec",
    ],
    "traffic_light_secondary_attribute_cls": [
        f"{root}/train_dataset/traffic_light_secondary_attribute_cls/traffic_light_secondary_attribute_day_before_202212_4/20230315-155439/data.rec",
        f"{root}/train_dataset/traffic_light_secondary_attribute_cls/traffic_light_secondary_attribute_night_before_202212_2/20230315-153959/data.rec",
        f"{root}/train_dataset/traffic_light_secondary_attribute_cls/traffic_light_secondary_attribute_others_before_202212_5/20230316-145701/data.rec",
    ],
    "traffic_cone_attribute_cls": [
        f"{root}/train_dataset/traffic_cone_attribute_cls/traffic_cone_attribute_before_202212_day_test/20230317-073338/data.rec",
        f"{root}/train_dataset/traffic_cone_attribute_cls/traffic_cone_attribute_before_202212_night_test/20230317-140310/data.rec",
        f"{root}/train_dataset/traffic_cone_attribute_cls/traffic_cone_attribute_before_202212_other_test/20230317-125953/data.rec",
    ],
    "traffic_light_primary_attribute_cls": [
        # f"{root}/train_dataset/traffic_light_primary_attribute_cls/traffic_light_primary_attribute_day_before_202212_5/20230314-003631/data.rec",
        f"{root}/train_dataset/traffic_light_primary_attribute_cls/traffic_light_primary_attribute_night_before_202212_2/20230314-003019/data.rec",
        f"{root}/train_dataset/traffic_light_primary_attribute_cls/traffic_light_primary_attribute_others_before_202212_6/20230314-032616/data.rec",
        f"{root}/train_dataset/traffic_light_primary_attribute_cls/traffic_light_primary_attribute_day_before_202212_5_1/20230315-203659/data.rec",
    ],
    "person_attribute_cls": [
        f"{root}/train_dataset/person_attribute_cls/person_pilot3_pilot5_day_202208_202212_attribute_9/20230104-172844/data.rec",
        f"{root}/train_dataset/person_attribute_cls/person_pilot3_pilot5_night_202208_202212_attribute_3/20230104-205534/data.rec",
        f"{root}/train_dataset/person_attribute_cls/person_mono3_PDT2020005_day_202208_202212_attribute_3/20230104-192254/data.rec",
        f"{root}/train_dataset/person_attribute_cls/person_mono3_PDT2020005_night_202208_202212_attribute_2/20230104-193638/data.rec",
    ],
    "traffic_sign_medium_attribute_cls": [
        f"{root}/train_dataset/traffic_sign_medium_attribute_cls/traffic_sign_medium_attribute_before_202212_day_test/20230317-161422/data.rec",
        f"{root}/train_dataset/traffic_sign_medium_attribute_cls/traffic_sign_medium_attribute_before_202212_night_test/20230317-144827/data.rec",
        f"{root}/train_dataset/traffic_sign_medium_attribute_cls/traffic_sign_medium_attribute_before_202212_other_test/20230317-150221/data.rec",
    ],
    "vehicle_rear_attribute_cls": [
        f"{root}/train_dataset/vehicle_rear_attribute_cls/vehiclerear_mono3_PDT2020005_day_202208_202212_attribute_4/20230104-151955/data.rec",
        f"{root}/train_dataset/vehicle_rear_attribute_cls/vehiclerear_mono3_PDT2020005_night_202208_202212_attribute_3/20230104-154403/data.rec",
        f"{root}/train_dataset/vehicle_rear_attribute_cls/vehiclerear_pilot3_PDT2020008_day_202208_202212_attribute_2/20230104-124722/data.rec",
        f"{root}/train_dataset/vehicle_rear_attribute_cls/vehiclerear_pilot3_PDT2020008_night_202208_202212_attribute_2/20230104-132053/data.rec",
        # f"{root}/train_dataset/vehicle_rear_attribute_cls/vehiclerear_B_C_D_pilot_day/20230510-201629/data.rec",
    ],
    "vehicle_attribute_cls": [
        # f"{root}/train_dataset/vehicle_again/vehicle_attribute_cls/testset_pilot_AS33_day_crop_eval/20230227-153031/data.rec",
        f"{root}/train_dataset/vehicle_attribute_cls/vehicle_mono3_PDT2020005_day_202208_202212_attribute_44cls_9/20230104-074656/data.rec",
        f"{root}/train_dataset/vehicle_attribute_cls/vehicle_mono3_PDT2020005_night_202208_202212_attribute_44cls_4/20230104-081523/data.rec",
        f"{root}/train_dataset/vehicle_attribute_cls/vehicle_pilot5_PDT20220001_day_202208_202212_attribute_44cls_3/20230104-045228/data.rec",
        f"{root}/train_dataset/vehicle_attribute_cls/vehicle_pilot5_PDT20220001_night_202208_202212_attribute_44cls_4/20230104-063136/data.rec",
        # f"{root}/train_dataset/vehicle_again/vehicle_attribute_cls/testset_mono_6026770/20230227-150103/data.rec",
        # f"{root}/train_dataset/vehicle_again/vehicle_attribute_cls/testset_mono_6028372/20230227-154359/data.rec",
        # f"{root}/train_dataset/vehicle_again/vehicle_attribute_cls/testset_mono_6029367/20230227-152720/data.rec",
        # f"{root}/train_dataset/vehicle_again/vehicle_attribute_cls/testset_pilot_AS33_day_crop_eval/20230227-153031/data.rec",
        # f"{root}/train_dataset/vehicle_again/vehicle_attribute_cls/testset_pilot_AS33_day_eval/20230227-155827/data.rec",
        # f"{root}/train_dataset/vehicle_again/vehicle_attribute_cls/testset_pilot_AS33_night_crop_eval/20230227-162220/data.rec",
        # f"{root}/train_dataset/vehicle_again/vehicle_attribute_cls/testset_pilot_AS33_night_eval/20230227-170119/data.rec",
        # f"{root}/train_dataset/vehicle_again/vehicle_attribute_cls/testset_pilot_C385_x3c_day_crop_eval/20230227-160930/data.rec",
        # f"{root}/train_dataset/vehicle_again/vehicle_attribute_cls/testset_pilot_C385_x3c_day_eval/20230227-163032/data.rec",
        # f"{root}/train_dataset/vehicle_again/vehicle_attribute_cls/testset_pilot_C385_x3c_night_crop_eval/20230227-164128/data.rec",
        # f"{root}/train_dataset/vehicle_again/vehicle_attribute_cls/testset_pilot_C385_x3c_night_eval/20230227-163802/data.rec",
        # f"{root}/train_dataset/vehicle_again/vehicle_attribute_cls/testset_pilot_C385_x3c_parking_eval/20230227-164623/data.rec",
        # f"{root}/train_dataset/vehicle_again/vehicle_attribute_cls/testset_pilot_cc02_x3c_day_eval/20230227-163351/data.rec",
        # f"{root}/train_dataset/vehicle_again/vehicle_attribute_cls/testset_pilot_cc02_x3c_night_eval/20230227-164436/data.rec",
        # f"{root}/train_dataset/vehicle_again/vehicle_attribute_cls/testset_pilot_galaxy_0233_day_crop_eval/20230227-170132/data.rec",
        # f"{root}/train_dataset/vehicle_again/vehicle_attribute_cls/testset_pilot_galaxy_0233_day_eval/20230227-183100/data.rec",
        # f"{root}/train_dataset/vehicle_again/vehicle_attribute_cls/testset_pilot_galaxy_0233_night_crop_eval/20230227-183548/data.rec",
        # f"{root}/train_dataset/vehicle_again/vehicle_attribute_cls/testset_pilot_galaxy_0233_night_eval/20230227-185016/data.rec",
        # f"{root}/train_dataset/vehicle_again/vehicle_attribute_cls/testset_pilot_galaxy_x3c_day_eval/20230227-182755/data.rec",
        # f"{root}/train_dataset/vehicle_again/vehicle_attribute_cls/testset_pilot_galaxy_x3c_eval/20230227-192005/data.rec",
        # f"{root}/train_dataset/vehicle_again/vehicle_attribute_cls/testset_pilot_galaxy_x3c_left_day_crop_eval/20230227-190438/data.rec",
        # f"{root}/train_dataset/vehicle_again/vehicle_attribute_cls/testset_pilot_galaxy_x3c_left_night_crop_eval/20230227-180548/data.rec",
        # f"{root}/train_dataset/vehicle_again/vehicle_attribute_cls/testset_pilot_galaxy_x3c_night_eval/20230227-182249/data.rec",
        # f"{root}/train_dataset/vehicle_again/vehicle_attribute_cls/testset_pilot_galaxy_x3c_right_day_crop_eval/20230227-191922/data.rec",
        # f"{root}/train_dataset/vehicle_again/vehicle_attribute_cls/testset_pilot_galaxy_x3c_right_night_crop_eval/20230227-192512/data.rec"
    ],
    "vertical_pole_attribute_cls": [
        f"{root}/train_dataset/vertical_pole_attribute_cls/vertical_pole_attribute_2/20221214-232059/data.rec"
    ],
    "face_attribute_cls": [
        f"{root}/train_dataset/face_attribute_cls/face_attribute_before_202212_day_test/20230321-172321/data.rec",
        f"{root}/train_dataset/face_attribute_cls/face_attribute_before_202212_night_test/20230321-172904/data.rec",
        f"{root}/train_dataset/face_attribute_cls/face_attribute_before_202212_other_test/20230321-174314/data.rec",
    ],
    "person_head_attribute_cls": [
        f"{root}/train_dataset/person_head_attribute_cls/person_head_attribute_before_202212_other_test/20230321-193911/data.rec"
    ],
    "wheel_attribute_cls": [
        f"{root}/train_dataset/wheel_attribute_cls/wheel_attribute_before_202212_noSP_other_test/20230321-133358/data.rec"
    ],
    "cyclist_wheel_attribute_cls": [
        f"{root}/train_dataset/cyclist_wheel_attribute_cls/cyclist_wheel_attribute_before_202212_day_test/20230322-153913/data.rec",
        f"{root}/train_dataset/cyclist_wheel_attribute_cls/cyclist_wheel_attribute_before_202212_night_test/20230322-155252/data.rec",
    ],
    "vehicle_light_detection_attribute_cls": [
        f"{root}/train_dataset/vehicle_light_detection_attribute_cls/vehicle_light_detection_attribute_before_202212_other_test/20230322-165354/data.rec"
    ],
    "vehicle_rear_recognition": [
        f"{root}/train_dataset/vehicle_rear_recognition/vehicle_rear_recognition_before_202212_day_test/20230323-142820/data.rec",
        f"{root}/train_dataset/vehicle_rear_recognition/vehicle_rear_recognition_before_202212_other_test/20230323-144519/data.rec",
    ],
    "vehicle_light_recognition": [
        f"{root}/train_dataset/vehicle_light_recognition/vehicle_light_recognition_before_202212_day_test/20230323-175351/data.rec",
        f"{root}/train_dataset/vehicle_light_recognition/vehicle_light_recognition_before_202212_night_test/20230323-182144/data.rec",
        f"{root}/train_dataset/vehicle_light_recognition/vehicle_light_recognition_before_202212_other_test/20230323-181856/data.rec",
    ],
    "traffic_sign_recognition": [
        f"{root}/train_dataset/traffic_sign_recognition/traffic_sign_recognition_before_202212_day_test/20230323-213617/data.rec",
        f"{root}/train_dataset/traffic_sign_recognition/traffic_sign_recognition_before_202212_night_test/20230323-214551/data.rec",
        f"{root}/train_dataset/traffic_sign_recognition/traffic_sign_recognition_before_202212_other_test/20230323-220346/data.rec",
    ],
    "vehicle_recognition": [
        f"{root}/train_dataset/vehicle_recognition/vehicle_recognition_before_202212_day_test/20230323-230547/data.rec",
        f"{root}/train_dataset/vehicle_recognition/vehicle_recognition_before_202212_night_test/20230323-225953/data.rec",
        f"{root}/train_dataset/vehicle_recognition/vehicle_recognition_before_202212_other_test/20230323-230612/data.rec",
    ],
    "person_negative_recognition": [
        f"{root}/train_dataset/person_negative_recognition/person_negative_recognition_before_202212_day_test/20230324-113101/data.rec",
        f"{root}/train_dataset/person_negative_recognition/person_negative_recognition_before_202212_other_test/20230324-120911/data.rec",
    ],
    "vehicle_negative_recognition": [
        f"{root}/train_dataset/vehicle_negative_recognition/vehicle_negative_recognition_before_202212_day_test/20230324-135100/data.rec",
        f"{root}/train_dataset/vehicle_negative_recognition/vehicle_negative_recognition_before_202212_night_test/20230324-145017/data.rec",
        f"{root}/train_dataset/vehicle_negative_recognition/vehicle_negative_recognition_before_202212_other_test/20230324-135901/data.rec",
    ],
    "person_orientation_recognition": [
        f"{root}/train_dataset/person_orientation_recognition/person_orientation_recognition_before_202212_other_test/20230324-161244/data.rec"
    ],
}

val_anno_rec = {
    "traffic_sign_US_medium_attribute_cls": [
        f"{root}/train_dataset/traffic_sign_US_medium_attribute_cls/traffic_sign_US_medium_attribute_2/20221006-095717/data.anno.pb_rec"
    ],
    "light_attribute_cls": [
        f"{root}/train_dataset/light_attribute_cls/light_attribute_2/20221005-222140/data.anno.pb_rec",
    ],
    "road_arrow_attribute_cls": [
        f"{root}/train_dataset/road_arrow_attribute_cls/road_arrow_attribute_before_202212_day_test/20230317-110352/data.anno.pb_rec",
        f"{root}/train_dataset/road_arrow_attribute_cls/road_arrow_attribute_before_202212_night_test/20230317-080613/data.anno.pb_rec",
        f"{root}/train_dataset/road_arrow_attribute_cls/road_arrow_attribute_before_202212_other_test/20230317-120127/data.anno.pb_rec",
    ],
    "traffic_sign_sub_attribute_cls": [
        f"{root}/train_dataset/traffic_sign_sub_attribute_cls/traffic_sign_sub_attribute_4/20221006-164347/data.anno.pb_rec",
    ],
    "traffic_light_secondary_attribute_cls": [
        f"{root}/train_dataset/traffic_light_secondary_attribute_cls/traffic_light_secondary_attribute_day_before_202212_4/20230315-155439/data.anno.pb_rec",
        f"{root}/train_dataset/traffic_light_secondary_attribute_cls/traffic_light_secondary_attribute_night_before_202212_2/20230315-153959/data.anno.pb_rec",
        f"{root}/train_dataset/traffic_light_secondary_attribute_cls/traffic_light_secondary_attribute_others_before_202212_5/20230316-145701/data.anno.pb_rec",
    ],
    "traffic_cone_attribute_cls": [
        f"{root}/train_dataset/traffic_cone_attribute_cls/traffic_cone_attribute_before_202212_day_test/20230317-073338/data.anno.pb_rec",
        f"{root}/train_dataset/traffic_cone_attribute_cls/traffic_cone_attribute_before_202212_night_test/20230317-140310/data.anno.pb_rec",
        f"{root}/train_dataset/traffic_cone_attribute_cls/traffic_cone_attribute_before_202212_other_test/20230317-125953/data.anno.pb_rec",
    ],
    "traffic_light_primary_attribute_cls": [
        # f"{root}/train_dataset/traffic_light_primary_attribute_cls/traffic_light_primary_attribute_day_before_202212_5/20230314-003631/data.anno.pb_rec",
        f"{root}/train_dataset/traffic_light_primary_attribute_cls/traffic_light_primary_attribute_night_before_202212_2/20230314-003019/data.anno.pb_rec",
        f"{root}/train_dataset/traffic_light_primary_attribute_cls/traffic_light_primary_attribute_others_before_202212_6/20230314-032616/data.anno.pb_rec",
        f"{root}/train_dataset/traffic_light_primary_attribute_cls/traffic_light_primary_attribute_day_before_202212_5_1/20230315-203659/data.anno.pb_rec",
    ],
    "person_attribute_cls": [
        f"{root}/train_dataset/person_attribute_cls/person_pilot3_pilot5_day_202208_202212_attribute_9/20230104-172844/data.anno.pb_rec",
        f"{root}/train_dataset/person_attribute_cls/person_pilot3_pilot5_night_202208_202212_attribute_3/20230104-205534/data.anno.pb_rec",
        f"{root}/train_dataset/person_attribute_cls/person_mono3_PDT2020005_day_202208_202212_attribute_3/20230104-192254/data.anno.pb_rec",
        f"{root}/train_dataset/person_attribute_cls/person_mono3_PDT2020005_night_202208_202212_attribute_2/20230104-193638/data.anno.pb_rec",
    ],
    "traffic_sign_medium_attribute_cls": [
        f"{root}/train_dataset/traffic_sign_medium_attribute_cls/traffic_sign_medium_attribute_before_202212_day_test/20230317-161422/data.anno.pb_rec",
        f"{root}/train_dataset/traffic_sign_medium_attribute_cls/traffic_sign_medium_attribute_before_202212_night_test/20230317-144827/data.anno.pb_rec",
        f"{root}/train_dataset/traffic_sign_medium_attribute_cls/traffic_sign_medium_attribute_before_202212_other_test/20230317-150221/data.anno.pb_rec",
    ],
    "vehicle_rear_attribute_cls": [
        f"{root}/train_dataset/vehicle_rear_attribute_cls/vehiclerear_mono3_PDT2020005_day_202208_202212_attribute_4/20230104-151955/data.anno.pb_rec",
        f"{root}/train_dataset/vehicle_rear_attribute_cls/vehiclerear_mono3_PDT2020005_night_202208_202212_attribute_3/20230104-154403/data.anno.pb_rec",
        f"{root}/train_dataset/vehicle_rear_attribute_cls/vehiclerear_pilot3_PDT2020008_day_202208_202212_attribute_2/20230104-124722/data.anno.pb_rec",
        f"{root}/train_dataset/vehicle_rear_attribute_cls/vehiclerear_pilot3_PDT2020008_night_202208_202212_attribute_2/20230104-132053/data.anno.pb_rec",
        # f"{root}/train_dataset/vehicle_rear_attribute_cls/vehiclerear_B_C_D_pilot_day/20230510-201629/data.anno.pb_rec",
    ],
    "vehicle_attribute_cls": [
        # f"{root}/train_dataset/vehicle_again/vehicle_attribute_cls/testset_pilot_AS33_day_crop_eval/20230227-153031/data.anno.pb_rec",
        f"{root}/train_dataset/vehicle_attribute_cls/vehicle_mono3_PDT2020005_day_202208_202212_attribute_44cls_9/20230104-074656/data.anno.pb_rec",
        f"{root}/train_dataset/vehicle_attribute_cls/vehicle_mono3_PDT2020005_night_202208_202212_attribute_44cls_4/20230104-081523/data.anno.pb_rec",
        f"{root}/train_dataset/vehicle_attribute_cls/vehicle_pilot5_PDT20220001_day_202208_202212_attribute_44cls_3/20230104-045228/data.anno.pb_rec",
        f"{root}/train_dataset/vehicle_attribute_cls/vehicle_pilot5_PDT20220001_night_202208_202212_attribute_44cls_4/20230104-063136/data.anno.pb_rec",
        # f"{root}/train_dataset/vehicle_again/vehicle_attribute_cls/testset_mono_6026770/20230227-150103/data.anno.pb_rec",
        # f"{root}/train_dataset/vehicle_again/vehicle_attribute_cls/testset_mono_6028372/20230227-154359/data.anno.pb_rec",
        # f"{root}/train_dataset/vehicle_again/vehicle_attribute_cls/testset_mono_6029367/20230227-152720/data.anno.pb_rec",
        # f"{root}/train_dataset/vehicle_again/vehicle_attribute_cls/testset_pilot_AS33_day_crop_eval/20230227-153031/data.anno.pb_rec",
        # f"{root}/train_dataset/vehicle_again/vehicle_attribute_cls/testset_pilot_AS33_day_eval/20230227-155827/data.anno.pb_rec",
        # f"{root}/train_dataset/vehicle_again/vehicle_attribute_cls/testset_pilot_AS33_night_crop_eval/20230227-162220/data.anno.pb_rec",
        # f"{root}/train_dataset/vehicle_again/vehicle_attribute_cls/testset_pilot_AS33_night_eval/20230227-170119/data.anno.pb_rec",
        # f"{root}/train_dataset/vehicle_again/vehicle_attribute_cls/testset_pilot_C385_x3c_day_crop_eval/20230227-160930/data.anno.pb_rec",
        # f"{root}/train_dataset/vehicle_again/vehicle_attribute_cls/testset_pilot_C385_x3c_day_eval/20230227-163032/data.anno.pb_rec",
        # f"{root}/train_dataset/vehicle_again/vehicle_attribute_cls/testset_pilot_C385_x3c_night_crop_eval/20230227-164128/data.anno.pb_rec",
        # f"{root}/train_dataset/vehicle_again/vehicle_attribute_cls/testset_pilot_C385_x3c_night_eval/20230227-163802/data.anno.pb_rec",
        # f"{root}/train_dataset/vehicle_again/vehicle_attribute_cls/testset_pilot_C385_x3c_parking_eval/20230227-164623/data.anno.pb_rec",
        # f"{root}/train_dataset/vehicle_again/vehicle_attribute_cls/testset_pilot_cc02_x3c_day_eval/20230227-163351/data.anno.pb_rec",
        # f"{root}/train_dataset/vehicle_again/vehicle_attribute_cls/testset_pilot_cc02_x3c_night_eval/20230227-164436/data.anno.pb_rec",
        # f"{root}/train_dataset/vehicle_again/vehicle_attribute_cls/testset_pilot_galaxy_0233_day_crop_eval/20230227-170132/data.anno.pb_rec",
        # f"{root}/train_dataset/vehicle_again/vehicle_attribute_cls/testset_pilot_galaxy_0233_day_eval/20230227-183100/data.anno.pb_rec",
        # f"{root}/train_dataset/vehicle_again/vehicle_attribute_cls/testset_pilot_galaxy_0233_night_crop_eval/20230227-183548/data.anno.pb_rec",
        # f"{root}/train_dataset/vehicle_again/vehicle_attribute_cls/testset_pilot_galaxy_0233_night_eval/20230227-185016/data.anno.pb_rec",
        # f"{root}/train_dataset/vehicle_again/vehicle_attribute_cls/testset_pilot_galaxy_x3c_day_eval/20230227-182755/data.anno.pb_rec",
        # f"{root}/train_dataset/vehicle_again/vehicle_attribute_cls/testset_pilot_galaxy_x3c_eval/20230227-192005/data.anno.pb_rec",
        # f"{root}/train_dataset/vehicle_again/vehicle_attribute_cls/testset_pilot_galaxy_x3c_left_day_crop_eval/20230227-190438/data.anno.pb_rec",
        # f"{root}/train_dataset/vehicle_again/vehicle_attribute_cls/testset_pilot_galaxy_x3c_left_night_crop_eval/20230227-180548/data.anno.pb_rec",
        # f"{root}/train_dataset/vehicle_again/vehicle_attribute_cls/testset_pilot_galaxy_x3c_night_eval/20230227-182249/data.anno.pb_rec",
        # f"{root}/train_dataset/vehicle_again/vehicle_attribute_cls/testset_pilot_galaxy_x3c_right_day_crop_eval/20230227-191922/data.anno.pb_rec",
        # f"{root}/train_dataset/vehicle_again/vehicle_attribute_cls/testset_pilot_galaxy_x3c_right_night_crop_eval/20230227-192512/data.anno.pb_rec"
    ],
    "vertical_pole_attribute_cls": [
        f"{root}/train_dataset/vertical_pole_attribute_cls/vertical_pole_attribute_2/20221214-232059/data.anno.pb_rec",
    ],
    "face_attribute_cls": [
        f"{root}/train_dataset/face_attribute_cls/face_attribute_before_202212_day_test/20230321-172321/data.anno.pb_rec",
        f"{root}/train_dataset/face_attribute_cls/face_attribute_before_202212_night_test/20230321-172904/data.anno.pb_rec",
        f"{root}/train_dataset/face_attribute_cls/face_attribute_before_202212_other_test/20230321-174314/data.anno.pb_rec",
    ],
    "person_head_attribute_cls": [
        f"{root}/train_dataset/person_head_attribute_cls/person_head_attribute_before_202212_other_test/20230321-193911/data.anno.pb_rec"
    ],
    "wheel_attribute_cls": [
        f"{root}/train_dataset/wheel_attribute_cls/wheel_attribute_before_202212_noSP_other_test/20230321-133358/data.anno.pb_rec"
    ],
    "cyclist_wheel_attribute_cls": [
        f"{root}/train_dataset/cyclist_wheel_attribute_cls/cyclist_wheel_attribute_before_202212_day_test/20230322-153913/data.anno.pb_rec",
        f"{root}/train_dataset/cyclist_wheel_attribute_cls/cyclist_wheel_attribute_before_202212_night_test/20230322-155252/data.anno.pb_rec",
    ],
    "vehicle_light_detection_attribute_cls": [
        f"{root}/train_dataset/vehicle_light_detection_attribute_cls/vehicle_light_detection_attribute_before_202212_other_test/20230322-165354/data.anno.pb_rec"
    ],
    "vehicle_rear_recognition": [
        f"{root}/train_dataset/vehicle_rear_recognition/vehicle_rear_recognition_before_202212_day_test/20230323-142820/data.anno.pb_rec",
        f"{root}/train_dataset/vehicle_rear_recognition/vehicle_rear_recognition_before_202212_other_test/20230323-144519/data.anno.pb_rec",
    ],
    "vehicle_light_recognition": [
        f"{root}/train_dataset/vehicle_light_recognition/vehicle_light_recognition_before_202212_day_test/20230323-175351/data.anno.pb_rec",
        f"{root}/train_dataset/vehicle_light_recognition/vehicle_light_recognition_before_202212_night_test/20230323-182144/data.anno.pb_rec",
        f"{root}/train_dataset/vehicle_light_recognition/vehicle_light_recognition_before_202212_other_test/20230323-181856/data.anno.pb_rec",
    ],
    "traffic_sign_recognition": [
        f"{root}/train_dataset/traffic_sign_recognition/traffic_sign_recognition_before_202212_day_test/20230323-213617/data.anno.pb_rec",
        f"{root}/train_dataset/traffic_sign_recognition/traffic_sign_recognition_before_202212_night_test/20230323-214551/data.anno.pb_rec",
        f"{root}/train_dataset/traffic_sign_recognition/traffic_sign_recognition_before_202212_other_test/20230323-220346/data.anno.pb_rec",
    ],
    "vehicle_recognition": [
        f"{root}/train_dataset/vehicle_recognition/vehicle_recognition_before_202212_day_test/20230323-230547/data.anno.pb_rec",
        f"{root}/train_dataset/vehicle_recognition/vehicle_recognition_before_202212_night_test/20230323-225953/data.anno.pb_rec",
        f"{root}/train_dataset/vehicle_recognition/vehicle_recognition_before_202212_other_test/20230323-230612/data.anno.pb_rec",
    ],
    "person_negative_recognition": [
        f"{root}/train_dataset/person_negative_recognition/person_negative_recognition_before_202212_day_test/20230324-113101/data.anno.pb_rec",
        f"{root}/train_dataset/person_negative_recognition/person_negative_recognition_before_202212_other_test/20230324-120911/data.anno.pb_rec",
    ],
    "vehicle_negative_recognition": [
        f"{root}/train_dataset/vehicle_negative_recognition/vehicle_negative_recognition_before_202212_day_test/20230324-135100/data.anno.pb_rec",
        f"{root}/train_dataset/vehicle_negative_recognition/vehicle_negative_recognition_before_202212_night_test/20230324-145017/data.anno.pb_rec",
        f"{root}/train_dataset/vehicle_negative_recognition/vehicle_negative_recognition_before_202212_other_test/20230324-135901/data.anno.pb_rec",
    ],
    "person_orientation_recognition": [
        f"{root}/train_dataset/person_orientation_recognition/person_orientation_recognition_before_202212_other_test/20230324-161244/data.anno.pb_rec"
    ],
}


datapaths = dict(
    vehicle_attribute=dict(
        train_batch_size_per_ctx=128,
        train_data_paths=[
            dict(
                rec_path=data_rec_path,
                anno_path=anno_rec_path,
                sample_weight=1,
            )
            for data_rec_path, anno_rec_path in zip(
                train_data_rec["vehicle_attribute_cls"],
                train_anno_rec["vehicle_attribute_cls"],
            )
        ],
        val_batch_size_per_ctx=256,  # 128,
        val_data_paths=[
            dict(
                rec_path=data_rec_path,
                anno_path=anno_rec_path,
                sample_weight=1,
            )
            for data_rec_path, anno_rec_path in zip(
                val_data_rec["vehicle_attribute_cls"],
                val_anno_rec["vehicle_attribute_cls"],
            )
        ],
    ),
    vehicle_rear_attribute=dict(
        train_batch_size_per_ctx=128,
        train_data_paths=[
            dict(
                rec_path=data_rec_path,
                anno_path=anno_rec_path,
                sample_weight=1,
            )
            for data_rec_path, anno_rec_path in zip(
                train_data_rec["vehicle_rear_attribute_cls"],
                train_anno_rec["vehicle_rear_attribute_cls"],
            )
        ],
        val_batch_size_per_ctx=256,  # 128,
        val_data_paths=[
            dict(
                rec_path=data_rec_path,
                anno_path=anno_rec_path,
                sample_weight=1,
            )
            for data_rec_path, anno_rec_path in zip(
                val_data_rec["vehicle_rear_attribute_cls"],
                val_anno_rec["vehicle_rear_attribute_cls"],
            )
        ],
    ),
    person_attribute=dict(
        train_batch_size_per_ctx=128,
        train_data_paths=[
            dict(
                rec_path=data_rec_path,
                anno_path=anno_rec_path,
                sample_weight=1,
            )
            for data_rec_path, anno_rec_path in zip(
                train_data_rec["person_attribute_cls"],
                train_anno_rec["person_attribute_cls"],
            )
        ],
        val_batch_size_per_ctx=128,
        val_data_paths=[
            dict(
                rec_path=data_rec_path,
                anno_path=anno_rec_path,
                sample_weight=1,
            )
            for data_rec_path, anno_rec_path in zip(
                val_data_rec["person_attribute_cls"],
                val_anno_rec["person_attribute_cls"],
            )
        ],
    ),
    traffic_light_primary_attribute=dict(
        train_batch_size_per_ctx=128,
        train_data_paths=[
            dict(
                rec_path=data_rec_path,
                anno_path=anno_rec_path,
                sample_weight=1,
            )
            for data_rec_path, anno_rec_path in zip(
                train_data_rec["traffic_light_primary_attribute_cls"],
                train_anno_rec["traffic_light_primary_attribute_cls"],
            )
        ],
        val_batch_size_per_ctx=128,
        val_data_paths=[
            dict(
                rec_path=data_rec_path,
                anno_path=anno_rec_path,
                sample_weight=1,
            )
            for data_rec_path, anno_rec_path in zip(
                val_data_rec["traffic_light_primary_attribute_cls"],
                val_anno_rec["traffic_light_primary_attribute_cls"],
            )
        ],
    ),
    traffic_light_secondary_attribute=dict(
        train_batch_size_per_ctx=128,
        train_data_paths=[
            dict(
                rec_path=data_rec_path,
                anno_path=anno_rec_path,
                sample_weight=1,
            )
            for data_rec_path, anno_rec_path in zip(
                train_data_rec["traffic_light_secondary_attribute_cls"],
                train_anno_rec["traffic_light_secondary_attribute_cls"],
            )
        ],
        val_batch_size_per_ctx=128,
        val_data_paths=[
            dict(
                rec_path=data_rec_path,
                anno_path=anno_rec_path,
                sample_weight=1,
            )
            for data_rec_path, anno_rec_path in zip(
                val_data_rec["traffic_light_secondary_attribute_cls"],
                val_anno_rec["traffic_light_secondary_attribute_cls"],
            )
        ],
    ),
    traffic_sign_medium_attribute=dict(
        train_batch_size_per_ctx=128,
        train_data_paths=[
            dict(
                rec_path=data_rec_path,
                anno_path=anno_rec_path,
                sample_weight=1,
            )
            for data_rec_path, anno_rec_path in zip(
                train_data_rec["traffic_sign_medium_attribute_cls"],
                train_anno_rec["traffic_sign_medium_attribute_cls"],
            )
        ],
        val_batch_size_per_ctx=128,
        val_data_paths=[
            dict(
                rec_path=data_rec_path,
                anno_path=anno_rec_path,
                sample_weight=1,
            )
            for data_rec_path, anno_rec_path in zip(
                val_data_rec["traffic_sign_medium_attribute_cls"],
                val_anno_rec["traffic_sign_medium_attribute_cls"],
            )
        ],
    ),
    traffic_sign_sub_attribute=dict(
        train_batch_size_per_ctx=64,  # 128,
        train_data_paths=[
            dict(
                rec_path=data_rec_path,
                anno_path=anno_rec_path,
                sample_weight=1,
            )
            for data_rec_path, anno_rec_path in zip(
                train_data_rec["traffic_sign_sub_attribute_cls"],
                train_anno_rec["traffic_sign_sub_attribute_cls"],
            )
        ],
        val_batch_size_per_ctx=128,
        val_data_paths=[
            dict(
                rec_path=data_rec_path,
                anno_path=anno_rec_path,
                sample_weight=1,
            )
            for data_rec_path, anno_rec_path in zip(
                val_data_rec["traffic_sign_sub_attribute_cls"],
                val_anno_rec["traffic_sign_sub_attribute_cls"],
            )
        ],
    ),
    traffic_sign_US_medium_attribute=dict(
        train_batch_size_per_ctx=128,
        train_data_paths=[
            dict(
                rec_path=data_rec_path,
                anno_path=anno_rec_path,
                sample_weight=1,
            )
            for data_rec_path, anno_rec_path in zip(
                train_data_rec["traffic_sign_US_medium_attribute_cls"],
                train_anno_rec["traffic_sign_US_medium_attribute_cls"],
            )
        ],
        val_batch_size_per_ctx=128,
        val_data_paths=[
            dict(
                rec_path=data_rec_path,
                anno_path=anno_rec_path,
                sample_weight=1,
            )
            for data_rec_path, anno_rec_path in zip(
                val_data_rec["traffic_sign_US_medium_attribute_cls"],
                val_anno_rec["traffic_sign_US_medium_attribute_cls"],
            )
        ],
    ),
    light_attribute=dict(
        train_batch_size_per_ctx=128,
        train_data_paths=[
            dict(
                rec_path=data_rec_path,
                anno_path=anno_rec_path,
                sample_weight=1,
            )
            for data_rec_path, anno_rec_path in zip(
                train_data_rec["light_attribute_cls"],
                train_anno_rec["light_attribute_cls"],
            )
        ],
        val_batch_size_per_ctx=128,
        val_data_paths=[
            dict(
                rec_path=data_rec_path,
                anno_path=anno_rec_path,
                sample_weight=1,
            )
            for data_rec_path, anno_rec_path in zip(
                val_data_rec["light_attribute_cls"],
                val_anno_rec["light_attribute_cls"],
            )
        ],
    ),
    road_arrow_attribute=dict(
        train_batch_size_per_ctx=128,
        train_data_paths=[
            dict(
                rec_path=data_rec_path,
                anno_path=anno_rec_path,
                sample_weight=1,
            )
            for data_rec_path, anno_rec_path in zip(
                train_data_rec["road_arrow_attribute_cls"],
                train_anno_rec["road_arrow_attribute_cls"],
            )
        ],
        val_batch_size_per_ctx=128,
        val_data_paths=[
            dict(
                rec_path=data_rec_path,
                anno_path=anno_rec_path,
                sample_weight=1,
            )
            for data_rec_path, anno_rec_path in zip(
                val_data_rec["road_arrow_attribute_cls"],
                val_anno_rec["road_arrow_attribute_cls"],
            )
        ],
    ),
    traffic_cone_attribute=dict(
        train_batch_size_per_ctx=128,
        train_data_paths=[
            dict(
                rec_path=data_rec_path,
                anno_path=anno_rec_path,
                sample_weight=1,
            )
            for data_rec_path, anno_rec_path in zip(
                train_data_rec["traffic_cone_attribute_cls"],
                train_anno_rec["traffic_cone_attribute_cls"],
            )
        ],
        val_batch_size_per_ctx=128,
        val_data_paths=[
            dict(
                rec_path=data_rec_path,
                anno_path=anno_rec_path,
                sample_weight=1,
            )
            for data_rec_path, anno_rec_path in zip(
                val_data_rec["traffic_cone_attribute_cls"],
                val_anno_rec["traffic_cone_attribute_cls"],
            )
        ],
    ),
    vertical_pole_attribute=dict(
        train_batch_size_per_ctx=128,
        train_data_paths=[
            dict(
                rec_path=data_rec_path,
                anno_path=anno_rec_path,
                sample_weight=1,
            )
            for data_rec_path, anno_rec_path in zip(
                train_data_rec["vertical_pole_attribute_cls"],
                train_anno_rec["vertical_pole_attribute_cls"],
            )
        ],
        val_batch_size_per_ctx=128,
        val_data_paths=[
            dict(
                rec_path=data_rec_path,
                anno_path=anno_rec_path,
                sample_weight=1,
            )
            for data_rec_path, anno_rec_path in zip(
                val_data_rec["vertical_pole_attribute_cls"],
                val_anno_rec["vertical_pole_attribute_cls"],
            )
        ],
    ),
    face_attribute=dict(
        train_batch_size_per_ctx=128,
        train_data_paths=[
            dict(
                rec_path=data_rec_path,
                anno_path=anno_rec_path,
                sample_weight=1,
            )
            for data_rec_path, anno_rec_path in zip(
                train_data_rec["face_attribute_cls"],
                train_anno_rec["face_attribute_cls"],
            )
        ],
        val_batch_size_per_ctx=128,
        val_data_paths=[
            dict(
                rec_path=data_rec_path,
                anno_path=anno_rec_path,
                sample_weight=1,
            )
            for data_rec_path, anno_rec_path in zip(
                val_data_rec["face_attribute_cls"],
                val_anno_rec["face_attribute_cls"],
            )
        ],
    ),
    person_head_attribute=dict(
        train_batch_size_per_ctx=128,
        train_data_paths=[
            dict(
                rec_path=data_rec_path,
                anno_path=anno_rec_path,
                sample_weight=1,
            )
            for data_rec_path, anno_rec_path in zip(
                train_data_rec["person_head_attribute_cls"],
                train_anno_rec["person_head_attribute_cls"],
            )
        ],
        val_batch_size_per_ctx=128,
        val_data_paths=[
            dict(
                rec_path=data_rec_path,
                anno_path=anno_rec_path,
                sample_weight=1,
            )
            for data_rec_path, anno_rec_path in zip(
                val_data_rec["person_head_attribute_cls"],
                val_anno_rec["person_head_attribute_cls"],
            )
        ],
    ),
    wheel_attribute=dict(
        train_batch_size_per_ctx=128,
        train_data_paths=[
            dict(
                rec_path=data_rec_path,
                anno_path=anno_rec_path,
                sample_weight=1,
            )
            for data_rec_path, anno_rec_path in zip(
                train_data_rec["wheel_attribute_cls"],
                train_anno_rec["wheel_attribute_cls"],
            )
        ],
        val_batch_size_per_ctx=128,
        val_data_paths=[
            dict(
                rec_path=data_rec_path,
                anno_path=anno_rec_path,
                sample_weight=1,
            )
            for data_rec_path, anno_rec_path in zip(
                val_data_rec["wheel_attribute_cls"],
                val_anno_rec["wheel_attribute_cls"],
            )
        ],
    ),
    cyclist_wheel_attribute=dict(
        train_batch_size_per_ctx=128,
        train_data_paths=[
            dict(
                rec_path=data_rec_path,
                anno_path=anno_rec_path,
                sample_weight=1,
            )
            for data_rec_path, anno_rec_path in zip(
                train_data_rec["cyclist_wheel_attribute_cls"],
                train_anno_rec["cyclist_wheel_attribute_cls"],
            )
        ],
        val_batch_size_per_ctx=128,
        val_data_paths=[
            dict(
                rec_path=data_rec_path,
                anno_path=anno_rec_path,
                sample_weight=1,
            )
            for data_rec_path, anno_rec_path in zip(
                val_data_rec["cyclist_wheel_attribute_cls"],
                val_anno_rec["cyclist_wheel_attribute_cls"],
            )
        ],
    ),
    vehicle_light_detection_attribute=dict(
        train_batch_size_per_ctx=128,
        train_data_paths=[
            dict(
                rec_path=data_rec_path,
                anno_path=anno_rec_path,
                sample_weight=1,
            )
            for data_rec_path, anno_rec_path in zip(
                train_data_rec["vehicle_light_detection_attribute_cls"],
                train_anno_rec["vehicle_light_detection_attribute_cls"],
            )
        ],
        val_batch_size_per_ctx=128,
        val_data_paths=[
            dict(
                rec_path=data_rec_path,
                anno_path=anno_rec_path,
                sample_weight=1,
            )
            for data_rec_path, anno_rec_path in zip(
                val_data_rec["vehicle_light_detection_attribute_cls"],
                val_anno_rec["vehicle_light_detection_attribute_cls"],
            )
        ],
    ),
    vehicle_rear_recognition=dict(
        train_batch_size_per_ctx=128,
        train_data_paths=[
            dict(
                rec_path=data_rec_path,
                anno_path=anno_rec_path,
                sample_weight=1,
            )
            for data_rec_path, anno_rec_path in zip(
                train_data_rec["vehicle_rear_recognition"],
                train_anno_rec["vehicle_rear_recognition"],
            )
        ],
        val_batch_size_per_ctx=128,
        val_data_paths=[
            dict(
                rec_path=data_rec_path,
                anno_path=anno_rec_path,
                sample_weight=1,
            )
            for data_rec_path, anno_rec_path in zip(
                val_data_rec["vehicle_rear_recognition"],
                val_anno_rec["vehicle_rear_recognition"],
            )
        ],
    ),
    vehicle_light_recognition=dict(
        train_batch_size_per_ctx=128,
        train_data_paths=[
            dict(
                rec_path=data_rec_path,
                anno_path=anno_rec_path,
                sample_weight=1,
            )
            for data_rec_path, anno_rec_path in zip(
                train_data_rec["vehicle_light_recognition"],
                train_anno_rec["vehicle_light_recognition"],
            )
        ],
        val_batch_size_per_ctx=128,
        val_data_paths=[
            dict(
                rec_path=data_rec_path,
                anno_path=anno_rec_path,
                sample_weight=1,
            )
            for data_rec_path, anno_rec_path in zip(
                val_data_rec["vehicle_light_recognition"],
                val_anno_rec["vehicle_light_recognition"],
            )
        ],
    ),
    traffic_sign_recognition=dict(
        train_batch_size_per_ctx=128,
        train_data_paths=[
            dict(
                rec_path=data_rec_path,
                anno_path=anno_rec_path,
                sample_weight=1,
            )
            for data_rec_path, anno_rec_path in zip(
                train_data_rec["traffic_sign_recognition"],
                train_anno_rec["traffic_sign_recognition"],
            )
        ],
        val_batch_size_per_ctx=128,
        val_data_paths=[
            dict(
                rec_path=data_rec_path,
                anno_path=anno_rec_path,
                sample_weight=1,
            )
            for data_rec_path, anno_rec_path in zip(
                val_data_rec["traffic_sign_recognition"],
                val_anno_rec["traffic_sign_recognition"],
            )
        ],
    ),
    vehicle_recognition=dict(
        train_batch_size_per_ctx=128,
        train_data_paths=[
            dict(
                rec_path=data_rec_path,
                anno_path=anno_rec_path,
                sample_weight=1,
            )
            for data_rec_path, anno_rec_path in zip(
                train_data_rec["vehicle_recognition"],
                train_anno_rec["vehicle_recognition"],
            )
        ],
        val_batch_size_per_ctx=128,
        val_data_paths=[
            dict(
                rec_path=data_rec_path,
                anno_path=anno_rec_path,
                sample_weight=1,
            )
            for data_rec_path, anno_rec_path in zip(
                val_data_rec["vehicle_recognition"],
                val_anno_rec["vehicle_recognition"],
            )
        ],
    ),
    person_negative_recognition=dict(
        train_batch_size_per_ctx=128,
        train_data_paths=[
            dict(
                rec_path=data_rec_path,
                anno_path=anno_rec_path,
                sample_weight=1,
            )
            for data_rec_path, anno_rec_path in zip(
                train_data_rec["person_negative_recognition"],
                train_anno_rec["person_negative_recognition"],
            )
        ],
        val_batch_size_per_ctx=128,
        val_data_paths=[
            dict(
                rec_path=data_rec_path,
                anno_path=anno_rec_path,
                sample_weight=1,
            )
            for data_rec_path, anno_rec_path in zip(
                val_data_rec["person_negative_recognition"],
                val_anno_rec["person_negative_recognition"],
            )
        ],
    ),
    vehicle_negative_recognition=dict(
        train_batch_size_per_ctx=128,
        train_data_paths=[
            dict(
                rec_path=data_rec_path,
                anno_path=anno_rec_path,
                sample_weight=1,
            )
            for data_rec_path, anno_rec_path in zip(
                train_data_rec["vehicle_negative_recognition"],
                train_anno_rec["vehicle_negative_recognition"],
            )
        ],
        val_batch_size_per_ctx=128,
        val_data_paths=[
            dict(
                rec_path=data_rec_path,
                anno_path=anno_rec_path,
                sample_weight=1,
            )
            for data_rec_path, anno_rec_path in zip(
                val_data_rec["vehicle_negative_recognition"],
                val_anno_rec["vehicle_negative_recognition"],
            )
        ],
    ),
    person_orientation_recognition=dict(
        train_batch_size_per_ctx=128,
        train_data_paths=[
            dict(
                rec_path=data_rec_path,
                anno_path=anno_rec_path,
                sample_weight=1,
            )
            for data_rec_path, anno_rec_path in zip(
                train_data_rec["person_orientation_recognition"],
                train_anno_rec["person_orientation_recognition"],
            )
        ],
        val_batch_size_per_ctx=128,
        val_data_paths=[
            dict(
                rec_path=data_rec_path,
                anno_path=anno_rec_path,
                sample_weight=1,
            )
            for data_rec_path, anno_rec_path in zip(
                val_data_rec["person_orientation_recognition"],
                val_anno_rec["person_orientation_recognition"],
            )
        ],
    ),
)

datapaths = EasyDict(datapaths)
