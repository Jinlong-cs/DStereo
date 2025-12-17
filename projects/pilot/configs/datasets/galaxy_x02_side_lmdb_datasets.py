from easydict import EasyDict

root = "dmpv2://matrix2"
datapaths = dict()
datapaths.update(
    {
        "vehicle_3d_detection": {
            "train_data_paths": [
                # x02 常规数据
                {
                    "data_path": [
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/LX513/4v_vehicle_v2/with_calib_all/data_LX513_20220723_v05__no_front__day_filtered.anno",
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/LX513/4v_vehicle_v2/with_calib_all/data_LX513_20220724_v05__no_front__day_filtered.anno",
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/LX513/4v_vehicle_v2/with_calib_all/data_LX513_20220725_v05__no_front__day_filtered.anno",
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/LX513/4v_vehicle_v2/with_calib_all/data_LX513_20220726_v05__no_front__day_filtered.anno",
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/LX513/4v_vehicle_v2/with_calib_all/data_LX513_20220816_v05__no_front__day_filtered.anno",
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/LX513/4v_vehicle_v2/with_calib_all/data_LX513_20220817_v05__no_front__day_filtered.anno",
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/LX513/4v_vehicle_v2/with_calib_all/data_LX513_20220818_v05__no_front__day_filtered.anno",
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/LX513/4v_vehicle_v2/with_calib_all/data_LX513_20220819_v05__no_front__day_filtered.anno",
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/LX513/4v_vehicle_v2/with_calib_all/data_LX513_20220820_v05__no_front__day_filtered.anno",
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/LX513/4v_vehicle_v2/with_calib_all/data_LX513_20220723_v05__no_front__night_filtered.anno",
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/LX513/4v_vehicle_v2/with_calib_all/data_LX513_20220724_v05__no_front__night_filtered.anno",
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/LX513/4v_vehicle_v2/with_calib_all/data_LX513_20220725_v05__no_front__night_filtered.anno",
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/LX513/4v_vehicle_v2/with_calib_all/data_LX513_20220726_v05__no_front__night_filtered.anno",
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/LX513/4v_vehicle_v2/with_calib_all/data_LX513_20220814_v05__no_front__night_filtered.anno",
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/LX513/4v_vehicle_v2/with_calib_all/data_LX513_20220816_v05__no_front__night_filtered.anno",
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/LX513/4v_vehicle_v2/with_calib_all/data_LX513_20220817_v05__no_front__night_filtered.anno",
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/LX513/4v_vehicle_v2/with_calib_all/data_LX513_20220818_v05__no_front__night_filtered.anno",
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/LX513/4v_vehicle_v2/with_calib_all/data_LX513_20220819_v05__no_front__night_filtered.anno",
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/LX513/4v_vehicle_v2/with_calib_all/data_LX513_20220820_v05__no_front__night_filtered.anno",
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/LX513/4v_vehicle_v2/with_calib_all/data_LX513_20220822_v05__no_front__day_filtered.anno",
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/LX513/4v_vehicle_v2/with_calib_all/data_LX513_20220823_v05__no_front__day_filtered.anno",
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/LX513/4v_vehicle_v2/with_calib_all/data_LX513_20220824_v05__no_front__day_filtered.anno",
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/LX513/4v_vehicle_v2/with_calib_all/data_LX513_20220825_v05__no_front__day_filtered.anno",
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/LX513/4v_vehicle_v2/with_calib_all/data_LX513_20220826_v05__no_front__day_filtered.anno",
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/LX513/4v_vehicle_v2/with_calib_all/data_LX513_20220901_v05__no_front__day_filtered.anno",
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/LX513/4v_vehicle_v2/with_calib_all/data_LX513_20220902_v05__no_front__day_filtered.anno",
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/LX513/4v_vehicle_v2/with_calib_all/data_LX513_20220903_v05__no_front__day_filtered.anno",
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/LX513/4v_vehicle_v2/with_calib_all/data_LX513_20220904_v05__no_front__day_filtered.anno",
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/LX513/4v_vehicle_v2/with_calib_all/data_LX513_20220905_v05__no_front__day_filtered.anno",
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/LX513/4v_vehicle_v2/with_calib_all/data_LX513_20220906_v05__no_front__day_filtered.anno",
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/LX513/4v_vehicle_v2/with_calib_all/data_LX513_20220907_v05__no_front__day_filtered.anno",
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/LX513/4v_vehicle_v2/with_calib_all/data_LX513_20220908_v05__no_front__day_filtered.anno",
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/LX513/4v_vehicle_v2/with_calib_all/data_LX513_20220909_v05__no_front__day_filtered.anno",
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/LX513/4v_vehicle_v2/with_calib_all/data_LX513_20220821_v05__no_front__night_filtered.anno",
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/LX513/4v_vehicle_v2/with_calib_all/data_LX513_20220822_v05__no_front__night_filtered.anno",
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/LX513/4v_vehicle_v2/with_calib_all/data_LX513_20220823_v05__no_front__night_filtered.anno",
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/LX513/4v_vehicle_v2/with_calib_all/data_LX513_20220824_v05__no_front__night_filtered.anno",
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/LX513/4v_vehicle_v2/with_calib_all/data_LX513_20220825_v05__no_front__night_filtered.anno",
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/LX513/4v_vehicle_v2/with_calib_all/data_LX513_20220826_v05__no_front__night_filtered.anno",
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/LX513/4v_vehicle_v2/with_calib_all/data_LX513_20220901_v05__no_front__night_filtered.anno",
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/LX513/4v_vehicle_v2/with_calib_all/data_LX513_20220902_v05__no_front__night_filtered.anno",
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/LX513/4v_vehicle_v2/with_calib_all/data_LX513_20220905_v05__no_front__night_filtered.anno",
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/LX513/4v_vehicle_v2/with_calib_all/data_LX513_20220906_v05__no_front__night_filtered.anno",
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/LX513/4v_vehicle_v2/with_calib_all/data_LX513_20220907_v05__no_front__night_filtered.anno",
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/LX513/4v_vehicle_v2/with_calib_all/data_LX513_20220908_v05__no_front__night_filtered.anno",
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/LX553/4v_vehicle_v2/with_calib_all/data_LX553_20220811_v05__no_front__day_filtered.anno",
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/LX553/4v_vehicle_v2/with_calib_all/data_LX553_20220812_v05__no_front__day_filtered.anno",
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/LX553/4v_vehicle_v2/with_calib_all/data_LX553_20220813_v05__no_front__day_filtered.anno",
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/LX553/4v_vehicle_v2/with_calib_all/data_LX553_20220815_v05__no_front__day_filtered.anno",
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/LX553/4v_vehicle_v2/with_calib_all/data_LX553_20220816_v05__no_front__day_filtered.anno",
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/LX553/4v_vehicle_v2/with_calib_all/data_LX553_20220817_v05__no_front__day_filtered.anno",
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/LX553/4v_vehicle_v2/with_calib_all/data_LX553_20220818_v05__no_front__day_filtered.anno",
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/LX553/4v_vehicle_v2/with_calib_all/data_LX553_20220819_v05__no_front__day_filtered.anno",
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/LX553/4v_vehicle_v2/with_calib_all/data_LX553_20220820_v05__no_front__day_filtered.anno",
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/LX553/4v_vehicle_v2/with_calib_all/data_LX553_20220821_v05__no_front__day_filtered.anno",
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/LX553/4v_vehicle_v2/with_calib_all/data_LX553_20220822_v05__no_front__day_filtered.anno",
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/LX553/4v_vehicle_v2/with_calib_all/data_LX553_20220824_v05__no_front__day_filtered.anno",
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/LX553/4v_vehicle_v2/with_calib_all/data_LX553_20220811_v05__no_front__night_filtered.anno",
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/LX553/4v_vehicle_v2/with_calib_all/data_LX553_20220812_v05__no_front__night_filtered.anno",
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/LX553/4v_vehicle_v2/with_calib_all/data_LX553_20220813_v05__no_front__night_filtered.anno",
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/LX553/4v_vehicle_v2/with_calib_all/data_LX553_20220814_v05__no_front__night_filtered.anno",
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/LX553/4v_vehicle_v2/with_calib_all/data_LX553_20220815_v05__no_front__night_filtered.anno",
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/LX553/4v_vehicle_v2/with_calib_all/data_LX553_20220816_v05__no_front__night_filtered.anno",
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/LX553/4v_vehicle_v2/with_calib_all/data_LX553_20220817_v05__no_front__night_filtered.anno",
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/LX553/4v_vehicle_v2/with_calib_all/data_LX553_20220818_v05__no_front__night_filtered.anno",
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/LX553/4v_vehicle_v2/with_calib_all/data_LX553_20220819_v05__no_front__night_filtered.anno",
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/LX553/4v_vehicle_v2/with_calib_all/data_LX553_20220820_v05__no_front__night_filtered.anno",
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/LX553/4v_vehicle_v2/with_calib_all/data_LX553_20220821_v05__no_front__night_filtered.anno",
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/LX553/4v_vehicle_v2/with_calib_all/data_LX553_20220822_v05__no_front__night_filtered.anno",
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/LX553/4v_vehicle_v2/with_calib_all/data_LX553_20220824_v05__no_front__night_filtered.anno",
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/LX553/4v_vehicle_v2/with_calib_all/data_LX553_20220802_v05__no_front__day_filtered.anno",
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/LX553/4v_vehicle_v2/with_calib_all/data_LX553_20220803_v05__no_front__day_filtered.anno",
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/LX553/4v_vehicle_v2/with_calib_all/data_LX553_20220804_v05__no_front__day_filtered.anno",
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/LX553/4v_vehicle_v2/with_calib_all/data_LX553_20220805_v05__no_front__day_filtered.anno",
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/LX553/4v_vehicle_v2/with_calib_all/data_LX553_20220806_v05__no_front__day_filtered.anno",
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/LX553/4v_vehicle_v2/with_calib_all/data_LX553_20220807_v05__no_front__day_filtered.anno",
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/LX553/4v_vehicle_v2/with_calib_all/data_LX553_20220808_v05__no_front__day_filtered.anno",
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/LX553/4v_vehicle_v2/with_calib_all/data_LX553_20220825_v05__no_front__day_filtered.anno",
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/LX553/4v_vehicle_v2/with_calib_all/data_LX553_20220826_v05__no_front__day_filtered.anno",
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/LX553/4v_vehicle_v2/with_calib_all/data_LX553_20220827_v05__no_front__day_filtered.anno",
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/LX553/4v_vehicle_v2/with_calib_all/data_LX553_20220829_v05__no_front__day_filtered.anno",
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/LX553/4v_vehicle_v2/with_calib_all/data_LX553_20220830_v05__no_front__day_filtered.anno",
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/LX553/4v_vehicle_v2/with_calib_all/data_LX553_20220831_v05__no_front__day_filtered.anno",
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/LX553/4v_vehicle_v2/with_calib_all/data_LX553_20220901_v05__no_front__day_filtered.anno",
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/LX553/4v_vehicle_v2/with_calib_all/data_LX553_20220902_v05__no_front__day_filtered.anno",
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/LX553/4v_vehicle_v2/with_calib_all/data_LX553_20220903_v05__no_front__day_filtered.anno",
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/LX553/4v_vehicle_v2/with_calib_all/data_LX553_20220802_v05__no_front__night_filtered.anno",
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/LX553/4v_vehicle_v2/with_calib_all/data_LX553_20220803_v05__no_front__night_filtered.anno",
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/LX553/4v_vehicle_v2/with_calib_all/data_LX553_20220804_v05__no_front__night_filtered.anno",
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/LX553/4v_vehicle_v2/with_calib_all/data_LX553_20220805_v05__no_front__night_filtered.anno",
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/LX553/4v_vehicle_v2/with_calib_all/data_LX553_20220806_v05__no_front__night_filtered.anno",
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/LX553/4v_vehicle_v2/with_calib_all/data_LX553_20220807_v05__no_front__night_filtered.anno",
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/LX553/4v_vehicle_v2/with_calib_all/data_LX553_20220808_v05__no_front__night_filtered.anno",
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/LX553/4v_vehicle_v2/with_calib_all/data_LX553_20220825_v05__no_front__night_filtered.anno",
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/LX553/4v_vehicle_v2/with_calib_all/data_LX553_20220826_v05__no_front__night_filtered.anno",
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/LX553/4v_vehicle_v2/with_calib_all/data_LX553_20220827_v05__no_front__night_filtered.anno",
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/LX553/4v_vehicle_v2/with_calib_all/data_LX553_20220828_v05__no_front__night_filtered.anno",
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/LX553/4v_vehicle_v2/with_calib_all/data_LX553_20220829_v05__no_front__night_filtered.anno",
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/LX553/4v_vehicle_v2/with_calib_all/data_LX553_20220830_v05__no_front__night_filtered.anno",
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/LX553/4v_vehicle_v2/with_calib_all/data_LX553_20220831_v05__no_front__night_filtered.anno",
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/LX553/4v_vehicle_v2/with_calib_all/data_LX553_20220901_v05__no_front__night_filtered.anno",
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/LX553/4v_vehicle_v2/with_calib_all/data_LX553_20220902_v05__no_front__night_filtered.anno",
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/LX553/4v_vehicle_v2/with_calib_all/data_LX553_20220903_v05__no_front__night_filtered.anno",
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/LX513/4v_vehicle_v2/with_calib_all/data_LX513_20220910_v05__no_front__day_filtered.anno",
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/LX513/4v_vehicle_v2/with_calib_all/data_LX513_20220914_v05__no_front__day_filtered.anno",
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/LX513/4v_vehicle_v2/with_calib_all/data_LX513_20220915_v05__no_front__day_filtered.anno",
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/LX513/4v_vehicle_v2/with_calib_all/data_LX513_20220917_v05__no_front__day_filtered.anno",
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/LX513/4v_vehicle_v2/with_calib_all/data_LX513_20220919_v05__no_front__day_filtered.anno",
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/LX513/4v_vehicle_v2/with_calib_all/data_LX513_20220920_v05__no_front__day_filtered.anno",
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/LX513/4v_vehicle_v2/with_calib_all/data_LX513_20220921_v05__no_front__day_filtered.anno",
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/LX513/4v_vehicle_v2/with_calib_all/data_LX513_20220922_v05__no_front__day_filtered.anno",
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/LX513/4v_vehicle_v2/with_calib_all/data_LX513_20220925_v05__no_front__day_filtered.anno",
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/LX513/4v_vehicle_v2/with_calib_all/data_LX513_20220915_v05__no_front__night_filtered.anno",
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/LX513/4v_vehicle_v2/with_calib_all/data_LX513_20220921_v05__no_front__night_filtered.anno",
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/LX513/4v_vehicle_v2/with_calib_all/data_LX513_20220922_v05__no_front__night_filtered.anno",
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/LX513/4v_vehicle_v2/with_calib_all/data_LX513_20220925_v05__no_front__night_filtered.anno",
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/LX553/4v_vehicle_v2/with_calib_all/data_LX553_20220905_v05__no_front__day_filtered.anno",
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/LX553/4v_vehicle_v2/with_calib_all/data_LX553_20220906_v05__no_front__day_filtered.anno",
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/LX553/4v_vehicle_v2/with_calib_all/data_LX553_20220916_v05__no_front__day_filtered.anno",
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/LX553/4v_vehicle_v2/with_calib_all/data_LX553_20220917_v05__no_front__day_filtered.anno",
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/LX553/4v_vehicle_v2/with_calib_all/data_LX553_20220919_v05__no_front__day_filtered.anno",
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/LX553/4v_vehicle_v2/with_calib_all/data_LX553_20220920_v05__no_front__day_filtered.anno",
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/LX553/4v_vehicle_v2/with_calib_all/data_LX553_20220905_v05__no_front__night_filtered.anno",
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/LX553/4v_vehicle_v2/with_calib_all/data_LX553_20220906_v05__no_front__night_filtered.anno",
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/LX513/4v_vehicle_v2/with_calib_all/data_LX513_20221009_v05__no_front__day_filtered.anno",
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/LX513/4v_vehicle_v2/with_calib_all/data_LX513_20221011_v05__no_front__day_filtered.anno",
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/LX513/4v_vehicle_v2/with_calib_all/data_LX513_20221013_v05__no_front__day_filtered.anno",
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/LX513/4v_vehicle_v2/with_calib_all/data_LX513_20221015_v05__no_front__day_filtered.anno",
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/LX513/4v_vehicle_v2/with_calib_all/data_LX513_20221016_v05__no_front__day_filtered.anno",
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/LX513/4v_vehicle_v2/with_calib_all/data_LX513_20221009_v05__no_front__night_filtered.anno",
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/LX513/4v_vehicle_v2/with_calib_all/data_LX513_20221011_v05__no_front__night_filtered.anno",
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/LX513/4v_vehicle_v2/with_calib_all/data_LX513_20221013_v05__no_front__night_filtered.anno",
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/LX513/4v_vehicle_v2/with_calib_all/data_LX513_20221015_v05__no_front__night_filtered.anno",
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/LX513/4v_vehicle_v2/with_calib_all/data_LX513_20221016_v05__no_front__night_filtered.anno",
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/LX553/4v_vehicle_v2/with_calib_all/data_LX553_20221013_v05__no_front__day_filtered.anno",
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/LX553/4v_vehicle_v2/with_calib_all/data_LX553_20221014_v05__no_front__day_filtered.anno",
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/LX553/4v_vehicle_v2/with_calib_all/data_LX553_20221015_v05__no_front__day_filtered.anno",
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/LX553/4v_vehicle_v2/with_calib_all/data_LX553_20221016_v05__no_front__day_filtered.anno",
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/LX553/4v_vehicle_v2/with_calib_all/data_LX553_20221017_v05__no_front__day_filtered.anno",
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/LX553/4v_vehicle_v2/with_calib_all/data_LX553_20221018_v05__no_front__day_filtered.anno",
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/LX553/4v_vehicle_v2/with_calib_all/data_LX553_20221019_v05__no_front__day_filtered.anno",
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/LX553/4v_vehicle_v2/with_calib_all/data_LX553_20221020_v05__no_front__day_filtered.anno",
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/LX553/4v_vehicle_v2/with_calib_all/data_LX553_20221013_v05__no_front__night_filtered.anno",
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/LX553/4v_vehicle_v2/with_calib_all/data_LX553_20221014_v05__no_front__night_filtered.anno",
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/LX553/4v_vehicle_v2/with_calib_all/data_LX553_20221015_v05__no_front__night_filtered.anno",
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/LX553/4v_vehicle_v2/with_calib_all/data_LX553_20221016_v05__no_front__night_filtered.anno",
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/LX553/4v_vehicle_v2/with_calib_all/data_LX553_20221017_v05__no_front__night_filtered.anno",
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/LX553/4v_vehicle_v2/with_calib_all/data_LX553_20221018_v05__no_front__night_filtered.anno",
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/LX553/4v_vehicle_v2/with_calib_all/data_LX553_20221019_v05__no_front__night_filtered.anno",
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/LX553/4v_vehicle_v2/with_calib_all/data_LX553_20221020_v05__no_front__night_filtered.anno",
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/LX513/4v_vehicle_v2/with_calib_all/data_LX513_20221023_v05__no_front__day_filtered.anno",
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/LX513/4v_vehicle_v2/with_calib_all/data_LX513_20221024_v05__no_front__day_filtered.anno",
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/LX513/4v_vehicle_v2/with_calib_all/data_LX513_20221025_v05__no_front__day_filtered.anno",
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/LX513/4v_vehicle_v2/with_calib_all/data_LX513_20221027_v05__no_front__day_filtered.anno",
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/LX513/4v_vehicle_v2/with_calib_all/data_LX513_20221029_v05__no_front__day_filtered.anno",
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/LX513/4v_vehicle_v2/with_calib_all/data_LX513_20221031_v05__no_front__day_filtered.anno",
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/LX513/4v_vehicle_v2/with_calib_all/data_LX513_20221023_v05__no_front__night_filtered.anno",
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/LX513/4v_vehicle_v2/with_calib_all/data_LX513_20221024_v05__no_front__night_filtered.anno",
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/LX513/4v_vehicle_v2/with_calib_all/data_LX513_20221025_v05__no_front__night_filtered.anno",
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/LX513/4v_vehicle_v2/with_calib_all/data_LX513_20221029_v05__no_front__night_filtered.anno",
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/LX553/4v_vehicle_v2/with_calib_all/data_LX553_20221021_v05__no_front__day_filtered.anno",
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/LX553/4v_vehicle_v2/with_calib_all/data_LX553_20221022_v05__no_front__day_filtered.anno",
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/LX553/4v_vehicle_v2/with_calib_all/data_LX553_20221023_v05__no_front__day_filtered.anno",
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/LX553/4v_vehicle_v2/with_calib_all/data_LX553_20221024_v05__no_front__day_filtered.anno",
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/LX553/4v_vehicle_v2/with_calib_all/data_LX553_20221025_v05__no_front__day_filtered.anno",
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/LX553/4v_vehicle_v2/with_calib_all/data_LX553_20221027_v05__no_front__day_filtered.anno",
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/LX553/4v_vehicle_v2/with_calib_all/data_LX553_20221028_v05__no_front__day_filtered.anno",
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/LX553/4v_vehicle_v2/with_calib_all/data_LX553_20221029_v05__no_front__day_filtered.anno",
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/LX553/4v_vehicle_v2/with_calib_all/data_LX553_20221031_v05__no_front__day_filtered.anno",
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/LX553/4v_vehicle_v2/with_calib_all/data_LX553_20221102_v05__no_front__day_filtered.anno",
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/LX553/4v_vehicle_v2/with_calib_all/data_LX553_20221103_v05__no_front__day_filtered.anno",
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/LX553/4v_vehicle_v2/with_calib_all/data_LX553_20221104_v05__no_front__day_filtered.anno",
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/LX553/4v_vehicle_v2/with_calib_all/data_LX553_20221105_v05__no_front__day_filtered.anno",
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/LX553/4v_vehicle_v2/with_calib_all/data_LX553_20221107_v05__no_front__day_filtered.anno",
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/LX553/4v_vehicle_v2/with_calib_all/data_LX553_20221109_v05__no_front__day_filtered.anno",
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/LX553/4v_vehicle_v2/with_calib_all/data_LX553_20221110_v05__no_front__day_filtered.anno",
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/LX553/4v_vehicle_v2/with_calib_all/data_LX553_20221021_v05__no_front__night_filtered.anno",
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/LX553/4v_vehicle_v2/with_calib_all/data_LX553_20221022_v05__no_front__night_filtered.anno",
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/LX553/4v_vehicle_v2/with_calib_all/data_LX553_20221023_v05__no_front__night_filtered.anno",
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/LX553/4v_vehicle_v2/with_calib_all/data_LX553_20221024_v05__no_front__night_filtered.anno",
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/LX553/4v_vehicle_v2/with_calib_all/data_LX553_20221025_v05__no_front__night_filtered.anno",
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/LX553/4v_vehicle_v2/with_calib_all/data_LX553_20221027_v05__no_front__night_filtered.anno",
                        f"{root}/multicam_pilot/data/lmdb_datasets/data/real_3d/filtered/v05/LX553/4v_vehicle_v2/with_calib_all/data_LX553_20221029_v05__no_front__night_filtered.anno",
                    ],
                    "sample_weight": 58.417,
                    "partition": "normal",
                },
                # x02 截断数据
                {
                    "data_path": [
                        # LX513 4.0 day
                        f"{root}/multicam_pilot/data/lmdb_datasets/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX513/data_vehicle_LX513_20220723_v05__truncation__day_.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX513/data_vehicle_LX513_20220724_v05__truncation__day_.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX513/data_vehicle_LX513_20220725_v05__truncation__day_.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX513/data_vehicle_LX513_20220726_v05__truncation__day_.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX513/data_vehicle_LX513_20220816_v05__truncation__day_.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX513/data_vehicle_LX513_20220817_v05__truncation__day_.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX513/data_vehicle_LX513_20220818_v05__truncation__day_.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX513/data_vehicle_LX513_20220819_v05__truncation__day_.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX513/data_vehicle_LX513_20220820_v05__truncation__day_.anno",  # noqa
                        # LX513 4.0 night
                        f"{root}/multicam_pilot/data/lmdb_datasets/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX513/data_vehicle_LX513_20220723_v05__truncation__night_.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX513/data_vehicle_LX513_20220724_v05__truncation__night_.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX513/data_vehicle_LX513_20220725_v05__truncation__night_.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX513/data_vehicle_LX513_20220726_v05__truncation__night_.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX513/data_vehicle_LX513_20220814_v05__truncation__night_.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX513/data_vehicle_LX513_20220816_v05__truncation__night_.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX513/data_vehicle_LX513_20220817_v05__truncation__night_.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX513/data_vehicle_LX513_20220818_v05__truncation__night_.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX513/data_vehicle_LX513_20220819_v05__truncation__night_.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX513/data_vehicle_LX513_20220820_v05__truncation__night_.anno",  # noqa
                        # LX513 5.0 day
                        f"{root}/multicam_pilot/data/lmdb_datasets/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX513/data_vehicle_LX513_20220822_v05__truncation__day_.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX513/data_vehicle_LX513_20220823_v05__truncation__day_.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX513/data_vehicle_LX513_20220824_v05__truncation__day_.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX513/data_vehicle_LX513_20220825_v05__truncation__day_.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX513/data_vehicle_LX513_20220826_v05__truncation__day_.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX513/data_vehicle_LX513_20220901_v05__truncation__day_.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX513/data_vehicle_LX513_20220902_v05__truncation__day_.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX513/data_vehicle_LX513_20220903_v05__truncation__day_.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX513/data_vehicle_LX513_20220904_v05__truncation__day_.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX513/data_vehicle_LX513_20220905_v05__truncation__day_.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX513/data_vehicle_LX513_20220906_v05__truncation__day_.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX513/data_vehicle_LX513_20220907_v05__truncation__day_.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX513/data_vehicle_LX513_20220908_v05__truncation__day_.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX513/data_vehicle_LX513_20220909_v05__truncation__day_.anno",  # noqa
                        # LX513 5.0 night
                        f"{root}/multicam_pilot/data/lmdb_datasets/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX513/data_vehicle_LX513_20220821_v05__truncation__night_.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX513/data_vehicle_LX513_20220822_v05__truncation__night_.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX513/data_vehicle_LX513_20220823_v05__truncation__night_.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX513/data_vehicle_LX513_20220824_v05__truncation__night_.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX513/data_vehicle_LX513_20220825_v05__truncation__night_.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX513/data_vehicle_LX513_20220826_v05__truncation__night_.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX513/data_vehicle_LX513_20220901_v05__truncation__night_.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX513/data_vehicle_LX513_20220902_v05__truncation__night_.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX513/data_vehicle_LX513_20220905_v05__truncation__night_.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX513/data_vehicle_LX513_20220906_v05__truncation__night_.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX513/data_vehicle_LX513_20220907_v05__truncation__night_.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX513/data_vehicle_LX513_20220908_v05__truncation__night_.anno",  # noqa
                        # LX553 4.0 day
                        f"{root}/multicam_pilot/data/lmdb_datasets/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX553/data_vehicle_LX553_20220811_v05__truncation__day_.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX553/data_vehicle_LX553_20220812_v05__truncation__day_.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX553/data_vehicle_LX553_20220813_v05__truncation__day_.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX553/data_vehicle_LX553_20220815_v05__truncation__day_.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX553/data_vehicle_LX553_20220816_v05__truncation__day_.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX553/data_vehicle_LX553_20220817_v05__truncation__day_.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX553/data_vehicle_LX553_20220818_v05__truncation__day_.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX553/data_vehicle_LX553_20220819_v05__truncation__day_.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX553/data_vehicle_LX553_20220820_v05__truncation__day_.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX553/data_vehicle_LX553_20220821_v05__truncation__day_.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX553/data_vehicle_LX553_20220822_v05__truncation__day_.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX553/data_vehicle_LX553_20220824_v05__truncation__day_.anno",  # noqa
                        # LX553 4.0 night
                        f"{root}/multicam_pilot/data/lmdb_datasets/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX553/data_vehicle_LX553_20220811_v05__truncation__night_.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX553/data_vehicle_LX553_20220812_v05__truncation__night_.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX553/data_vehicle_LX553_20220813_v05__truncation__night_.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX553/data_vehicle_LX553_20220814_v05__truncation__night_.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX553/data_vehicle_LX553_20220815_v05__truncation__night_.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX553/data_vehicle_LX553_20220816_v05__truncation__night_.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX553/data_vehicle_LX553_20220817_v05__truncation__night_.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX553/data_vehicle_LX553_20220818_v05__truncation__night_.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX553/data_vehicle_LX553_20220819_v05__truncation__night_.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX553/data_vehicle_LX553_20220820_v05__truncation__night_.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX553/data_vehicle_LX553_20220821_v05__truncation__night_.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX553/data_vehicle_LX553_20220822_v05__truncation__night_.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX553/data_vehicle_LX553_20220824_v05__truncation__night_.anno",  # noqa
                        # LX553 5.0 day
                        f"{root}/multicam_pilot/data/lmdb_datasets/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX553/data_vehicle_LX553_20220802_v05__truncation__day_.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX553/data_vehicle_LX553_20220803_v05__truncation__day_.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX553/data_vehicle_LX553_20220804_v05__truncation__day_.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX553/data_vehicle_LX553_20220805_v05__truncation__day_.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX553/data_vehicle_LX553_20220806_v05__truncation__day_.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX553/data_vehicle_LX553_20220807_v05__truncation__day_.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX553/data_vehicle_LX553_20220808_v05__truncation__day_.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX553/data_vehicle_LX553_20220825_v05__truncation__day_.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX553/data_vehicle_LX553_20220826_v05__truncation__day_.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX553/data_vehicle_LX553_20220827_v05__truncation__day_.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX553/data_vehicle_LX553_20220829_v05__truncation__day_.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX553/data_vehicle_LX553_20220830_v05__truncation__day_.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX553/data_vehicle_LX553_20220831_v05__truncation__day_.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX553/data_vehicle_LX553_20220901_v05__truncation__day_.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX553/data_vehicle_LX553_20220902_v05__truncation__day_.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX553/data_vehicle_LX553_20220903_v05__truncation__day_.anno",  # noqa
                        # LX553 5.0 night
                        f"{root}/multicam_pilot/data/lmdb_datasets/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX553/data_vehicle_LX553_20220802_v05__truncation__night_.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX553/data_vehicle_LX553_20220803_v05__truncation__night_.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX553/data_vehicle_LX553_20220804_v05__truncation__night_.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX553/data_vehicle_LX553_20220805_v05__truncation__night_.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX553/data_vehicle_LX553_20220806_v05__truncation__night_.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX553/data_vehicle_LX553_20220807_v05__truncation__night_.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX553/data_vehicle_LX553_20220808_v05__truncation__night_.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX553/data_vehicle_LX553_20220825_v05__truncation__night_.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX553/data_vehicle_LX553_20220826_v05__truncation__night_.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX553/data_vehicle_LX553_20220827_v05__truncation__night_.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX553/data_vehicle_LX553_20220828_v05__truncation__night_.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX553/data_vehicle_LX553_20220829_v05__truncation__night_.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX553/data_vehicle_LX553_20220830_v05__truncation__night_.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX553/data_vehicle_LX553_20220831_v05__truncation__night_.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX553/data_vehicle_LX553_20220901_v05__truncation__night_.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX553/data_vehicle_LX553_20220902_v05__truncation__night_.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX553/data_vehicle_LX553_20220903_v05__truncation__night_.anno",  # noqa
                        # LX513 6.0 day
                        f"{root}/multicam_pilot/data/lmdb_datasets/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX513/data_vehicle_LX513_20220910_v05__truncation__day_.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX513/data_vehicle_LX513_20220914_v05__truncation__day_.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX513/data_vehicle_LX513_20220915_v05__truncation__day_.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX513/data_vehicle_LX513_20220917_v05__truncation__day_.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX513/data_vehicle_LX513_20220919_v05__truncation__day_.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX513/data_vehicle_LX513_20220920_v05__truncation__day_.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX513/data_vehicle_LX513_20220921_v05__truncation__day_.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX513/data_vehicle_LX513_20220922_v05__truncation__day_.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX513/data_vehicle_LX513_20220925_v05__truncation__day_.anno",  # noqa
                        # LX513 6.0 night
                        f"{root}/multicam_pilot/data/lmdb_datasets/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX513/data_vehicle_LX513_20220915_v05__truncation__night_.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX513/data_vehicle_LX513_20220921_v05__truncation__night_.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX513/data_vehicle_LX513_20220922_v05__truncation__night_.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX513/data_vehicle_LX513_20220925_v05__truncation__night_.anno",  # noqa
                        # LX553 6.0 day
                        f"{root}/multicam_pilot/data/lmdb_datasets/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX553/data_vehicle_LX553_20220905_v05__truncation__day_.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX553/data_vehicle_LX553_20220906_v05__truncation__day_.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX553/data_vehicle_LX553_20220916_v05__truncation__day_.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX553/data_vehicle_LX553_20220917_v05__truncation__day_.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX553/data_vehicle_LX553_20220919_v05__truncation__day_.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX553/data_vehicle_LX553_20220920_v05__truncation__day_.anno",  # noqa
                        # LX553 6.0 night
                        f"{root}/multicam_pilot/data/lmdb_datasets/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX553/data_vehicle_LX553_20220905_v05__truncation__night_.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX553/data_vehicle_LX553_20220906_v05__truncation__night_.anno",  # noqa
                        # LX513 7.0 day
                        f"{root}/multicam_pilot/data/lmdb_datasets/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX513/data_vehicle_LX513_20221009_v05__truncation__day_.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX513/data_vehicle_LX513_20221011_v05__truncation__day_.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX513/data_vehicle_LX513_20221013_v05__truncation__day_.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX513/data_vehicle_LX513_20221015_v05__truncation__day_.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX513/data_vehicle_LX513_20221016_v05__truncation__day_.anno",  # noqa
                        # LX513 7.0 night
                        f"{root}/multicam_pilot/data/lmdb_datasets/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX513/data_vehicle_LX513_20221009_v05__truncation__night_.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX513/data_vehicle_LX513_20221011_v05__truncation__night_.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX513/data_vehicle_LX513_20221013_v05__truncation__night_.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX513/data_vehicle_LX513_20221015_v05__truncation__night_.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX513/data_vehicle_LX513_20221016_v05__truncation__night_.anno",  # noqa
                        # LX553 7.0 day
                        f"{root}/multicam_pilot/data/lmdb_datasets/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX553/data_vehicle_LX553_20221013_v05__truncation__day_.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX553/data_vehicle_LX553_20221014_v05__truncation__day_.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX553/data_vehicle_LX553_20221015_v05__truncation__day_.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX553/data_vehicle_LX553_20221016_v05__truncation__day_.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX553/data_vehicle_LX553_20221017_v05__truncation__day_.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX553/data_vehicle_LX553_20221018_v05__truncation__day_.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX553/data_vehicle_LX553_20221019_v05__truncation__day_.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX553/data_vehicle_LX553_20221020_v05__truncation__day_.anno",  # noqa
                        # LX553 7.0 night
                        f"{root}/multicam_pilot/data/lmdb_datasets/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX553/data_vehicle_LX553_20221013_v05__truncation__night_.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX553/data_vehicle_LX553_20221014_v05__truncation__night_.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX553/data_vehicle_LX553_20221015_v05__truncation__night_.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX553/data_vehicle_LX553_20221016_v05__truncation__night_.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX553/data_vehicle_LX553_20221017_v05__truncation__night_.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX553/data_vehicle_LX553_20221018_v05__truncation__night_.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX553/data_vehicle_LX553_20221019_v05__truncation__night_.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX553/data_vehicle_LX553_20221020_v05__truncation__night_.anno",  # noqa
                        # LX513 8.0 day
                        f"{root}/multicam_pilot/data/lmdb_datasets/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX513/data_vehicle_LX513_20221024_v05__truncation__day_.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX513/data_vehicle_LX513_20221025_v05__truncation__day_.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX513/data_vehicle_LX513_20221027_v05__truncation__day_.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX513/data_vehicle_LX513_20221029_v05__truncation__day_.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX513/data_vehicle_LX513_20221031_v05__truncation__day_.anno",  # noqa
                        # LX513 8.0 night
                        f"{root}/multicam_pilot/data/lmdb_datasets/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX513/data_vehicle_LX513_20221023_v05__truncation__night_.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX513/data_vehicle_LX513_20221024_v05__truncation__night_.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX513/data_vehicle_LX513_20221025_v05__truncation__night_.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX513/data_vehicle_LX513_20221029_v05__truncation__night_.anno",  # noqa
                        # LX553 8.0 day
                        f"{root}/multicam_pilot/data/lmdb_datasets/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX553/data_vehicle_LX553_20221021_v05__truncation__day_.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX553/data_vehicle_LX553_20221022_v05__truncation__day_.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX553/data_vehicle_LX553_20221023_v05__truncation__day_.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX553/data_vehicle_LX553_20221024_v05__truncation__day_.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX553/data_vehicle_LX553_20221025_v05__truncation__day_.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX553/data_vehicle_LX553_20221027_v05__truncation__day_.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX553/data_vehicle_LX553_20221028_v05__truncation__day_.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX553/data_vehicle_LX553_20221029_v05__truncation__day_.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX553/data_vehicle_LX553_20221031_v05__truncation__day_.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX553/data_vehicle_LX553_20221102_v05__truncation__day_.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX553/data_vehicle_LX553_20221103_v05__truncation__day_.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX553/data_vehicle_LX553_20221104_v05__truncation__day_.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX553/data_vehicle_LX553_20221105_v05__truncation__day_.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX553/data_vehicle_LX553_20221107_v05__truncation__day_.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX553/data_vehicle_LX553_20221109_v05__truncation__day_.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX553/data_vehicle_LX553_20221110_v05__truncation__day_.anno",  # noqa
                        # LX553 8.0 night
                        f"{root}/multicam_pilot/data/lmdb_datasets/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX553/data_vehicle_LX553_20221021_v05__truncation__night_.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX553/data_vehicle_LX553_20221022_v05__truncation__night_.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX553/data_vehicle_LX553_20221023_v05__truncation__night_.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX553/data_vehicle_LX553_20221024_v05__truncation__night_.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX553/data_vehicle_LX553_20221025_v05__truncation__night_.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX553/data_vehicle_LX553_20221027_v05__truncation__night_.anno",  # noqa
                        f"{root}/multicam_pilot/data/lmdb_datasets/users/wenyuan.zeng/galaxy_truncate_badcase_4v/v05/LX553/data_vehicle_LX553_20221029_v05__truncation__night_.anno",  # noqa
                    ],
                    "sample_weight": 13.52,
                    "partition": "truncation",
                },
            ]
        },
    }
)

datapaths = EasyDict(datapaths)
