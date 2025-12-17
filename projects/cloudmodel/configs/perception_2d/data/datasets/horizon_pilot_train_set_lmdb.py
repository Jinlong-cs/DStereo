from easydict import EasyDict

pilot_data_paths = {
    "vehicle": {
        "train_data_paths": [
            {
                "data_path": "dmpv2://matrix2/multicam_pilot/data/lmdb_datasets/data/vehicle_full_detection/matrix_2.1_v3.0.0_train_data/newdata_rec/densebox.train.basenewv1",
                "sample_weight": 10,
                "data_path_md5sum": "894e59a34d3b4b07168dc9e674f0f1bd",
            },
            {
                "data_path": "dmpv2://matrix2/multicam_pilot/data/lmdb_datasets/data/vehicle_full_detection/matrix_2.1_v3.0.0_train_data/newdata_rec/densebox.train.12camnewv1",
                "sample_weight": 10,
                "data_path_md5sum": "8861e3899a0f80cf925cdfc02e4a1ad1",
            },
            {
                "data_path": "dmpv2://matrix2/multicam_pilot/data/lmdb_datasets/data/vehicle_full_detection/matrix_2.1_v3.0.0_train_data/newdata_rec/densebox.train.newdatav2",
                "sample_weight": 2,
                "data_path_md5sum": "576665f581e4f3318ee3f15305bd5308",
            },
            {
                "data_path": "dmpv2://matrix2/multicam_pilot/data/lmdb_datasets/data/vehicle_full_detection/matrix_2.1_v3.0.0_train_data/vehicle_full_data_1080_cam3900/det_vehicle_full_v1_0-20200527_v1/train.mayv9",
                "sample_weight": 5,
                "data_path_md5sum": "b160b3ae1681f2d7dd2e1470e6dda8ef",
            },
            {
                "data_path": "dmpv2://matrix2/multicam_pilot/data/lmdb_datasets/data/vehicle_full_detection/matrix_2.1_v3.0.0_train_data/vehicle_full_data_1080/det_vehicle_full_v1_0-20200102_v1/train",
                "sample_weight": 5,
                "data_path_md5sum": "815bb660d68a128acfea99cb225e5068",
            },
            {
                "data_path": "dmpv2://matrix2/multicam_pilot/data/lmdb_datasets/mono_0323_data/user/yonggang.yang/data/vehicle_full_data_0323/det_vehicle_full_v1_0-20201118_v1/train",
                "sample_weight": 5,
                "data_path_md5sum": "98d6ee3eccfe2a7e9a9b3fc65fd53ee9",
            },
            {
                "data_path": "dmpv2://matrix2/multicam_pilot/data/lmdb_datasets/mono_0323_data/user/yonggang.yang/data/vehicle_full_data_1080/det_vehicle_full_v1_0-20200706_v1/train",
                "sample_weight": 5,
                "data_path_md5sum": "718c62d34f8c994146b66c6888799d17",
            },
            {
                "data_path": "dmpv2://matrix2/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/vehicle/vehicle_full_detection/pilot3.0_0233_data_before_0714_day/train",
                "sample_weight": 20,
                "data_path_md5sum": "dedd563feef3c08c8435e8fb5d4860b3",
            },
            {
                "data_path": "dmpv2://matrix2/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/vehicle/vehicle_full_detection_tmp/pilot3.0_0233_data_before_0406_bigtruck_new/data",
                "sample_weight": 5,
                "data_path_md5sum": "c5b4bfcdb781db741541e569de748e5f",
            },
            {
                "data_path": "dmpv2://matrix2/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/vehicle/vehicle_full_detection/vehicle_full_packing_day_local/20210827-195610/data",
                "sample_weight": 10,
                "data_path_md5sum": "46b569e8d7008f79b74f7c28913f60ef",
            },
            {
                "data_path": "dmpv2://matrix2/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/vehicle/vehicle_full_detection/vehicle_full_packing_day_local_local/20210928-120519/data",
                "sample_weight": 10,
                "data_path_md5sum": "2042309d23ca4ce501d830bc39fbc5f0",
            },
            {
                "data_path": "dmpv2://matrix2/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/vehicle/vehicle_full_detection/badcase/full_vehicle_badcase_self_camera_local/20210827-212337/data",
                "sample_weight": 1,
                "data_path_md5sum": "15f5a81b28560ed4236ab5782bd75261",
            },
            {
                "data_path": "dmpv2://matrix2/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/vehicle/vehicle_full_detection/vehicle_full_packing_badcase_local/20211019-194438/data",
                "sample_weight": 6,
                "data_path_md5sum": "2223b6e1b3d2ae0fc620474ab16746d3",
            },
            {
                "data_path": "dmpv2://matrix2/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/vehicle/vehicle_full_detection/vehicle_full_packing_day_trace/20211124-163642/data",
                "sample_weight": 3,
                "data_path_md5sum": "180bad5959e703117aa18ecdb27b9016",
            },
            {
                "data_path": "dmpv2://matrix2/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/vehicle/vehicle_full_detection/vehicle_full_packing_day_trace/20211129-163640/data",
                "sample_weight": 3,
                "data_path_md5sum": "23c25a5807b8774fc3c9652babb5aabb",
            },
            {
                "data_path": "dmpv2://matrix2/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/vehicle/vehicle_full_detection/vehicle_full_packing_day_x3c/20211222-124627/data",
                "sample_weight": 4,
                "data_path_md5sum": "d5916eaa7c39bb8f0865cb1a3622a94c",
            },
            {
                "data_path": "dmpv2://matrix2/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/vehicle/vehicle_full_detection/vehicle_full_packing_badcase/20211222-121533/data",
                "sample_weight": 2,
                "data_path_md5sum": "b06a9b3702415315579c2364a5e69ae0",
            },
            {
                "data_path": "dmpv2://matrix2/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/vehicle/vehicle_full_detection/vehicle_full_packing_night_x3c/20220121-114854/data",
                "sample_weight": 3,
                "data_path_md5sum": "f5e1c1a43a8f054e2018efefb2109e09",
            },
            {
                "data_path": "dmpv2://matrix2/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/vehicle/vehicle_full_detection/pilot_vehicle_data_5v_32393_2.1w_CAMERA_CO_OX3GB_A100_L_TIME_DAY_day/20220406-125044/data",
                "sample_weight": 2,
                "data_path_md5sum": "a817cfec0da6fc6b3a68af83647ff658",
            },
            {
                "data_path": "dmpv2://matrix2/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/vehicle/vehicle_full_detection/pilot_vehicle_DG4W_data_5v_CAMERA_CO_OX3GB_D100_L_TIME_DAY_day/20220401-184015/data",
                "sample_weight": 4,
                "data_path_md5sum": "42d9aef475edd5c0cb7f2ca78c8f1079",
            },
            {
                "data_path": "dmpv2://matrix2/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/vehicle/vehicle_full_detection/pilot_vehicle_LIGHT07_data_5v_CAMERA_CP_OX3GB_D100_L_TIME_DAY_day/20220401-164602/data",
                "sample_weight": 2,
                "data_path_md5sum": "67ec8e0ba9aadfd202e0d636a1f68abe",
            },
            {
                "data_path": "dmpv2://matrix2/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/vehicle/vehicle_full_detection/pilot_vehicle_29384_10763_CAMERA_CP_OX3GB_D100_L_TIME_DAY_day/20220426-101418/data",
                "sample_weight": 1,
                "data_path_md5sum": "6835c20312ef332272c26abaeaf37c6a",
            },
            {
                "data_path": "dmpv2://matrix2/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/vehicle/vehicle_full_detection/pilot_vehicle_data_5v_as33_25469_2.2w_CAMERA_CW_A23GB_A114_L_TIME_DAY_day/20220413-111602/data",
                "sample_weight": 3,
                "data_path_md5sum": "c47f0252ed3c6fab74dc37100603cb7d",
            },
            {
                "data_path": "dmpv2://matrix2/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/vehicle/vehicle_full_detection/polit3.0_v9.0_0315_0233_day/20220316-142817/data",
                "sample_weight": 3,
                "data_path_md5sum": "599b23ceb2392195bccb2675d13b26ef",
            },
            {
                "data_path": "dmpv2://matrix2/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/vehicle/det_vehicle_full_v1_0-20220407_v1/train",
                "sample_weight": 2,
                "data_path_md5sum": "a140fa0b1a79ebda95ec7a6d8bbdb2ac",
            },
            {
                "data_path": "dmpv2://matrix2/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/pilot_vehicle_1v_CW-A23GB-A100-L_day/20220518-142747/data",
                "sample_weight": 1,
                "data_path_md5sum": "1b02fdafa25d109181b1fa2faf2c61e5",
            },
            {
                "data_path": "dmpv2://matrix2/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/vehicle/pilot_vehicle_5v_CAMERA_CW_A23GB_A100_L_TIME_BIG_CAR_DAY_day/20220621-202410/data",
                "sample_weight": 4,
                "data_path_md5sum": "5c71336d60ec3f35a6a6d0d5227bbc1d",
            },
            {
                "data_path": "dmpv2://matrix2/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/vehicle/pilot_vehicle_5v_CAMERA_CO_OX3GB_A100_L_TIME_BIG_CAR_DAY_day/20220621-192421/data",
                "sample_weight": 1,
                "data_path_md5sum": "acac29f1c4bb6cbc1730eeecffe0df3a",
            },
            {
                "data_path": "dmpv2://matrix2/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/pilot_vehicle_5v_CAMERA_CW_A23GB_A100_L_day/20220617-131419/data",
                "sample_weight": 1,
                "data_path_md5sum": "b078c1968ce37b4b95ed064710d9a013",
            },
            {
                "data_path": "dmpv2://matrix2/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/vehicle_detection/pilot_vehicle_5v_CAMERA_CW_A23GB_A114_L_day_normal/20220706-134742/data",
                "sample_weight": 2,
                "data_path_md5sum": "c7aa98a0d24f46a37e5eab6387ef03ab",
            },
            {
                "data_path": "dmpv2://matrix2/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/vehicle_detection/pilot_vehicle_5v_CAMERA_CW_A23GB_A114_L_day/20220621-161421/data",
                "sample_weight": 1,
                "data_path_md5sum": "9ed1a5795d5424a322770bdddb25bf2c",
            },
            {
                "data_path": "dmpv2://matrix2/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/kVehBBox2D/kVehBBox2D_project_as33_5v_CW-A23GB-A114-L_train_day_overlap_vehicle_Cam_5V/20220826-161108/data",
                "sample_weight": 0.5,
                "data_path_md5sum": "4da49735da86281aa3768dfacd7a54f4",
            },
            {
                "data_path": "dmpv2://matrix2/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/kVehBBox2D/kVehBBox2D_project_as33_5v_CW-A23GB-A114-L_train_day_overlap_vehicle_Cam_5V/20220826-163619/data",
                "sample_weight": 0.5,
                "data_path_md5sum": "f71bce1be83268f50d7dfdfc126fcc7f",
            },
            {
                "data_path": "dmpv2://matrix2/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/kVehBBox2D/kVehBBox2D_project_pilot_pdt_5v_CW-A23GB-A100-L_train_day_glare_Cam_5V/20220729-121524/data",
                "sample_weight": 2,
                "data_path_md5sum": "6ec260511b3d0391cafd873f36b9c931",
            },
            {
                "data_path": "dmpv2://matrix2/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/vehicle_detection/pilot_vehicle_5v_54665_10259_CAMERA_CP_OX3GB_D100_L_TIME_DAY_day/20220707-160146/data",
                "sample_weight": 2,
                "data_path_md5sum": "21d13cfdd66a0da4837bdfd760b98c0f",
            },
            {
                "data_path": "dmpv2://matrix2/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/vehicle_detection/pilot_vehicle_5v_CAMERA_CW_A23GB_A100_L_TIME_DAY_day/20220711-121816/data",
                "sample_weight": 1,
                "data_path_md5sum": "d1aedf504267ea11ce5dcf13304e4527",
            },
            {
                "data_path": "dmpv2://matrix2/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/vehicle_detection/pilot_vehicle_5v_CAMERA_CW_A23GB_A100_L_TIME_DAY_day/20220711-121723/data",
                "sample_weight": 1,
                "data_path_md5sum": "f1ccfcbae35b9d3c7ae4b6cdcb7a3a3e",
            },
            {
                "data_path": "dmpv2://matrix2/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/vehicle_detection/pilot_vehicle_5v_CAMERA_CW_A23GB_A100_L_TIME_DAY_day/20220711-134219/data",
                "sample_weight": 4,
                "data_path_md5sum": "217daa5fc1bdbce04e110876537a026a",
            },
            {
                "data_path": "dmpv2://matrix2/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/vehicle_detection/pilot_vehicle_5v_CAMERA_CW_A23GB_A100_L_TIME_DAY_day/20220711-160123/data",
                "sample_weight": 4,
                "data_path_md5sum": "b1802491798ae2214742586943ea57b7",
            },
            {
                "data_path": "dmpv2://matrix2/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/vehicle_detection/pilot_vehicle_1v_OV10652_TIME_DAY_day/20220713-053741/data",
                "sample_weight": 7,
                "data_path_md5sum": "656555a2458e1e1049430e1ab99acce1",
            },
            {
                "data_path": "dmpv2://matrix2/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/vehicle_detection/pilot_vehicle_1v_OV10652_TIME_DAY_day/20220713-061904/data",
                "sample_weight": 11,
                "data_path_md5sum": "3a34ee6009073fcc77bc520ee0d2ae18",
            },
            {
                "data_path": "dmpv2://matrix2/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/kVehBBox2D/kVehBBox2D_project_c385_5v_CO-OX3GB-D100-L_train_day_Cam_5V/20220902-200336/data",
                "sample_weight": 1,
                "data_path_md5sum": "34679abf8fea49207b2c3bcde45fde44",
            },
            {
                "data_path": "dmpv2://matrix2/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/vehicle_detection/pilot_vehicle_5v_CAMERA_AS33_SCENE-FP_day/20221115-182150/data",
                "sample_weight": 4,
                "data_path_md5sum": "c7a241a3ca73e246327baf4abbfaaed6",
            },
            {
                "data_path": "dmpv2://matrix2/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/kVehBBox2D/kVehBBox2D_project_maxfa_5v_CP-OX3GB-D100-L_train_day_special_vehicle_Cam_5V/20220926-103720/data",
                "sample_weight": 1,
                "data_path_md5sum": "2ca70dff275b86c682e213decd28db48",
            },
            {
                "data_path": "dmpv2://matrix2/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/kVehBBox2D/kVehBBox2D_project_c385_5v_CO-OX3GB-D100-L_train_day_Cam_5V/20221012-122031/data",
                "sample_weight": 1,
                "data_path_md5sum": "81cb3d2cd8b4d197d1cc815519580c23",
            },
            {
                "data_path": "dmpv2://matrix2/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/vehicle_detection/pilot_vehicle_5v_CAMERA_CO-OX3GB-D100-L_JIRA-badcase_all/20230329-101353",
                "sample_weight": 5,
                "data_path_md5sum": "e1f8a740bd02448ca5811709e7e0e478",
            },
            {
                "data_path": "dmpv2://matrix2/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/vehicle_detection/pilot_vehicle_5v_CAMERA_GALAXY_SCENE-FP_day/20221115-190526/data",
                "sample_weight": 2,
                "data_path_md5sum": "7d5f8964a093341850462b6d3becafe4",
            },
            {
                "data_path": "dmpv2://matrix2/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/vehicle_detection/pilot_vehicle_5v_CAMERA_AS33_JIRA-badcase_all/20221110-192849/data",
                "sample_weight": 1,
                "data_path_md5sum": "6cc522f10f93dcc1a8e4972c04c3cae5",
            },
            {
                "data_path": "dmpv2://matrix2/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/vehicle_detection/pilot_vehicle_5v_CAMERA_GALAXY_SCENE-FP_day/20221219-120427/data",
                "sample_weight": 2,
                "data_path_md5sum": "ee0ca4736235c28aedf5fa32acafbd4f",
            },
            {
                "data_path": "dmpv2://matrix2/multicam_pilot/data/train/vehicle_detection/project_maxfa_5v_CP-OX3GB-D100-L_train_day_special_vehicle_Cam_5V/20230320_111109",
                "sample_weight": 2,
                "data_path_md5sum": "b864895a9e61c5b87864c6d480114b2c",
            },
            {
                "data_path": "dmpv2://matrix2/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/vehicle_detection/pilot_vehicle_5v_CAMERA_CO_OX3GB_D100_L_day_all/20230118-115653/data",
                "sample_weight": 1,
                "data_path_md5sum": "c4903a92c4c1462c510e453346d9b886",
            },
            {
                "data_path": "dmpv2://matrix2/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/vehicle_detection/pilot_vehicle_5v_CAMERA_CO_OX3GB_D100_L_jira_badcase_day_all/20230118-114938/data",
                "sample_weight": 2,
                "data_path_md5sum": "26eac1048dc0d568b03b35905dc5a8f8",
            },
            {
                "data_path": "dmpv2://matrix2/multicam_pilot/data/train/vehicle_detection/CP-OX3GB-D100-L_train_day_Cam_5V/20230317_153229",
                "sample_weight": 1,
                "data_path_md5sum": "bf11c0d91250fd12bdbd9c3e7d0d2be7",
            },
            {
                "data_path": "dmpv2://matrix2/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/vehicle_detection/pilot_vehicle_5v_night_0233/20220119-002848/data",
                "sample_weight": 60,
                "data_path_md5sum": "e1c1f95c7878ad3a2b11e7d0c899d847",
            },
            {
                "data_path": "dmpv2://matrix2/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/vehicle/vehicle_full_detection/vehicle_full_packing_night_x3c/20211222-132550/data",
                "sample_weight": 3,
                "data_path_md5sum": "c81dea6ed13f0d95675808893612cfcf",
            },
            {
                "data_path": "dmpv2://matrix2/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/vehicle/vehicle_full_detection/pilot_vehicle_data_5v_32394_0.68w_CAMERA_CO_OX3GB_A100_L_TIME_NIGHT_night/20220406-124214/data",
                "sample_weight": 1,
                "data_path_md5sum": "b66944d9547d5557cae7262e27719f09",
            },
            {
                "data_path": "dmpv2://matrix2/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/kVehBBox2D/kVehBBox2D_CW-A23GB-A100-L_train_night_dark_night/20220610-160750/data",
                "sample_weight": 4,
                "data_path_md5sum": "db772e2418bfc0bb262a9e7ab2d9144a",
            },
            {
                "data_path": "dmpv2://matrix2/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/kVehBBox2D/kVehBBox2D_project_maxfa_5v_CP-OX3GB-D100-L_train_night_dark_night/20220610-144900/data",
                "sample_weight": 1,
                "data_path_md5sum": "ce4e2e39a9ba5b67bc226088706d9c9e",
            },
            {
                "data_path": "dmpv2://matrix2/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/kVehBBox2D/kVehBBox2D_project_as33_5v_CW-A23GB-A114-L_train_night_dark_night/20220610-144130/data",
                "sample_weight": 1,
                "data_path_md5sum": "3928580b45228880277ab25bb89e8661",
            },
            {
                "data_path": "dmpv2://matrix2/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/pilot_vehicle_5v_CAMERA_CW_A23GB_A100_L_day/20220620-181555/data",
                "sample_weight": 1,
                "data_path_md5sum": "266eb29ba6ced9a01237412da2d7a16d",
            },
            {
                "data_path": "dmpv2://matrix2/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/vehicle/vehicle_full_detection/pilot_vehicle_33333_6661_CAMERA_CO_OX3GB_A100-L_TIME_NIGHT_night/20220527-145352/data",
                "sample_weight": 1,
                "data_path_md5sum": "07ce67dc9a6d0e8ffc18e57d39d21a02",
            },
            {
                "data_path": "dmpv2://matrix2/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/vehicle/vehicle_full_detection/pilot_vehicle_32658_9964_CAMERA_CO_OX3GB_A100-L_TIME_NIGHT_night/20220527-145938/data",
                "sample_weight": 1,
                "data_path_md5sum": "d5fca6b6930454a135a4af2f8bc1a1d9",
            },
            {
                "data_path": "dmpv2://matrix2/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/vehicle/vehicle_full_detection/pilot_vehicle_34231_12284_CAMERA_CO_OX3GB_A100-L_TIME_NIGHT_night/20220527-150019/data",
                "sample_weight": 2,
                "data_path_md5sum": "775ccab4214c7d77d5d872ea6463c499",
            },
            {
                "data_path": "dmpv2://matrix2/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/vehicle/pilot_vehicle_5v_CAMERA_CW_A23GB_A100_L_TIME_BIG_CAR_night/20220629-121819/data",
                "sample_weight": 16,
                "data_path_md5sum": "05b9bc6986d0b8bb1192b461fe77d315",
            },
            {
                "data_path": "dmpv2://matrix2/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/vehicle_detection/pilot_vehicle_5v_CAMERA_CW_A23GB_A114_L_day_bigcar/20220704-185847/data",
                "sample_weight": 1,
                "data_path_md5sum": "6b6ce6baa54ca1d6e2071ef9f25a973f",
            },
            {
                "data_path": "dmpv2://matrix2/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/kVehBBox2D/kVehBBox2D_project_maxfa_5v_CP-OX3GB-D100-L_train_night_badcase_dark_night_Cam_5V/20220709-175425/data",
                "sample_weight": 1,
                "data_path_md5sum": "9d076baea73ff1b657938e60fc44aa94",
            },
            {
                "data_path": "dmpv2://matrix2/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/kVehBBox2D/kVehBBox2D_project_as33_5v_CW-A23GB-A114-L_train_night_badcase_dark_night_Cam_5V/20220709-173429/data",
                "sample_weight": 3,
                "data_path_md5sum": "ce722526fefbd19d152787811703e8e9",
            },
            {
                "data_path": "dmpv2://matrix2/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/vehicle_detection/pilot_vehicle_5v_CAMERA_CO-OX3GB-A100-L_TIME_BIG_CAR_night/20220725-143019/data",
                "sample_weight": 2,
                "data_path_md5sum": "78f0bacc2cd6610360b039ecd51a6161",
            },
            {
                "data_path": "dmpv2://matrix2/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/vehicle_detection/pilot_vehicle_5v_CAMERA_CW_A23GB_A114_L_night_dark_0736/20220726-165515/data",
                "sample_weight": 1,
                "data_path_md5sum": "4cf88c351411a722b3ee0b5a34d2b51f",
            },
            {
                "data_path": "dmpv2://matrix2/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/vehicle_detection/pilot_vehicle_5v_CAMERA_CW_A23GB_A114_L_night_bigcar_0715/20220715-134225/data",
                "sample_weight": 1,
                "data_path_md5sum": "29e264ecfa75a004def392f4591a6db9",
            },
            {
                "data_path": "dmpv2://matrix2/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/vehicle_detection/pilot_vehicle_5v_73388_73389_73390_7858_CAMERA_CW_A23GB_A114_L_night/20220726-181251/data",
                "sample_weight": 2,
                "data_path_md5sum": "59d1f877d661962dd1412ca165b25aaa",
            },
            {
                "data_path": "dmpv2://matrix2/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/vehicle_detection/pilot_vehicle_5v_64258_64255_65416_9091_CAMERA_CW_A23GB_A114_L_TIME_NIGHT_night/20220707-143748/data",
                "sample_weight": 2,
                "data_path_md5sum": "0e9f181f3f253c61615e82de1521e16b",
            },
            {
                "data_path": "dmpv2://matrix2/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/vehicle_detection/pilot_vehicle_5v_CAMERA_CW_A23GB_A100_L_TIME_DARK_NIGHT_night/20220731-090659/data",
                "sample_weight": 1,
                "data_path_md5sum": "50b6e8a7d6dc8a48ea113179c5951c40",
            },
            {
                "data_path": "dmpv2://matrix2/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/vehicle_detection/pilot_vehicle_5v_CAMERA_AS33_SCENE-FP_night/20221115-182152/data",
                "sample_weight": 4,
                "data_path_md5sum": "bc15d229667cd860f28ee36931cf88bd",
            },
            {
                "data_path": "dmpv2://matrix2/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/kVehBBox2D/kVehBBox2D_project_c385_5v_CO-OX3GB-D100-L_train_night_Cam_5V/20221012-114409/data",
                "sample_weight": 1,
                "data_path_md5sum": "26df6cb055ae70479ae2e8efc95fbfb6",
            },
            {
                "data_path": "dmpv2://matrix2/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/vehicle_detection/pilot_vehicle_5v_CAMERA_CO-OX3GB-D100-L_JIRA-badcase_all/20221229-173836/data",
                "sample_weight": 5,
                "data_path_md5sum": "be8267dfbb28a746de1b0d29c9ba9f1a",
            },
            {
                "data_path": "dmpv2://matrix2/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/kVehBBox2D/kVehBBox2D_project_as33_5v_CW-A23GB-A114-L_train_night_glare_Cam_5V/20221009-193824/data",
                "sample_weight": 1,
                "data_path_md5sum": "68868a2e9ba9512d737c3d261c118b21",
            },
            {
                "data_path": "dmpv2://matrix2/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/vehicle_detection/pilot_vehicle_5v_CAMERA_GALAXY_SCENE-FP_night/20221115-190527/data",
                "sample_weight": 2,
                "data_path_md5sum": "aff509d94935f9053bb9683dcdc73d54",
            },
            {
                "data_path": "dmpv2://matrix2/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/kVehBBox2D/kVehBBox2D_project_c385_5v_CO-OX3GB-D100-L_train_night_glare_Cam_5V/20221103-204258/data",
                "sample_weight": 2,
                "data_path_md5sum": "5cd3bcd9df6d3a4634c69a520d39f0a8",
            },
            {
                "data_path": "dmpv2://matrix2/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/kVehBBox2D/kVehBBox2D_project_as33_5v_CW-A23GB-A114-L_train_night_special_vehicle_Cam_5V/20221103-204024/data",
                "sample_weight": 2,
                "data_path_md5sum": "5750d4f4569848ab837e8a70ba9ffac4",
            },
            {
                "data_path": "dmpv2://matrix2/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/kVehBBox2D/kVehBBox2D_project_as33_5v_CW-A23GB-A114-L_train_night_glare_Cam_5V/20221103-205303/data",
                "sample_weight": 2,
                "data_path_md5sum": "ed90526761fea823a7ef7992c4119d27",
            },
            {
                "data_path": "dmpv2://matrix2/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/kVehBBox2D/kVehBBox2D_project_as33_5v_CW-A23GB-A114-L_train_night_glare_Cam_5V/20221114-194733/data",
                "sample_weight": 1,
                "data_path_md5sum": "b9f5d020f114fa3c89807efb60c6e1c4",
            },
            {
                "data_path": "dmpv2://matrix2/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/vehicle_detection/pilot_vehicle_5v_CAMERA_AS33_night-glareV4_all/20221205-170615/data",
                "sample_weight": 1,
                "data_path_md5sum": "623b5b356b66cd1baaf2420304beeeb6",
            },
            {
                "data_path": "dmpv2://matrix2/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/vehicle_detection/pilot_vehicle_5v_CAMERA_GALAXY_SCENE-FP_night/20221219-120553/data",
                "sample_weight": 2,
                "data_path_md5sum": "5c8ad2093f2e765d5837f6a6f338a567",
            },
            {
                "data_path": "dmpv2://matrix2/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/vehicle_detection/pilot_vehicle_5v_CAMERA_GALAXY_SCENE-FP_night/20230221-104721",
                "sample_weight": 1,
                "data_path_md5sum": "1419d99c2d35eab9e20398d514846204",
            },
            {
                "data_path": "dmpv2://matrix2/multicam_pilot/data/train/vehicle_detection/CP-OX3GB-D100-L_train_night_Cam_5V/20230317_155826",
                "sample_weight": 1,
                "data_path_md5sum": "b45f6de01dcf08e2d35ac248eabbd777",
            },
            {
                "data_path": "dmpv2://matrix2/multicam_pilot/data/train/vehicle_detection/project_as33_5v_CW-A23GB-A114-L_train_night_glare_Cam_5V/20230317_202437",
                "sample_weight": 1,
                "data_path_md5sum": "30a99c245f93f74090152eb83ab1284c",
            },
            {
                "data_path": "dmpv2://matrix2/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/vehicle_detection/pilot_vehicle_350703_CAMERA_CO_OX3GB_D100_L_TIME_ALL_all/20220818-202734/data",
                "sample_weight": 32,
                "data_path_md5sum": "4700f0cb566be765fbf0669b62f9c466",
            },
            {
                "data_path": "dmpv2://matrix2/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/vehicle_detection/pilot_vehicle_common_20220905_CAMERA_CO_OX3GB_D100_L_TIME_ALL_all/20220905-164533/data",
                "sample_weight": 32,
                "data_path_md5sum": "b4d32115b99d94081a3c4edd795e8931",
            },
            {
                "data_path": "dmpv2://matrix2/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/vehicle_detection/pilot_vehicle_350703_CAMERA_CO_OX3GB_D100_L_TIME_ALL_all/20220831-181342/data",
                "sample_weight": 0.5,
                "data_path_md5sum": "44e6bed552df22833d9ea117dc2ba7d1",
            },
            {
                "data_path": "dmpv2://matrix2/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/vehicle_detection/pilot_vehicle_common_20221008_CAMERA_CO_OX3GB_D100_L_TIME_ALL_all/20221008-161126/data",
                "sample_weight": 16,
                "data_path_md5sum": "46ae64804f5fe4ddb0fb7ddb5ad0018f",
            },
            {
                "data_path": "dmpv2://matrix2/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/vehicle_detection/pilot_vehicle_common_20221114_CAMERA_CO_OX3GB_D100_L_TIME_ALL_all/20221114-204456/data",
                "sample_weight": 16,
                "data_path_md5sum": "34f0b37ddcb79ec7c8b1b64a87b6725d",
            },
            {
                "data_path": "dmpv2://matrix2/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/vehicle_detection/pilot_vehicle_wall_FP_20221125_CAMERA_CO_OX3GB_D100_L_TIME_ALL_all/20221125-085017/data",
                "sample_weight": 0.5,
                "data_path_md5sum": "cb866f7d20f463770e9a59572b361579",
            },
            {
                "data_path": "dmpv2://matrix2/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/vehicle_detection/pilot_vehicle_common_20221008_CAMERA_CO_OX3GB_D100_L_TIME_ALL_all/20230120-155231/data",
                "sample_weight": 16,
                "data_path_md5sum": "51ab20ecf654839e40ea616e9825e969",
            },
            {
                "data_path": "dmpv2://matrix2/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/kVehBBox2D/kVehBBox2D_project_galaxy_5v_train_day_Cam_Rear/20220622-150758/data",
                "sample_weight": 1,
                "data_path_md5sum": "d5c0b18cfa99b232500784e41a09a010",
            },
            {
                "data_path": "dmpv2://matrix2/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/kVehBBox2D/kVehBBox2D_project_galaxy_5v_train_night_Cam_Rear/20220622-145944/data",
                "sample_weight": 1,
                "data_path_md5sum": "18b83529cdcd5a21aeb484254a5e797c",
            },
            {
                "data_path": "dmpv2://matrix2/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/kVehBBox2D/kVehBBox2D_project_galaxy_5v_train_day_Cam_Side/20220622-150945/data",
                "sample_weight": 1,
                "data_path_md5sum": "da3c44a7a2f7fb8b4b04912ee67a8609",
            },
            {
                "data_path": "dmpv2://matrix2/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/kVehBBox2D/kVehBBox2D_project_galaxy_5v_train_night_Cam_Side/20220622-145829/data",
                "sample_weight": 1,
                "data_path_md5sum": "d93e552917f3e5a2a492fb8d097940f5",
            },
            {
                "data_path": "dmpv2://matrix2/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/kVehBBox2D/kVehBBox2D_project_galaxy_5v_train_day_Cam_Rear/20220719-114309/data",
                "sample_weight": 2,
                "data_path_md5sum": "49a6b629a5b2dadbc5f74aa78572e55d",
            },
            {
                "data_path": "dmpv2://matrix2/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/kVehBBox2D/kVehBBox2D_project_galaxy_5v_train_day_Cam_Side/20220719-111957/data",
                "sample_weight": 2,
                "data_path_md5sum": "70cfe0f6780a41a1c0503b8b4e395595",
            },
            {
                "data_path": "dmpv2://matrix2/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/kVehBBox2D/kVehBBox2D_project_galaxy_5v_train_night_Cam_Side/20220719-115156/data",
                "sample_weight": 1,
                "data_path_md5sum": "748c46bf94c7bf36600282b61e18602f",
            },
            {
                "data_path": "dmpv2://matrix2/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/kVehBBox2D/kVehBBox2D_project_galaxy_5v_train_night_Cam_Rear/20220719-115248/data",
                "sample_weight": 1,
                "data_path_md5sum": "7757c3c850d5179ed55481f5aa233ff9",
            },
            {
                "data_path": "dmpv2://matrix2/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/kVehBBox2D/kVehBBox2D_project_galaxy_5v_train_day_Cam_Side/20220805-225859/data",
                "sample_weight": 1,
                "data_path_md5sum": "a0655b2ca06549be360b45cd1cb8c0d3",
            },
            {
                "data_path": "dmpv2://matrix2/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/kVehBBox2D/kVehBBox2D_project_galaxy_5v_train_day_Cam_Rear/20220805-230512/data",
                "sample_weight": 1,
                "data_path_md5sum": "033dd5b78c01e8d3300e5dac1e0ea617",
            },
            {
                "data_path": "dmpv2://matrix2/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/kVehBBox2D/kVehBBox2D_project_galaxy_5v_train_night_Cam_Rear/20220805-230658/data",
                "sample_weight": 1,
                "data_path_md5sum": "19561ede11f0982b6744ea8eb17bf891",
            },
            {
                "data_path": "dmpv2://matrix2/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/kVehBBox2D/kVehBBox2D_project_galaxy_5v_train_night_Cam_Side/20220805-230919/data",
                "sample_weight": 1,
                "data_path_md5sum": "056d848333dd6a41cd53a0654547d326",
            },
            {
                "data_path": "dmpv2://matrix2/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/kVehBBox2D/kVehBBox2D_project_galaxy_5v_train_day_Cam_Side/20221001-093910/data",
                "sample_weight": 3,
                "data_path_md5sum": "12add3c64a0db61c309cd6bba9ffb5eb",
            },
            {
                "data_path": "dmpv2://matrix2/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/kVehBBox2D/kVehBBox2D_project_galaxy_5v_train_night_Cam_Rear/20221001-093514/data",
                "sample_weight": 1,
                "data_path_md5sum": "e68efa76b06458d797f3fd70e7ffdc69",
            },
            {
                "data_path": "dmpv2://matrix2/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/kVehBBox2D/kVehBBox2D_project_galaxy_5v_train_day_Cam_Rear/20221001-092648/data",
                "sample_weight": 2,
                "data_path_md5sum": "607d0d87d248dec0fff5c95f8335b7a5",
            },
            {
                "data_path": "dmpv2://matrix2/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/kVehBBox2D/kVehBBox2D_project_galaxy_5v_train_night_Cam_Side/20221001-092213/data",
                "sample_weight": 2,
                "data_path_md5sum": "73f190e074e2273fc63ca721838d585e",
            },
            {
                "data_path": "dmpv2://matrix2/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/vehicle_detection/pilot_vehicle_5v_CAMERA_GALAXY_JIRA-badcase_all/20230329-094447",
                "sample_weight": 12,
                "data_path_md5sum": "a4011dc1c77a48e2dd77c1e89730981d",
            },
            {
                "data_path": "dmpv2://matrix2/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/vehicle_detection/pilot_vehicle_5v_CAMERA_CO-OX3GB-D100-L_JIRA-badcase_all/20221230-150000",
                "sample_weight": 5,
                "data_path_md5sum": "1a03e7face0c7fb8a229e9d07aada35c",
            },
            {
                "data_path": "dmpv2://matrix2/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/kVehBBox2D/kVehBBox2D_project_c385_5v_CO-OX3GB-D100-L_train_day_special_vehicle_Cam_5V/20221009-193855/data",
                "sample_weight": 1,
                "data_path_md5sum": "32ee664bb0577601a2a44416fe04efe7",
            },
            {
                "data_path": "dmpv2://matrix2/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/kVehBBox2D/kVehBBox2D_project_c385_5v_CO-OX3GB-D100-L_train_day_special_vehicle_Cam_5V/20221103-203708/data",
                "sample_weight": 2,
                "data_path_md5sum": "a61aa0c0d07575dbedb165c9ed7bee2b",
            },
            {
                "data_path": "dmpv2://matrix2/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/vehicle_detection/pilot_vehicle_5v_CAMERA_CO_OX3GB_D100_L_TIME_DAY_SPECIAL_CAR_all/20221122-103953/data",
                "sample_weight": 2,
                "data_path_md5sum": "709c0455b47a2e2fcc908ca66465f685",
            },
            {
                "data_path": "dmpv2://matrix2/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/vehicle_detection/pilot_vehicle_5v_CAMERA_GALAXY_SCENE-FP_day/20221230-151426",
                "sample_weight": 2,
                "data_path_md5sum": "6a1ebb15ddd9252f5c40942a2653698d",
            },
            {
                "data_path": "dmpv2://matrix2/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/vehicle_detection/pilot_vehicle_5v_CAMERA_GALAXY_SCENE-FP_night/20221230-161307",
                "sample_weight": 2,
                "data_path_md5sum": "d670f81991f986f6454faca875af2356",
            },
        ],
        "val_data_paths": [],
    },
    "rear": {
        "train_data_paths": [
            {
                "data_path": "dmpv2://matrix2/multicam_pilot/data/lmdb_datasets/data/vehicle_rear_detection/matrix_2.1_v3.0.0_train_data/det_vehicle_rear_v3-20190726-train_base/data",
                "sample_weight": 8,
                "data_path_md5sum": "79961b668dc695bb741fb248995fcbc1",
            },
            {
                "data_path": "dmpv2://matrix2/multicam_pilot/data/lmdb_datasets/data/vehicle_rear_detection/matrix_2.1_v3.0.0_train_data/det_vehicle_rear_v3-20190726-train_gray/data",
                "sample_weight": 6,
                "data_path_md5sum": "c734a90882584c649cf96fdbdc3e1ee1",
            },
            {
                "data_path": "dmpv2://matrix2/multicam_pilot/data/lmdb_datasets/data/vehicle_rear_detection/matrix_2.1_v3.0.0_train_data/det_vehicle_rear_v3-20190726-train_night/data",
                "sample_weight": 3,
                "data_path_md5sum": "929c4b99db885064df305ea40d128b6d",
            },
            {
                "data_path": "dmpv2://matrix2/multicam_pilot/data/lmdb_datasets/data/vehicle_rear_detection/matrix_2.1_v3.0.0_train_data/det_vehicle_rear_v6-20200624-train_390_5200_day_FrontRear/data",
                "sample_weight": 2,
                "data_path_md5sum": "c887530baa6cfe78557ae9f26426269b",
            },
            {
                "data_path": "dmpv2://matrix2/multicam_pilot/data/lmdb_datasets/data/vehicle_rear_detection/matrix_2.1_v3.0.0_train_data/det_vehicle_rear_v6-20200624-train_390_5200_day_NonFrontRear/data",
                "sample_weight": 3,
                "data_path_md5sum": "3db8cef65eb16fbb6e090f56703d24bf",
            },
            {
                "data_path": "dmpv2://matrix2/multicam_pilot/data/lmdb_datasets/data/vehicle_rear_detection/matrix_2.1_v3.0.0_train_data/det_vehicle_rear_v6-20200624-train_390_5200_night/data",
                "sample_weight": 2,
                "data_path_md5sum": "c13ec3a5ece49607a30fabb8bce6d2d8",
            },
            {
                "data_path": "dmpv2://matrix2/multicam_pilot/data/lmdb_datasets/mono_0323_data/user/ni.jiang/data/vehicle_rear_data/det_vehicle_rear_v8-20201119-train_part_4_0323_0220_exclude_20200601_exclude_part33/data",
                "sample_weight": 12,
                "data_path_md5sum": "3d224c0955aae4e80c06b2ac4e0410ef",
            },
            {
                "data_path": "dmpv2://matrix2/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/rear_detection/pilot_rear_5v_day_0233/20220119-012241/data",
                "sample_weight": 10,
                "data_path_md5sum": "77bc9e2a9faaf8d84511f1727391fc87",
            },
            {
                "data_path": "dmpv2://matrix2/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/rear_detection/pilot_rear_5v_day_X3C/20220224-113552/data",
                "sample_weight": 1,
                "data_path_md5sum": "9d82b56b10ee15a1b3193939ec177551",
            },
            {
                "data_path": "dmpv2://matrix2/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/rear_detection/pilot_rear_5v_CAMERA_X3C_day/20220315-163248/data",
                "sample_weight": 1,
                "data_path_md5sum": "8f0b5a8af10bfc2bb89dc8f72c99d4fc",
            },
            {
                "data_path": "dmpv2://matrix2/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/rear_detection/pilot_rear_5v_CAMERA_X3C_day/20220401-140046/data",
                "sample_weight": 2,
                "data_path_md5sum": "57565640339bb77da9cef1c4f6618e61",
            },
            {
                "data_path": "dmpv2://matrix2/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/tmp/pilot_rear_5v_AS33_day/20220412-220556/data",
                "sample_weight": 3,
                "data_path_md5sum": "78a6a61cf4b1bce0048ee708d7be5651",
            },
            {
                "data_path": "dmpv2://matrix2/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/tmp/pilot_rear_5v_CW_A23GB_A100_L_big_car_day/20220412-212012/data",
                "sample_weight": 1,
                "data_path_md5sum": "f0b2efa7a78ffafb71af17a6c715a0d5",
            },
            {
                "data_path": "dmpv2://matrix2/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/rear_detection/pilot_rear_5v_PROJECT_CC02_5V_day/20220407-195106/data",
                "sample_weight": 1,
                "data_path_md5sum": "6fd674c12dd96f5f15fb386b4213e5d2",
            },
            {
                "data_path": "dmpv2://matrix2/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/rear_detection/pilot_rear_5v_PROJECT_CC02_5V_day/20220510-214104/data",
                "sample_weight": 1,
                "data_path_md5sum": "390c94df83120e063b1a48063869508a",
            },
            {
                "data_path": "dmpv2://matrix2/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/rear_detection/pilot_rear_5v_CW-A23GB-A114-L_day/20220510-103100/data",
                "sample_weight": 1,
                "data_path_md5sum": "b13c10cacf7a57e948ae71a689141fb2",
            },
            {
                "data_path": "dmpv2://matrix2/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/vehicle_rear/pilot_rear_5v_CO-OX3GB-D100-L_day/20221127-140301/data",
                "sample_weight": 1,
                "data_path_md5sum": "ac09127e2c23e47ac9b5c7080d33d0bb",
            },
            {
                "data_path": "dmpv2://matrix2/multicam_pilot/data/train/rear_detection/CP-OX3GB-D100-L_train_day_Cam_5V/20230317_142227",
                "sample_weight": 1,
                "data_path_md5sum": "81684d56fc592eaad2b13d6a8065dd91",
            },
            {
                "data_path": "dmpv2://matrix2/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/rear_detection/pilot_rear_5v_night_0233/20220118-233709/data",
                "sample_weight": 40,
                "data_path_md5sum": "c18415c455c5eb2b4e1c6ada81729535",
            },
            {
                "data_path": "dmpv2://matrix2/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/rear_detection/pilot_rear_5v_day_X3C/20220119-111048/data",
                "sample_weight": 5,
                "data_path_md5sum": "3d17e494d21d86d130e578cd461dc2f0",
            },
            {
                "data_path": "dmpv2://matrix2/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/rear_detection/pilot_rear_5v_night_X3C/20220119-105348/data",
                "sample_weight": 8,
                "data_path_md5sum": "0c81563c7ab3c04ac99f10ef8710df7e",
            },
            {
                "data_path": "dmpv2://matrix2/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/rear_detection/pilot_rear_5v_night_X3C/20220224-115939/data",
                "sample_weight": 8,
                "data_path_md5sum": "2a67a1546a8e6f00aec7572c01999594",
            },
            {
                "data_path": "dmpv2://matrix2/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/rear_detection/pilot_rear_5v_CAMERA_X3C_night/20220315-164230/data",
                "sample_weight": 1,
                "data_path_md5sum": "ab21a9908b584c1ab1a079f8da462591",
            },
            {
                "data_path": "dmpv2://matrix2/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/rear_detection/pilot_rear_5v_PROJECT_CC02_5V_night/20220407-195117/data",
                "sample_weight": 1,
                "data_path_md5sum": "ae6bc188bf8d099c05df60ad0a9825b8",
            },
            {
                "data_path": "dmpv2://matrix2/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/rear_detection/pilot_rear_5v_CAMERA_CW_A23GB_A100_L_night/20220425-140822/data",
                "sample_weight": 1,
                "data_path_md5sum": "8f075a57a2d29a2eff753c5374e6d2dd",
            },
            {
                "data_path": "dmpv2://matrix2/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/rear_detection/pilot_rear_5v_AS33_night/20220412-213815/data",
                "sample_weight": 1,
                "data_path_md5sum": "548597062169c2561d40e5fc9d57b964",
            },
            {
                "data_path": "dmpv2://matrix2/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/rear_detection/pilot_rear_5v_CW_A23GB_A100_L_big_car_night/20220412-212057/data",
                "sample_weight": 1,
                "data_path_md5sum": "58da63a506d9f43b7b3b343f2cb86e69",
            },
            {
                "data_path": "dmpv2://matrix2/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/rear_detection/pilot_rear_5v_PROJECT_CC02_5V_night/20220510-213158/data",
                "sample_weight": 1,
                "data_path_md5sum": "7085d116b8066877eba6cc95267651ef",
            },
            {
                "data_path": "dmpv2://matrix2/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/rear_detection/pilot_rear_5v_CW-A23GB-A114-L_night/20220510-115328/data",
                "sample_weight": 1,
                "data_path_md5sum": "7a63b3ebbf6aaf2484b12202a7d44993",
            },
            {
                "data_path": "dmpv2://matrix2/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/tmp/pilot_rear_5v_CAMERA_X3C_night/20220621-162154/data",
                "sample_weight": 1,
                "data_path_md5sum": "84baa2e2b9e214f70b3d0a61288915ac",
            },
            {
                "data_path": "dmpv2://matrix2/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/rear/det_vehicle_rear_v10-20211229-10652-split_special_and_BigTrucks/data",
                "sample_weight": 10,
                "data_path_md5sum": "72bcd29d2b60c0665c20cab828390cd2",
            },
            {
                "data_path": "dmpv2://matrix2/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/rear/det_vehicle_rear_v10-20220129-jira_and_close_vehicle_data/data",
                "sample_weight": 10,
                "data_path_md5sum": "f303f35688ac53d64271f6d1c4826406",
            },
            {
                "data_path": "dmpv2://matrix2/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/rear/det_vehicle_rear_v8-0220_0323_10635_bigcar-20210917/data",
                "sample_weight": 10,
                "data_path_md5sum": "18247ba140f6bdfcd6101ce1af73cd7e",
            },
            {
                "data_path": "dmpv2://matrix2/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/vehicle_rear/pilot_rear_5v_CAMERA_CW_A23GB_A114_L_night_glare_0831/20220831-181141/data",
                "sample_weight": 1,
                "data_path_md5sum": "5e189197597098f1a269183ad3000233",
            },
            {
                "data_path": "dmpv2://matrix2/multicam_pilot/data/train/rear_detection/CP-OX3GB-D100-L_train_night_Cam_5V/20230317_140605",
                "sample_weight": 0.5,
                "data_path_md5sum": "ec0ed002e743f892876e71e2b749addf",
            },
            {
                "data_path": "dmpv2://matrix2/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/rear_detection/pilot_rear_5v_PROJECT_C385_5V_PARKING_all/20220524-122038/data",
                "sample_weight": 2,
                "data_path_md5sum": "4565f0e69aef1bbbd57c02233f0d9414",
            },
            {
                "data_path": "dmpv2://matrix2/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/tmp/pilot_vehicle_5v_CAMERA_CW_A23GB_A100_L_day/20220621-011046/data",
                "sample_weight": 4,
                "data_path_md5sum": "167302d30b008df8896c912fbe567121",
            },
            {
                "data_path": "dmpv2://matrix2/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/vehicle_rear/pilot_rear_5v_CO-OX3GB-D100-L_day/20220812-004728/data",
                "sample_weight": 1,
                "data_path_md5sum": "67d7a1d50c634dcf035afcbfccf3260d",
            },
            {
                "data_path": "dmpv2://matrix2/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/vehicle_rear/pilot_rear_5v_CO-OX3GB-D100-L_all/20220908-190201/data",
                "sample_weight": 2,
                "data_path_md5sum": "0030c07a4b3e1e72a16f77159c8c310b",
            },
            {
                "data_path": "dmpv2://matrix2/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/vehicle_rear/pilot_rear_5v_CO-OX3GB-D100-L_all/20221122-121515/data",
                "sample_weight": 1,
                "data_path_md5sum": "6f699d964b3631a82b090c816267f974",
            },
            {
                "data_path": "dmpv2://matrix2/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/vehicle_rear/pilot_rear_5v_CO-OX3GB-D100-L_all/20230127-003401/data",
                "sample_weight": 1,
                "data_path_md5sum": "7704050168322a5389d2c5817b6aca96",
            },
            {
                "data_path": "dmpv2://matrix2/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/kVehRearBBox2D/kVehRearBBox2D_project_galaxy_5v_train_day_Cam_Rear/20220622-150345/data",
                "sample_weight": 1,
                "data_path_md5sum": "6e813c6988c0bce469062902ea2ab62b",
            },
            {
                "data_path": "dmpv2://matrix2/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/kVehRearBBox2D/kVehRearBBox2D_project_galaxy_5v_train_night_Cam_Rear/20220622-171850/data",
                "sample_weight": 1,
                "data_path_md5sum": "e1e88870250fc722ab0fdb79a5c32bb2",
            },
            {
                "data_path": "dmpv2://matrix2/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/kVehRearBBox2D/kVehRearBBox2D_project_galaxy_5v_train_day_Cam_Side/20220622-150745/data",
                "sample_weight": 1,
                "data_path_md5sum": "28e818c01658fae77d721502f07c9931",
            },
            {
                "data_path": "dmpv2://matrix2/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/kVehRearBBox2D/kVehRearBBox2D_project_galaxy_5v_train_night_Cam_Side/20220622-172038/data",
                "sample_weight": 1,
                "data_path_md5sum": "f4c21650c7509f76928fe300bd7d313b",
            },
            {
                "data_path": "dmpv2://matrix2/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/kVehRearBBox2D/kVehRearBBox2D_project_galaxy_5v_train_night_Cam_Rear/20220719-115310/data",
                "sample_weight": 1,
                "data_path_md5sum": "879e170b7cac3288234c52b552e1b72d",
            },
            {
                "data_path": "dmpv2://matrix2/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/kVehRearBBox2D/kVehRearBBox2D_project_galaxy_5v_train_day_Cam_Rear/20220719-114719/data",
                "sample_weight": 2,
                "data_path_md5sum": "4fed6abd8cd00ad182e349fb2c0a1c3a",
            },
            {
                "data_path": "dmpv2://matrix2/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/kVehRearBBox2D/kVehRearBBox2D_project_galaxy_5v_train_day_Cam_Side/20220719-111918/data",
                "sample_weight": 1,
                "data_path_md5sum": "d5277eb04f1b525418c7339aff8b306d",
            },
            {
                "data_path": "dmpv2://matrix2/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/kVehRearBBox2D/kVehRearBBox2D_project_galaxy_5v_train_night_Cam_Side/20220805-224901/data",
                "sample_weight": 1,
                "data_path_md5sum": "c997ea0b874cd64bd95b3c29b49fdf24",
            },
            {
                "data_path": "dmpv2://matrix2/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/kVehRearBBox2D/kVehRearBBox2D_project_galaxy_5v_train_night_Cam_Rear/20220805-230106/data",
                "sample_weight": 2,
                "data_path_md5sum": "b1cbdbefd4b87062ae97956362e3e1a1",
            },
            {
                "data_path": "dmpv2://matrix2/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/kVehRearBBox2D/kVehRearBBox2D_project_galaxy_5v_train_day_Cam_Side/20220805-233211/data",
                "sample_weight": 1,
                "data_path_md5sum": "f629b305559720534933e078ac28d392",
            },
            {
                "data_path": "dmpv2://matrix2/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/kVehRearBBox2D/kVehRearBBox2D_project_galaxy_5v_train_day_Cam_Rear/20220806-000857/data",
                "sample_weight": 2,
                "data_path_md5sum": "3fcc79f0ad20d6a18e3f5cda47c405ed",
            },
            {
                "data_path": "dmpv2://matrix2/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/kVehRearBBox2D/kVehRearBBox2D_project_galaxy_5v_train_day_Cam_Side/20221001-094341/data",
                "sample_weight": 3,
                "data_path_md5sum": "bba0acca10879f5b23443049cc64e0da",
            },
            {
                "data_path": "dmpv2://matrix2/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/kVehRearBBox2D/kVehRearBBox2D_project_galaxy_5v_train_day_Cam_Rear/20221001-093816/data",
                "sample_weight": 2,
                "data_path_md5sum": "137598f6cd5b5c9db365889fd8162d02",
            },
            {
                "data_path": "dmpv2://matrix2/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/kVehRearBBox2D/kVehRearBBox2D_project_galaxy_5v_train_night_Cam_Rear/20221001-093022/data",
                "sample_weight": 1,
                "data_path_md5sum": "18c993491ea5f038ae3d40e61fb46153",
            },
            {
                "data_path": "dmpv2://matrix2/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/kVehRearBBox2D/kVehRearBBox2D_project_galaxy_5v_train_night_Cam_Side/20221001-092532/data",
                "sample_weight": 1,
                "data_path_md5sum": "7fc3511920bfdafc3f089e3d1c76435c",
            },
            {
                "data_path": "dmpv2://matrix2/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/kVehRearBBox2D/kVehRearBBox2D_project_galaxy_5v_train_day_Cam_Side/20221027-205819/data",
                "sample_weight": 1,
                "data_path_md5sum": "a7fd1d2e6234a2917066b5250f3091b9",
            },
            {
                "data_path": "dmpv2://matrix2/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/kVehRearBBox2D/kVehRearBBox2D_project_galaxy_5v_train_night_Cam_Side/20221027-205945/data",
                "sample_weight": 1,
                "data_path_md5sum": "451f1c54940c1db1709f6d3b4a3854a1",
            },
            {
                "data_path": "dmpv2://matrix2/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/kVehRearBBox2D/kVehRearBBox2D_project_galaxy_5v_train_day_Cam_Rear/20221027-210957/data",
                "sample_weight": 1,
                "data_path_md5sum": "5a2cb4baf2087aa4e433bf656c7c1ded",
            },
            {
                "data_path": "dmpv2://matrix2/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/kVehRearBBox2D/kVehRearBBox2D_project_galaxy_5v_train_night_Cam_Rear/20221027-211254/data",
                "sample_weight": 1,
                "data_path_md5sum": "315512446538885de8a24a9a1d9e1819",
            },
        ],
        "val_data_paths": [],
    },
    "person": {
        "train_data_paths": [
            {
                "data_path": "dmpv2://matrix2/multicam_pilot/data/lmdb_datasets/data/pedestrain_detection/majiajia.data/rare_pose_exp/Pilot_person_5v_CW-A23GB-A100-L_day/20220602-180009/data",
                "sample_weight": 10,
                "data_path_md5sum": "63216b1c37fef8795c7ce9753853bbcf",
            },
            {
                "data_path": "dmpv2://matrix2/multicam_pilot/data/lmdb_datasets/data/pedestrain_detection/majiajia.data/CC02_V5/Pilot_person_5v_CO-OX3GB-A100-L_day/20220611-170110/data",
                "sample_weight": 5,
                "data_path_md5sum": "443cd1971956a2753214cf4c55599b7b",
            },
            {
                "data_path": "dmpv2://matrix2/multicam_pilot/data/lmdb_datasets/data/pedestrain_detection/han.shen.data/ped_data/ped_20181210_train_12cam_update",
                "sample_weight": 5,
                "data_path_md5sum": "201ada8693bc01c1f6971cd7f043cdf2",
            },
            {
                "data_path": "dmpv2://matrix2/multicam_pilot/data/lmdb_datasets/data/pedestrain_detection/han.shen.data/ped_data/ped_20190605_train_part_2_update",
                "sample_weight": 4,
                "data_path_md5sum": "76c0dc8a4a3b9bded4590e3a9ecc9f61",
            },
            {
                "data_path": "dmpv2://matrix2/multicam_pilot/data/lmdb_datasets/data/pedestrain_detection/han.shen.data/ped_data/ped_20190918",
                "sample_weight": 2,
                "data_path_md5sum": "fa490ef60a652b46059795f2ed76cc5e",
            },
            {
                "data_path": "dmpv2://matrix2/multicam_pilot/data/lmdb_datasets/data/pedestrain_detection/zhengwei.data/ped_data/ped_20200613_cn_390_person_tain_history",
                "sample_weight": 5,
                "data_path_md5sum": "1ea8520bef690a2fa3ee48d219de01d9",
            },
            {
                "data_path": "dmpv2://matrix2/multicam_pilot/data/lmdb_datasets/data/pedestrain_detection/han.shen.data/ped_data/ped_street_people_data",
                "sample_weight": 2,
                "data_path_md5sum": "03e27325e4dc8ae69157a9ad660cc663",
            },
            {
                "data_path": "dmpv2://matrix2/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/pedestrain_detection/20210915_sensor0233_train_id_4218-9655_part6_day_upsample3_gt63_remove_empty/train",
                "sample_weight": 20,
                "data_path_md5sum": "09e266c95c2b6991202f877c0d07bade",
            },
            {
                "data_path": "dmpv2://matrix2/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/pedestrain_detection/20211119-153000-big-person/data",
                "sample_weight": 10,
                "data_path_md5sum": "4052a4f7f3d17a4ebcc6b841eba78d7a",
            },
            {
                "data_path": "dmpv2://matrix2/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/pedestrain_detection/Pilot_person_5v_CAMERA_CW_A23GB_A100_L_day_big_person/20220620-201707/data",
                "sample_weight": 2,
                "data_path_md5sum": "4856a7fedd4515aa7094e52c25de1bf7",
            },
            {
                "data_path": "dmpv2://matrix2/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/pedestrain_detection/20220214-031030-0233-toll-gate/data",
                "sample_weight": 20,
                "data_path_md5sum": "92604eb87b5ee5d86628e89ee4ca88a3",
            },
            {
                "data_path": "dmpv2://matrix2/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/pedestrain_detection/20220329-151248-x3c-tunnel/data",
                "sample_weight": 8,
                "data_path_md5sum": "b28ee79b81d0c0e73e8faba2d5c3e2b2",
            },
            {
                "data_path": "dmpv2://matrix2/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/pedestrain_detection/tunnel_ped_by_zhongpeng/data",
                "sample_weight": 8,
                "data_path_md5sum": "9a385035c6e40384dfd01daa1eec58d9",
            },
            {
                "data_path": "dmpv2://matrix2/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/pedestrain_detection/Pilot_person_5v_CAMERA_CW_A23GB_A114_L_day_5w/20220804-122458/data",
                "sample_weight": 24,
                "data_path_md5sum": "6602380aa25f432f643d46ae092a8d22",
            },
            {
                "data_path": "dmpv2://matrix2/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/pedestrain_detection/20220121-192622-x3c-all/data",
                "sample_weight": 20,
                "data_path_md5sum": "880048a494176e11b0ceb6625a84cd16",
            },
            {
                "data_path": "dmpv2://matrix2/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/pedestrain_detection/20220510/person/day/20220511-142910/data",
                "sample_weight": 20,
                "data_path_md5sum": "e3349c51d85ef74bfd00f7f3b7e6e053",
            },
            {
                "data_path": "dmpv2://matrix2/multicam_pilot/data/lmdb_datasets/data/pedestrain_detection/majiajia.data/NonTarget_exp/exp5/Pilot_person_5v__all/20220708-063813/data",
                "sample_weight": 20,
                "data_path_md5sum": "eb32d5ab9f15d8b7d47eafc54f7cbd46",
            },
            {
                "data_path": "dmpv2://matrix2/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/kPedBBox2D/kPedBBox2D_project_c385_5v_CO-OX3GB-D100-L_train_day_Cam_5V/20220907-200310/data",
                "sample_weight": 2,
                "data_path_md5sum": "c735129c1b39f5416458420b3a0864d4",
            },
            {
                "data_path": "dmpv2://matrix2/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/pedestrain_detection/fp/Veh_Head/Pilot_person_5v_CW-A23GB-A114-L_day_fp/20220926-152912/data",
                "sample_weight": 20,
                "data_path_md5sum": "3cda06ad6d0ab6418edd2fe3d39abd3e",
            },
            {
                "data_path": "dmpv2://matrix2/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/kPedBBox2D/kPedBBox2D_project_c385_5v_CO-OX3GB-D100-L_train_day_Cam_5V/20221011-223223/data",
                "sample_weight": 10,
                "data_path_md5sum": "13ef102bf3deacdc241ddca8d82d53e7",
            },
            {
                "data_path": "dmpv2://matrix2/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/kPedBBox2D/kPedBBox2D_project_c385_5v_CO-OX3GB-D100-L_train_day_Jira_Cam_5V/20221014-201823/data",
                "sample_weight": 1,
                "data_path_md5sum": "4cce7f6eedf6f6157a20a886a14f819a",
            },
            {
                "data_path": "dmpv2://matrix2/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/kPedBBox2D/Pilot_person_5v_all_all_Child/20221015-081347/data",
                "sample_weight": 6,
                "data_path_md5sum": "ae34edc3413327d6692ba90d62e818c3",
            },
            {
                "data_path": "dmpv2://matrix2/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/kPedBBox2D/kPedBBox2D_project_c385_5v_CO-OX3GB-D100-L_train_day_Cam_5V/20221124-233317/data",
                "sample_weight": 3,
                "data_path_md5sum": "0db5429fd18c6abd6e02515b9a84606b",
            },
            {
                "data_path": "dmpv2://matrix2/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/kPedBBox2D/kPedBBox2D_project_c385_5v_CO-OX3GB-D100-L_train_day_Cam_5V/20221127-201047/data",
                "sample_weight": 10,
                "data_path_md5sum": "97cfc68aa3cac37df933894ab3f0cb31",
            },
            {
                "data_path": "dmpv2://matrix2/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/kPedBBox2D/kPedBBox2D_project_c385_5v_CO-OX3GB-D100-L_train_day_Jira_Cam_5V/20221127-194559/data",
                "sample_weight": 1,
                "data_path_md5sum": "9aac46cc193dec85cafc9de1673254f6",
            },
            {
                "data_path": "dmpv2://matrix2/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/kPedBBox2D/kPedBBox2D_project_c385_5v_CO-OX3GB-D100-L_train_day_Jira_Cam_5V/20230126-144504/data",
                "sample_weight": 10,
                "data_path_md5sum": "5d9257ed9be6841657c57ec4fda3dd2a",
            },
            {
                "data_path": "dmpv2://matrix2/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/kPedBBox2D/kPedBBox2D_project_c385_5v_CO-OX3GB-D100-L_train_day_Jira_Cam_5V/20230126-145220/data",
                "sample_weight": 5,
                "data_path_md5sum": "dd3c36985bc4a6cee17577787966a31d",
            },
            {
                "data_path": "dmpv2://matrix2/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/kPedBBox2D/kPedBBox2D_project_c385_5v_CO-OX3GB-D100-L_train_day_Cam_5V/20230126-161618/data",
                "sample_weight": 20,
                "data_path_md5sum": "0d36bf6873c83319f3f33af1cc68bb86",
            },
            {
                "data_path": "dmpv2://matrix2/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/kPedBBox2D/kPedBBox2D_project_c385_5v_CO-OX3GB-D100-L_train_day_Cam_5V/20230126-160449/data",
                "sample_weight": 10,
                "data_path_md5sum": "e625d84b641559a832ffd1e1a4987eaa",
            },
            {
                "data_path": "dmpv2://matrix2/multicam_pilot/data/train/person_detection/CP-OX3GB-D100-L_train_day_Cam_5V/20230317_195129",
                "sample_weight": 1,
                "data_path_md5sum": "90ca17b36d26d2a6840c1043207a2676",
            },
            {
                "data_path": "dmpv2://matrix2/multicam_pilot/data/train/person_detection/project_maxfa_5v_CP-OX3GB-D100-L_train_day_Cam_5V/20230313_142610",
                "sample_weight": 1,
                "data_path_md5sum": "34b0bb99299d7558144580d88a1b877a",
            },
            {
                "data_path": "dmpv2://matrix2/multicam_pilot/data/train/person_detection/project_cc02_5v_train_day_Cam_5V/20230313_142957",
                "sample_weight": 3,
                "data_path_md5sum": "b1ff073d8ccd9250767abdd6ceab08e7",
            },
            {
                "data_path": "dmpv2://matrix2/multicam_pilot/data/train/person_detection/project_as33_5v_train_day_Cam_5V/20230313_143511",
                "sample_weight": 4,
                "data_path_md5sum": "c65fa6fea0575e09e3991b7b49e3f04b",
            },
            {
                "data_path": "dmpv2://matrix2/multicam_pilot/data/lmdb_datasets/data/pedestrain_detection/majiajia.data/CC02_V5/Pilot_person_5v_CO-OX3GB-A100-L_night/20220611-164225/data",
                "sample_weight": 5,
                "data_path_md5sum": "40c6f64bc09ed5142d789c01e9f17ceb",
            },
            {
                "data_path": "dmpv2://matrix2/multicam_pilot/data/lmdb_datasets/data/pedestrain_detection/han.shen.data/ped_data/ped_20180814_train_gray_update",
                "sample_weight": 3,
                "data_path_md5sum": "0e2b1f6d6bf84bac53e29989032b6d5d",
            },
            {
                "data_path": "dmpv2://matrix2/multicam_pilot/data/lmdb_datasets/mono_0323_data/user/kaijun01.zhang/data/data_2020/0323_0220_ped_20201119/0323_0220_ped_20201119",
                "sample_weight": 3,
                "data_path_md5sum": "d9e3663404c378ce08e8316a8dc5973c",
            },
            {
                "data_path": "dmpv2://matrix2/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/pedestrain_detection/20220510/person/night/20220511-124631/data",
                "sample_weight": 20,
                "data_path_md5sum": "e0df8d731a461004a5faa5a1424a78af",
            },
            {
                "data_path": "dmpv2://matrix2/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/pedestrain_detection/Pilot_person_5v_CAMERA_CW_A23GB_A114_L_day_5w/20220826-121734/data",
                "sample_weight": 24,
                "data_path_md5sum": "41af26da1694b2f7708ff44d89544389",
            },
            {
                "data_path": "dmpv2://matrix2/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/pedestrain_detection/Pilot_person_5v_CAMERA_CW_A23GB_A100_L_night_big_person/20220620-201624/data",
                "sample_weight": 1,
                "data_path_md5sum": "ecc9737085b14d3eef734a0a8e1ef195",
            },
            {
                "data_path": "dmpv2://matrix2/multicam_pilot/data/lmdb_datasets/data/pedestrain_detection/majiajia.data/NonTarget_exp/exp6/Pilot_person_5v_all_night/20220713-233017/data",
                "sample_weight": 20,
                "data_path_md5sum": "c67a873a844db484ac7aa102bf483d3b",
            },
            {
                "data_path": "dmpv2://matrix2/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/kPedBBox2D/kPedBBox2D_project_c385_5v_CO-OX3GB-D100-L_train_night_Cam_5V/20221128-104636/data",
                "sample_weight": 3,
                "data_path_md5sum": "6be761a998ae0705c557e0eabbacfa30",
            },
            {
                "data_path": "dmpv2://matrix2/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/kPedBBox2D/kPedBBox2D_project_c385_5v_CO-OX3GB-D100-L_train_night_Jira_Cam_5V/20230126-152323/data",
                "sample_weight": 5,
                "data_path_md5sum": "805fea11f7554aff30f210783e787183",
            },
            {
                "data_path": "dmpv2://matrix2/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/kPedBBox2D/kPedBBox2D_project_c385_5v_CO-OX3GB-D100-L_train_night_Cam_5V/20230126-162729/data",
                "sample_weight": 10,
                "data_path_md5sum": "a08f49a632f1d8e05c28eb6c6fef2cfa",
            },
            {
                "data_path": "dmpv2://matrix2/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/kPedBBox2D/kPedBBox2D_project_AS33_glare_fp_train_night/20230130-121601/data",
                "sample_weight": 4,
                "data_path_md5sum": "9a2c0802fc676eec2b64eae59ca08758",
            },
            {
                "data_path": "dmpv2://matrix2/multicam_pilot/data/train/person_detection/CP-OX3GB-D100-L_train_night_Cam_5V/20230317_195254",
                "sample_weight": 1,
                "data_path_md5sum": "78ed69e02cfeb510a3a06e79281e0b97",
            },
            {
                "data_path": "dmpv2://matrix2/multicam_pilot/data/lmdb_datasets/data/pedestrain_detection/majiajia.data/C385_V1/Pilot_person_5v_CAMERA_CO_OX3GB_D100_L_all/20220621-003109/data",
                "sample_weight": 20,
                "data_path_md5sum": "a0a58dceea7a02c5f9ed57ef185bf27e",
            },
            {
                "data_path": "dmpv2://matrix2/multicam_pilot/data/lmdb_datasets/data/pedestrain_detection/majiajia.data/C385_v05/Pilot_person_5v_CAMERA_CO_OX3GB_D100_L_day/20220523-232136/data",
                "sample_weight": 20,
                "data_path_md5sum": "547e7bf468be2ac6922199b452cda46e",
            },
            {
                "data_path": "dmpv2://matrix2/multicam_pilot/data/lmdb_datasets/data/pedestrain_detection/majiajia.data/C385_v05/Pilot_person_5v_CAMERA_CO_OX3GB_D100_L_night/20220523-231054/data",
                "sample_weight": 20,
                "data_path_md5sum": "65bb0d638b39f64851a6db9b5abc3020",
            },
            {
                "data_path": "dmpv2://matrix2/multicam_pilot/data/lmdb_datasets/data/pedestrain_detection/jianhang.he.data/C385_V2/Pilot_person_5v_CO_OX3GB_D100_L_all/20220805-205614/data",
                "sample_weight": 20,
                "data_path_md5sum": "875d573b6c609f56a6c6b7686152ed73",
            },
            {
                "data_path": "dmpv2://matrix2/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/kPedBBox2D/kPedBBox2D_project_c385_5v_CO-OX3GB-D100-L_train_underground_park_Cam_5V/20221014-155712/data",
                "sample_weight": 10,
                "data_path_md5sum": "180471bd68e40472f197709564182282",
            },
            {
                "data_path": "dmpv2://matrix2/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/kPedBBox2D/kPedBBox2D_project_c385_5v_CO-OX3GB-D100-L_train_underground_park_Cam_5V/20220908-135320/data",
                "sample_weight": 10,
                "data_path_md5sum": "e7f46e7210196b909f90dbd300ce892c",
            },
            {
                "data_path": "dmpv2://matrix2/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/kPedBBox2D/Pilot_person_5v_all_all_Child/20221012-170449/data",
                "sample_weight": 5,
                "data_path_md5sum": "dd2294050fdb669e8931d6e4b6ec86b6",
            },
            {
                "data_path": "dmpv2://matrix2/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/kPedBBox2D/kPedBBox2D_project_c385_5v_CO-OX3GB-D100-L_train_parking_Jira_Cam_5V/20221014-213958/data",
                "sample_weight": 1,
                "data_path_md5sum": "6a9e52c966200b0eb5f355211acdbd28",
            },
            {
                "data_path": "dmpv2://matrix2/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/kPedBBox2D/kPedBBox2D_project_c385_5v_CO-OX3GB-D100-L_train_underground_park_Cam_5V/20221124-220020/data",
                "sample_weight": 10,
                "data_path_md5sum": "16d5687bde41bc62c048513daca844af",
            },
            {
                "data_path": "dmpv2://matrix2/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/kPedBBox2D/kPedBBox2D_project_c385_5v_CO-OX3GB-D100-L_train_underground_park_Cam_5V/20221124-222923/data",
                "sample_weight": 10,
                "data_path_md5sum": "02e55095580277d8650ce3f3296e2801",
            },
            {
                "data_path": "dmpv2://matrix2/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/kPedBBox2D/kPedBBox2D_project_c385_5v_CO-OX3GB-D100-L_train_parking_Jira_Cam_5V/20221127-194944/data",
                "sample_weight": 1,
                "data_path_md5sum": "fee2ee5886cebfa1dbcabd8d028895f6",
            },
            {
                "data_path": "dmpv2://matrix2/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/kPedBBox2D/kPedBBox2D_project_c385_5v_CO-OX3GB-D100-L_train_parking_Jira_Cam_5V/20230126-151952/data",
                "sample_weight": 5,
                "data_path_md5sum": "2718c50a0ad34abe4cf2d33e5947ce83",
            },
            {
                "data_path": "dmpv2://matrix2/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/kPedBBox2D/kPedBBox2D_project_c385_5v_CO-OX3GB-D100-L_train_parking_Cam_5V/20230126-154426/data",
                "sample_weight": 10,
                "data_path_md5sum": "2c6cab4cad527e9f8528aa3690507483",
            },
            {
                "data_path": "dmpv2://matrix2/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/kPedBBox2D/kPedBBox2D_project_galaxy_5v_train_day_Cam_Rear/20220622-145832/data",
                "sample_weight": 1,
                "data_path_md5sum": "87970e25457018a87a334f12a75e499c",
            },
            {
                "data_path": "dmpv2://matrix2/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/kPedBBox2D/kPedBBox2D_project_galaxy_5v_train_night_Cam_Rear/20220622-145002/data",
                "sample_weight": 1,
                "data_path_md5sum": "9241a53741bf0c7a0a5bfe1b117308eb",
            },
            {
                "data_path": "dmpv2://matrix2/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/kPedBBox2D/kPedBBox2D_project_galaxy_5v_train_day_Cam_Side/20220622-150324/data",
                "sample_weight": 1,
                "data_path_md5sum": "528cc90fa74c44c8e458e157169d13a7",
            },
            {
                "data_path": "dmpv2://matrix2/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/kPedBBox2D/kPedBBox2D_project_galaxy_5v_train_night_Cam_Side/20220622-150103/data",
                "sample_weight": 1,
                "data_path_md5sum": "0789f8d0adc5db8742f7035b263a97de",
            },
            {
                "data_path": "dmpv2://matrix2/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/kPedBBox2D/kPedBBox2D_project_galaxy_5v_train_day_Cam_Rear/20220719-114353/data",
                "sample_weight": 2,
                "data_path_md5sum": "3540a5ce85ea3dfe3c1e2b4d8f133909",
            },
            {
                "data_path": "dmpv2://matrix2/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/kPedBBox2D/kPedBBox2D_project_galaxy_5v_train_day_Cam_Side/20220719-102053/data",
                "sample_weight": 1,
                "data_path_md5sum": "d851e38e3f61bfd3fb0832d01e60d2ab",
            },
            {
                "data_path": "dmpv2://matrix2/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/kPedBBox2D/kPedBBox2D_project_galaxy_5v_train_day_Cam_Side/20220805-235334/data",
                "sample_weight": 2,
                "data_path_md5sum": "dfc9fea198312c84519cfb79e9b0dbb9",
            },
            {
                "data_path": "dmpv2://matrix2/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/kPedBBox2D/kPedBBox2D_project_galaxy_5v_train_day_Cam_Rear/20220805-234732/data",
                "sample_weight": 2,
                "data_path_md5sum": "eaa1aa18a54985cc78d041be9635a275",
            },
            {
                "data_path": "dmpv2://matrix2/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/kPedBBox2D/kPedBBox2D_project_galaxy_5v_train_day_Cam_Side/kPedBBox2D_project_galaxy_5v_train_day_Cam_Side/20220928-171508/data",
                "sample_weight": 2,
                "data_path_md5sum": "d066f8a1d52833abfcb5a8ff6f5c63e9",
            },
            {
                "data_path": "dmpv2://matrix2/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/kPedBBox2D/kPedBBox2D_project_galaxy_5v_train_night_Cam_Side/kPedBBox2D_project_galaxy_5v_train_night_Cam_Side/20220928-155340/data",
                "sample_weight": 2,
                "data_path_md5sum": "b67085cad26588ce50855c808ce26b18",
            },
            {
                "data_path": "dmpv2://matrix2/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/kPedBBox2D/kPedBBox2D_project_galaxy_5v_train_day_Cam_Rear/kPedBBox2D_project_galaxy_5v_train_day_Cam_Rear/20220928-170750/data",
                "sample_weight": 2,
                "data_path_md5sum": "522ffe2736f20dbc5547d177c36efc0e",
            },
            {
                "data_path": "dmpv2://matrix2/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/kPedBBox2D/kPedBBox2D_project_galaxy_5v_train_night_Cam_Rear/kPedBBox2D_project_galaxy_5v_train_night_Cam_Rear/20220928-155540/data",
                "sample_weight": 2,
                "data_path_md5sum": "46b1b1339e06bee856bd0c8ac32f13fb",
            },
            {
                "data_path": "dmpv2://matrix2/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/kPedBBox2D/kPedBBox2D_project_galaxy_5v_train_day_Cam_Side/20221028-213019/data",
                "sample_weight": 2,
                "data_path_md5sum": "33758ca32c8aa563072a3856aed5c941",
            },
            {
                "data_path": "dmpv2://matrix2/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/kPedBBox2D/kPedBBox2D_project_galaxy_5v_train_day_Cam_Rear/20221028-215546/data",
                "sample_weight": 2,
                "data_path_md5sum": "ebebfba24bfdd2e02a442989a0dafa50",
            },
            {
                "data_path": "dmpv2://matrix2/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/kPedBBox2D/kPedBBox2D_project_galaxy_5v_train_night_Cam_Side/20221028-225802/data",
                "sample_weight": 1,
                "data_path_md5sum": "d1e4b14c862b3e3932c6d285d7c1df14",
            },
            {
                "data_path": "dmpv2://matrix2/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/kPedBBox2D/kPedBBox2D_project_galaxy_5v_train_night_Cam_Rear/20221028-230726/data",
                "sample_weight": 1,
                "data_path_md5sum": "d22c3b68a0b3b3a83d167885dbd529c1",
            },
            {
                "data_path": "dmpv2://matrix2/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/kPedBBox2D/kPedBBox2D_project_galaxy_5v_train_all_Cam_Rear/20221119-132747/data",
                "sample_weight": 0.25,
                "data_path_md5sum": "a44150d834fac7978ca3eee931b47942",
            },
            {
                "data_path": "dmpv2://matrix2/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/kPedBBox2D/kPedBBox2D_project_galaxy_5v_train_all_Cam_Side/20221119-134707/data",
                "sample_weight": 0.25,
                "data_path_md5sum": "5b795e8950447f22e063e305bcc08e9b",
            },
            {
                "data_path": "dmpv2://matrix2/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/kPedBBox2D/kPedBBox2D_project_galaxy_5v_train_day_JIRA_badcase_Cam_All/20221119-142245/data",
                "sample_weight": 1,
                "data_path_md5sum": "32d1e883aaccaf46563e49243fac4867",
            },
            {
                "data_path": "dmpv2://matrix2/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/kPedBBox2D/kPedBBox2D_project_galaxy_5v_train_night_JIRA_badcase_Cam_All/20221119-142917/data",
                "sample_weight": 1.5,
                "data_path_md5sum": "6820de3b4467c30e1c832cefa07af193",
            },
            {
                "data_path": "dmpv2://matrix2/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/kPedBBox2D/kPedBBox2D_project_galaxy_5v_train_all_Cam_Side/20221212-132212/data",
                "sample_weight": 1,
                "data_path_md5sum": "5368f971d6bb569fcedc352461e536ad",
            },
            {
                "data_path": "dmpv2://matrix2/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/kPedBBox2D/kPedBBox2D_project_galaxy_5v_train_all_Cam_Rear/20221212-132630/data",
                "sample_weight": 1,
                "data_path_md5sum": "8bafa7efa4363e9107a5deaeed16d7dc",
            },
            {
                "data_path": "dmpv2://matrix2/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/kPedBBox2D/kPedBBox2D_project_galaxy_5v_CO-OX3GB-D100-L_train_night_jira_Cam_5V/20230128-143411/data",
                "sample_weight": 4,
                "data_path_md5sum": "09ed1f0df7c0fe0af7f59d06b11dd7ea",
            },
            {
                "data_path": "dmpv2://matrix2/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/kPedBBox2D/kPedBBox2D_project_galaxy_train_night_Cam_Rear/20230130-211620/data",
                "sample_weight": 3,
                "data_path_md5sum": "959f0df12bcc7fdeadfe116016ad9a0f",
            },
            {
                "data_path": "dmpv2://matrix2/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/kPedBBox2D/kPedBBox2D_project_galaxy_train_night_Cam_Rear/20230130-221528/data",
                "sample_weight": 1,
                "data_path_md5sum": "f1329b291b7bcad8652a002b08fda916",
            },
        ],
        "val_data_paths": [],
    },
    "cyclist": {
        "train_data_paths": [
            {
                "data_path": "dmpv2://matrix2/multicam_pilot/data/lmdb_datasets/data/cyclist_detection/majiajia.data/CC02_V5.0/Pilot_cyclist_5v_CO-OX3GB-A100-L_day/20220611-151613/data",
                "sample_weight": 6,
                "data_path_md5sum": "07beb4e6e0c26399821dc040a18b8e2c",
            },
            {
                "data_path": "dmpv2://matrix2/multicam_pilot/data/lmdb_datasets/data/cyclist_detection/han.shen.data/cyc_data/20190522-remove-truck/20190522-remove-truck",
                "sample_weight": 6,
                "data_path_md5sum": "fd1bbb05930e5d0422d28d1475108aa0",
            },
            {
                "data_path": "dmpv2://matrix2/multicam_pilot/data/lmdb_datasets/data/cyclist_detection/han.shen.data/cyc_data/cyc_20190918",
                "sample_weight": 4,
                "data_path_md5sum": "306f96e224ae1331472cd1fe94c3c04f",
            },
            {
                "data_path": "dmpv2://matrix2/multicam_pilot/data/lmdb_datasets/data/cyclist_detection/zhengwei.data/cyc_data/cyc_20200613_cn_390_person_tain_history",
                "sample_weight": 6,
                "data_path_md5sum": "3796277e7aeb6d0460fa8d67ac9016fc",
            },
            {
                "data_path": "dmpv2://matrix2/multicam_pilot/data/lmdb_datasets/mono_0323_data/user/kaijun01.zhang/data/data_2020/0323_10635_quad_cyc_20200723/0323_10635_quad_cyc_20200723",
                "sample_weight": 4,
                "data_path_md5sum": "e301f3a723b996aa6a7d22f5814c98fd",
            },
            {
                "data_path": "dmpv2://matrix2/multicam_pilot/data/lmdb_datasets/mono_0323_data/user/kaijun01.zhang/data/data_2020/0323_0220_cyc_20201119/0323_0220_cyc_20201119",
                "sample_weight": 4,
                "data_path_md5sum": "1ac773491f6005aa2b66102a42f73617",
            },
            {
                "data_path": "dmpv2://matrix2/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/cyclist_detection/20210915_sensor0233_train_id_4218-9655_part6_day_upsample3_gt63_remove_empty_up300pix_30_heavyocc_normal/train",
                "sample_weight": 20,
                "data_path_md5sum": "6fc3bd80d973da660781d72bbf60ed79",
            },
            {
                "data_path": "dmpv2://matrix2/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/cyclist_detection/20220121-190855-x3c-all/data",
                "sample_weight": 20,
                "data_path_md5sum": "88d149062ca6c65a65b0028bf15ff1fa",
            },
            {
                "data_path": "dmpv2://matrix2/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/pedestrain_detection/20220510/cyclist/day/20220511-142409/data",
                "sample_weight": 30,
                "data_path_md5sum": "718790466bb70f2fabec117a04bb5dba",
            },
            {
                "data_path": "dmpv2://matrix2/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/cyclist_detection/Pilot_cyclist_5v_CAMERA_CW_A23GB_A114_L_day_5w/20220804-134412/data",
                "sample_weight": 24,
                "data_path_md5sum": "e32b5389b3fd54a65286896ce2a328c0",
            },
            {
                "data_path": "dmpv2://matrix2/multicam_pilot/data/lmdb_datasets/data/cyclist_detection/majiajia.data/NonTarget_exp/exp5/Pilot_cyclist_5v__all/20220712-043858/data",
                "sample_weight": 20,
                "data_path_md5sum": "951106fbe27a453dd5180352d3b6417e",
            },
            {
                "data_path": "dmpv2://matrix2/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/kCycBBox2D/kCycBBox2D_project_c385_5v_CO-OX3GB-D100-L_train_day_Cam_5V/20220907-200731/data",
                "sample_weight": 2,
                "data_path_md5sum": "2001d8e5a514867c7a267f81dbe5b8ae",
            },
            {
                "data_path": "dmpv2://matrix2/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/cyclist_detection/fp/Veh_Head/Pilot_cyclist_5v_CW-A23GB-A114-L_day_fp/20220926-150141/data",
                "sample_weight": 20,
                "data_path_md5sum": "2573dc55bcfa7f8b5237a04ce904caf9",
            },
            {
                "data_path": "dmpv2://matrix2/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/kCycBBox2D/kCycBBox2D_project_c385_5v_CO-OX3GB-D100-L_train_day_Cam_5V/20221011-224648/data",
                "sample_weight": 10,
                "data_path_md5sum": "272653458f167a7fcb2c7d947bdaa232",
            },
            {
                "data_path": "dmpv2://matrix2/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/kCycBBox2D/kCycBBox2D_project_c385_5v_CO-OX3GB-D100-L_train_day_Cam_5V/20221124-230709/data",
                "sample_weight": 3,
                "data_path_md5sum": "7959c8beab252acf8c463820da22311d",
            },
            {
                "data_path": "dmpv2://matrix2/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/kCycBBox2D/kCycBBox2D_project_c385_5v_CO-OX3GB-D100-L_train_day_Cam_5V/20221127-200817/data",
                "sample_weight": 3,
                "data_path_md5sum": "cdb7071087c436e9944465c63e9c0ba9",
            },
            {
                "data_path": "dmpv2://matrix2/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/kCycBBox2D/kCycBBox2D_project_c385_5v_CO-OX3GB-D100-L_train_day_Jira_Cam_5V/20221014-200602/data",
                "sample_weight": 1,
                "data_path_md5sum": "058e123f7b26b0702e80f9fbd677edf9",
            },
            {
                "data_path": "dmpv2://matrix2/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/kCycBBox2D/kCycBBox2D_project_c385_5v_CO-OX3GB-D100-L_train_day_Jira_Cam_5V/20221127-194742/data",
                "sample_weight": 1,
                "data_path_md5sum": "88c5f2dc22f89a773e69c17fcd2885a2",
            },
            {
                "data_path": "dmpv2://matrix2/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/kCycBBox2D/kCycBBox2D_project_c385_5v_CO-OX3GB-D100-L_train_day_Jira_Cam_5V/20230126-144639/data",
                "sample_weight": 2,
                "data_path_md5sum": "67be3d21127f86a10777124655d7bd25",
            },
            {
                "data_path": "dmpv2://matrix2/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/kCycBBox2D/kCycBBox2D_project_c385_5v_CO-OX3GB-D100-L_train_day_Cam_5V/20230126-161858/data",
                "sample_weight": 10,
                "data_path_md5sum": "e74f25553c86e3ab9bc84500704025a6",
            },
            {
                "data_path": "dmpv2://matrix2/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/kCycBBox2D/kCycBBox2D_project_c385_5v_CO-OX3GB-D100-L_train_day_Cam_5V/20230126-161135/data",
                "sample_weight": 10,
                "data_path_md5sum": "3fd5990ca2067dd24a8c2d32cc28f1d8",
            },
            {
                "data_path": "dmpv2://matrix2/multicam_pilot/data/train/cyclist_detection/CP-OX3GB-D100-L_train_day_Cam_5V/20230317_195457",
                "sample_weight": 1,
                "data_path_md5sum": "53267bf4c4a34a49c9d86d02fb4babb2",
            },
            {
                "data_path": "dmpv2://matrix2/multicam_pilot/data/lmdb_datasets/data/cyclist_detection/majiajia.data/CC02_V5/Pilot_cyclist_5v_CO-OX3GB-A100-L_night/20220611-163932/data",
                "sample_weight": 40,
                "data_path_md5sum": "ae374243d4f3f0017c8e069387a43a84",
            },
            {
                "data_path": "dmpv2://matrix2/multicam_pilot/data/lmdb_datasets/data/cyclist_detection/majiajia.data/MaxFaV10.0/Cyclist_lighted_night/Pilot_cyclist_5v_MaxFaV10_Cyclist_Lighted_Night_night/20220425-182351/data",
                "sample_weight": 40,
                "data_path_md5sum": "3bab35defc58eb9809eee51aa4238483",
            },
            {
                "data_path": "dmpv2://matrix2/multicam_pilot/data/lmdb_datasets/data/cyclist_detection/majiajia.data/cyc_light_CW_A23GB_A100_L/pilot_cyclist_packing_data_publish/20220330-211259/data",
                "sample_weight": 40,
                "data_path_md5sum": "0807fd89d2b98702c2531531e85568d4",
            },
            {
                "data_path": "dmpv2://matrix2/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/pedestrain_detection/20220510/cyclist/night/20220511-124542/data",
                "sample_weight": 20,
                "data_path_md5sum": "9358e5e9183430032eee96a2e5bf7aca",
            },
            {
                "data_path": "dmpv2://matrix2/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/cyclist_detection/Pilot_cyclist_5v_CAMERA_CW_A23GB_A114_L_night/20220624-203955/data",
                "sample_weight": 20,
                "data_path_md5sum": "f970ed45e36e9ed7fcfdcc763aab5bab",
            },
            {
                "data_path": "dmpv2://matrix2/multicam_pilot/data/lmdb_datasets/data/cyclist_detection/majiajia.data/NonTarget_exp/exp6/Pilot_cyclist_5v_all_night/20220713-225739/data",
                "sample_weight": 20,
                "data_path_md5sum": "b466bbb796dd08948137fe1f838fff42",
            },
            {
                "data_path": "dmpv2://matrix2/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/cyclist_detection/glare/lighted_cyclist/debug/Pilot_cyclist_5v_all_night_glare/20220923-222554/data",
                "sample_weight": 60,
                "data_path_md5sum": "c584286bf7b4f1c0366c39db029d5a37",
            },
            {
                "data_path": "dmpv2://matrix2/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/person/cyclist_detection/glare/Pilot_cyclist_5v_CAMERA_CW_A23GB_A100_L_Night_night/20220929-145754/data",
                "sample_weight": 5,
                "data_path_md5sum": "f82fc2157848c07e14095616cc817a7d",
            },
            {
                "data_path": "dmpv2://matrix2/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/kCycBBox2D/kCycBBox2D_project_c385_5v_CO-OX3GB-D100-L_train_night_Cam_5V/20221128-104916/data",
                "sample_weight": 3,
                "data_path_md5sum": "8240325dac889d993b9db17335c9826e",
            },
            {
                "data_path": "dmpv2://matrix2/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/kCycBBox2D/kCycBBox2D_project_AS33_glare_fp_train_night/20230131-114216/data",
                "sample_weight": 1,
                "data_path_md5sum": "a7a7798825c51056366a2fb718c902d8",
            },
            {
                "data_path": "dmpv2://matrix2/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/kCycBBox2D/kCycBBox2D_project_c385_5v_CO-OX3GB-D100-L_train_night_Jira_Cam_5V/20230126-152414/data",
                "sample_weight": 5,
                "data_path_md5sum": "b4fcce6c0211d91cd2c45dfcf8b2efad",
            },
            {
                "data_path": "dmpv2://matrix2/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/kCycBBox2D/kCycBBox2D_project_c385_5v_CO-OX3GB-D100-L_train_night_Cam_5V/20230126-163331/data",
                "sample_weight": 5,
                "data_path_md5sum": "95a42c8ddaeb62511fe67cdcff7dabfd",
            },
            {
                "data_path": "dmpv2://matrix2/multicam_pilot/data/train/cyclist_detection/CP-OX3GB-D100-L_train_night_Cam_5V/20230317_195347",
                "sample_weight": 1,
                "data_path_md5sum": "ccafe4837bd8552ccce8dd17b4b342b5",
            },
            {
                "data_path": "dmpv2://matrix2/multicam_pilot/data/lmdb_datasets/data/cyclist_detection/majiajia.data/C385_V1/Pilot_cyclist_5v_CAMERA_CO_OX3GB_D100_L_all/20220621-171148/data",
                "sample_weight": 20,
                "data_path_md5sum": "5a5742567d9db30c6af16900cb8fb27f",
            },
            {
                "data_path": "dmpv2://matrix2/multicam_pilot/data/lmdb_datasets/data/cyclist_detection/jianhang.he.data/C385_V2/Pilot_cyclist_5v_CO_OX3GB_D100_L_all/20220805-210426/data",
                "sample_weight": 20,
                "data_path_md5sum": "94ba7e6550e94e758ed50bedce44e7dd",
            },
            {
                "data_path": "dmpv2://matrix2/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/kCycBBox2D/kCycBBox2D_project_c385_5v_CO-OX3GB-D100-L_train_underground_park_Cam_5V/20221014-155654/data",
                "sample_weight": 10,
                "data_path_md5sum": "0e2688763d80b1fe7713a181f21a4bcc",
            },
            {
                "data_path": "dmpv2://matrix2/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/kCycBBox2D/kCycBBox2D_project_c385_5v_CO-OX3GB-D100-L_train_underground_park_Cam_5V/20220908-135342/data",
                "sample_weight": 10,
                "data_path_md5sum": "61b3b78770724318f512ca6b9ab7ccb9",
            },
            {
                "data_path": "dmpv2://matrix2/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/kCycBBox2D/kCycBBox2D_project_c385_5v_CO-OX3GB-D100-L_train_parking_Jira_Cam_5V/20221014-214402/data",
                "sample_weight": 1,
                "data_path_md5sum": "34ef3e910aef48fdf885fdfcd0081deb",
            },
            {
                "data_path": "dmpv2://matrix2/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/kCycBBox2D/kCycBBox2D_project_c385_5v_CO-OX3GB-D100-L_train_underground_park_Cam_5V/20221124-224155/data",
                "sample_weight": 10,
                "data_path_md5sum": "8fb231aa6271ded67539d6b385d0ebd9",
            },
            {
                "data_path": "dmpv2://matrix2/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/kCycBBox2D/kCycBBox2D_project_c385_5v_CO-OX3GB-D100-L_train_parking_Jira_Cam_5V/20221127-195043/data",
                "sample_weight": 2,
                "data_path_md5sum": "a9fb4561553e87bf260db38b857196af",
            },
            {
                "data_path": "dmpv2://matrix2/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/kCycBBox2D/kCycBBox2D_project_galaxy_5v_train_day_Cam_Rear/20220622-145223/data",
                "sample_weight": 1,
                "data_path_md5sum": "ac6d52efd87f38b089411648020c2d4e",
            },
            {
                "data_path": "dmpv2://matrix2/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/kCycBBox2D/kCycBBox2D_project_galaxy_5v_train_night_Cam_Rear/20220622-145424/data",
                "sample_weight": 1,
                "data_path_md5sum": "a071c41cbd57cf72433e1b26406d387a",
            },
            {
                "data_path": "dmpv2://matrix2/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/kCycBBox2D/kCycBBox2D_project_galaxy_5v_train_day_Cam_Side/20220622-150123/data",
                "sample_weight": 1,
                "data_path_md5sum": "200508e041db183518144cd37b0ec904",
            },
            {
                "data_path": "dmpv2://matrix2/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/kCycBBox2D/kCycBBox2D_project_galaxy_5v_train_night_Cam_Side/20220622-145741/data",
                "sample_weight": 1,
                "data_path_md5sum": "e91fdeea1ed5777217191243c62e23c7",
            },
            {
                "data_path": "dmpv2://matrix2/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/kCycBBox2D/kCycBBox2D_project_galaxy_5v_train_day_Cam_Rear/20220719-115017/data",
                "sample_weight": 2,
                "data_path_md5sum": "46e052906b0cddd79f81bbc832691d1c",
            },
            {
                "data_path": "dmpv2://matrix2/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/kCycBBox2D/kCycBBox2D_project_galaxy_5v_train_day_Cam_Side/20220719-102656/data",
                "sample_weight": 1,
                "data_path_md5sum": "76adc53467d8eeadc1d558c31019628c",
            },
            {
                "data_path": "dmpv2://matrix2/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/kCycBBox2D/kCycBBox2D_project_galaxy_5v_train_day_Cam_Rear/20220806-002734/data",
                "sample_weight": 2,
                "data_path_md5sum": "365d4826340d99868a98d9ddfd1a96e5",
            },
            {
                "data_path": "dmpv2://matrix2/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/kCycBBox2D/kCycBBox2D_project_galaxy_5v_train_day_Cam_Side/20220805-233757/data",
                "sample_weight": 2,
                "data_path_md5sum": "466a8771f4b6add9833fe5419bac770d",
            },
            {
                "data_path": "dmpv2://matrix2/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/kCycBBox2D/kCycBBox2D_project_galaxy_5v_train_day_Cam_Side/kCycBBox2D_project_galaxy_5v_train_day_Cam_Side/20220928-170533/data",
                "sample_weight": 2,
                "data_path_md5sum": "43ab02a9075d570cba15105407243ab5",
            },
            {
                "data_path": "dmpv2://matrix2/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/kCycBBox2D/kCycBBox2D_project_galaxy_5v_train_night_Cam_Side/kCycBBox2D_project_galaxy_5v_train_night_Cam_Side/20220928-155907/data",
                "sample_weight": 1,
                "data_path_md5sum": "f295181123d6b2e60fff3c8e720cf201",
            },
            {
                "data_path": "dmpv2://matrix2/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/kCycBBox2D/kCycBBox2D_project_galaxy_5v_train_day_Cam_Rear/kCycBBox2D_project_galaxy_5v_train_day_Cam_Rear/20220928-172750/data",
                "sample_weight": 2,
                "data_path_md5sum": "0bd94c5ea1f560f8be51c7e4782bfd71",
            },
            {
                "data_path": "dmpv2://matrix2/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/kCycBBox2D/kCycBBox2D_project_galaxy_5v_train_night_Cam_Rear/kCycBBox2D_project_galaxy_5v_train_night_Cam_Rear/20220928-160112/data",
                "sample_weight": 2,
                "data_path_md5sum": "e6d2bcfcf89b4ec324d9c673aeab0910",
            },
            {
                "data_path": "dmpv2://matrix2/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/kCycBBox2D/kCycBBox2D_project_galaxy_5v_train_day_Cam_Side/20221028-222207/data",
                "sample_weight": 2,
                "data_path_md5sum": "75d6c06dd43333f268a6a4c38edf01db",
            },
            {
                "data_path": "dmpv2://matrix2/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/kCycBBox2D/kCycBBox2D_project_galaxy_5v_train_day_Cam_Rear/20221028-235854/data",
                "sample_weight": 2,
                "data_path_md5sum": "b1fecc0328d760a8db71ad1f90972376",
            },
            {
                "data_path": "dmpv2://matrix2/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/kCycBBox2D/kCycBBox2D_project_galaxy_5v_train_night_Cam_Side/20221028-231600/data",
                "sample_weight": 2,
                "data_path_md5sum": "69cbd6b4733833c4a494a9c4d7f5edd5",
            },
            {
                "data_path": "dmpv2://matrix2/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/kCycBBox2D/kCycBBox2D_project_galaxy_5v_train_night_Cam_Rear/20221028-232324/data",
                "sample_weight": 1,
                "data_path_md5sum": "a1519f022aeb183fd333ab78268e5bb7",
            },
            {
                "data_path": "dmpv2://matrix2/multicam_pilot/data/lmdb_datasets/data/cyclist_detection/jianhang.he.data/Cutin_exp/debug/Pilot_cyclist_5v_all_night_cutin/20220928-194559/data",
                "sample_weight": 1,
                "data_path_md5sum": "d3a9328f002353cbb96b137110ecbc3e",
            },
            {
                "data_path": "dmpv2://matrix2/multicam_pilot/data/lmdb_datasets/data/cyclist_detection/jianhang.he.data/Cutin_exp/debug/Pilot_cyclist_5v_all_night_cutin/20220928-200950/data",
                "sample_weight": 1,
                "data_path_md5sum": "dff2b5d77a6f828c692e8e0deadb77f5",
            },
            {
                "data_path": "dmpv2://matrix2/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/kCycBBox2D/kCycBBox2D_project_galaxy_5v_train_night_JIRA_badcase_Cam_All/20221028-230453/data",
                "sample_weight": 1,
                "data_path_md5sum": "212e4cfcb688f1e0344fb9acf586c730",
            },
            {
                "data_path": "dmpv2://matrix2/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/kCycBBox2D/kCycBBox2D_project_galaxy_5v_train_all_Cam_Rear/20221119-132909/data",
                "sample_weight": 0.25,
                "data_path_md5sum": "69786c4389dc07f1c6a8ae92146d7a7c",
            },
            {
                "data_path": "dmpv2://matrix2/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/kCycBBox2D/kCycBBox2D_project_galaxy_5v_train_all_Cam_Side/20221119-134819/data",
                "sample_weight": 0.25,
                "data_path_md5sum": "ace9b83bf0ce576bf25bd7d4249eb5f4",
            },
            {
                "data_path": "dmpv2://matrix2/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/kCycBBox2D/kCycBBox2D_project_galaxy_5v_train_day_JIRA_badcase_Cam_All/20221119-142707/data",
                "sample_weight": 1,
                "data_path_md5sum": "57575bcb876f19821f3cf63e46527795",
            },
            {
                "data_path": "dmpv2://matrix2/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/kCycBBox2D/kCycBBox2D_project_galaxy_5v_train_night_JIRA_badcase_Cam_All/20221119-143020/data",
                "sample_weight": 1.5,
                "data_path_md5sum": "dffd80dbb253d290d176c7f8acb3e7bc",
            },
            {
                "data_path": "dmpv2://matrix2/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/kCycBBox2D/kCycBBox2D_project_galaxy_5v_train_all_Cam_Side/20221212-133412/data",
                "sample_weight": 1,
                "data_path_md5sum": "9b6d5fac7e99b97da58f269e88553296",
            },
            {
                "data_path": "dmpv2://matrix2/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/kCycBBox2D/kCycBBox2D_project_galaxy_5v_train_all_Cam_Rear/20221212-134154/data",
                "sample_weight": 1,
                "data_path_md5sum": "0ad729d1449f1f4d41daba2c9c77d4fd",
            },
            {
                "data_path": "dmpv2://matrix2/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/kCycBBox2D/kCycBBox2D_project_galaxy_5v_CO-OX3GB-D100-L_train_night_jira_Cam_5V/20230128-143617/data",
                "sample_weight": 4,
                "data_path_md5sum": "1ce0ec1308f9acefe8d72f649fc2cb69",
            },
            {
                "data_path": "dmpv2://matrix2/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/kCycBBox2D/kCycBBox2D_project_galaxy_train_night_Cam_Rear/20230130-211816/data",
                "sample_weight": 3,
                "data_path_md5sum": "decc152d9ca254186e7419fa6bb9c544",
            },
            {
                "data_path": "dmpv2://matrix2/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/kCycBBox2D/kCycBBox2D_project_galaxy_train_night_Cam_Rear/20230130-221935/data",
                "sample_weight": 1,
                "data_path_md5sum": "0745a124dee2e39b713d9fca9e82cc7c",
            },
            {
                "data_path": "dmpv2://matrix2/multicam_pilot/data/train/cyclist_detection/kCycBBox2D_project_galaxy_5v_train_day_JIRA_FN_Cam_All/20230324-142145",
                "sample_weight": 1,
                "data_path_md5sum": "ea435ecdedb8dd1d2d879df1bbd17025",
            },
        ],
        "val_data_paths": [],
    },
}

pilot_data_paths_lmdb = EasyDict(pilot_data_paths)
