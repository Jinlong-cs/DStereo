import os

from easydict import EasyDict

is_local_train = not os.path.exists("/running_package")
bucket_root = "/horizon-bucket" if is_local_train else "/bucket/input"
root = os.path.join(bucket_root, "matrix")

datapaths = dict(
    multiview_dynamic_3d_detection=dict(
        train_data_paths=[
            dict(  # galaxy special vehicle
                lmdb_path=[
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_special_veh/20230117/normal/bev3d_packing_5v_galaxy_special_veh_20230524_145000/20230524_145358",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_special_veh/20230112/normal/bev3d_packing_5v_galaxy_special_veh_20230524_145000/20230524_152003",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_special_veh/20221223/normal/bev3d_packing_5v_galaxy_special_veh_20230524_145000/20230524_150643",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_special_veh/20230116/normal/bev3d_packing_5v_galaxy_special_veh_20230524_145000/20230524_150833",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_special_veh/20230114/normal/bev3d_packing_5v_galaxy_special_veh_20230524_145000/20230524_145603",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_special_veh/20221222/normal/bev3d_packing_5v_galaxy_special_veh_20230524_145000/20230524_151400",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_special_veh/20221225/normal/bev3d_packing_5v_galaxy_special_veh_20230524_145000/20230524_145419",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_special_veh/20230118/normal/bev3d_packing_5v_galaxy_special_veh_20230524_145000/20230524_145859",  # noqa
                    # f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_special_veh/20230222/normal/bev3d_packing_5v_galaxy_special_veh_20230524_145000/20230524_145434", # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_special_veh/20230102/normal/bev3d_packing_5v_galaxy_special_veh_20230524_145000/20230524_151721",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_special_veh/20230109/normal/bev3d_packing_5v_galaxy_special_veh_20230524_145000/20230524_152140",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_special_veh/20221231/normal/bev3d_packing_5v_galaxy_special_veh_20230524_145000/20230524_151415",  # noqa
                    # f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_special_veh/20230218/normal/bev3d_packing_5v_galaxy_special_veh_20230524_145000/20230524_145403", # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_special_veh/20230120/normal/bev3d_packing_5v_galaxy_special_veh_20230524_145000/20230524_145414",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_special_veh/20221217/normal/bev3d_packing_5v_galaxy_special_veh_20230524_145000/20230524_145726",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_special_veh/20221227/normal/bev3d_packing_5v_galaxy_special_veh_20230524_145000/20230524_145426",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_special_veh/20230225/normal/bev3d_packing_5v_galaxy_special_veh_20230524_145000/20230524_152002",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_special_veh/20221228/normal/bev3d_packing_5v_galaxy_special_veh_20230524_145000/20230524_145528",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_special_veh/20230103/normal/bev3d_packing_5v_galaxy_special_veh_20230524_145000/20230524_145405",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_special_veh/20230227/normal/bev3d_packing_5v_galaxy_special_veh_20230524_145000/20230524_152726",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_special_veh/20230111/normal/bev3d_packing_5v_galaxy_special_veh_20230524_145000/20230524_151327",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_special_veh/20230215/normal/bev3d_packing_5v_galaxy_special_veh_20230524_145000/20230524_153105",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_special_veh/20230216/normal/bev3d_packing_5v_galaxy_special_veh_20230524_145000/20230524_145406",  # noqa
                    # f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_special_veh/20230219/normal/bev3d_packing_5v_galaxy_special_veh_20230524_145000/20230524_145409", # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_special_veh/20230226/normal/bev3d_packing_5v_galaxy_special_veh_20230524_145000/20230524_153348",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_special_veh/20230214/normal/bev3d_packing_5v_galaxy_special_veh_20230524_145000/20230524_153540",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_special_veh/20230223/normal/bev3d_packing_5v_galaxy_special_veh_20230524_145000/20230524_145410",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_special_veh/20221221/normal/bev3d_packing_5v_galaxy_special_veh_20230524_145000/20230524_152426",  # noqa
                    # f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_special_veh/20230217/normal/bev3d_packing_5v_galaxy_special_veh_20230524_145000/20230524_152510", # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_special_veh/20230228/normal/bev3d_packing_5v_galaxy_special_veh_20230524_145000/20230524_152404",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_special_veh/20221216/normal/bev3d_packing_5v_galaxy_special_veh_20230524_145000/20230524_152009",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_special_veh/20221229/normal/bev3d_packing_5v_galaxy_special_veh_20230524_145000/20230524_153524",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_special_veh/20230105/normal/bev3d_packing_5v_galaxy_special_veh_20230524_145000/20230524_145422",  # noqa
                    # f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_special_veh/20230220/normal/bev3d_packing_5v_galaxy_special_veh_20230524_145000/20230524_152906", # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_special_veh/20230110/normal/bev3d_packing_5v_galaxy_special_veh_20230524_145000/20230524_145359",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_special_veh/20221226/normal/bev3d_packing_5v_galaxy_special_veh_20230524_145000/20230524_152904",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_special_veh/20230115/normal/bev3d_packing_5v_galaxy_special_veh_20230524_145000/20230524_145401",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_special_veh/20230108/normal/bev3d_packing_5v_galaxy_special_veh_20230524_145000/20230524_145419",  # noqa
                    # f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_special_veh/20230221/normal/bev3d_packing_5v_galaxy_special_veh_20230524_145000/20230524_145414", # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_special_veh/20230104/normal/bev3d_packing_5v_galaxy_special_veh_20230524_145000/20230524_145402",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_special_veh/20221219/normal/bev3d_packing_5v_galaxy_special_veh_20230524_145000/20230524_145409",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_special_veh/20221224/normal/bev3d_packing_5v_galaxy_special_veh_20230524_145000/20230524_153122",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_special_veh/20230107/normal/bev3d_packing_5v_galaxy_special_veh_20230524_145000/20230524_145402",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_special_veh/20221230/normal/bev3d_packing_5v_galaxy_special_veh_20230524_145000/20230524_152623",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_special_veh/20230224/normal/bev3d_packing_5v_galaxy_special_veh_20230524_145000/20230524_150601",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_special_veh/20230119/normal/bev3d_packing_5v_galaxy_special_veh_20230524_145000/20230524_151913",  # noqa
                ],
            ),
            dict(  # galaxy night glare
                lmdb_path=[
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_night_glare/20221122/normal/bev3d_packing_5v_galaxy_night_glare_20230524_010650/20230524_013658",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_night_glare/20221208/normal/bev3d_packing_5v_galaxy_night_glare_20230524_010650/20230524_015247",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_night_glare/20221124/normal/bev3d_packing_5v_galaxy_night_glare_20230524_010650/20230524_011556",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_night_glare/20230210/normal/bev3d_packing_5v_galaxy_night_glare_20230524_010650/20230524_014016",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_night_glare/20221126/normal/bev3d_packing_5v_galaxy_night_glare_20230524_010650/20230524_012915",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_night_glare/20221207/normal/bev3d_packing_5v_galaxy_night_glare_20230524_010650/20230524_011301",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_night_glare/20230116/normal/bev3d_packing_5v_galaxy_night_glare_20230524_010650/20230524_011314",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_night_glare/20230225/normal/bev3d_packing_5v_galaxy_night_glare_20230524_010650/20230524_021859",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_night_glare/20221105/normal/bev3d_packing_5v_galaxy_night_glare_20230524_010650/20230524_021611",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_night_glare/20230204/normal/bev3d_packing_5v_galaxy_night_glare_20230524_010650/20230524_013655",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_night_glare/20221028/normal/bev3d_packing_5v_galaxy_night_glare_20230524_010650/20230524_011322",  # noqa
                    # f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_night_glare/20230222/normal/bev3d_packing_5v_galaxy_night_glare_20230524_010650/20230524_022246", # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_night_glare/20230205/normal/bev3d_packing_5v_galaxy_night_glare_20230524_010650/20230524_013738",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_night_glare/20221021/normal/bev3d_packing_5v_galaxy_night_glare_20230524_010650/20230524_011315",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_night_glare/20221026/normal/bev3d_packing_5v_galaxy_night_glare_20230524_010650/20230524_014314",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_night_glare/20230104/normal/bev3d_packing_5v_galaxy_night_glare_20230524_010650/20230524_015838",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_night_glare/20221017/normal/bev3d_packing_5v_galaxy_night_glare_20230524_010650/20230524_022009",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_night_glare/20221024/normal/bev3d_packing_5v_galaxy_night_glare_20230524_010650/20230524_011323",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_night_glare/20221204/normal/bev3d_packing_5v_galaxy_night_glare_20230524_010650/20230524_015625",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_night_glare/20230214/normal/bev3d_packing_5v_galaxy_night_glare_20230524_010650/20230524_013833",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_night_glare/20221104/normal/bev3d_packing_5v_galaxy_night_glare_20230524_010650/20230524_011256",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_night_glare/20230201/normal/bev3d_packing_5v_galaxy_night_glare_20230524_010650/20230524_020925",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_night_glare/20230211/normal/bev3d_packing_5v_galaxy_night_glare_20230524_010650/20230524_020619",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_night_glare/20230227/normal/bev3d_packing_5v_galaxy_night_glare_20230524_010650/20230524_011313",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_night_glare/20221116/normal/bev3d_packing_5v_galaxy_night_glare_20230524_010650/20230524_021907",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_night_glare/20230226/normal/bev3d_packing_5v_galaxy_night_glare_20230524_010650/20230524_011316",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_night_glare/20221018/normal/bev3d_packing_5v_galaxy_night_glare_20230524_010650/20230524_011316",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_night_glare/20221225/normal/bev3d_packing_5v_galaxy_night_glare_20230524_010650/20230524_014255",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_night_glare/20221110/normal/bev3d_packing_5v_galaxy_night_glare_20230524_010650/20230524_021305",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_night_glare/20230228/normal/bev3d_packing_5v_galaxy_night_glare_20230524_010650/20230524_021651",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_night_glare/20221031/normal/bev3d_packing_5v_galaxy_night_glare_20230524_010650/20230524_014729",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_night_glare/20221016/normal/bev3d_packing_5v_galaxy_night_glare_20230524_010650/20230524_015133",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_night_glare/20230130/normal/bev3d_packing_5v_galaxy_night_glare_20230524_010650/20230524_012929",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_night_glare/20230115/normal/bev3d_packing_5v_galaxy_night_glare_20230524_010650/20230524_011155",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_night_glare/20221127/normal/bev3d_packing_5v_galaxy_night_glare_20230524_010650/20230524_014617",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_night_glare/20221218/normal/bev3d_packing_5v_galaxy_night_glare_20230524_010650/20230524_022323",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_night_glare/20221220/normal/bev3d_packing_5v_galaxy_night_glare_20230524_010650/20230524_015353",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_night_glare/20230208/normal/bev3d_packing_5v_galaxy_night_glare_20230524_010650/20230524_013122",  # noqa
                    # f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_night_glare/20230218/normal/bev3d_packing_5v_galaxy_night_glare_20230524_010650/20230524_013251", # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_night_glare/20230202/normal/bev3d_packing_5v_galaxy_night_glare_20230524_010650/20230524_020101",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_night_glare/20221216/normal/bev3d_packing_5v_galaxy_night_glare_20230524_010650/20230524_015247",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_night_glare/20221231/normal/bev3d_packing_5v_galaxy_night_glare_20230524_010650/20230524_013916",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_night_glare/20230131/normal/bev3d_packing_5v_galaxy_night_glare_20230524_010650/20230524_012313",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_night_glare/20230207/normal/bev3d_packing_5v_galaxy_night_glare_20230524_010650/20230524_014143",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_night_glare/20230212/normal/bev3d_packing_5v_galaxy_night_glare_20230524_010650/20230524_012709",  # noqa
                    # f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_night_glare/20230206/normal/bev3d_packing_5v_galaxy_night_glare_20230524_010650/20230524_014308", # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_night_glare/20230223/normal/bev3d_packing_5v_galaxy_night_glare_20230524_010650/20230524_012840",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_night_glare/20221212/normal/bev3d_packing_5v_galaxy_night_glare_20230524_010650/20230524_015421",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_night_glare/20230209/normal/bev3d_packing_5v_galaxy_night_glare_20230524_010650/20230524_013313",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_night_glare/20221129/normal/bev3d_packing_5v_galaxy_night_glare_20230524_010650/20230524_011301",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_night_glare/20221029/normal/bev3d_packing_5v_galaxy_night_glare_20230524_010650/20230524_015102",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_night_glare/20221230/normal/bev3d_packing_5v_galaxy_night_glare_20230524_010650/20230524_021701",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_night_glare/20221209/normal/bev3d_packing_5v_galaxy_night_glare_20230524_010650/20230524_014013",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_night_glare/20221210/normal/bev3d_packing_5v_galaxy_night_glare_20230524_010650/20230524_011912",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_night_glare/20221217/normal/bev3d_packing_5v_galaxy_night_glare_20230524_010650/20230524_015423",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_night_glare/20221103/normal/bev3d_packing_5v_galaxy_night_glare_20230524_010650/20230524_021331",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_night_glare/20230114/normal/bev3d_packing_5v_galaxy_night_glare_20230524_010650/20230524_015839",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_night_glare/20230224/normal/bev3d_packing_5v_galaxy_night_glare_20230524_010650/20230524_011323",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_night_glare/20230215/normal/bev3d_packing_5v_galaxy_night_glare_20230524_010650/20230524_021044",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_night_glare/20221223/normal/bev3d_packing_5v_galaxy_night_glare_20230524_010650/20230524_021825",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_night_glare/20221222/normal/bev3d_packing_5v_galaxy_night_glare_20230524_010650/20230524_011929",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_night_glare/20221228/normal/bev3d_packing_5v_galaxy_night_glare_20230524_010650/20230524_011205",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_night_glare/20221027/normal/bev3d_packing_5v_galaxy_night_glare_20230524_010650/20230524_013631",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_night_glare/20230213/normal/bev3d_packing_5v_galaxy_night_glare_20230524_010650/20230524_012836",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_night_glare/20221221/normal/bev3d_packing_5v_galaxy_night_glare_20230524_010650/20230524_011255",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_night_glare/20221219/normal/bev3d_packing_5v_galaxy_night_glare_20230524_010650/20230524_020515",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_night_glare/20230112/normal/bev3d_packing_5v_galaxy_night_glare_20230524_010650/20230524_012112",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_night_glare/20221206/normal/bev3d_packing_5v_galaxy_night_glare_20230524_010650/20230524_020528",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_night_glare/20230203/normal/bev3d_packing_5v_galaxy_night_glare_20230524_010650/20230524_011314",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_night_glare/20221102/normal/bev3d_packing_5v_galaxy_night_glare_20230524_010650/20230524_022335",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_night_glare/20221025/normal/bev3d_packing_5v_galaxy_night_glare_20230524_010650/20230524_012549",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_night_glare/20221224/normal/bev3d_packing_5v_galaxy_night_glare_20230524_010650/20230524_013542",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_night_glare/20230129/normal/bev3d_packing_5v_galaxy_night_glare_20230524_010650/20230524_014621",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_night_glare/20221226/normal/bev3d_packing_5v_galaxy_night_glare_20230524_010650/20230524_021430",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_night_glare/20221109/normal/bev3d_packing_5v_galaxy_night_glare_20230524_010650/20230524_015321",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_night_glare/20221022/normal/bev3d_packing_5v_galaxy_night_glare_20230524_010650/20230524_011309",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_night_glare/20221015/normal/bev3d_packing_5v_galaxy_night_glare_20230524_010650/20230524_022133",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_night_glare/20230103/normal/bev3d_packing_5v_galaxy_night_glare_20230524_010650/20230524_011302",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_night_glare/20230102/normal/bev3d_packing_5v_galaxy_night_glare_20230524_010650/20230524_022433",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_night_glare/20221211/normal/bev3d_packing_5v_galaxy_night_glare_20230524_010650/20230524_014725",  # noqa
                    # f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_night_glare/20230217/normal/bev3d_packing_5v_galaxy_night_glare_20230524_010650/20230524_011309", # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_night_glare/20221019/normal/bev3d_packing_5v_galaxy_night_glare_20230524_010650/20230524_011952",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_night_glare/20221203/normal/bev3d_packing_5v_galaxy_night_glare_20230524_010650/20230524_021416",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_night_glare/20221205/normal/bev3d_packing_5v_galaxy_night_glare_20230524_010650/20230524_020420",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_night_glare/20221101/normal/bev3d_packing_5v_galaxy_night_glare_20230524_010650/20230524_014209",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_night_glare/20221107/normal/bev3d_packing_5v_galaxy_night_glare_20230524_010650/20230524_011309",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_night_glare/20221128/normal/bev3d_packing_5v_galaxy_night_glare_20230524_010650/20230524_011507",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_night_glare/20221120/normal/bev3d_packing_5v_galaxy_night_glare_20230524_010650/20230524_013657",  # noqa
                    # f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_night_glare/20230219/normal/bev3d_packing_5v_galaxy_night_glare_20230524_010650/20230524_011309", # noqa
                    # f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_night_glare/20230221/normal/bev3d_packing_5v_galaxy_night_glare_20230524_010650/20230524_020456", # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_night_glare/20221229/normal/bev3d_packing_5v_galaxy_night_glare_20230524_010650/20230524_020350",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_night_glare/20221123/normal/bev3d_packing_5v_galaxy_night_glare_20230524_010650/20230524_014741",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_night_glare/20221020/normal/bev3d_packing_5v_galaxy_night_glare_20230524_010650/20230524_021600",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_night_glare/20221215/normal/bev3d_packing_5v_galaxy_night_glare_20230524_010650/20230524_020001",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_night_glare/20230216/normal/bev3d_packing_5v_galaxy_night_glare_20230524_010650/20230524_021850",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_night_glare/20221227/normal/bev3d_packing_5v_galaxy_night_glare_20230524_010650/20230524_015606",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_night_glare/20221125/normal/bev3d_packing_5v_galaxy_night_glare_20230524_010650/20230524_014611",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_night_glare/20221121/normal/bev3d_packing_5v_galaxy_night_glare_20230524_010650/20230524_014757",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_night_glare/20230113/normal/bev3d_packing_5v_galaxy_night_glare_20230524_010650/20230524_012325",  # noqa
                ],
            ),
            dict(  # galaxy roty
                lmdb_path=[
                    # f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_roty/20230217/normal/bev3d_packing_5v_galaxy_roty_20230530_111631/20230530_112204", # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_roty/20230120/normal/bev3d_packing_5v_galaxy_roty_20230530_111631/20230530_113012",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_roty/20230211/normal/bev3d_packing_5v_galaxy_roty_20230530_111631/20230530_112207",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_roty/20230210/normal/bev3d_packing_5v_galaxy_roty_20230530_111631/20230530_112203",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_roty/20230215/normal/bev3d_packing_5v_galaxy_roty_20230530_111631/20230530_112203",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_roty/20221224/normal/bev3d_packing_5v_galaxy_roty_20230530_111631/20230530_113217",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_roty/20230201/normal/bev3d_packing_5v_galaxy_roty_20230530_111631/20230530_112206",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_roty/20230113/normal/bev3d_packing_5v_galaxy_roty_20230530_111631/20230530_112214",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_roty/20230105/normal/bev3d_packing_5v_galaxy_roty_20230530_111631/20230530_112210",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_roty/20221216/normal/bev3d_packing_5v_galaxy_roty_20230530_111631/20230530_112209",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_roty/20230119/normal/bev3d_packing_5v_galaxy_roty_20230530_111631/20230530_113617",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_roty/20230225/normal/bev3d_packing_5v_galaxy_roty_20230530_111631/20230530_114511",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_roty/20230112/normal/bev3d_packing_5v_galaxy_roty_20230530_111631/20230530_112205",  # noqa
                    # f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_roty/20230221/normal/bev3d_packing_5v_galaxy_roty_20230530_111631/20230530_114248", # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_roty/20230110/normal/bev3d_packing_5v_galaxy_roty_20230530_111631/20230530_113313",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_roty/20230227/normal/bev3d_packing_5v_galaxy_roty_20230530_111631/20230530_112208",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_roty/20230102/normal/bev3d_packing_5v_galaxy_roty_20230530_111631/20230530_112211",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_roty/20230207/normal/bev3d_packing_5v_galaxy_roty_20230530_111631/20230530_114214",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_roty/20230111/normal/bev3d_packing_5v_galaxy_roty_20230530_111631/20230530_112209",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_roty/20230116/normal/bev3d_packing_5v_galaxy_roty_20230530_111631/20230530_112916",  # noqa
                    # f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_roty/20230220/normal/bev3d_packing_5v_galaxy_roty_20230530_111631/20230530_112220", # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_roty/20221222/normal/bev3d_packing_5v_galaxy_roty_20230530_111631/20230530_112222",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_roty/20221217/normal/bev3d_packing_5v_galaxy_roty_20230530_111631/20230530_112219",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_roty/20221218/normal/bev3d_packing_5v_galaxy_roty_20230530_111631/20230530_112210",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_roty/20221209/normal/bev3d_packing_5v_galaxy_roty_20230530_111631/20230530_114044",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_roty/20221223/normal/bev3d_packing_5v_galaxy_roty_20230530_111631/20230530_112208",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_roty/20221219/normal/bev3d_packing_5v_galaxy_roty_20230530_111631/20230530_112205",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_roty/20221230/normal/bev3d_packing_5v_galaxy_roty_20230530_111631/20230530_112205",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_roty/20221228/normal/bev3d_packing_5v_galaxy_roty_20230530_111631/20230530_112212",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_roty/20230212/normal/bev3d_packing_5v_galaxy_roty_20230530_111631/20230530_113645",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_roty/20230114/normal/bev3d_packing_5v_galaxy_roty_20230530_111631/20230530_112215",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_roty/20230204/normal/bev3d_packing_5v_galaxy_roty_20230530_111631/20230530_112216",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_roty/20230224/normal/bev3d_packing_5v_galaxy_roty_20230530_111631/20230530_112217",  # noqa
                    # f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_roty/20230218/normal/bev3d_packing_5v_galaxy_roty_20230530_111631/20230530_112654", # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_roty/20230128/normal/bev3d_packing_5v_galaxy_roty_20230530_111631/20230530_112205",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_roty/20230228/normal/bev3d_packing_5v_galaxy_roty_20230530_111631/20230530_112215",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_roty/20221227/normal/bev3d_packing_5v_galaxy_roty_20230530_111631/20230530_112216",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_roty/20230107/normal/bev3d_packing_5v_galaxy_roty_20230530_111631/20230530_112213",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_roty/20221226/normal/bev3d_packing_5v_galaxy_roty_20230530_111631/20230530_112212",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_roty/20230103/normal/bev3d_packing_5v_galaxy_roty_20230530_111631/20230530_112206",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_roty/20230209/normal/bev3d_packing_5v_galaxy_roty_20230530_111631/20230530_112208",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_roty/20230208/normal/bev3d_packing_5v_galaxy_roty_20230530_111631/20230530_112211",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_roty/20221229/normal/bev3d_packing_5v_galaxy_roty_20230530_111631/20230530_114138",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_roty/20230226/normal/bev3d_packing_5v_galaxy_roty_20230530_111631/20230530_114349",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_roty/20230108/normal/bev3d_packing_5v_galaxy_roty_20230530_111631/20230530_112211",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_roty/20230203/normal/bev3d_packing_5v_galaxy_roty_20230530_111631/20230530_112202",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_roty/20230214/normal/bev3d_packing_5v_galaxy_roty_20230530_111631/20230530_112210",  # noqa
                    # f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_roty/20230219/normal/bev3d_packing_5v_galaxy_roty_20230530_111631/20230530_112215", # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_roty/20221225/normal/bev3d_packing_5v_galaxy_roty_20230530_111631/20230530_112217",  # noqa
                    # f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_roty/20230222/normal/bev3d_packing_5v_galaxy_roty_20230530_111631/20230530_113617", # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_roty/20221231/normal/bev3d_packing_5v_galaxy_roty_20230530_111631/20230530_112208",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_roty/20230216/normal/bev3d_packing_5v_galaxy_roty_20230530_111631/20230530_112204",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_roty/20230118/normal/bev3d_packing_5v_galaxy_roty_20230530_111631/20230530_112213",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_roty/20230117/normal/bev3d_packing_5v_galaxy_roty_20230530_111631/20230530_113100",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_roty/20230223/normal/bev3d_packing_5v_galaxy_roty_20230530_111631/20230530_112210",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_roty/20230205/normal/bev3d_packing_5v_galaxy_roty_20230530_111631/20230530_112213",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_roty/20230202/normal/bev3d_packing_5v_galaxy_roty_20230530_111631/20230530_112925",  # noqa
                    # f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_roty/20230206/normal/bev3d_packing_5v_galaxy_roty_20230530_111631/20230530_112210", # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_roty/20230115/normal/bev3d_packing_5v_galaxy_roty_20230530_111631/20230530_112202",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_roty/20230131/normal/bev3d_packing_5v_galaxy_roty_20230530_111631/20230530_112757",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_roty/20230104/normal/bev3d_packing_5v_galaxy_roty_20230530_111631/20230530_112204",  # noqa
                ],
            ),
            dict(  # galaxy big car
                lmdb_path=[
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_bigcar_v3/20221208/normal/bev3d_packing_5v_galaxy_bigcar_20230626_210907/20230626_211332",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_bigcar_v3/20221101/normal/bev3d_packing_5v_galaxy_bigcar_20230626_210907/20230626_211333",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_bigcar_v3/20221129/normal/bev3d_packing_5v_galaxy_bigcar_20230626_210907/20230626_211323",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_bigcar_v3/20221226/normal/bev3d_packing_5v_galaxy_bigcar_20230626_210907/20230626_211344",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_bigcar_v3/20221123/normal/bev3d_packing_5v_galaxy_bigcar_20230626_210907/20230626_211323",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_bigcar_v3/20230227/normal/bev3d_packing_5v_galaxy_bigcar_20230626_210907/20230626_211319",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_bigcar_v3/20221121/normal/bev3d_packing_5v_galaxy_bigcar_20230626_210907/20230626_211330",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_bigcar_v3/20221128/normal/bev3d_packing_5v_galaxy_bigcar_20230626_210907/20230626_211329",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_bigcar_v3/20230203/normal/bev3d_packing_5v_galaxy_bigcar_20230626_210907/20230626_211447",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_bigcar_v3/20221109/normal/bev3d_packing_5v_galaxy_bigcar_20230626_210907/20230626_211320",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_bigcar_v3/20221116/normal/bev3d_packing_5v_galaxy_bigcar_20230626_210907/20230626_211318",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_bigcar_v3/20221223/normal/bev3d_packing_5v_galaxy_bigcar_20230626_210907/20230626_211324",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_bigcar_v3/20221222/normal/bev3d_packing_5v_galaxy_bigcar_20230626_210907/20230626_211329",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_bigcar_v3/20221022/normal/bev3d_packing_5v_galaxy_bigcar_20230626_210907/20230626_211323",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_bigcar_v3/20230115/normal/bev3d_packing_5v_galaxy_bigcar_20230626_210907/20230626_211320",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_bigcar_v3/20221127/normal/bev3d_packing_5v_galaxy_bigcar_20230626_210907/20230626_212021",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_bigcar_v3/20230210/normal/bev3d_packing_5v_galaxy_bigcar_20230626_210907/20230626_211342",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_bigcar_v3/20221202/normal/bev3d_packing_5v_galaxy_bigcar_20230626_210907/20230626_211324",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_bigcar_v3/20221205/normal/bev3d_packing_5v_galaxy_bigcar_20230626_210907/20230626_212031",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_bigcar_v3/20221220/normal/bev3d_packing_5v_galaxy_bigcar_20230626_210907/20230626_211322",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_bigcar_v3/20230214/normal/bev3d_packing_5v_galaxy_bigcar_20230626_210907/20230626_211323",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_bigcar_v3/20221115/normal/bev3d_packing_5v_galaxy_bigcar_20230626_210907/20230626_212319",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_bigcar_v3/20221124/normal/bev3d_packing_5v_galaxy_bigcar_20230626_210907/20230626_211323",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_bigcar_v3/20221203/normal/bev3d_packing_5v_galaxy_bigcar_20230626_210907/20230626_211326",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_bigcar_v3/20221031/normal/bev3d_packing_5v_galaxy_bigcar_20230626_210907/20230626_211433",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_bigcar_v3/20221126/normal/bev3d_packing_5v_galaxy_bigcar_20230626_210907/20230626_211322",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_bigcar_v3/20230224/normal/bev3d_packing_5v_galaxy_bigcar_20230626_210907/20230626_211328",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_bigcar_v3/20221204/normal/bev3d_packing_5v_galaxy_bigcar_20230626_210907/20230626_211342",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_bigcar_v3/20230208/normal/bev3d_packing_5v_galaxy_bigcar_20230626_210907/20230626_211320",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_bigcar_v3/20230207/normal/bev3d_packing_5v_galaxy_bigcar_20230626_210907/20230626_212015",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_bigcar_v3/20221227/normal/bev3d_packing_5v_galaxy_bigcar_20230626_210907/20230626_211325",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_bigcar_v3/20221026/normal/bev3d_packing_5v_galaxy_bigcar_20230626_210907/20230626_211328",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_bigcar_v3/20230129/normal/bev3d_packing_5v_galaxy_bigcar_20230626_210907/20230626_211328",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_bigcar_v3/20221207/normal/bev3d_packing_5v_galaxy_bigcar_20230626_210907/20230626_211321",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_bigcar_v3/20221017/normal/bev3d_packing_5v_galaxy_bigcar_20230626_210907/20230626_211320",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_bigcar_v3/20221206/normal/bev3d_packing_5v_galaxy_bigcar_20230626_210907/20230626_211330",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_bigcar_v3/20221020/normal/bev3d_packing_5v_galaxy_bigcar_20230626_210907/20230626_211319",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_bigcar_v3/20230202/normal/bev3d_packing_5v_galaxy_bigcar_20230626_210907/20230626_212113",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_bigcar_v3/20221210/normal/bev3d_packing_5v_galaxy_bigcar_20230626_210907/20230626_212013",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_bigcar_v3/20221211/normal/bev3d_packing_5v_galaxy_bigcar_20230626_210907/20230626_211330",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_bigcar_v3/20230112/normal/bev3d_packing_5v_galaxy_bigcar_20230626_210907/20230626_211327",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_bigcar_v3/20221209/normal/bev3d_packing_5v_galaxy_bigcar_20230626_210907/20230626_211333",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_bigcar_v3/20230113/normal/bev3d_packing_5v_galaxy_bigcar_20230626_210907/20230626_211329",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_bigcar_v3/20230213/normal/bev3d_packing_5v_galaxy_bigcar_20230626_210907/20230626_212129",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_bigcar_v3/20221018/normal/bev3d_packing_5v_galaxy_bigcar_20230626_210907/20230626_211338",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_bigcar_v3/20230110/normal/bev3d_packing_5v_galaxy_bigcar_20230626_210907/20230626_212406",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_bigcar_v3/20221201/normal/bev3d_packing_5v_galaxy_bigcar_20230626_210907/20230626_211326",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_bigcar_v3/20221221/normal/bev3d_packing_5v_galaxy_bigcar_20230626_210907/20230626_211329",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_bigcar_v3/20230104/normal/bev3d_packing_5v_galaxy_bigcar_20230626_210907/20230626_211322",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_bigcar_v3/20221027/normal/bev3d_packing_5v_galaxy_bigcar_20230626_210907/20230626_212317",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_bigcar_v3/20230105/normal/bev3d_packing_5v_galaxy_bigcar_20230626_210907/20230626_211329",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_bigcar_v3/20221120/normal/bev3d_packing_5v_galaxy_bigcar_20230626_210907/20230626_211330",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_bigcar_v3/20221104/normal/bev3d_packing_5v_galaxy_bigcar_20230626_210907/20230626_211322",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_bigcar_v3/20221102/normal/bev3d_packing_5v_galaxy_bigcar_20230626_210907/20230626_211325",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_bigcar_v3/20221019/normal/bev3d_packing_5v_galaxy_bigcar_20230626_210907/20230626_211325",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_bigcar_v3/20221122/normal/bev3d_packing_5v_galaxy_bigcar_20230626_210907/20230626_211326",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_bigcar_v3/20221215/normal/bev3d_packing_5v_galaxy_bigcar_20230626_210907/20230626_211324",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_bigcar_v3/20230223/normal/bev3d_packing_5v_galaxy_bigcar_20230626_210907/20230626_211330",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_bigcar_v3/20221013/normal/bev3d_packing_5v_galaxy_bigcar_20230626_210907/20230626_211314",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_bigcar_v3/20221028/normal/bev3d_packing_5v_galaxy_bigcar_20230626_210907/20230626_211634",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_bigcar_v3/20230118/normal/bev3d_packing_5v_galaxy_bigcar_20230626_210907/20230626_211320",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_bigcar_v3/20221212/normal/bev3d_packing_5v_galaxy_bigcar_20230626_210907/20230626_211320",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_bigcar_v3/20230119/normal/bev3d_packing_5v_galaxy_bigcar_20230626_210907/20230626_211325",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_bigcar_v3/20221216/normal/bev3d_packing_5v_galaxy_bigcar_20230626_210907/20230626_211322",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_bigcar_v3/20230107/normal/bev3d_packing_5v_galaxy_bigcar_20230626_210907/20230626_211331",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_bigcar_v3/20230116/normal/bev3d_packing_5v_galaxy_bigcar_20230626_210907/20230626_211313",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_bigcar_v3/20230205/normal/bev3d_packing_5v_galaxy_bigcar_20230626_210907/20230626_212137",  # noqa
                    # f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_bigcar_v3/20230222/normal/bev3d_packing_5v_galaxy_bigcar_20230626_210907/20230626_211444", # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_bigcar_v3/20221130/normal/bev3d_packing_5v_galaxy_bigcar_20230626_210907/20230626_211319",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_bigcar_v3/20230226/normal/bev3d_packing_5v_galaxy_bigcar_20230626_210907/20230626_211326",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_bigcar_v3/20221023/normal/bev3d_packing_5v_galaxy_bigcar_20230626_210907/20230626_211340",  # noqa
                    # f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_bigcar_v3/20230206/normal/bev3d_packing_5v_galaxy_bigcar_20230626_210907/20230626_211320", # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_bigcar_v3/20230102/normal/bev3d_packing_5v_galaxy_bigcar_20230626_210907/20230626_212114",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_bigcar_v3/20221229/normal/bev3d_packing_5v_galaxy_bigcar_20230626_210907/20230626_211430",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_bigcar_v3/20230216/normal/bev3d_packing_5v_galaxy_bigcar_20230626_210907/20230626_211324",  # noqa
                    # f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_bigcar_v3/20230221/normal/bev3d_packing_5v_galaxy_bigcar_20230626_210907/20230626_211323", # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_bigcar_v3/20221024/normal/bev3d_packing_5v_galaxy_bigcar_20230626_210907/20230626_211319",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_bigcar_v3/20230209/normal/bev3d_packing_5v_galaxy_bigcar_20230626_210907/20230626_211336",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_bigcar_v3/20230211/normal/bev3d_packing_5v_galaxy_bigcar_20230626_210907/20230626_211328",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_bigcar_v3/20230212/normal/bev3d_packing_5v_galaxy_bigcar_20230626_210907/20230626_211329",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_bigcar_v3/20230130/normal/bev3d_packing_5v_galaxy_bigcar_20230626_210907/20230626_211442",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_bigcar_v3/20230108/normal/bev3d_packing_5v_galaxy_bigcar_20230626_210907/20230626_211321",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_bigcar_v3/20230204/normal/bev3d_packing_5v_galaxy_bigcar_20230626_210907/20230626_211326",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_bigcar_v3/20221010/normal/bev3d_packing_5v_galaxy_bigcar_20230626_210907/20230626_211325",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_bigcar_v3/20221103/normal/bev3d_packing_5v_galaxy_bigcar_20230626_210907/20230626_211334",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_bigcar_v3/20221225/normal/bev3d_packing_5v_galaxy_bigcar_20230626_210907/20230626_211339",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_bigcar_v3/20230109/normal/bev3d_packing_5v_galaxy_bigcar_20230626_210907/20230626_211326",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_bigcar_v3/20221029/normal/bev3d_packing_5v_galaxy_bigcar_20230626_210907/20230626_212017",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_bigcar_v3/20230215/normal/bev3d_packing_5v_galaxy_bigcar_20230626_210907/20230626_211317",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_bigcar_v3/20221009/normal/bev3d_packing_5v_galaxy_bigcar_20230626_210907/20230626_212012",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_bigcar_v3/20221218/normal/bev3d_packing_5v_galaxy_bigcar_20230626_210907/20230626_211325",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_bigcar_v3/20230128/normal/bev3d_packing_5v_galaxy_bigcar_20230626_210907/20230626_211327",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_bigcar_v3/20221016/normal/bev3d_packing_5v_galaxy_bigcar_20230626_210907/20230626_211325",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_bigcar_v3/20221025/normal/bev3d_packing_5v_galaxy_bigcar_20230626_210907/20230626_211328",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_bigcar_v3/20230225/normal/bev3d_packing_5v_galaxy_bigcar_20230626_210907/20230626_211324",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_bigcar_v3/20221110/normal/bev3d_packing_5v_galaxy_bigcar_20230626_210907/20230626_211329",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_bigcar_v3/20230114/normal/bev3d_packing_5v_galaxy_bigcar_20230626_210907/20230626_211322",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_bigcar_v3/20221012/normal/bev3d_packing_5v_galaxy_bigcar_20230626_210907/20230626_212041",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_bigcar_v3/20221219/normal/bev3d_packing_5v_galaxy_bigcar_20230626_210907/20230626_211641",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_bigcar_v3/20221011/normal/bev3d_packing_5v_galaxy_bigcar_20230626_210907/20230626_211318",  # noqa
                    # f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_bigcar_v3/20230218/normal/bev3d_packing_5v_galaxy_bigcar_20230626_210907/20230626_211324", # noqa
                    # f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_bigcar_v3/20230220/normal/bev3d_packing_5v_galaxy_bigcar_20230626_210907/20230626_211327", # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_bigcar_v3/20221107/normal/bev3d_packing_5v_galaxy_bigcar_20230626_210907/20230626_211340",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_bigcar_v3/20221224/normal/bev3d_packing_5v_galaxy_bigcar_20230626_210907/20230626_211329",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_bigcar_v3/20221231/normal/bev3d_packing_5v_galaxy_bigcar_20230626_210907/20230626_211321",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_bigcar_v3/20230228/normal/bev3d_packing_5v_galaxy_bigcar_20230626_210907/20230626_212128",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_bigcar_v3/20221214/normal/bev3d_packing_5v_galaxy_bigcar_20230626_210907/20230626_211322",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_bigcar_v3/20221230/normal/bev3d_packing_5v_galaxy_bigcar_20230626_210907/20230626_211346",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_bigcar_v3/20230131/normal/bev3d_packing_5v_galaxy_bigcar_20230626_210907/20230626_211330",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_bigcar_v3/20221125/normal/bev3d_packing_5v_galaxy_bigcar_20230626_210907/20230626_211341",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_bigcar_v3/20221105/normal/bev3d_packing_5v_galaxy_bigcar_20230626_210907/20230626_211319",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_bigcar_v3/20221119/normal/bev3d_packing_5v_galaxy_bigcar_20230626_210907/20230626_211640",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_bigcar_v3/20221217/normal/bev3d_packing_5v_galaxy_bigcar_20230626_210907/20230626_211322",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_bigcar_v3/20230117/normal/bev3d_packing_5v_galaxy_bigcar_20230626_210907/20230626_211318",  # noqa
                    # f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_bigcar_v3/20230217/normal/bev3d_packing_5v_galaxy_bigcar_20230626_210907/20230626_211318", # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_bigcar_v3/20221021/normal/bev3d_packing_5v_galaxy_bigcar_20230626_210907/20230626_212146",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_bigcar_v3/20221213/normal/bev3d_packing_5v_galaxy_bigcar_20230626_210907/20230626_212404",  # noqa
                    # f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_bigcar_v3/20230219/normal/bev3d_packing_5v_galaxy_bigcar_20230626_210907/20230626_211337", # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_bigcar_v3/20230201/normal/bev3d_packing_5v_galaxy_bigcar_20230626_210907/20230626_211321",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_bigcar_v3/20221228/normal/bev3d_packing_5v_galaxy_bigcar_20230626_210907/20230626_212113",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_bigcar_v3/20230103/normal/bev3d_packing_5v_galaxy_bigcar_20230626_210907/20230626_211326",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_bigcar_v3/20230111/normal/bev3d_packing_5v_galaxy_bigcar_20230626_210907/20230626_212401",  # noqa
                ],
            ),
            dict(  # galaxy day glare
                lmdb_path=[
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_day_glare/20221213/normal/bev3d_packing_5v_galaxy_bigcar_20230626_221532/20230626_221920",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_day_glare/20221123/normal/bev3d_packing_5v_galaxy_bigcar_20230626_221532/20230626_221909",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_day_glare/20221209/normal/bev3d_packing_5v_galaxy_bigcar_20230626_221532/20230626_221910",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_day_glare/20221129/normal/bev3d_packing_5v_galaxy_bigcar_20230626_221532/20230626_221908",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_day_glare/20221219/normal/bev3d_packing_5v_galaxy_bigcar_20230626_221532/20230626_221913",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_day_glare/20230102/normal/bev3d_packing_5v_galaxy_bigcar_20230626_221532/20230626_221919",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_day_glare/20230104/normal/bev3d_packing_5v_galaxy_bigcar_20230626_221532/20230626_222447",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_day_glare/20221105/normal/bev3d_packing_5v_galaxy_bigcar_20230626_221532/20230626_221913",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_day_glare/20221229/normal/bev3d_packing_5v_galaxy_bigcar_20230626_221532/20230626_221924",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_day_glare/20221016/normal/bev3d_packing_5v_galaxy_bigcar_20230626_221532/20230626_221918",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_day_glare/20221205/normal/bev3d_packing_5v_galaxy_bigcar_20230626_221532/20230626_221931",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_day_glare/20221026/normal/bev3d_packing_5v_galaxy_bigcar_20230626_221532/20230626_221909",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_day_glare/20230225/normal/bev3d_packing_5v_galaxy_bigcar_20230626_221532/20230626_221920",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_day_glare/20230227/normal/bev3d_packing_5v_galaxy_bigcar_20230626_221532/20230626_221914",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_day_glare/20221207/normal/bev3d_packing_5v_galaxy_bigcar_20230626_221532/20230626_221916",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_day_glare/20221024/normal/bev3d_packing_5v_galaxy_bigcar_20230626_221532/20230626_221912",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_day_glare/20221130/normal/bev3d_packing_5v_galaxy_bigcar_20230626_221532/20230626_221920",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_day_glare/20221120/normal/bev3d_packing_5v_galaxy_bigcar_20230626_221532/20230626_221917",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_day_glare/20221109/normal/bev3d_packing_5v_galaxy_bigcar_20230626_221532/20230626_221914",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_day_glare/20221218/normal/bev3d_packing_5v_galaxy_bigcar_20230626_221532/20230626_221912",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_day_glare/20221101/normal/bev3d_packing_5v_galaxy_bigcar_20230626_221532/20230626_221917",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_day_glare/20221217/normal/bev3d_packing_5v_galaxy_bigcar_20230626_221532/20230626_221916",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_day_glare/20221110/normal/bev3d_packing_5v_galaxy_bigcar_20230626_221532/20230626_221928",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_day_glare/20221231/normal/bev3d_packing_5v_galaxy_bigcar_20230626_221532/20230626_221911",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_day_glare/20221023/normal/bev3d_packing_5v_galaxy_bigcar_20230626_221532/20230626_221913",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_day_glare/20221020/normal/bev3d_packing_5v_galaxy_bigcar_20230626_221532/20230626_221914",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_day_glare/20221225/normal/bev3d_packing_5v_galaxy_bigcar_20230626_221532/20230626_222220",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_day_glare/20221216/normal/bev3d_packing_5v_galaxy_bigcar_20230626_221532/20230626_221910",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_day_glare/20221212/normal/bev3d_packing_5v_galaxy_bigcar_20230626_221532/20230626_221914",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_day_glare/20221028/normal/bev3d_packing_5v_galaxy_bigcar_20230626_221532/20230626_221909",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_day_glare/20221025/normal/bev3d_packing_5v_galaxy_bigcar_20230626_221532/20230626_221920",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_day_glare/20221103/normal/bev3d_packing_5v_galaxy_bigcar_20230626_221532/20230626_221928",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_day_glare/20221208/normal/bev3d_packing_5v_galaxy_bigcar_20230626_221532/20230626_221935",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_day_glare/20221215/normal/bev3d_packing_5v_galaxy_bigcar_20230626_221532/20230626_221919",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_day_glare/20221018/normal/bev3d_packing_5v_galaxy_bigcar_20230626_221532/20230626_221916",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_day_glare/20230223/normal/bev3d_packing_5v_galaxy_bigcar_20230626_221532/20230626_221918",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_day_glare/20221211/normal/bev3d_packing_5v_galaxy_bigcar_20230626_221532/20230626_221931",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_day_glare/20221228/normal/bev3d_packing_5v_galaxy_bigcar_20230626_221532/20230626_221919",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_day_glare/20221201/normal/bev3d_packing_5v_galaxy_bigcar_20230626_221532/20230626_221916",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_day_glare/20221204/normal/bev3d_packing_5v_galaxy_bigcar_20230626_221532/20230626_221914",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_day_glare/20221210/normal/bev3d_packing_5v_galaxy_bigcar_20230626_221532/20230626_221922",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_day_glare/20221029/normal/bev3d_packing_5v_galaxy_bigcar_20230626_221532/20230626_221914",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_day_glare/20221022/normal/bev3d_packing_5v_galaxy_bigcar_20230626_221532/20230626_221919",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_day_glare/20221223/normal/bev3d_packing_5v_galaxy_bigcar_20230626_221532/20230626_221919",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_day_glare/20230103/normal/bev3d_packing_5v_galaxy_bigcar_20230626_221532/20230626_221925",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_day_glare/20221015/normal/bev3d_packing_5v_galaxy_bigcar_20230626_221532/20230626_221912",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_day_glare/20221126/normal/bev3d_packing_5v_galaxy_bigcar_20230626_221532/20230626_221923",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_day_glare/20221221/normal/bev3d_packing_5v_galaxy_bigcar_20230626_221532/20230626_221914",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_day_glare/20221127/normal/bev3d_packing_5v_galaxy_bigcar_20230626_221532/20230626_221914",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_day_glare/20221021/normal/bev3d_packing_5v_galaxy_bigcar_20230626_221532/20230626_222227",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_day_glare/20221104/normal/bev3d_packing_5v_galaxy_bigcar_20230626_221532/20230626_221916",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_day_glare/20221128/normal/bev3d_packing_5v_galaxy_bigcar_20230626_221532/20230626_221910",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_day_glare/20221203/normal/bev3d_packing_5v_galaxy_bigcar_20230626_221532/20230626_221908",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_day_glare/20221019/normal/bev3d_packing_5v_galaxy_bigcar_20230626_221532/20230626_221921",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_day_glare/20221220/normal/bev3d_packing_5v_galaxy_bigcar_20230626_221532/20230626_221922",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_day_glare/20221017/normal/bev3d_packing_5v_galaxy_bigcar_20230626_221532/20230626_221916",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_day_glare/20221202/normal/bev3d_packing_5v_galaxy_bigcar_20230626_221532/20230626_221920",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_day_glare/20221121/normal/bev3d_packing_5v_galaxy_bigcar_20230626_221532/20230626_221912",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_day_glare/20221027/normal/bev3d_packing_5v_galaxy_bigcar_20230626_221532/20230626_221924",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_day_glare/20221230/normal/bev3d_packing_5v_galaxy_bigcar_20230626_221532/20230626_222346",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_day_glare/20221227/normal/bev3d_packing_5v_galaxy_bigcar_20230626_221532/20230626_221920",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_day_glare/20221206/normal/bev3d_packing_5v_galaxy_bigcar_20230626_221532/20230626_222200",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_day_glare/20221124/normal/bev3d_packing_5v_galaxy_bigcar_20230626_221532/20230626_221932",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_day_glare/20221031/normal/bev3d_packing_5v_galaxy_bigcar_20230626_221532/20230626_221919",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_day_glare/20221226/normal/bev3d_packing_5v_galaxy_bigcar_20230626_221532/20230626_221911",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_day_glare/20221125/normal/bev3d_packing_5v_galaxy_bigcar_20230626_221532/20230626_221922",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_day_glare/20221102/normal/bev3d_packing_5v_galaxy_bigcar_20230626_221532/20230626_221931",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_day_glare/20221116/normal/bev3d_packing_5v_galaxy_bigcar_20230626_221532/20230626_221919",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_day_glare/20221107/normal/bev3d_packing_5v_galaxy_bigcar_20230626_221532/20230626_222433",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_day_glare/20221122/normal/bev3d_packing_5v_galaxy_bigcar_20230626_221532/20230626_222316",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_day_glare/20230224/normal/bev3d_packing_5v_galaxy_bigcar_20230626_221532/20230626_221910",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_day_glare/20221224/normal/bev3d_packing_5v_galaxy_bigcar_20230626_221532/20230626_221916",  # noqa
                ],
            ),
            dict(  # galaxy normal
                lmdb_path=[
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_normal/20221231/normal/bev3d_packing_5v_galaxy_normal_20230630_104909/20230630_105508",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_normal/20221211/normal/bev3d_packing_5v_galaxy_normal_20230630_104909/20230630_105403",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_normal/20230106/normal/bev3d_packing_5v_galaxy_normal_20230630_104909/20230630_105354",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_normal/20221010/normal/bev3d_packing_5v_galaxy_normal_20230630_104909/20230630_105406",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_normal/20221204/normal/bev3d_packing_5v_galaxy_normal_20230630_104909/20230630_105408",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_normal/20221201/normal/bev3d_packing_5v_galaxy_normal_20230630_104909/20230630_105408",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_normal/20230108/normal/bev3d_packing_5v_galaxy_normal_20230630_104909/20230630_105359",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_normal/20230102/normal/bev3d_packing_5v_galaxy_normal_20230630_104909/20230630_105356",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_normal/20221101/normal/bev3d_packing_5v_galaxy_normal_20230630_104909/20230630_105505",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_normal/20221207/normal/bev3d_packing_5v_galaxy_normal_20230630_104909/20230630_105359",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_normal/20230114/normal/bev3d_packing_5v_galaxy_normal_20230630_104909/20230630_105403",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_normal/20221107/normal/bev3d_packing_5v_galaxy_normal_20230630_104909/20230630_105405",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_normal/20221221/normal/bev3d_packing_5v_galaxy_normal_20230630_104909/20230630_105352",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_normal/20221210/normal/bev3d_packing_5v_galaxy_normal_20230630_104909/20230630_105357",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_normal/20221215/normal/bev3d_packing_5v_galaxy_normal_20230630_104909/20230630_105409",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_normal/20230113/normal/bev3d_packing_5v_galaxy_normal_20230630_104909/20230630_105401",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_normal/20221218/normal/bev3d_packing_5v_galaxy_normal_20230630_104909/20230630_105516",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_normal/20230203/normal/bev3d_packing_5v_galaxy_normal_20230630_104909/20230630_105500",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_normal/20221120/normal/bev3d_packing_5v_galaxy_normal_20230630_104909/20230630_105359",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_normal/20221016/normal/bev3d_packing_5v_galaxy_normal_20230630_104909/20230630_105359",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_normal/20221125/normal/bev3d_packing_5v_galaxy_normal_20230630_104909/20230630_105402",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_normal/20221011/normal/bev3d_packing_5v_galaxy_normal_20230630_104909/20230630_105405",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_normal/20221122/normal/bev3d_packing_5v_galaxy_normal_20230630_104909/20230630_105400",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_normal/20221129/normal/bev3d_packing_5v_galaxy_normal_20230630_104909/20230630_105401",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_normal/20230224/normal/bev3d_packing_5v_galaxy_normal_20230630_104909/20230630_105356",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_normal/20230115/normal/bev3d_packing_5v_galaxy_normal_20230630_104909/20230630_105406",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_normal/20230210/normal/bev3d_packing_5v_galaxy_normal_20230630_104909/20230630_105359",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_normal/20221121/normal/bev3d_packing_5v_galaxy_normal_20230630_104909/20230630_105405",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_normal/20230112/normal/bev3d_packing_5v_galaxy_normal_20230630_104909/20230630_105350",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_normal/20221222/normal/bev3d_packing_5v_galaxy_normal_20230630_104909/20230630_105408",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_normal/20230116/normal/bev3d_packing_5v_galaxy_normal_20230630_104909/20230630_105410",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_normal/20230227/normal/bev3d_packing_5v_galaxy_normal_20230630_104909/20230630_105354",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_normal/20221116/normal/bev3d_packing_5v_galaxy_normal_20230630_104909/20230630_105359",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_normal/20230225/normal/bev3d_packing_5v_galaxy_normal_20230630_104909/20230630_105356",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_normal/20221206/normal/bev3d_packing_5v_galaxy_normal_20230630_104909/20230630_105412",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_normal/20221026/normal/bev3d_packing_5v_galaxy_normal_20230630_104909/20230630_105406",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_normal/20230216/normal/bev3d_packing_5v_galaxy_normal_20230630_104909/20230630_105356",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_normal/20221025/normal/bev3d_packing_5v_galaxy_normal_20230630_104909/20230630_105404",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_normal/20230209/normal/bev3d_packing_5v_galaxy_normal_20230630_104909/20230630_105411",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_normal/20221009/normal/bev3d_packing_5v_galaxy_normal_20230630_104909/20230630_105402",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_normal/20221020/normal/bev3d_packing_5v_galaxy_normal_20230630_104909/20230630_105401",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_normal/20230207/normal/bev3d_packing_5v_galaxy_normal_20230630_104909/20230630_105400",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_normal/20221217/normal/bev3d_packing_5v_galaxy_normal_20230630_104909/20230630_105406",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_normal/20221104/normal/bev3d_packing_5v_galaxy_normal_20230630_104909/20230630_105511",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_normal/20221216/normal/bev3d_packing_5v_galaxy_normal_20230630_104909/20230630_105514",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_normal/20221024/normal/bev3d_packing_5v_galaxy_normal_20230630_104909/20230630_105406",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_normal/20230215/normal/bev3d_packing_5v_galaxy_normal_20230630_104909/20230630_105359",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_normal/20230206/normal/bev3d_packing_5v_galaxy_normal_20230630_104909/20230630_105403",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_normal/20221127/normal/bev3d_packing_5v_galaxy_normal_20230630_104909/20230630_105412",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_normal/20221220/normal/bev3d_packing_5v_galaxy_normal_20230630_104909/20230630_105512",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_normal/20221027/normal/bev3d_packing_5v_galaxy_normal_20230630_104909/20230630_105359",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_normal/20230118/normal/bev3d_packing_5v_galaxy_normal_20230630_104909/20230630_105359",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_normal/20230130/normal/bev3d_packing_5v_galaxy_normal_20230630_104909/20230630_105407",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_normal/20221130/normal/bev3d_packing_5v_galaxy_normal_20230630_104909/20230630_105407",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_normal/20221223/normal/bev3d_packing_5v_galaxy_normal_20230630_104909/20230630_105414",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_normal/20230104/normal/bev3d_packing_5v_galaxy_normal_20230630_104909/20230630_105400",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_normal/20230109/normal/bev3d_packing_5v_galaxy_normal_20230630_104909/20230630_105402",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_normal/20221124/normal/bev3d_packing_5v_galaxy_normal_20230630_104909/20230630_105403",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_normal/20221208/normal/bev3d_packing_5v_galaxy_normal_20230630_104909/20230630_105400",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_normal/20230111/normal/bev3d_packing_5v_galaxy_normal_20230630_104909/20230630_105511",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_normal/20221203/normal/bev3d_packing_5v_galaxy_normal_20230630_104909/20230630_105401",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_normal/20221103/normal/bev3d_packing_5v_galaxy_normal_20230630_104909/20230630_105359",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_normal/20221202/normal/bev3d_packing_5v_galaxy_normal_20230630_104909/20230630_105406",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_normal/20221109/normal/bev3d_packing_5v_galaxy_normal_20230630_104909/20230630_105352",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_normal/20221105/normal/bev3d_packing_5v_galaxy_normal_20230630_104909/20230630_105357",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_normal/20221022/normal/bev3d_packing_5v_galaxy_normal_20230630_104909/20230630_105352",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_normal/20221227/normal/bev3d_packing_5v_galaxy_normal_20230630_104909/20230630_105359",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_normal/20221213/normal/bev3d_packing_5v_galaxy_normal_20230630_104909/20230630_105403",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_normal/20230128/normal/bev3d_packing_5v_galaxy_normal_20230630_104909/20230630_105404",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_normal/20221212/normal/bev3d_packing_5v_galaxy_normal_20230630_104909/20230630_105403",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_normal/20230211/normal/bev3d_packing_5v_galaxy_normal_20230630_104909/20230630_105501",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_normal/20221021/normal/bev3d_packing_5v_galaxy_normal_20230630_104909/20230630_105359",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_normal/20221110/normal/bev3d_packing_5v_galaxy_normal_20230630_104909/20230630_105353",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_normal/20230212/normal/bev3d_packing_5v_galaxy_normal_20230630_104909/20230630_105400",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_normal/20221015/normal/bev3d_packing_5v_galaxy_normal_20230630_104909/20230630_105412",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_normal/20221017/normal/bev3d_packing_5v_galaxy_normal_20230630_104909/20230630_105405",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_normal/20221012/normal/bev3d_packing_5v_galaxy_normal_20230630_104909/20230630_105359",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_normal/20230110/normal/bev3d_packing_5v_galaxy_normal_20230630_104909/20230630_105353",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_normal/20221228/normal/bev3d_packing_5v_galaxy_normal_20230630_104909/20230630_105400",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_normal/20230105/normal/bev3d_packing_5v_galaxy_normal_20230630_104909/20230630_105407",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_normal/20221029/normal/bev3d_packing_5v_galaxy_normal_20230630_104909/20230630_105357",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_normal/20230205/normal/bev3d_packing_5v_galaxy_normal_20230630_104909/20230630_105406",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_normal/20221023/normal/bev3d_packing_5v_galaxy_normal_20230630_104909/20230630_105351",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_normal/20221219/normal/bev3d_packing_5v_galaxy_normal_20230630_104909/20230630_105412",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_normal/20221102/normal/bev3d_packing_5v_galaxy_normal_20230630_104909/20230630_105356",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_normal/20221209/normal/bev3d_packing_5v_galaxy_normal_20230630_104909/20230630_105401",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_normal/20230131/normal/bev3d_packing_5v_galaxy_normal_20230630_104909/20230630_105402",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_normal/20221013/normal/bev3d_packing_5v_galaxy_normal_20230630_104909/20230630_105401",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_normal/20221224/normal/bev3d_packing_5v_galaxy_normal_20230630_104909/20230630_105350",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_normal/20230103/normal/bev3d_packing_5v_galaxy_normal_20230630_104909/20230630_105357",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_normal/20221225/normal/bev3d_packing_5v_galaxy_normal_20230630_104909/20230630_105407",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_normal/20221128/normal/bev3d_packing_5v_galaxy_normal_20230630_104909/20230630_105355",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_normal/20221028/normal/bev3d_packing_5v_galaxy_normal_20230630_104909/20230630_105352",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_normal/20230117/normal/bev3d_packing_5v_galaxy_normal_20230630_104909/20230630_105408",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_normal/20230217/normal/bev3d_packing_5v_galaxy_normal_20230630_104909/20230630_105359",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_normal/20221018/normal/bev3d_packing_5v_galaxy_normal_20230630_104909/20230630_105508",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_normal/20221014/normal/bev3d_packing_5v_galaxy_normal_20230630_104909/20230630_105358",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_normal/20221229/normal/bev3d_packing_5v_galaxy_normal_20230630_104909/20230630_105401",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_normal/20221230/normal/bev3d_packing_5v_galaxy_normal_20230630_104909/20230630_105356",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_normal/20221126/normal/bev3d_packing_5v_galaxy_normal_20230630_104909/20230630_105353",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_normal/20221226/normal/bev3d_packing_5v_galaxy_normal_20230630_104909/20230630_105357",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_normal/20221123/normal/bev3d_packing_5v_galaxy_normal_20230630_104909/20230630_105406",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_normal/20221031/normal/bev3d_packing_5v_galaxy_normal_20230630_104909/20230630_105507",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_normal/20230129/normal/bev3d_packing_5v_galaxy_normal_20230630_104909/20230630_105403",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_normal/20230223/normal/bev3d_packing_5v_galaxy_normal_20230630_104909/20230630_105500",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_normal/20221205/normal/bev3d_packing_5v_galaxy_normal_20230630_104909/20230630_105507",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_normal/20221019/normal/bev3d_packing_5v_galaxy_normal_20230630_104909/20230630_105412",  # noqa
                    f"{bucket_root}/matrix2/data/bev3d/packing_data/galaxy_normal/20230107/normal/bev3d_packing_5v_galaxy_normal_20230630_104909/20230630_105404",  # noqa
                ],
            ),
        ],
        val_data_paths=[
            dict(
                rec_path=[
                    f"{root}/data/real_3d/multi_view/v01/DG101/data_DG101_20211210_v06__no_front__.rec",  # noqa
                ],
            ),
        ],
    )
)

datapaths = EasyDict(datapaths)
