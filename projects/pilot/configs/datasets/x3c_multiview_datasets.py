import os

from easydict import EasyDict

is_local_train = not os.path.exists("/running_package")
bucket_root = "/horizon-bucket" if is_local_train else "/bucket/input"
root = os.path.join(bucket_root, "matrix")

datapaths = dict(
    multiview_dynamic_3d_detection=dict(
        train_data_paths=[
            dict(
                rec_path=[
                    f"{root}/data/real_3d/multi_view/v01/DG101/data_DG101_20211015_v06__no_front__.rec",  # noqa
                    f"{root}/data/real_3d/multi_view/v01/DG101/data_DG101_20211016_v06__no_front__.rec",  # noqa
                    f"{root}/data/real_3d/multi_view/v01/DG101/data_DG101_20211017_v06__no_front__.rec",  # noqa
                    f"{root}/data/real_3d/multi_view/v01/DG101/data_DG101_20211018_v06__no_front__.rec",  # noqa
                    f"{root}/data/real_3d/multi_view/v01/DG101/data_DG101_20211019_v06__no_front__.rec",  # noqa
                    f"{root}/data/real_3d/multi_view/v01/DG101/data_DG101_20211020_v06__no_front__.rec",  # noqa
                    f"{root}/data/real_3d/multi_view/v01/DG101/data_DG101_20211021_v06__no_front__.rec",  # noqa
                    f"{root}/data/real_3d/multi_view/v01/DG101/data_DG101_20211022_v06__no_front__.rec",  # noqa
                    f"{root}/data/real_3d/multi_view/v01/DG101/data_DG101_20211023_v06__no_front__.rec",  # noqa
                    f"{root}/data/real_3d/multi_view/v01/DG101/data_DG101_20211024_v06__no_front__.rec",  # noqa
                    f"{root}/data/real_3d/multi_view/v01/DG101/data_DG101_20211025_v06__no_front__.rec",  # noqa
                    f"{root}/data/real_3d/multi_view/v01/DG101/data_DG101_20211110_v06__no_front__.rec",  # noqa
                    f"{root}/data/real_3d/multi_view/v01/DG101/data_DG101_20211111_v06__no_front__.rec",  # noqa
                    f"{root}/data/real_3d/multi_view/v01/DG101/data_DG101_20211112_v06__no_front__.rec",  # noqa
                    f"{root}/data/real_3d/multi_view/v01/DG101/data_DG101_20211113_v06__no_front__.rec",  # noqa
                    f"{root}/data/real_3d/multi_view/v01/DG101/data_DG101_20211114_v06__no_front__.rec",  # noqa
                    f"{root}/data/real_3d/multi_view/v01/DG101/data_DG101_20211115_v06__no_front__.rec",  # noqa
                    f"{root}/data/real_3d/multi_view/v01/DG101/data_DG101_20211116_v06__no_front__.rec",  # noqa
                    f"{root}/data/real_3d/multi_view/v01/DG101/data_DG101_20211117_v06__no_front__.rec",  # noqa
                    f"{root}/data/real_3d/multi_view/v01/DG101/data_DG101_20211118_v06__no_front__.rec",  # noqa
                    f"{root}/data/real_3d/multi_view/v01/DG101/data_DG101_20211119_v06__no_front__.rec",  # noqa
                    f"{root}/data/real_3d/multi_view/v01/DG101/data_DG101_20211126_v06__no_front__.rec",  # noqa
                    f"{root}/data/real_3d/multi_view/v01/DG101/data_DG101_20211128_v06__no_front__.rec",  # noqa
                    f"{root}/data/real_3d/multi_view/v01/DG101/data_DG101_20211129_v06__no_front__.rec",  # noqa
                    f"{root}/data/real_3d/multi_view/v01/DG101/data_DG101_20211201_v06__no_front__.rec",  # noqa
                    f"{root}/data/real_3d/multi_view/v01/DG101/data_DG101_20211202_v06__no_front__.rec",  # noqa
                    f"{root}/data/real_3d/multi_view/v01/DG101/data_DG101_20211203_v06__no_front__.rec",  # noqa
                    f"{root}/data/real_3d/multi_view/v01/DG101/data_DG101_20211204_v06__no_front__.rec",  # noqa
                    f"{root}/data/real_3d/multi_view/v01/DG101/data_DG101_20211205_v06__no_front__.rec",  # noqa
                    f"{root}/data/real_3d/multi_view/v01/DG101/data_DG101_20211206_v06__no_front__.rec",  # noqa
                    f"{root}/data/real_3d/multi_view/v01/DG101/data_DG101_20211207_v06__no_front__.rec",  # noqa
                    f"{root}/data/real_3d/multi_view/v01/DG101/data_DG101_20211208_v06__no_front__.rec",  # noqa
                    f"{root}/data/real_3d/multi_view/v01/DG101/data_DG101_20211209_v06__no_front__.rec",  # noqa
                    f"{root}/data/real_3d/multi_view/v01/DG101/data_DG101_20211211_v06__no_front__.rec",  # noqa
                    f"{root}/data/real_3d/multi_view/v01/DG101/data_DG101_20211212_v06__no_front__.rec",  # noqa
                    f"{root}/data/real_3d/multi_view/v01/DG101/data_DG101_20211213_v06__no_front__.rec",  # noqa
                    f"{root}/data/real_3d/multi_view/v01/DG101/data_DG101_20211214_v06__no_front__.rec",
                    f"{root}/data/real_3d/multi_view/v01/DG101/data_DG101_20220101_v06__no_front__.rec",  # noqa
                    f"{root}/data/real_3d/multi_view/v01/DG101/data_DG101_20220102_v06__no_front__.rec",  # noqa
                    f"{root}/data/real_3d/multi_view/v01/DG101/data_DG101_20220103_v06__no_front__.rec",  # noqa
                    f"{root}/data/real_3d/multi_view/v01/DG101/data_DG101_20220104_v06__no_front__.rec",  # noqa
                    f"{root}/data/real_3d/multi_view/v01/DG101/data_DG101_20220105_v06__no_front__.rec",  # noqa
                    f"{root}/data/real_3d/multi_view/v01/DG101/data_DG101_20220106_v06__no_front__.rec",  # noqa
                    f"{root}/data/real_3d/multi_view/v01/DG101/data_DG101_20220107_v06__no_front__.rec",  # noqa
                    f"{root}/data/real_3d/multi_view/v01/DG101/data_DG101_20220108_v06__no_front__.rec",  # noqa
                    f"{root}/data/real_3d/multi_view/v01/DG101/data_DG101_20220109_v06__no_front__.rec",  # noqa
                    f"{root}/data/real_3d/multi_view/v01/DG101/data_DG101_20220110_v06__no_front__.rec",  # noqa
                    f"{root}/data/real_3d/multi_view/v01/DG101/data_DG101_20220111_v06__no_front__.rec",  # noqa
                    f"{root}/data/real_3d/multi_view/v01/DG101/data_DG101_20220112_v06__no_front__.rec",  # noqa
                    f"{root}/data/real_3d/multi_view/v01/DG101/data_DG101_20220113_v06__no_front__.rec",  # noqa
                    f"{root}/data/real_3d/multi_view/v01/DG101/data_DG101_20220114_v06__no_front__.rec",  # noqa
                    f"{root}/data/real_3d/multi_view/v01/DG101/data_DG101_20220115_v06__no_front__.rec",  # noqa
                    f"{root}/data/real_3d/multi_view/v01/DG101/data_DG101_20220116_v06__no_front__.rec",  # noqa
                    f"{root}/data/real_3d/multi_view/v01/DG101/data_DG101_20220117_v06__no_front__.rec",  # noqa
                    f"{root}/data/real_3d/multi_view/v01/DG101/data_DG101_20220118_v06__no_front__.rec",  # noqa
                    f"{root}/data/real_3d/multi_view/v01/DG101/data_DG101_20220119_v06__no_front__.rec",  # noqa
                    f"{root}/data/real_3d/multi_view/v01/DG101/data_DG101_20220120_v06__no_front__.rec",  # noqa
                    f"{root}/data/real_3d/multi_view/v01/DG101/data_DG101_20220121_v06__no_front__.rec",  # noqa
                    f"{root}/data/real_3d/multi_view/v01/DG101/data_DG101_20220122_v06__no_front__.rec",  # noqa
                    f"{root}/data/real_3d/multi_view/v01/DG101/data_DG101_20220123_v06__no_front__.rec",  # noqa
                    f"{root}/data/real_3d/multi_view/v01/DG101/data_DG101_20220124_v06__no_front__.rec",  # noqa
                    f"{root}/data/real_3d/multi_view/v01/DG101/data_DG101_20220125_v06__no_front__.rec",  # noqa
                ]
            ),
            dict(
                rec_path=[
                    # BYD V2.1
                    f"{root}/data/real_3d/multi_view/v03/BYD72/data_BYD72_20221109_v03__no_front__.rec",  # noqa
                    f"{root}/data/real_3d/multi_view/v03/BYD72/data_BYD72_20221110_v03__no_front__.rec",  # noqa
                    f"{root}/data/real_3d/multi_view/v03/BYD72/data_BYD72_20221111_v03__no_front__.rec",  # noqa
                    f"{root}/data/real_3d/multi_view/v03/BYD72/data_BYD72_20221113_v03__no_front__.rec",  # noqa
                    f"{root}/data/real_3d/multi_view/v03/BYD19/data_BYD19_20221105_v03__no_front__.rec",  # noqa
                    f"{root}/data/real_3d/multi_view/v03/BYD19/data_BYD19_20221106_v03__no_front__.rec",  # noqa
                    f"{root}/data/real_3d/multi_view/v03/BYD19/data_BYD19_20221107_v03__no_front__.rec",  # noqa
                    f"{root}/data/real_3d/multi_view/v03/BYD19/data_BYD19_20221108_v03__no_front__.rec",  # noqa
                    f"{root}/data/real_3d/multi_view/v03/BYD19/data_BYD19_20221109_v03__no_front__.rec",  # noqa
                    f"{root}/data/real_3d/multi_view/v03/BYD19/data_BYD19_20221110_v03__no_front__.rec",  # noqa
                    f"{root}/data/real_3d/multi_view/v03/BYD19/data_BYD19_20221111_v03__no_front__.rec",  # noqa
                    f"{root}/data/real_3d/multi_view/v03/BYD19/data_BYD19_20221113_v03__no_front__.rec",  # noqa
                    # BYD V3.0
                    f"{root}/data/real_3d/multi_view/v03/BYD72/data_BYD72_20221112_v03__no_front__.rec",  # noqa
                    f"{root}/data/real_3d/multi_view/v03/BYD72/data_BYD72_20221114_v03__no_front__.rec",  # noqa
                    f"{root}/data/real_3d/multi_view/v03/BYD72/data_BYD72_20221115_v03__no_front__.rec",  # noqa
                    f"{root}/data/real_3d/multi_view/v03/BYD72/data_BYD72_20221116_v03__no_front__.rec",  # noqa
                    f"{root}/data/real_3d/multi_view/v03/BYD72/data_BYD72_20221117_v03__no_front__.rec",  # noqa
                    f"{root}/data/real_3d/multi_view/v03/BYD72/data_BYD72_20221118_v03__no_front__.rec",  # noqa
                    f"{root}/data/real_3d/multi_view/v03/BYD72/data_BYD72_20221119_v03__no_front__.rec",  # noqa
                    f"{root}/data/real_3d/multi_view/v03/BYD72/data_BYD72_20221120_v03__no_front__.rec",  # noqa
                    f"{root}/data/real_3d/multi_view/v03/BYD72/data_BYD72_20221121_v03__no_front__.rec",  # noqa
                    f"{root}/data/real_3d/multi_view/v03/BYD72/data_BYD72_20221122_v03__no_front__.rec",  # noqa
                    f"{root}/data/real_3d/multi_view/v03/BYD72/data_BYD72_20221123_v03__no_front__.rec",  # noqa
                    f"{root}/data/real_3d/multi_view/v03/BYD72/data_BYD72_20221125_v03__no_front__.rec",  # noqa
                    f"{root}/data/real_3d/multi_view/v03/BYD72/data_BYD72_20221126_v03__no_front__.rec",  # noqa
                    f"{root}/data/real_3d/multi_view/v03/BYD72/data_BYD72_20221128_v03__no_front__.rec",  # noqa
                    f"{root}/data/real_3d/multi_view/v03/BYD19/data_BYD19_20221112_v03__no_front__.rec",  # noqa
                    f"{root}/data/real_3d/multi_view/v03/BYD19/data_BYD19_20221117_v03__no_front__.rec",  # noqa
                    f"{root}/data/real_3d/multi_view/v03/BYD19/data_BYD19_20221118_v03__no_front__.rec",  # noqa
                    f"{root}/data/real_3d/multi_view/v03/BYD19/data_BYD19_20221119_v03__no_front__.rec",  # noqa
                    f"{root}/data/real_3d/multi_view/v03/BYD19/data_BYD19_20221120_v03__no_front__.rec",  # noqa
                    f"{root}/data/real_3d/multi_view/v03/BYD19/data_BYD19_20221121_v03__no_front__.rec",  # noqa
                    f"{root}/data/real_3d/multi_view/v03/BYD19/data_BYD19_20221122_v03__no_front__.rec",  # noqa
                    f"{root}/data/real_3d/multi_view/v03/BYD19/data_BYD19_20221123_v03__no_front__.rec",  # noqa
                    f"{root}/data/real_3d/multi_view/v03/BYD19/data_BYD19_20221125_v03__no_front__.rec",  # noqa
                    f"{root}/data/real_3d/multi_view/v03/BYD18/data_BYD18_20221120_v03__no_front__.rec",  # noqa
                    f"{root}/data/real_3d/multi_view/v03/BYD18/data_BYD18_20221121_v03__no_front__.rec",  # noqa
                    f"{root}/data/real_3d/multi_view/v03/BYD18/data_BYD18_20221122_v03__no_front__.rec",  # noqa
                    f"{root}/data/real_3d/multi_view/v03/BYD18/data_BYD18_20221124_v03__no_front__.rec",  # noqa
                ]
            ),
            # dict(
            #     rec_path=[
            #         # BYD V2.0
            #         f"{root}/users/zixiang.pei/data/3d/train/v221105/ut126/UT126_20220811.rec",  # noqa
            #         f"{root}/users/zixiang.pei/data/3d/train/v221105/ut126/UT126_20220818.rec",  # noqa
            #         f"{root}/users/zixiang.pei/data/3d/train/v221105/ut126/UT126_20220819.rec",  # noqa
            #         f"{root}/users/zixiang.pei/data/3d/train/v221105/ut126/UT126_20220820.rec",  # noqa
            #         f"{root}/users/zixiang.pei/data/3d/train/v221105/ut126/UT126_20220821_20220824_index0_index1500.rec",  # noqa
            #         f"{root}/users/zixiang.pei/data/3d/train/v221105/ut126/UT126_20220821_20220824_index1500_index3000.rec",  # noqa
            #         f"{root}/users/zixiang.pei/data/3d/train/v221105/ut126/UT126_20220821_20220824_index3000_index4466.rec",  # noqa
            #         f"{root}/users/zixiang.pei/data/3d/train/v221105/ut126/UT126_20220825.rec",  # noqa
            #         f"{root}/users/zixiang.pei/data/3d/train/v221105/ut126/UT126_20220826.rec",  # noqa
            #         f"{root}/users/zixiang.pei/data/3d/train/v221105/ut126/UT126_20220827_20220829_index0_index1500.rec",  # noqa
            #         f"{root}/users/zixiang.pei/data/3d/train/v221105/ut126/UT126_20220827_20220829_index1500_index3000.rec",  # noqa
            #         f"{root}/users/zixiang.pei/data/3d/train/v221105/ut126/UT126_20220830_20220831.rec",  # noqa
            #     ]
            # ),
            dict(  # BYD 4D GT experiment
                rec_path=[
                    f"{bucket_root}/matrix2/users/zixiang.pei/bev3d/packing_data/4DGT/BYD/BYD72/20221118/bev3d_packing_5v_BYD72_20221117To20221119_20230328_193553/20230329_071704/data_sort_v2.rec",
                    f"{bucket_root}/matrix2/users/zixiang.pei/bev3d/packing_data/4DGT/BYD/BYD18/20221121/bev3d_packing_5v_BYD18_20221121To20221123_20230328_190114/20230329_052028/data.rec",
                    f"{bucket_root}/matrix2/users/zixiang.pei/bev3d/packing_data/4DGT/BYD/BYD18/20221123/bev3d_packing_5v_BYD18_20221121To20221123_20230328_190114/20230329_052024/data.rec",
                    f"{bucket_root}/matrix2/users/zixiang.pei/bev3d/packing_data/4DGT/BYD/BYD18/20221124/bev3d_packing_5v_BYD18_20221124To20221124_20230328_193323/20230329_060532/data.rec",
                    f"{bucket_root}/matrix2/users/zixiang.pei/bev3d/packing_data/4DGT/BYD/BYD72/20221117/bev3d_packing_5v_BYD72_20221117To20221119_20230328_193553/20230329_060817/data.rec",
                    # f"{bucket_root}/matrix2/users/zixiang.pei/bev3d/packing_data/4DGT/BYD/BYD72/20221119/bev3d_packing_5v_BYD72_20221119To20221119_20230330_204300/20230330_204635/data.rec",
                    f"{bucket_root}/matrix2/users/zixiang.pei/bev3d/packing_data/4DGT/BYD/BYD72/20221123/bev3d_packing_5v_BYD72_20221120To20221123_20230328_193853/20230329_071700_fix/data_v2.rec",
                    f"{bucket_root}/matrix2/users/zixiang.pei/bev3d/packing_data/4DGT/BYD/BYD72/20221123/bev3d_packing_5v_BYD72_20221120To20221123_20230328_193853/20230329_071700_fix/data_2_v2.rec",
                    f"{bucket_root}/matrix2/users/zixiang.pei/bev3d/packing_data/4DGT/BYD/BYD72/20221119/bev3d_packing_5v_BYD72_20221119To20221119_20230330_204300/20230330_204635/data_0to68503.rec",
                    f"{bucket_root}/matrix2/users/zixiang.pei/bev3d/packing_data/4DGT/BYD/BYD72/20221119/bev3d_packing_5v_BYD72_20221119To20221119_20230330_204300/20230330_204635/data_68503to137006.rec",
                    f"{bucket_root}/matrix2/users/zixiang.pei/bev3d/packing_data/4DGT/BYD/BYD72/20221119/bev3d_packing_5v_BYD72_20221119To20221119_20230330_204300/20230330_204635/data_137006to205509.rec",
                    f"{bucket_root}/matrix2/users/zixiang.pei/bev3d/packing_data/4DGT/BYD/BYD72/20221119/bev3d_packing_5v_BYD72_20221119To20221119_20230330_204300/20230330_204635/data_205509to274015.rec",
                ]
            ),
            dict(
                rec_path=[
                    # C385 CA110 1134689
                    f"{bucket_root}/matrix/users/zixiang.pei/real_3d/multi_view/v02/CA110/data_CA110_20220609_v05__no_front__.rec",  # noqa
                    f"{bucket_root}/matrix/users/zixiang.pei/real_3d/multi_view/v02/CA110/data_CA110_20220610_v05__no_front__.rec",  # noqa
                    f"{bucket_root}/matrix/users/zixiang.pei/real_3d/multi_view/v02/CA110/data_CA110_20220611_v05__no_front__.rec",  # noqa
                    f"{bucket_root}/matrix/users/zixiang.pei/real_3d/multi_view/v02/CA110/data_CA110_20220613_v05__no_front__.rec",  # noqa
                    f"{bucket_root}/matrix/users/zixiang.pei/real_3d/multi_view/v02/CA110/data_CA110_20220614_v05__no_front__.rec",  # noqa
                    f"{bucket_root}/matrix/users/zixiang.pei/real_3d/multi_view/v02/CA110/data_CA110_20220615_v05__no_front__.rec",  # noqa
                    f"{bucket_root}/matrix/users/zixiang.pei/real_3d/multi_view/v02/CA110/data_CA110_20220616_v05__no_front__.rec",  # noqa
                    f"{bucket_root}/matrix/users/zixiang.pei/real_3d/multi_view/v02/CA110/data_CA110_20220617_v05__no_front__.rec",  # noqa
                    f"{bucket_root}/matrix/users/zixiang.pei/real_3d/multi_view/v02/CA110/data_CA110_20220618_v05__no_front__.rec",  # noqa
                    f"{bucket_root}/matrix/users/zixiang.pei/real_3d/multi_view/v02/CA110/data_CA110_20220620_v05__no_front__.rec",  # noqa
                    f"{bucket_root}/matrix/users/zixiang.pei/real_3d/multi_view/v02/CA110/data_CA110_20220622_v05__no_front__.rec",  # noqa
                    f"{bucket_root}/matrix/users/zixiang.pei/real_3d/multi_view/v02/CA110/data_CA110_20220623_v05__no_front__.rec",  # noqa
                    f"{bucket_root}/matrix/users/zixiang.pei/real_3d/multi_view/v02/CA110/data_CA110_20220629_v05__no_front__.rec",  # noqa
                    f"{bucket_root}/matrix/users/zixiang.pei/real_3d/multi_view/v02/CA110/data_CA110_20220630_v05__no_front__.rec",  # noqa
                    f"{bucket_root}/matrix/users/zixiang.pei/real_3d/multi_view/v02/CA110/data_CA110_20220701_v05__no_front__.rec",  # noqa
                    f"{bucket_root}/matrix/users/zixiang.pei/real_3d/multi_view/v02/CA110/data_CA110_20220702_v05__no_front__.rec",  # noqa
                    f"{bucket_root}/matrix/users/zixiang.pei/real_3d/multi_view/v02/CA110/data_CA110_20220703_v05__no_front__.rec",  # noqa
                    f"{bucket_root}/matrix/users/zixiang.pei/real_3d/multi_view/v02/CA110/data_CA110_20220704_v05__no_front__.rec",  # noqa
                    f"{bucket_root}/matrix/users/zixiang.pei/real_3d/multi_view/v02/CA110/data_CA110_20220705_v05__no_front__.rec",  # noqa
                    f"{bucket_root}/matrix/users/zixiang.pei/real_3d/multi_view/v02/CA110/data_CA110_20220706_v05__no_front__.rec",  # noqa
                    f"{bucket_root}/matrix/users/zixiang.pei/real_3d/multi_view/v02/CA110/data_CA110_20220707_v05__no_front__.rec",  # noqa
                    f"{bucket_root}/matrix/users/zixiang.pei/real_3d/multi_view/v02/CA110/data_CA110_20220708_v05__no_front__.rec",  # noqa
                    f"{bucket_root}/matrix/users/zixiang.pei/real_3d/multi_view/v02/CA110/data_CA110_20220709_v05__no_front__.rec",  # noqa
                    f"{bucket_root}/matrix/users/zixiang.pei/real_3d/multi_view/v02/CA110/data_CA110_20220710_v05__no_front__.rec",  # noqa
                    f"{bucket_root}/matrix/users/zixiang.pei/real_3d/multi_view/v02/CA110/data_CA110_20220711_v05__no_front__.rec",  # noqa
                    f"{bucket_root}/matrix/users/zixiang.pei/real_3d/multi_view/v02/CA110/data_CA110_20220712_v05__no_front__.rec",  # noqa
                    f"{bucket_root}/matrix/users/zixiang.pei/real_3d/multi_view/v02/CA110/data_CA110_20220713_v05__no_front__.rec",  # noqa
                    f"{bucket_root}/matrix/users/zixiang.pei/real_3d/multi_view/v02/CA110/data_CA110_20220714_v05__no_front__.rec",  # noqa
                    f"{bucket_root}/matrix/users/zixiang.pei/real_3d/multi_view/v02/CA110/data_CA110_20220715_v05__no_front__.rec",  # noqa
                    # f"{bucket_root}/auto_tmp_2/users/zixiang.pei/real_3d/multi_view/v02/CA110/data_CA110_20220723_v05__no_front__.rec", val  # noqa
                    # f"{bucket_root}/auto_tmp_2/users/zixiang.pei/real_3d/multi_view/v02/CA110/data_CA110_20220724_v05__no_front__.rec", val  # noqa
                    # f"{bucket_root}/auto_tmp_2/users/zixiang.pei/real_3d/multi_view/v02/CA110/data_CA110_20220725_v05__no_front__.rec", val  # noqa
                    # f"{bucket_root}/auto_tmp_2/users/zixiang.pei/real_3d/multi_view/v02/CA110/data_CA110_20220726_v05__no_front__.rec", val  # noqa
                    f"{bucket_root}/matrix/users/zixiang.pei/real_3d/multi_view/v02/CA110/data_CA110_20220729_v05__no_front__.rec",  # noqa
                    f"{bucket_root}/matrix/users/zixiang.pei/real_3d/multi_view/v02/CA110/data_CA110_20220730_v05__no_front__.rec",  # noqa
                    f"{bucket_root}/matrix/users/zixiang.pei/real_3d/multi_view/v02/CA110/data_CA110_20220731_v05__no_front__.rec",  # noqa
                    f"{bucket_root}/matrix/users/zixiang.pei/real_3d/multi_view/v02/CA110/data_CA110_20220801_v05__no_front__.rec",  # noqa
                    f"{bucket_root}/matrix/users/zixiang.pei/real_3d/multi_view/v02/CA110/data_CA110_20220802_v05__no_front__.rec",  # noqa
                    f"{bucket_root}/matrix/users/zixiang.pei/real_3d/multi_view/v02/CA110/data_CA110_20220803_v05__no_front__.rec",  # noqa
                    f"{bucket_root}/matrix/users/zixiang.pei/real_3d/multi_view/v02/CA110/data_CA110_20220804_v05__no_front__.rec",  # noqa
                    f"{bucket_root}/matrix/users/zixiang.pei/real_3d/multi_view/v02/CA110/data_CA110_20220805_v05__no_front__.rec",  # noqa
                    f"{bucket_root}/matrix/users/zixiang.pei/real_3d/multi_view/v02/CA110/data_CA110_20220807_v05__no_front__.rec",  # noqa
                    f"{bucket_root}/matrix/users/zixiang.pei/real_3d/multi_view/v02/CA110/data_CA110_20220808_v05__no_front__.rec",  # noqa
                    f"{bucket_root}/matrix/users/zixiang.pei/real_3d/multi_view/v02/CA110/data_CA110_20220809_v05__no_front__.rec",  # noqa
                    f"{bucket_root}/matrix/users/zixiang.pei/real_3d/multi_view/v02/CA110/data_CA110_20220810_v05__no_front__.rec",  # noqa
                    f"{bucket_root}/matrix/users/zixiang.pei/real_3d/multi_view/v02/CA110/data_CA110_20220811_v05__no_front__.rec",  # noqa
                    f"{bucket_root}/matrix/users/zixiang.pei/real_3d/multi_view/v02/CA110/data_CA110_20220812_v05__no_front__.rec",  # noqa
                    f"{bucket_root}/matrix/users/zixiang.pei/real_3d/multi_view/v02/CA110/data_CA110_20220813_v05__no_front__.rec",  # noqa
                    f"{bucket_root}/matrix/users/zixiang.pei/real_3d/multi_view/v02/CA110/data_CA110_20220814_v05__no_front__.rec",  # noqa
                    f"{bucket_root}/matrix/users/zixiang.pei/real_3d/multi_view/v02/CA110/data_CA110_20220815_v05__no_front__.rec",  # noqa
                    f"{bucket_root}/matrix/users/zixiang.pei/real_3d/multi_view/v02/CA110/data_CA110_20220816_v05__no_front__.rec",  # noqa
                    f"{bucket_root}/matrix/users/zixiang.pei/real_3d/multi_view/v02/CA110/data_CA110_20220817_v05__no_front__.rec",  # noqa
                    f"{bucket_root}/matrix/users/zixiang.pei/real_3d/multi_view/v02/CA110/data_CA110_20220818_v05__no_front__.rec",  # noqa
                    f"{bucket_root}/matrix/users/zixiang.pei/real_3d/multi_view/v02/CA110/data_CA110_20220821_v05__no_front__.rec",  # noqa
                    f"{bucket_root}/matrix/users/zixiang.pei/real_3d/multi_view/v02/CA110/data_CA110_20220822_v05__no_front__.rec",  # noqa
                    f"{bucket_root}/matrix/users/zixiang.pei/real_3d/multi_view/v02/CA110/data_CA110_20220823_v05__no_front__.rec",  # noqa
                    f"{bucket_root}/matrix/users/zixiang.pei/real_3d/multi_view/v02/CA110/data_CA110_20220824_v05__no_front__.rec",  # noqa
                    f"{bucket_root}/matrix/users/zixiang.pei/real_3d/multi_view/v02/CA110/data_CA110_20220825_v05__no_front__.rec",  # noqa
                    f"{bucket_root}/matrix/users/zixiang.pei/real_3d/multi_view/v02/CA110/data_CA110_20220826_v05__no_front__.rec",  # noqa
                    f"{bucket_root}/matrix/users/zixiang.pei/real_3d/multi_view/v02/CA110/data_CA110_20220827_v05__no_front__.rec",  # noqa
                    f"{bucket_root}/matrix/users/zixiang.pei/real_3d/multi_view/v02/CA110/data_CA110_20220828_v05__no_front__.rec",  # noqa
                    f"{bucket_root}/matrix/users/zixiang.pei/real_3d/multi_view/v02/CA110/data_CA110_20220829_v05__no_front__.rec",  # noqa
                    f"{bucket_root}/matrix/users/zixiang.pei/real_3d/multi_view/v02/CA110/data_CA110_20220830_v05__no_front__.rec",  # noqa
                    f"{bucket_root}/matrix/users/zixiang.pei/real_3d/multi_view/v02/CA110/data_CA110_20220831_v05__no_front__.rec",  # noqa
                    f"{bucket_root}/matrix/users/zixiang.pei/real_3d/multi_view/v02/CA110/data_CA110_20220901_v05__no_front__.rec",  # noqa
                    f"{bucket_root}/matrix/users/zixiang.pei/real_3d/multi_view/v02/CA110/data_CA110_20220902_v05__no_front__.rec",  # noqa
                    f"{bucket_root}/matrix/users/zixiang.pei/real_3d/multi_view/v02/CA110/data_CA110_20220903_v05__no_front__.rec",  # noqa
                    f"{bucket_root}/matrix/users/zixiang.pei/real_3d/multi_view/v02/CA110/data_CA110_20220904_v05__no_front__.rec",  # noqa
                    f"{bucket_root}/matrix/users/zixiang.pei/real_3d/multi_view/v02/CA110/data_CA110_20220905_v05__no_front__.rec",  # noqa
                    f"{bucket_root}/matrix/users/zixiang.pei/real_3d/multi_view/v02/CA110/data_CA110_20221009_v05__no_front__.rec",  # noqa
                    f"{bucket_root}/matrix/users/zixiang.pei/real_3d/multi_view/v02/CA110/data_CA110_20221010_v05__no_front__.rec",  # noqa
                    f"{bucket_root}/matrix/users/zixiang.pei/real_3d/multi_view/v02/CA110/data_CA110_20221025_v05__no_front__.rec",  # noqa
                    f"{bucket_root}/matrix/users/zixiang.pei/real_3d/multi_view/v02/CA110/data_CA110_20221026_v05__no_front__.rec",  # noqa
                    f"{bucket_root}/matrix/users/zixiang.pei/real_3d/multi_view/v02/CA110/data_CA110_20221028_v05__no_front__.rec",  # noqa
                    f"{bucket_root}/matrix/users/zixiang.pei/real_3d/multi_view/v02/CA110/data_CA110_20221030_v05__no_front__.rec",  # noqa
                    f"{bucket_root}/matrix/users/zixiang.pei/real_3d/multi_view/v02/CA110/data_CA110_20221031_v05__no_front__.rec",  # noqa
                    f"{bucket_root}/matrix/users/zixiang.pei/real_3d/multi_view/v02/CA110/data_CA110_20221101_v05__no_front__.rec",  # noqa
                    f"{bucket_root}/matrix/users/zixiang.pei/real_3d/multi_view/v02/CA110/data_CA110_20221102_v05__no_front__.rec",  # noqa
                    f"{bucket_root}/matrix/users/zixiang.pei/real_3d/multi_view/v02/CA110/data_CA110_20221103_v05__no_front__.rec",  # noqa
                    f"{bucket_root}/matrix/users/zixiang.pei/real_3d/multi_view/v02/CA110/data_CA110_20221104_v05__no_front__.rec",  # noqa
                    f"{bucket_root}/matrix/users/zixiang.pei/real_3d/multi_view/v02/CA110/data_CA110_20221105_v05__no_front__.rec",  # noqa
                    f"{bucket_root}/matrix/users/zixiang.pei/real_3d/multi_view/v02/CA110/data_CA110_20221106_v05__no_front__.rec",  # noqa
                    f"{bucket_root}/matrix/users/zixiang.pei/real_3d/multi_view/v02/CA110/data_CA110_20221107_v05__no_front__.rec",  # noqa
                    f"{bucket_root}/matrix/users/zixiang.pei/real_3d/multi_view/v02/CA110/data_CA110_20221108_v05__no_front__.rec",  # noqa
                    f"{bucket_root}/matrix/users/zixiang.pei/real_3d/multi_view/v02/CA110/data_CA110_20221109_v05__no_front__.rec",  # noqa
                    f"{bucket_root}/matrix/users/zixiang.pei/real_3d/multi_view/v02/CA110/data_CA110_20221110_v05__no_front__.rec",  # noqa
                    f"{bucket_root}/matrix/users/zixiang.pei/real_3d/multi_view/v02/CA110/data_CA110_20221122_v05__no_front__.rec",  # noqa
                ]
            ),
            dict(
                rec_path=[
                    # C385 CA038 499969
                    f"{bucket_root}/matrix/users/zixiang.pei/real_3d/multi_view/v02/CA038/data_CA038_20220817_v05__no_front__.rec",  # noqa
                    f"{bucket_root}/matrix/users/zixiang.pei/real_3d/multi_view/v02/CA038/data_CA038_20220818_v05__no_front__.rec",  # noqa
                    f"{bucket_root}/matrix/users/zixiang.pei/real_3d/multi_view/v02/CA038/data_CA038_20220819_v05__no_front__.rec",  # noqa
                    f"{bucket_root}/matrix/users/zixiang.pei/real_3d/multi_view/v02/CA038/data_CA038_20220820_v05__no_front__.rec",  # noqa
                    f"{bucket_root}/matrix/users/zixiang.pei/real_3d/multi_view/v02/CA038/data_CA038_20220821_v05__no_front__.rec",  # noqa
                    f"{bucket_root}/matrix/users/zixiang.pei/real_3d/multi_view/v02/CA038/data_CA038_20220822_v05__no_front__.rec",  # noqa
                    f"{bucket_root}/matrix/users/zixiang.pei/real_3d/multi_view/v02/CA038/data_CA038_20220823_v05__no_front__.rec",  # noqa
                    f"{bucket_root}/matrix/users/zixiang.pei/real_3d/multi_view/v02/CA038/data_CA038_20220824_v05__no_front__.rec",  # noqa
                    f"{bucket_root}/matrix/users/zixiang.pei/real_3d/multi_view/v02/CA038/data_CA038_20220825_v05__no_front__.rec",  # noqa
                    f"{bucket_root}/matrix/users/zixiang.pei/real_3d/multi_view/v02/CA038/data_CA038_20220826_v05__no_front__.rec",  # noqa
                    f"{bucket_root}/matrix/users/zixiang.pei/real_3d/multi_view/v02/CA038/data_CA038_20220827_v05__no_front__.rec",  # noqa
                    f"{bucket_root}/matrix/users/zixiang.pei/real_3d/multi_view/v02/CA038/data_CA038_20220828_v05__no_front__.rec",  # noqa
                    f"{bucket_root}/matrix/users/zixiang.pei/real_3d/multi_view/v02/CA038/data_CA038_20220829_v05__no_front__.rec",  # noqa
                    f"{bucket_root}/matrix/users/zixiang.pei/real_3d/multi_view/v02/CA038/data_CA038_20220830_v05__no_front__.rec",  # noqa
                    f"{bucket_root}/matrix/users/zixiang.pei/real_3d/multi_view/v02/CA038/data_CA038_20220901_v05__no_front__.rec",  # noqa
                    f"{bucket_root}/matrix/users/zixiang.pei/real_3d/multi_view/v02/CA038/data_CA038_20220902_v05__no_front__.rec",  # noqa
                    f"{bucket_root}/matrix/users/zixiang.pei/real_3d/multi_view/v02/CA038/data_CA038_20220903_v05__no_front__.rec",  # noqa
                    f"{bucket_root}/matrix/users/zixiang.pei/real_3d/multi_view/v02/CA038/data_CA038_20220904_v05__no_front__.rec",  # noqa
                    f"{bucket_root}/matrix/users/zixiang.pei/real_3d/multi_view/v02/CA038/data_CA038_20220905_v05__no_front__.rec",  # noqa
                    f"{bucket_root}/matrix/users/zixiang.pei/real_3d/multi_view/v02/CA038/data_CA038_20220906_v05__no_front__.rec",  # noqa
                    f"{bucket_root}/matrix/users/zixiang.pei/real_3d/multi_view/v02/CA038/data_CA038_20220907_v05__no_front__.rec",  # noqa
                    f"{bucket_root}/matrix/users/zixiang.pei/real_3d/multi_view/v02/CA038/data_CA038_20220908_v05__no_front__.rec",  # noqa
                    f"{bucket_root}/matrix/users/zixiang.pei/real_3d/multi_view/v02/CA038/data_CA038_20220909_v05__no_front__.rec",  # noqa
                ]
            ),
            dict(
                rec_path=[
                    # C385 CA138 253041
                    f"{bucket_root}/matrix/users/zixiang.pei/real_3d/multi_view/v02/CA138/data_CA138_20220810_v05__no_front__.rec",  # noqa
                    f"{bucket_root}/matrix/users/zixiang.pei/real_3d/multi_view/v02/CA138/data_CA138_20220811_v05__no_front__.rec",  # noqa
                    f"{bucket_root}/matrix/users/zixiang.pei/real_3d/multi_view/v02/CA138/data_CA138_20220812_v05__no_front__.rec",  # noqa
                    f"{bucket_root}/matrix/users/zixiang.pei/real_3d/multi_view/v02/CA138/data_CA138_20220814_v05__no_front__.rec",  # noqa
                    f"{bucket_root}/matrix/users/zixiang.pei/real_3d/multi_view/v02/CA138/data_CA138_20220817_v05__no_front__.rec",  # noqa
                    f"{bucket_root}/matrix/users/zixiang.pei/real_3d/multi_view/v02/CA138/data_CA138_20220818_v05__no_front__.rec",  # noqa
                    f"{bucket_root}/matrix/users/zixiang.pei/real_3d/multi_view/v02/CA138/data_CA138_20220819_v05__no_front__.rec",  # noqa
                    f"{bucket_root}/matrix/users/zixiang.pei/real_3d/multi_view/v02/CA138/data_CA138_20220820_v05__no_front__.rec",  # noqa
                    f"{bucket_root}/matrix/users/zixiang.pei/real_3d/multi_view/v02/CA138/data_CA138_20220821_v05__no_front__.rec",  # noqa
                    f"{bucket_root}/matrix/users/zixiang.pei/real_3d/multi_view/v02/CA138/data_CA138_20220822_v05__no_front__.rec",  # noqa
                    f"{bucket_root}/matrix/users/zixiang.pei/real_3d/multi_view/v02/CA138/data_CA138_20220823_v05__no_front__.rec",  # noqa
                    f"{bucket_root}/matrix/users/zixiang.pei/real_3d/multi_view/v02/CA138/data_CA138_20220824_v05__no_front__.rec",  # noqa
                    f"{bucket_root}/matrix/users/zixiang.pei/real_3d/multi_view/v02/CA138/data_CA138_20220825_v05__no_front__.rec",  # noqa
                    f"{bucket_root}/matrix/users/zixiang.pei/real_3d/multi_view/v02/CA138/data_CA138_20220826_v05__no_front__.rec",  # noqa
                    f"{bucket_root}/matrix/users/zixiang.pei/real_3d/multi_view/v02/CA138/data_CA138_20220827_v05__no_front__.rec",  # noqa
                    f"{bucket_root}/matrix/users/zixiang.pei/real_3d/multi_view/v02/CA138/data_CA138_20220828_v05__no_front__.rec",  # noqa
                    f"{bucket_root}/matrix/users/zixiang.pei/real_3d/multi_view/v02/CA138/data_CA138_20220829_v05__no_front__.rec",  # noqa
                    f"{bucket_root}/matrix/users/zixiang.pei/real_3d/multi_view/v02/CA138/data_CA138_20220830_v05__no_front__.rec",  # noqa
                    f"{bucket_root}/matrix/users/zixiang.pei/real_3d/multi_view/v02/CA138/data_CA138_20220831_v05__no_front__.rec",  # noqa
                    f"{bucket_root}/matrix/users/zixiang.pei/real_3d/multi_view/v02/CA138/data_CA138_20220901_v05__no_front__.rec",  # noqa
                    f"{bucket_root}/matrix/users/zixiang.pei/real_3d/multi_view/v02/CA138/data_CA138_20220902_v05__no_front__.rec",  # noqa
                    f"{bucket_root}/matrix/users/zixiang.pei/real_3d/multi_view/v02/CA138/data_CA138_20220903_v05__no_front__.rec",  # noqa
                    f"{bucket_root}/matrix/users/zixiang.pei/real_3d/multi_view/v02/CA138/data_CA138_20220904_v05__no_front__.rec",  # noqa
                    f"{bucket_root}/matrix/users/zixiang.pei/real_3d/multi_view/v02/CA138/data_CA138_20220905_v05__no_front__.rec",  # noqa
                    f"{bucket_root}/matrix/users/zixiang.pei/real_3d/multi_view/v02/CA138/data_CA138_20220906_v05__no_front__.rec",  # noqa
                    f"{bucket_root}/matrix/users/zixiang.pei/real_3d/multi_view/v02/CA138/data_CA138_20220907_v05__no_front__.rec",  # noqa
                    f"{bucket_root}/matrix/users/zixiang.pei/real_3d/multi_view/v02/CA138/data_CA138_20220908_v05__no_front__.rec",  # noqa
                    f"{bucket_root}/matrix/users/zixiang.pei/real_3d/multi_view/v02/CA138/data_CA138_20220923_v05__no_front__.rec",  # noqa
                    f"{bucket_root}/matrix/users/zixiang.pei/real_3d/multi_view/v02/CA138/data_CA138_20220924_v05__no_front__.rec",  # noqa
                    f"{bucket_root}/matrix/users/zixiang.pei/real_3d/multi_view/v02/CA138/data_CA138_20220925_v05__no_front__.rec",  # noqa
                    f"{bucket_root}/matrix/users/zixiang.pei/real_3d/multi_view/v02/CA138/data_CA138_20220926_v05__no_front__.rec",  # noqa
                    f"{bucket_root}/matrix/users/zixiang.pei/real_3d/multi_view/v02/CA138/data_CA138_20220927_v05__no_front__.rec",  # noqa
                    f"{bucket_root}/matrix/users/zixiang.pei/real_3d/multi_view/v02/CA138/data_CA138_20220928_v05__no_front__.rec",  # noqa
                    f"{bucket_root}/matrix/users/zixiang.pei/real_3d/multi_view/v02/CA138/data_CA138_20220929_v05__no_front__.rec",  # noqa
                    f"{bucket_root}/matrix/users/zixiang.pei/real_3d/multi_view/v02/CA138/data_CA138_20220930_v05__no_front__.rec",  # noqa
                    f"{bucket_root}/matrix/users/zixiang.pei/real_3d/multi_view/v02/CA138/data_CA138_20221019_v05__no_front__.rec",  # noqa
                    f"{bucket_root}/matrix/users/zixiang.pei/real_3d/multi_view/v02/CA138/data_CA138_20221025_v05__no_front__.rec",  # noqa
                    f"{bucket_root}/matrix/users/zixiang.pei/real_3d/multi_view/v02/CA138/data_CA138_20221027_v05__no_front__.rec",  # noqa
                    f"{bucket_root}/matrix/users/zixiang.pei/real_3d/multi_view/v02/CA138/data_CA138_20221028_v05__no_front__.rec",  # noqa
                    f"{bucket_root}/matrix/users/zixiang.pei/real_3d/multi_view/v02/CA138/data_CA138_20221029_v05__no_front__.rec",  # noqa
                    f"{bucket_root}/matrix/users/zixiang.pei/real_3d/multi_view/v02/CA138/data_CA138_20221030_v05__no_front__.rec",  # noqa
                    f"{bucket_root}/matrix/users/zixiang.pei/real_3d/multi_view/v02/CA138/data_CA138_20221031_v05__no_front__.rec",  # noqa
                    f"{bucket_root}/matrix/users/zixiang.pei/real_3d/multi_view/v02/CA138/data_CA138_20221101_v05__no_front__.rec",  # noqa
                    f"{bucket_root}/matrix/users/zixiang.pei/real_3d/multi_view/v02/CA138/data_CA138_20221102_v05__no_front__.rec",  # noqa
                    f"{bucket_root}/matrix/users/zixiang.pei/real_3d/multi_view/v02/CA138/data_CA138_20221103_v05__no_front__.rec",  # noqa
                    f"{bucket_root}/matrix/users/zixiang.pei/real_3d/multi_view/v02/CA138/data_CA138_20221104_v05__no_front__.rec",  # noqa
                    f"{bucket_root}/matrix/users/zixiang.pei/real_3d/multi_view/v02/CA138/data_CA138_20221105_v05__no_front__.rec",  # noqa
                    f"{bucket_root}/matrix/users/zixiang.pei/real_3d/multi_view/v02/CA138/data_CA138_20221106_v05__no_front__.rec",  # noqa
                    f"{bucket_root}/matrix/users/zixiang.pei/real_3d/multi_view/v02/CA138/data_CA138_20221107_v05__no_front__.rec",  # noqa
                    f"{bucket_root}/matrix/users/zixiang.pei/real_3d/multi_view/v02/CA138/data_CA138_20221108_v05__no_front__.rec",  # noqa
                    f"{bucket_root}/matrix/users/zixiang.pei/real_3d/multi_view/v02/CA138/data_CA138_20221109_v05__no_front__.rec",  # noqa
                    f"{bucket_root}/matrix/users/zixiang.pei/real_3d/multi_view/v02/CA138/data_CA138_20221110_v05__no_front__.rec",  # noqa
                    f"{bucket_root}/matrix/users/zixiang.pei/real_3d/multi_view/v02/CA138/data_CA138_20221111_v05__no_front__.rec",  # noqa
                ]
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
