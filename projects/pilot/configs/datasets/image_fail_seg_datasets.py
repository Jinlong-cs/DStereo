import os

from easydict import EasyDict

is_local_train = not os.path.exists("/running_package")
bucket_root = "/horizon-bucket" if is_local_train else "/bucket/input"
root = os.path.join(bucket_root, "matrix2")

# ----- Required -----

datapaths = dict(
    image_fail_parsing=dict(
        train_batch_size_per_ctx=16,  # for vargnet
        train_data_paths=[
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/image_fail/blur/20221216_225253/train_CW-OX3GB-O100-065-L_blur_cls7_day_num38_20221216_225253.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/image_fail/blur/20221216_225253/train_CW-OX3GB-O100-065-L_blur_cls7_day_num38_20221216_225253.json",  # noqa
                sample_weight=1,
            ),
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/image_fail/blur/train_CW-OX3GB-O100-065-L_blur_cls7_num13250_all_20221104_181556/20221104-184612/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/image_fail/blur/train_CW-OX3GB-O100-065-L_blur_cls7_num13250_all_20221104_181556/20221104-184612/data.json",  # noqa
                sample_weight=5,
            ),
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/image_fail/blockage/20220829_015057/train_X3C-OFT224_blockage_cls7_day_num2999_20220829_015057.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/image_fail/blockage/20220829_015057/train_X3C-OFT224_blockage_cls7_day_num2999_20220829_015057.json",  # noqa
                sample_weight=5,
            ),
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/image_fail/blockage/20220828_214231/train_X3C-OFT224_blockage_cls7_day_num226_20220828_214231.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/image_fail/blockage/20220828_214231/train_X3C-OFT224_blockage_cls7_day_num226_20220828_214231.json",  # noqa
                sample_weight=1,
            ),
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/image_fail/blockage/20220828_214329/train_X3C-OFT224_blockage_cls7_day_num144_20220828_214329.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/image_fail/blockage/20220828_214329/train_X3C-OFT224_blockage_cls7_day_num144_20220828_214329.json",  # noqa
                sample_weight=1,
            ),
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/image_fail/blockage/20220828_214635/train_X3C-OFT224_blockage_cls7_day_num1800_20220828_214635.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/image_fail/blockage/20220828_214635/train_X3C-OFT224_blockage_cls7_day_num1800_20220828_214635.json",  # noqa
                sample_weight=3,
            ),
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/image_fail/blockage/20220828_215234/train_X3C-OFT224_blockage_cls7_day_num1800_20220828_215234.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/image_fail/blockage/20220828_215234/train_X3C-OFT224_blockage_cls7_day_num1800_20220828_215234.json",  # noqa
                sample_weight=3,
            ),
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/image_fail/blockage/20220828_215632/train_X3C-OFT224_blockage_cls7_day_num1300_20220828_215632.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/image_fail/blockage/20220828_215632/train_X3C-OFT224_blockage_cls7_day_num1300_20220828_215632.json",  # noqa
                sample_weight=2,
            ),
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/image_fail/blockage/20220828_215837/train_X3C-OFT224_blockage_cls7_day_num378_20220828_215837.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/image_fail/blockage/20220828_215837/train_X3C-OFT224_blockage_cls7_day_num378_20220828_215837.json",  # noqa
                sample_weight=1,
            ),
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/image_fail/blockage/20220826_154623/train_X3C-OFT224_blockage_cls7_day_num1800_20220826_154623.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/image_fail/blockage/20220826_154623/train_X3C-OFT224_blockage_cls7_day_num1800_20220826_154623.json",  # noqa
                sample_weight=3,
            ),
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/image_fail/blockage/20220826_031402/train_X3C-OFT224_blockage_cls7_day_num3032_20220826_031402.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/image_fail/blockage/20220826_031402/train_X3C-OFT224_blockage_cls7_day_num3032_20220826_031402.json",  # noqa
                sample_weight=5,
            ),
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/image_fail/blockage/20220826_031615/train_X3C-OFT224_blockage_cls7_day_num3000_20220826_031615.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/image_fail/blockage/20220826_031615/train_X3C-OFT224_blockage_cls7_day_num3000_20220826_031615.json",  # noqa
                sample_weight=5,
            ),
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/image_fail/blur/20220628_143017/train_CO_OX3GB_D100_L_num612_blur_cls7_night_20220628_143017.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/image_fail/blur/20220628_143017/train_CO_OX3GB_D100_L_num612_blur_cls7_night_20220628_143017.json",  # noqa
                sample_weight=1,
            ),
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/image_fail/glare/20220629_143232/train_CW_A23GB_A114_L_num221_glare_cls7_night_20220629_143232.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/image_fail/glare/20220629_143232/train_CW_A23GB_A114_L_num221_glare_cls7_night_20220629_143232.json",  # noqa
                sample_weight=1,
            ),
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/image_fail/blur/20220629_143637/train_CW_A23GB_A114_L_num2189_blur_cls7_night_20220629_143637.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/image_fail/blur/20220629_143637/train_CW_A23GB_A114_L_num2189_blur_cls7_night_20220629_143637.json",  # noqa
                sample_weight=4,
            ),
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/image_fail/blockage/20220629_143903/train_CW_A23GB_A114_L_num97_blockage_cls7_day_20220629_143903.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/image_fail/blockage/20220629_143903/train_CW_A23GB_A114_L_num97_blockage_cls7_day_20220629_143903.json",  # noqa
                sample_weight=1,
            ),
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/image_fail/blur/20220629_144057/train_CW_A23GB_A114_L_num988_blur_cls7_day_20220629_144057.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/image_fail/blur/20220629_144057/train_CW_A23GB_A114_L_num988_blur_cls7_day_20220629_144057.json",  # noqa
                sample_weight=2,
            ),
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/image_fail/blockage/20220629_144219/train_X3C-OFT224_blockage_cls7_day_num96_20220629_144219.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/image_fail/blockage/20220629_144219/train_X3C-OFT224_blockage_cls7_day_num96_20220629_144219.json",  # noqa
                sample_weight=1,
            ),
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/image_fail/blur/20220629_151014/train_CO_OX3GB_D100_L_num1834_blur_cls7_day_20220629_151014.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/image_fail/blur/20220629_151014/train_CO_OX3GB_D100_L_num1834_blur_cls7_day_20220629_151014.json",  # noqa
                sample_weight=3,
            ),
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/image_fail/blur/20220629_152225/train_X3C-OFT224_blur_cls7_night_num6068_20220629_152225.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/image_fail/blur/20220629_152225/train_X3C-OFT224_blur_cls7_night_num6068_20220629_152225.json",  # noqa
                sample_weight=10,
            ),
            # galaxy rear blur
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/image_fail/blur/20220907_122609/train_0233-OF0110_blur_cls7_night_num1469_20220907_122609.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/image_fail/blur/20220907_122609/train_0233-OF0110_blur_cls7_night_num1469_20220907_122609.json",  # noqa
                sample_weight=3,
            ),
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/image_fail/blur/20220907_122952/train_0233-OF0110_blur_cls7_night_num1294_20220907_122952.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/image_fail/blur/20220907_122952/train_0233-OF0110_blur_cls7_night_num1294_20220907_122952.json",  # noqa
                sample_weight=3,
            ),
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/image_fail/blur/20220907_123357/train_0233-OF0110_blur_cls7_night_num1573_20220907_123357.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/image_fail/blur/20220907_123357/train_0233-OF0110_blur_cls7_night_num1573_20220907_123357.json",  # noqa
                sample_weight=4.5,
            ),
            # galaxy rainy blur
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/image_fail/blur/20220914_192728/train_0233-OF0110_blur_cls7_night_num2187_20220914_192728.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/image_fail/blur/20220914_192728/train_0233-OF0110_blur_cls7_night_num2187_20220914_192728.json",  # noqa
                sample_weight=3,
            ),
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/image_fail/blur/20221014_174647/train_0233-OF0110_blur_cls7_night_num2298_20221014_174647.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/image_fail/blur/20221014_174647/train_0233-OF0110_blur_cls7_night_num2298_20221014_174647.json",  # noqa
                sample_weight=3,
            ),
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/image_fail/blur/20220914_193425/train_0233-OF0110_blur_cls7_day_num2286_20220914_193425.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/image_fail/blur/20220914_193425/train_0233-OF0110_blur_cls7_day_num2286_20220914_193425.json",  # noqa
                sample_weight=3,
            ),
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/image_fail/blur/20221014_175439/train_0233-OF0110_blur_cls7_day_num1182_20221014_175439.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/image_fail/blur/20221014_175439/train_0233-OF0110_blur_cls7_day_num1182_20221014_175439.json",  # noqa
                sample_weight=1.5,
            ),
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/image_fail/blur/20221016_225716/train_0233-OF0110_blur_cls7_day_num5708_20221016_225716.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/image_fail/blur/20221016_225716/train_0233-OF0110_blur_cls7_day_num5708_20221016_225716.json",  # noqa
                sample_weight=6,
            ),
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/image_fail/blur/train_CW-OX3GB-O100-065-L_blur_cls7_num3844_day_20221104_170306/20221104-171137/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/image_fail/blur/train_CW-OX3GB-O100-065-L_blur_cls7_num3844_day_20221104_170306/20221104-171137/data.json",  # noqa
                sample_weight=4.5,
            ),
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/image_fail/blur/train_CW-A23GB-O123-066-L_blur_cls7_num7302_night_20221220_125404/20221220-132100/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/image_fail/blur/train_CW-A23GB-O123-066-L_blur_cls7_num7302_night_20221220_125404/20221220-132100/data.json",  # noqa
                sample_weight=9,
            ),
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/image_fail/blur/train_CW-A23GB-O123-066-L_blur_cls7_num7480_day_20221217_222121/20221217-224523/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/image_fail/blur/train_CW-A23GB-O123-066-L_blur_cls7_num7480_day_20221217_222121/20221217-224523/data.json",  # noqa
                sample_weight=9,
            ),
            # galaxy night blur
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/image_fail/blur/20221016_230612/train_0233-OF0110_blur_cls7_night_num2852_20221016_230612.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/image_fail/blur/20221016_230612/train_0233-OF0110_blur_cls7_night_num2852_20221016_230612.json",  # noqa
                sample_weight=2.5,
            ),
            # galaxy jira
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/image_fail/blur/train_CW-OX3GB-O100-066-L_blur_cls7_num316_night_20221129_161822/20221129-162631/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/image_fail/blur/train_CW-OX3GB-O100-066-L_blur_cls7_num316_night_20221129_161822/20221129-162631/data.json",  # noqa
                sample_weight=2,
            ),
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/image_fail/blur/train_CW-OX3GB-O100-066-L_blur_cls7_num1283_day_20221129_161546/20221129-162944/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/image_fail/blur/train_CW-OX3GB-O100-066-L_blur_cls7_num1283_day_20221129_161546/20221129-162944/data.json",  # noqa
                sample_weight=6,
            ),
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/image_fail/blur/20221220_155633/train_CW-A23GB-O123-066-L_blur_cls7_day_num135_20221220_155633.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/image_fail/blur/20221220_155633/train_CW-A23GB-O123-066-L_blur_cls7_day_num135_20221220_155633.json",  # noqa
                sample_weight=1,
            ),
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/image_fail/blur/train_CO-OX3GB-D100-L_blur_cls7_num1360_day_20230315_184305/20230315-184941/data.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/image_fail/blur/train_CO-OX3GB-D100-L_blur_cls7_num1360_day_20230315_184305/20230315-184941/data.json",  # noqa
                sample_weight=2,
            ),
            # blur
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/image_fail/blur/20210627_220752/train_ar0233-RGGB_blur_cls5_num2860_20210627_220752.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/image_fail/blur/20210627_220752/train_ar0233-RGGB_blur_cls5_num2860_20210627_220752.json",  # noqa
                sample_weight=4,
            ),
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/image_fail/blur/20210708_181516/train_ar0233-RGGB_blur_cls7_all_num1531_20210708_181516.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/image_fail/blur/20210708_181516/train_ar0233-RGGB_blur_cls7_all_num1531_20210708_181516.json",  # noqa
                sample_weight=2,
            ),
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/image_fail/blur/20210708_212718/train_ar0233-RGGB_blur_cls7_day_num2785_20210708_212718.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/image_fail/blur/20210708_212718/train_ar0233-RGGB_blur_cls7_day_num2785_20210708_212718.json",  # noqa
                sample_weight=4,
            ),
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/image_fail/blur/20210708_214348/train_ar0233-RGGB_blur_cls7_all_num2465_20210708_214348.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/image_fail/blur/20210708_214348/train_ar0233-RGGB_blur_cls7_all_num2465_20210708_214348.json",  # noqa
                sample_weight=4,
            ),
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/image_fail/blur/20210728_010308/train_ar0233-RGGB_blur_cls7_all_num2120_20210728_010308.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/image_fail/blur/20210728_010308/train_ar0233-RGGB_blur_cls7_all_num2120_20210728_010308.json",  # noqa
                sample_weight=4,
            ),
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/image_fail/blur/20210728_011114/train_ar0233-RGGB_blur_cls7_day_num1961_20210728_011114.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/image_fail/blur/20210728_011114/train_ar0233-RGGB_blur_cls7_day_num1961_20210728_011114.json",  # noqa
                sample_weight=3,
            ),
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/image_fail/blur/20210807_215030/train_ar0233-RGGB_blur_cls7_all_num9751_20210807_215030.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/image_fail/blur/20210807_215030/train_ar0233-RGGB_blur_cls7_all_num9751_20210807_215030.json",  # noqa
                sample_weight=16,
            ),
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/image_fail/blur/20210807_230644/train_ar0233-WISSEN_blur_cls7_day_num2990_20210807_230644.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/image_fail/blur/20210807_230644/train_ar0233-WISSEN_blur_cls7_day_num2990_20210807_230644.json",  # noqa
                sample_weight=5,
            ),
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/image_fail/blur/20210807_231606/train_ar0233-WISSEN_blur_cls7_night_num244_20210807_231606.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/image_fail/blur/20210807_231606/train_ar0233-WISSEN_blur_cls7_night_num244_20210807_231606.json",  # noqa
                sample_weight=1,
            ),
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/image_fail/blur/20210807_232342/train_ar0233-RGGB_blur_cls7_day_num3753_20210807_232342.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/image_fail/blur/20210807_232342/train_ar0233-RGGB_blur_cls7_day_num3753_20210807_232342.json",  # noqa
                sample_weight=7,
            ),
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/image_fail/blur/20210807_233108/train_ar0233-RGGB_blur_cls7_night_num533_20210807_233108.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/image_fail/blur/20210807_233108/train_ar0233-RGGB_blur_cls7_night_num533_20210807_233108.json",  # noqa
                sample_weight=2,
            ),
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/image_fail/blur/20210816_010056/train_ar0233-RGGB_blur_cls7_day_num5040_20210816_010056.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/image_fail/blur/20210816_010056/train_ar0233-RGGB_blur_cls7_day_num5040_20210816_010056.json",  # noqa
                sample_weight=8,
            ),
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/image_fail/blur/20211008_100708/train_ar0233-WISSEN_blur_cls7_day_num149_20211008_100708.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/image_fail/blur/20211008_100708/train_ar0233-WISSEN_blur_cls7_day_num149_20211008_100708.json",  # noqa
                sample_weight=1,
            ),
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/image_fail/blur/20211008_100928/train_ar0233-WISSEN_blur_cls7_night_num3494_20211008_100928.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/image_fail/blur/20211008_100928/train_ar0233-WISSEN_blur_cls7_night_num3494_20211008_100928.json",  # noqa
                sample_weight=6,
            ),
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/image_fail/blur/20211008_101249/train_ar0233-RGGB_blur_cls7_day_num1690_20211008_101249.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/image_fail/blur/20211008_101249/train_ar0233-RGGB_blur_cls7_day_num1690_20211008_101249.json",  # noqa
                sample_weight=3,
            ),
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/image_fail/blur/20211008_101710/train_ar0233-RGGB_blur_cls7_night_num4470_20211008_101710.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/image_fail/blur/20211008_101710/train_ar0233-RGGB_blur_cls7_night_num4470_20211008_101710.json",  # noqa
                sample_weight=7,
            ),
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/image_fail/blur/20211009_235508/train_ar0233-RGGB_blur_cls7_day_num1432_20211009_235508.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/image_fail/blur/20211009_235508/train_ar0233-RGGB_blur_cls7_day_num1432_20211009_235508.json",  # noqa
                sample_weight=2,
            ),
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/image_fail/blur/20211015_184244/train_ar0233-WISSEN_blur_cls7_day_num4551_20211015_184244.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/image_fail/blur/20211015_184244/train_ar0233-WISSEN_blur_cls7_day_num4551_20211015_184244.json",  # noqa
                sample_weight=7,
            ),
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/image_fail/blur/20211015_184719/train_ar0233-WISSEN_blur_cls7_night_num1805_20211015_184719.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/image_fail/blur/20211015_184719/train_ar0233-WISSEN_blur_cls7_night_num1805_20211015_184719.json",  # noqa
                sample_weight=3,
            ),
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/image_fail/blur/20211015_185125/train_ar0233-RGGB_blur_cls7_day_num4455_20211015_185125.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/image_fail/blur/20211015_185125/train_ar0233-RGGB_blur_cls7_day_num4455_20211015_185125.json",  # noqa
                sample_weight=7,
            ),
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/image_fail/blur/20211026_105958/train_ar0233-WISSEN_blur_cls7_day_num383_20211026_105958.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/image_fail/blur/20211026_105958/train_ar0233-WISSEN_blur_cls7_day_num383_20211026_105958.json",  # noqa
                sample_weight=1,
            ),
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/image_fail/blur/20211026_110648/train_ar0233-WISSEN_blur_cls7_night_num10117_20211026_110648.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/image_fail/blur/20211026_110648/train_ar0233-WISSEN_blur_cls7_night_num10117_20211026_110648.json",  # noqa
                sample_weight=16,
            ),
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/image_fail/blur/20211110_172204/train_ar0233-WISSEN_blur_cls7_night_num7357_20211110_172204.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/image_fail/blur/20211110_172204/train_ar0233-WISSEN_blur_cls7_night_num7357_20211110_172204.json",  # noqa
                sample_weight=12,
            ),
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/image_fail/blur/20211124_131318/train_ar0233-WISSEN_blur_cls7_night_num7887_20211124_131318.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/image_fail/blur/20211124_131318/train_ar0233-WISSEN_blur_cls7_night_num7887_20211124_131318.json",  # noqa
                sample_weight=13,
            ),
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/image_fail/blur/20211202_125733/train_x3c_blur_cls7_day_num1548_20211202_125733.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/image_fail/blur/20211202_125733/train_x3c_blur_cls7_day_num1548_20211202_125733.json",  # noqa
                sample_weight=2,
            ),
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/image_fail/blur/20211202_130146/train_ar0233-WISSEN_blur_cls7_night_num4234_20211202_130146.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/image_fail/blur/20211202_130146/train_ar0233-WISSEN_blur_cls7_night_num4234_20211202_130146.json",  # noqa
                sample_weight=7,
            ),
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/image_fail/blur/20211202_130732/train_x3c_blur_cls7_night_num4540_20211202_130732.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/image_fail/blur/20211202_130732/train_x3c_blur_cls7_night_num4540_20211202_130732.json",  # noqa
                sample_weight=7,
            ),
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/image_fail/blur/20211208_165446/train_x3c_blur_cls7_day_num9921_20211208_165446.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/image_fail/blur/20211208_165446/train_x3c_blur_cls7_day_num9921_20211208_165446.json",  # noqa
                sample_weight=16,
            ),
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/image_fail/blur/20211208_170626/train_x3c_blur_cls7_night_num7662_20211208_170626.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/image_fail/blur/20211208_170626/train_x3c_blur_cls7_night_num7662_20211208_170626.json",  # noqa
                sample_weight=12,
            ),
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/image_fail/blur/20211214_110951/train_x3c_blur_cls7_day_num7776_20211214_110951.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/image_fail/blur/20211214_110951/train_x3c_blur_cls7_day_num7776_20211214_110951.json",  # noqa
                sample_weight=12,
            ),
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/image_fail/blur/20211214_111833/train_x3c_blur_cls7_night_num5942_20211214_111833.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/image_fail/blur/20211214_111833/train_x3c_blur_cls7_night_num5942_20211214_111833.json",  # noqa
                sample_weight=10,
            ),
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/image_fail/blur/20211222_171010/train_x3c_blur_cls7_day_num4313_20211222_171010.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/image_fail/blur/20211222_171010/train_x3c_blur_cls7_day_num4313_20211222_171010.json",  # noqa
                sample_weight=7,
            ),
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/image_fail/blur/20211222_171321/train_x3c_blur_cls7_night_num1190_20211222_171321.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/image_fail/blur/20211222_171321/train_x3c_blur_cls7_night_num1190_20211222_171321.json",  # noqa
                sample_weight=2,
            ),
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/image_fail/blur/20211222_215121/train_x3c_blur_cls7_day_num3988_20211222_215121.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/image_fail/blur/20211222_215121/train_x3c_blur_cls7_day_num3988_20211222_215121.json",  # noqa
                sample_weight=6,
            ),
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/image_fail/blur/20211228_192820/train_x3c_blur_cls7_day_num3110_20211228_192820.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/image_fail/blur/20211228_192820/train_x3c_blur_cls7_day_num3110_20211228_192820.json",  # noqa
                sample_weight=5,
            ),
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/image_fail/blur/20211228_193513/train_x3c_blur_cls7_night_num7169_20211228_193513.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/image_fail/blur/20211228_193513/train_x3c_blur_cls7_night_num7169_20211228_193513.json",  # noqa
                sample_weight=11,
            ),
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/image_fail/blur/20220112_150702/train_x3c_blur_cls7_day_num4961_20220112_150702.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/image_fail/blur/20220112_150702/train_x3c_blur_cls7_day_num4961_20220112_150702.json",  # noqa
                sample_weight=8,
            ),
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/image_fail/blur/20220112_151404/train_x3c_blur_cls7_night_num3925_20220112_151404.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/image_fail/blur/20220112_151404/train_x3c_blur_cls7_night_num3925_20220112_151404.json",  # noqa
                sample_weight=6,
            ),
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/image_fail/blur/20220125_102315/train_ar0233-WISSEN_blur_cls7_day_num15758_20220125_102315.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/image_fail/blur/20220125_102315/train_ar0233-WISSEN_blur_cls7_day_num15758_20220125_102315.json",  # noqa
                sample_weight=25,
            ),
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/image_fail/blur/20220125_103635/train_x3c_blur_cls7_day_num1032_20220125_103635.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/image_fail/blur/20220125_103635/train_x3c_blur_cls7_day_num1032_20220125_103635.json",  # noqa
                sample_weight=2,
            ),
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/image_fail/blur/20220125_104108/train_ar0233-WISSEN_blur_cls7_night_num3794_20220125_104108.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/image_fail/blur/20220125_104108/train_ar0233-WISSEN_blur_cls7_night_num3794_20220125_104108.json",  # noqa
                sample_weight=6,
            ),
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/image_fail/blur/20220125_104722/train_x3c_blur_cls7_night_num2985_20220125_104722.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/image_fail/blur/20220125_104722/train_x3c_blur_cls7_night_num2985_20220125_104722.json",  # noqa
                sample_weight=5,
            ),
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/image_fail/blur/20220228_221544/train_x3c_blur_cls7_night_num14994_20220228_221544.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/image_fail/blur/20220228_221544/train_x3c_blur_cls7_night_num14994_20220228_221544.json",  # noqa
                sample_weight=24,
            ),
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/image_fail/blur/20220228_224910/train_x3c_blur_cls7_day_num13360_20220228_224910.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/image_fail/blur/20220228_224910/train_x3c_blur_cls7_day_num13360_20220228_224910.json",  # noqa
                sample_weight=21,
            ),
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/image_fail/blur/20220228_233335/train_ar0233-WISSEN_blur_cls7_night_num1784_20220228_233335.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/image_fail/blur/20220228_233335/train_ar0233-WISSEN_blur_cls7_night_num1784_20220228_233335.json",  # noqa
                sample_weight=3,
            ),
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/image_fail/blur/20220307_112550/train_ar0233-WISSEN_blur_cls7_day_num6460_20220307_112550.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/image_fail/blur/20220307_112550/train_ar0233-WISSEN_blur_cls7_day_num6460_20220307_112550.json",  # noqa
                sample_weight=10,
            ),
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/image_fail/blur/20220314_213658/train_x3c_blur_cls7_night_num7035_20220314_213658.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/image_fail/blur/20220314_213658/train_x3c_blur_cls7_night_num7035_20220314_213658.json",  # noqa
                sample_weight=11,
            ),
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/image_fail/blur/20220314_215023/train_x3c_blur_cls7_day_num6820_20220314_215023.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/image_fail/blur/20220314_215023/train_x3c_blur_cls7_day_num6820_20220314_215023.json",  # noqa
                sample_weight=11,
            ),
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/image_fail/blur/20220328_125658/train_ar0233-WISSEN_blur_cls7_night_num13329_20220328_125658.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/image_fail/blur/20220328_125658/train_ar0233-WISSEN_blur_cls7_night_num13329_20220328_125658.json",  # noqa
                sample_weight=21,
            ),
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/image_fail/blur/20220328_131538/train_ar0233-WISSEN_blur_cls7_day_num13491_20220328_131538.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/image_fail/blur/20220328_131538/train_ar0233-WISSEN_blur_cls7_day_num13491_20220328_131538.json",  # noqa
                sample_weight=22,
            ),
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/image_fail/blur/20220328_122021/train_x3c_blur_cls7_night_num4772_20220328_122021.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/image_fail/blur/20220328_122021/train_x3c_blur_cls7_night_num4772_20220328_122021.json",  # noqa
                sample_weight=8,
            ),
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/image_fail/blur/20220328_123413/train_x3c_blur_cls7_day_num8963_20220328_123413.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/image_fail/blur/20220328_123413/train_x3c_blur_cls7_day_num8963_20220328_123413.json",  # noqa
                sample_weight=14,
            ),
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/image_fail/blur/20220417_132204/train_ar0233-306_blur_cls7_night_num480_20220417_132204.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/image_fail/blur/20220417_132204/train_ar0233-306_blur_cls7_night_num480_20220417_132204.json",  # noqa
                sample_weight=1,
            ),
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/image_fail/blur/20220417_132318/train_ar0233-306_blur_cls7_day_num643_20220417_132318.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/image_fail/blur/20220417_132318/train_ar0233-306_blur_cls7_day_num643_20220417_132318.json",  # noqa
                sample_weight=1,
            ),
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/image_fail/blur/20220417_133307/train_x3c_blur_cls7_night_num5323_20220417_133307.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/image_fail/blur/20220417_133307/train_x3c_blur_cls7_night_num5323_20220417_133307.json",  # noqa
                sample_weight=9,
            ),
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/image_fail/blur/20220417_134547/train_x3c_blur_cls7_day_num7742_20220417_134547.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/image_fail/blur/20220417_134547/train_x3c_blur_cls7_day_num7742_20220417_134547.json",  # noqa
                sample_weight=12,
            ),
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/image_fail/blur/20220417_135958/train_ar0233-WISSEN_blur_cls7_night_num6055_20220417_135958.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/image_fail/blur/20220417_135958/train_ar0233-WISSEN_blur_cls7_night_num6055_20220417_135958.json",  # noqa
                sample_weight=10,
            ),
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/image_fail/blur/20220417_140834/train_ar0233-WISSEN_blur_cls7_day_num3828_20220417_140834.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/image_fail/blur/20220417_140834/train_ar0233-WISSEN_blur_cls7_day_num3828_20220417_140834.json",  # noqa
                sample_weight=6,
            ),
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/image_fail/blur/20220503_142704/train_ar0233-306_blur_cls7_night_num3229_20220503_142704.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/image_fail/blur/20220503_142704/train_ar0233-306_blur_cls7_night_num3229_20220503_142704.json",  # noqa
                sample_weight=5,
            ),
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/image_fail/blur/20220503_143834/train_ar0233-306_blur_cls7_day_num8040_20220503_143834.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/image_fail/blur/20220503_143834/train_ar0233-306_blur_cls7_day_num8040_20220503_143834.json",  # noqa
                sample_weight=13,
            ),
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/image_fail/blur/20220503_151213/train_x3c_blur_cls7_night_num9563_20220503_151213.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/image_fail/blur/20220503_151213/train_x3c_blur_cls7_night_num9563_20220503_151213.json",  # noqa
                sample_weight=15,
            ),
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/image_fail/blur/20220503_152032/train_x3c_blur_cls7_day_num1146_20220503_152032.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/image_fail/blur/20220503_152032/train_x3c_blur_cls7_day_num1146_20220503_152032.json",  # noqa
                sample_weight=2,
            ),
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/image_fail/blur/20220520_110000/train_x3c-CC02_blur_cls7_day_num3905_20220520_110000.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/image_fail/blur/20220520_110000/train_x3c-CC02_blur_cls7_day_num3905_20220520_110000.json",  # noqa
                sample_weight=6,
            ),
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/image_fail/blur/20220520_111817/train_ar0233-306_blur_cls7_night_num4484_20220520_111817.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/image_fail/blur/20220520_111817/train_ar0233-306_blur_cls7_night_num4484_20220520_111817.json",  # noqa
                sample_weight=7,
            ),
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/image_fail/blur/20220520_112200/train_ar0233-306_blur_cls7_day_num152_20220520_112200.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/image_fail/blur/20220520_112200/train_ar0233-306_blur_cls7_day_num152_20220520_112200.json",  # noqa
                sample_weight=1,
            ),
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/image_fail/blur/20220520_112708/train_x3c_blur_cls7_day_num3627_20220520_112708.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/image_fail/blur/20220520_112708/train_x3c_blur_cls7_day_num3627_20220520_112708.json",  # noqa
                sample_weight=6,
            ),
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/image_fail/blur/20220520_113406/train_x3c_blur_cls7_night_num2176_20220520_113406.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/image_fail/blur/20220520_113406/train_x3c_blur_cls7_night_num2176_20220520_113406.json",  # noqa
                sample_weight=3,
            ),
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/image_fail/blur/20220520_115011/train_ar0233-WISSEN_blur_cls7_day_num4870_20220520_115011.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/image_fail/blur/20220520_115011/train_ar0233-WISSEN_blur_cls7_day_num4870_20220520_115011.json",  # noqa
                sample_weight=8,
            ),
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/image_fail/blur/20220606_232942/train_x3c-CC02_blur_cls7_night_num7410_20220606_232942.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/image_fail/blur/20220606_232942/train_x3c-CC02_blur_cls7_night_num7410_20220606_232942.json",  # noqa
                sample_weight=12,
            ),
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/image_fail/blur/20220606_233739/train_ar0233-306_blur_cls7_night_num1911_20220606_233739.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/image_fail/blur/20220606_233739/train_ar0233-306_blur_cls7_night_num1911_20220606_233739.json",  # noqa
                sample_weight=3,
            ),
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/image_fail/blur/20220606_234032/train_x3c-CC02_blur_cls7_day_num502_20220606_234032.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/image_fail/blur/20220606_234032/train_x3c-CC02_blur_cls7_day_num502_20220606_234032.json",  # noqa
                sample_weight=1,
            ),
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/image_fail/blur/20220621_003203/train_ar0233-306_blur_cls7_night_num2587_20220621_003203.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/image_fail/blur/20220621_003203/train_ar0233-306_blur_cls7_night_num2587_20220621_003203.json",  # noqa
                sample_weight=4,
            ),
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/image_fail/blur/20220621_003757/train_x3c-CC02_blur_cls7_night_num3079_20220621_003757.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/image_fail/blur/20220621_003757/train_x3c-CC02_blur_cls7_night_num3079_20220621_003757.json",  # noqa
                sample_weight=5,
            ),
            # glare
            dict(
                # 686 images sun glare
                rec_path=f"{root}/multicam_pilot/data/train/image_fail/glare/20210618_000000/train_glare_cls5.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/image_fail/glare/20210618_000000/train_glare_cls5.json",  # noqa
                sample_weight=1,
            ),
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/image_fail/glare/20210708_220605/train_ar0233-RGGB_glare_cls7_day_num1468_20210708_220605.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/image_fail/glare/20210708_220605/train_ar0233-RGGB_glare_cls7_day_num1468_20210708_220605.json",  # noqa
                sample_weight=2,
            ),
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/image_fail/glare/20210708_221637/train_ar0233-RGGB_glare_cls7_night_num248_20210708_221637.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/image_fail/glare/20210708_221637/train_ar0233-RGGB_glare_cls7_night_num248_20210708_221637.json",  # noqa
                sample_weight=1,
            ),
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/image_fail/glare/20210728_004701/train_ar0233-RGGB_glare_cls7_night_num4603_20210728_004701.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/image_fail/glare/20210728_004701/train_ar0233-RGGB_glare_cls7_night_num4603_20210728_004701.json",  # noqa
                sample_weight=8,
            ),
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/image_fail/glare/20210807_224529/train_ar0233-RGGB_glare_cls7_night_num2643_20210807_224529.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/image_fail/glare/20210807_224529/train_ar0233-RGGB_glare_cls7_night_num2643_20210807_224529.json",  # noqa
                sample_weight=4,
            ),
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/image_fail/glare/20211008_100501/train_ar0233-WISSEN_glare_cls7_night_num1941_20211008_100501.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/image_fail/glare/20211008_100501/train_ar0233-WISSEN_glare_cls7_night_num1941_20211008_100501.json",  # noqa
                sample_weight=3,
            ),
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/image_fail/glare/20211202_131053/train_x3c_glare_cls7_day_num222_20211202_131053.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/image_fail/glare/20211202_131053/train_x3c_glare_cls7_day_num222_20211202_131053.json",  # noqa
                sample_weight=1,
            ),
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/image_fail/glare/20211202_131145/train_x3c_glare_cls7_night_num370_20211202_131145.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/image_fail/glare/20211202_131145/train_x3c_glare_cls7_night_num370_20211202_131145.json",  # noqa
                sample_weight=1,
            ),
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/image_fail/glare/20211208_171247/train_x3c_glare_cls7_day_num1034_20211208_171247.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/image_fail/glare/20211208_171247/train_x3c_glare_cls7_day_num1034_20211208_171247.json",  # noqa
                sample_weight=2,
            ),
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/image_fail/glare/20211208_171637/train_x3c_glare_cls7_night_num3412_20211208_171637.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/image_fail/glare/20211208_171637/train_x3c_glare_cls7_night_num3412_20211208_171637.json",  # noqa
                sample_weight=5,
            ),
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/image_fail/glare/20211222_170642/train_ar0233-WISSEN_glare_cls7_night_num477_20211222_170642.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/image_fail/glare/20211222_170642/train_ar0233-WISSEN_glare_cls7_night_num477_20211222_170642.json",  # noqa
                sample_weight=1,
            ),
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/image_fail/glare/20211222_171425/train_x3c_glare_cls7_night_num253_20211222_171425.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/image_fail/glare/20211222_171425/train_x3c_glare_cls7_night_num253_20211222_171425.json",  # noqa
                sample_weight=1,
            ),
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/image_fail/glare/20211222_215449/train_x3c_glare_cls7_day_num236_20211222_215449.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/image_fail/glare/20211222_215449/train_x3c_glare_cls7_day_num236_20211222_215449.json",  # noqa
                sample_weight=1,
            ),
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/image_fail/glare/20211222_215529/train_x3c_glare_cls7_night_num432_20211222_215529.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/image_fail/glare/20211222_215529/train_x3c_glare_cls7_night_num432_20211222_215529.json",  # noqa
                sample_weight=1,
            ),
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/image_fail/glare/20211228_194028/train_x3c_glare_cls7_day_num348_20211228_194028.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/image_fail/glare/20211228_194028/train_x3c_glare_cls7_day_num348_20211228_194028.json",  # noqa
                sample_weight=1,
            ),
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/image_fail/glare/20211228_194242/train_x3c_glare_cls7_night_num2232_20211228_194242.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/image_fail/glare/20211228_194242/train_x3c_glare_cls7_night_num2232_20211228_194242.json",  # noqa
                sample_weight=4,
            ),
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/image_fail/glare/20220112_151755/train_x3c_glare_cls7_day_num1188_20220112_151755.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/image_fail/glare/20220112_151755/train_x3c_glare_cls7_day_num1188_20220112_151755.json",  # noqa
                sample_weight=2,
            ),
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/image_fail/glare/20220112_152113/train_x3c_glare_cls7_night_num2594_20220112_152113.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/image_fail/glare/20220112_152113/train_x3c_glare_cls7_night_num2594_20220112_152113.json",  # noqa
                sample_weight=4,
            ),
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/image_fail/glare/20220112_190329/train_ar0233-WISSEN_glare_cls7_night_num4395_20220112_190329.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/image_fail/glare/20220112_190329/train_ar0233-WISSEN_glare_cls7_night_num4395_20220112_190329.json",  # noqa
                sample_weight=7,
            ),
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/image_fail/glare/20220125_105151/train_x3c_glare_cls7_day_num500_20220125_105151.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/image_fail/glare/20220125_105151/train_x3c_glare_cls7_day_num500_20220125_105151.json",  # noqa
                sample_weight=1,
            ),
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/image_fail/glare/20220125_105304/train_x3c_glare_cls7_night_num321_20220125_105304.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/image_fail/glare/20220125_105304/train_x3c_glare_cls7_night_num321_20220125_105304.json",  # noqa
                sample_weight=1,
            ),
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/image_fail/glare/20220228_223110/train_x3c_glare_cls7_night_num2771_20220228_223110.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/image_fail/glare/20220228_223110/train_x3c_glare_cls7_night_num2771_20220228_223110.json",  # noqa
                sample_weight=4,
            ),
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/image_fail/glare/20220228_233101/train_x3c_glare_cls7_day_num1177_20220228_233101.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/image_fail/glare/20220228_233101/train_x3c_glare_cls7_day_num1177_20220228_233101.json",  # noqa
                sample_weight=2,
            ),
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/image_fail/glare/20220314_212752/train_x3c_glare_cls7_day_num792_20220314_212752.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/image_fail/glare/20220314_212752/train_x3c_glare_cls7_day_num792_20220314_212752.json",  # noqa
                sample_weight=1,
            ),
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/image_fail/glare/20220314_215740/train_x3c_glare_cls7_night_num1391_20220314_215740.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/image_fail/glare/20220314_215740/train_x3c_glare_cls7_night_num1391_20220314_215740.json",  # noqa
                sample_weight=2,
            ),
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/image_fail/glare/20220314_215947/train_ar0233-WISSEN_glare_cls7_night_num496_20220314_215947.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/image_fail/glare/20220314_215947/train_ar0233-WISSEN_glare_cls7_night_num496_20220314_215947.json",  # noqa
                sample_weight=1,
            ),
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/image_fail/glare/20220328_113848/train_x3c_glare_cls7_night_num570_20220328_113848.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/image_fail/glare/20220328_113848/train_x3c_glare_cls7_night_num570_20220328_113848.json",  # noqa
                sample_weight=1,
            ),
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/image_fail/glare/20220328_121446/train_x3c_glare_cls7_day_num107_20220328_121446.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/image_fail/glare/20220328_121446/train_x3c_glare_cls7_day_num107_20220328_121446.json",  # noqa
                sample_weight=1,
            ),
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/image_fail/glare/20220328_124309/train_ar0233-WISSEN_glare_cls7_night_num1321_20220328_124309.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/image_fail/glare/20220328_124309/train_ar0233-WISSEN_glare_cls7_night_num1321_20220328_124309.json",  # noqa
                sample_weight=2,
            ),
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/image_fail/glare/20220328_124626/train_ar0233-WISSEN_glare_cls7_day_num844_20220328_124626.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/image_fail/glare/20220328_124626/train_ar0233-WISSEN_glare_cls7_day_num844_20220328_124626.json",  # noqa
                sample_weight=1,
            ),
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/image_fail/glare/20220417_132054/train_ar0233-306_glare_cls7_night_num232_20220417_132054.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/image_fail/glare/20220417_132054/train_ar0233-306_glare_cls7_night_num232_20220417_132054.json",  # noqa
                sample_weight=1,
            ),
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/image_fail/glare/20220417_132601/train_x3c_glare_cls7_night_num1542_20220417_132601.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/image_fail/glare/20220417_132601/train_x3c_glare_cls7_night_num1542_20220417_132601.json",  # noqa
                sample_weight=2,
            ),
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/image_fail/glare/20220417_135112/train_x3c_glare_cls7_day_num118_20220417_135112.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/image_fail/glare/20220417_135112/train_x3c_glare_cls7_day_num118_20220417_135112.json",  # noqa
                sample_weight=1,
            ),
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/image_fail/glare/20220417_135205/train_ar0233-WISSEN_glare_cls7_night_num419_20220417_135205.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/image_fail/glare/20220417_135205/train_ar0233-WISSEN_glare_cls7_night_num419_20220417_135205.json",  # noqa
                sample_weight=1,
            ),
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/image_fail/glare/20220417_135318/train_ar0233-WISSEN_glare_cls7_day_num615_20220417_135318.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/image_fail/glare/20220417_135318/train_ar0233-WISSEN_glare_cls7_day_num615_20220417_135318.json",  # noqa
                sample_weight=1,
            ),
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/image_fail/glare/20220503_144425/train_ar0233-306_glare_cls7_night_num239_20220503_144425.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/image_fail/glare/20220503_144425/train_ar0233-306_glare_cls7_night_num239_20220503_144425.json",  # noqa
                sample_weight=1,
            ),
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/image_fail/glare/20220503_144511/train_x3c_glare_cls7_day_num185_20220503_144511.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/image_fail/glare/20220503_144511/train_x3c_glare_cls7_day_num185_20220503_144511.json",  # noqa
                sample_weight=1,
            ),
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/image_fail/glare/20220503_145735/train_x3c_glare_cls7_night_num3059_20220503_145735.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/image_fail/glare/20220503_145735/train_x3c_glare_cls7_night_num3059_20220503_145735.json",  # noqa
                sample_weight=5,
            ),
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/image_fail/glare/20220520_111007/train_ar0233-306_glare_cls7_night_num1494_20220520_111007.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/image_fail/glare/20220520_111007/train_ar0233-306_glare_cls7_night_num1494_20220520_111007.json",  # noqa
                sample_weight=2,
            ),
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/image_fail/glare/20220520_114019/train_x3c_glare_cls7_night_num2665_20220520_114019.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/image_fail/glare/20220520_114019/train_x3c_glare_cls7_night_num2665_20220520_114019.json",  # noqa
                sample_weight=4,
            ),
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/image_fail/glare/20220606_233928/train_x3c-CC02_glare_cls7_night_num152_20220606_233928.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/image_fail/glare/20220606_233928/train_x3c-CC02_glare_cls7_night_num152_20220606_233928.json",  # noqa
                sample_weight=1,
            ),
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/image_fail/glare/20220607_000424/train_x3c_glare_cls7_night_num1579_20220607_000424.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/image_fail/glare/20220607_000424/train_x3c_glare_cls7_night_num1579_20220607_000424.json",  # noqa
                sample_weight=3,
            ),
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/image_fail/glare/20220621_002814/train_ar0233-306_glare_cls7_night_num308_20220621_002814.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/image_fail/glare/20220621_002814/train_ar0233-306_glare_cls7_night_num308_20220621_002814.json",  # noqa
                sample_weight=1,
            ),
            # blockage
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/image_fail/blockage/20210922_100235/train_ar0233-WISSEN_blockage_cls7_day_num1639_20210922_100235.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/image_fail/blockage/20210922_100235/train_ar0233-WISSEN_blockage_cls7_day_num1639_20210922_100235.json",  # noqa
                sample_weight=2,
            ),
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/image_fail/blockage/20210630_101019/train_ar0233-WISSEN_blockage_cls7_night_num462_20210630_101019.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/image_fail/blockage/20210630_101019/train_ar0233-WISSEN_blockage_cls7_night_num462_20210630_101019.json",  # noqa
                sample_weight=1,
            ),
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/image_fail/blockage/20210807_223121/train_ar0233-WISSEN_blockage_cls7_day_num8329_20210807_223121.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/image_fail/blockage/20210807_223121/train_ar0233-WISSEN_blockage_cls7_day_num8329_20210807_223121.json",  # noqa
                sample_weight=16,
            ),
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/image_fail/blockage/20210807_233439/train_ar0233-RGGB_blockage_cls7_all_num350_20210807_233439.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/image_fail/blockage/20210807_233439/train_ar0233-RGGB_blockage_cls7_all_num350_20210807_233439.json",  # noqa
                sample_weight=2,
            ),
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/image_fail/blockage/20210822_235801/train_ar0233-WISSEN_blockage_cls7_day_num4301_20210822_235801.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/image_fail/blockage/20210822_235801/train_ar0233-WISSEN_blockage_cls7_day_num4301_20210822_235801.json",  # noqa
                sample_weight=7,
            ),
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/image_fail/blockage/20210823_000353/train_ar0233-WISSEN_blockage_cls7_night_num4581_20210823_000353.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/image_fail/blockage/20210823_000353/train_ar0233-WISSEN_blockage_cls7_night_num4581_20210823_000353.json",  # noqa
                sample_weight=7,
            ),
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/image_fail/blockage/20210902_200116/train_ar0233-RGGB_blockage_cls7_day_num10852_20210902_200116.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/image_fail/blockage/20210902_200116/train_ar0233-RGGB_blockage_cls7_day_num10852_20210902_200116.json",  # noqa
                sample_weight=17,
            ),
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/image_fail/blockage/20211008_102302/train_ar0233-RGGB_blockage_cls7_day_num4240_20211008_102302.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/image_fail/blockage/20211008_102302/train_ar0233-RGGB_blockage_cls7_day_num4240_20211008_102302.json",  # noqa
                sample_weight=7,
            ),
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/image_fail/blockage/20211010_001929/train_ar0233-RGGB_blockage_cls7_day_num11365_20211010_001929.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/image_fail/blockage/20211010_001929/train_ar0233-RGGB_blockage_cls7_day_num11365_20211010_001929.json",  # noqa
                sample_weight=18,
            ),
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/image_fail/blockage/20211015_183543/train_ar0233-WISSEN_blockage_cls7_day_num3877_20211015_183543.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/image_fail/blockage/20211015_183543/train_ar0233-WISSEN_blockage_cls7_day_num3877_20211015_183543.json",  # noqa
                sample_weight=6,
            ),
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/image_fail/blockage/20211026_105851/train_ar0233-WISSEN_blockage_cls7_day_num204_20211026_105851.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/image_fail/blockage/20211026_105851/train_ar0233-WISSEN_blockage_cls7_day_num204_20211026_105851.json",  # noqa
                sample_weight=1,
            ),
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/image_fail/blockage/20211110_171446/train_ar0233-WISSEN_blockage_cls7_day_num1401_20211110_171446.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/image_fail/blockage/20211110_171446/train_ar0233-WISSEN_blockage_cls7_day_num1401_20211110_171446.json",  # noqa
                sample_weight=2,
            ),
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/image_fail/blockage/20211124_130539/train_ar0233-WISSEN_blockage_cls7_day_num1143_20211124_130539.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/image_fail/blockage/20211124_130539/train_ar0233-WISSEN_blockage_cls7_day_num1143_20211124_130539.json",  # noqa
                sample_weight=2,
            ),
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/image_fail/blockage/20211124_130703/train_ar0233-WISSEN_blockage_cls7_night_num115_20211124_130703.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/image_fail/blockage/20211124_130703/train_ar0233-WISSEN_blockage_cls7_night_num115_20211124_130703.json",  # noqa
                sample_weight=1,
            ),
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/image_fail/blockage/20211124_135143/train_ar0233-WISSEN_blockage_cls7_day_num4761_20211124_135143.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/image_fail/blockage/20211124_135143/train_ar0233-WISSEN_blockage_cls7_day_num4761_20211124_135143.json",  # noqa
                sample_weight=8,
            ),
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/image_fail/blockage/20211208_171916/train_x3c_blockage_cls7_day_num498_20211208_171916.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/image_fail/blockage/20211208_171916/train_x3c_blockage_cls7_day_num498_20211208_171916.json",  # noqa
                sample_weight=1,
            ),
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/image_fail/blockage/20211208_172036/train_x3c_blockage_cls7_night_num475_20211208_172036.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/image_fail/blockage/20211208_172036/train_x3c_blockage_cls7_night_num475_20211208_172036.json",  # noqa
                sample_weight=1,
            ),
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/image_fail/blockage/20211222_170428/train_ar0233-WISSEN_blockage_cls7_day_num2650_20211222_170428.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/image_fail/blockage/20211222_170428/train_ar0233-WISSEN_blockage_cls7_day_num2650_20211222_170428.json",  # noqa
                sample_weight=4,
            ),
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/image_fail/blockage/20211228_194508/train_x3c_blockage_cls7_day_num798_20211228_194508.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/image_fail/blockage/20211228_194508/train_x3c_blockage_cls7_day_num798_20211228_194508.json",  # noqa
                sample_weight=1,
            ),
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/image_fail/blockage/20211228_194648/train_x3c_blockage_cls7_night_num845_20211228_194648.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/image_fail/blockage/20211228_194648/train_x3c_blockage_cls7_night_num845_20211228_194648.json",  # noqa
                sample_weight=1,
            ),
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/image_fail/blockage/20220112_152409/train_x3c_blockage_cls7_day_num791_20220112_152409.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/image_fail/blockage/20220112_152409/train_x3c_blockage_cls7_day_num791_20220112_152409.json",  # noqa
                sample_weight=1,
            ),
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/image_fail/blockage/20220125_105400/train_x3c_blockage_cls7_day_num218_20220125_105400.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/image_fail/blockage/20220125_105400/train_x3c_blockage_cls7_day_num218_20220125_105400.json",  # noqa
                sample_weight=1,
            ),
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/image_fail/blockage/20220228_232649/train_x3c_blockage_cls7_night_num284_20220228_232649.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/image_fail/blockage/20220228_232649/train_x3c_blockage_cls7_night_num284_20220228_232649.json",  # noqa
                sample_weight=1,
            ),
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/image_fail/blockage/20220228_232810/train_x3c_blockage_cls7_day_num469_20220228_232810.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/image_fail/blockage/20220228_232810/train_x3c_blockage_cls7_day_num469_20220228_232810.json",  # noqa
                sample_weight=1,
            ),
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/image_fail/blockage/20220314_215905/train_x3c_blockage_cls7_day_num111_20220314_215905.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/image_fail/blockage/20220314_215905/train_x3c_blockage_cls7_day_num111_20220314_215905.json",  # noqa
                sample_weight=1,
            ),
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/image_fail/blockage/20220328_124130/train_x3c_blockage_cls7_day_num256_20220328_124130.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/image_fail/blockage/20220328_124130/train_x3c_blockage_cls7_day_num256_20220328_124130.json",  # noqa
                sample_weight=1,
            ),
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/image_fail/blockage/20220328_132454/train_ar0233-WISSEN_blockage_cls7_night_num477_20220328_132454.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/image_fail/blockage/20220328_132454/train_ar0233-WISSEN_blockage_cls7_night_num477_20220328_132454.json",  # noqa
                sample_weight=1,
            ),
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/image_fail/blockage/20220328_132756/train_ar0233-WISSEN_blockage_cls7_day_num3327_20220328_132756.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/image_fail/blockage/20220328_132756/train_ar0233-WISSEN_blockage_cls7_day_num3327_20220328_132756.json",  # noqa
                sample_weight=5,
            ),
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/image_fail/blockage/20220503_145007/train_x3c_blockage_cls7_night_num3681_20220503_145007.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/image_fail/blockage/20220503_145007/train_x3c_blockage_cls7_night_num3681_20220503_145007.json",  # noqa
                sample_weight=6,
            ),
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/image_fail/blockage/20220503_145311/train_x3c_blockage_cls7_day_num150_20220503_145311.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/image_fail/blockage/20220503_145311/train_x3c_blockage_cls7_day_num150_20220503_145311.json",  # noqa
                sample_weight=1,
            ),
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/image_fail/blockage/20220520_110519/train_x3c-CC02_blockage_cls7_day_num2180_20220520_110519.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/image_fail/blockage/20220520_110519/train_x3c-CC02_blockage_cls7_day_num2180_20220520_110519.json",  # noqa
                sample_weight=3,
            ),
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/image_fail/blockage/20220520_110714/train_ar0233-306_blockage_cls7_night_num139_20220520_110714.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/image_fail/blockage/20220520_110714/train_ar0233-306_blockage_cls7_night_num139_20220520_110714.json",  # noqa
                sample_weight=1,
            ),
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/image_fail/blockage/20220520_113002/train_x3c_blockage_cls7_day_num284_20220520_113002.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/image_fail/blockage/20220520_113002/train_x3c_blockage_cls7_day_num284_20220520_113002.json",  # noqa
                sample_weight=1,
            ),
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/image_fail/blockage/20220520_113620/train_ar0233-306_blockage_cls7_day_num286_20220520_113620.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/image_fail/blockage/20220520_113620/train_ar0233-306_blockage_cls7_day_num286_20220520_113620.json",  # noqa
                sample_weight=1,
            ),
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/image_fail/blockage/20220520_114251/train_ar0233-WISSEN_blockage_cls7_night_num268_20220520_114251.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/image_fail/blockage/20220520_114251/train_ar0233-WISSEN_blockage_cls7_night_num268_20220520_114251.json",  # noqa
                sample_weight=1,
            ),
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/image_fail/blockage/20220520_114419/train_ar0233-WISSEN_blockage_cls7_day_num819_20220520_114419.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/image_fail/blockage/20220520_114419/train_ar0233-WISSEN_blockage_cls7_day_num819_20220520_114419.json",  # noqa
                sample_weight=1,
            ),
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/image_fail/blockage/20220606_234152/train_x3c_blockage_cls7_night_num246_20220606_234152.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/image_fail/blockage/20220606_234152/train_x3c_blockage_cls7_night_num246_20220606_234152.json",  # noqa
                sample_weight=1,
            ),
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/image_fail/blockage/20220607_000149/train_x3c_blockage_cls7_day_num280_20220607_000149.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/image_fail/blockage/20220607_000149/train_x3c_blockage_cls7_day_num280_20220607_000149.json",  # noqa
                sample_weight=1,
            ),
            dict(
                rec_path=f"{root}/multicam_pilot/data/train/image_fail/blockage/20220621_003436/train_ar0233-306_blockage_cls7_night_num106_20220621_003436.rec",  # noqa
                anno_path=f"{root}/multicam_pilot/data/train/image_fail/blockage/20220621_003436/train_ar0233-306_blockage_cls7_night_num106_20220621_003436.json",  # noqa
                sample_weight=1,
            ),
        ],
        val_batch_size_per_ctx=4,
        val_data_paths=None,
    ),
)


datapaths = EasyDict(datapaths)
buckets = [
    "matrix2",
]
