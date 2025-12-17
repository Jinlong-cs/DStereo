from easydict import EasyDict

root = "dmpv2://matrix2"
datapaths = {
    "image_fail_parsing": {
        "train_data_paths": [
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/image_fail/blur/20221216_225253/train_CW-OX3GB-O100-065-L_blur_cls7_day_num38_20221216_225253",
                "sample_weight": 1,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/image_fail/blur/train_CW-OX3GB-O100-065-L_blur_cls7_num13250_all_20221104_181556/20221104-184612/data",
                "sample_weight": 5,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/image_fail/blockage/20220829_015057/train_X3C-OFT224_blockage_cls7_day_num2999_20220829_015057",
                "sample_weight": 5,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/image_fail/blockage/20220828_214231/train_X3C-OFT224_blockage_cls7_day_num226_20220828_214231",
                "sample_weight": 1,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/image_fail/blockage/20220828_214329/train_X3C-OFT224_blockage_cls7_day_num144_20220828_214329",
                "sample_weight": 1,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/image_fail/blockage/20220828_214635/train_X3C-OFT224_blockage_cls7_day_num1800_20220828_214635",
                "sample_weight": 3,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/image_fail/blockage/20220828_215234/train_X3C-OFT224_blockage_cls7_day_num1800_20220828_215234",
                "sample_weight": 3,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/image_fail/blockage/20220828_215632/train_X3C-OFT224_blockage_cls7_day_num1300_20220828_215632",
                "sample_weight": 2,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/image_fail/blockage/20220828_215837/train_X3C-OFT224_blockage_cls7_day_num378_20220828_215837",
                "sample_weight": 1,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/image_fail/blockage/20220826_154623/train_X3C-OFT224_blockage_cls7_day_num1800_20220826_154623",
                "sample_weight": 3,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/image_fail/blockage/20220826_031402/train_X3C-OFT224_blockage_cls7_day_num3032_20220826_031402",
                "sample_weight": 5,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/image_fail/blockage/20220826_031615/train_X3C-OFT224_blockage_cls7_day_num3000_20220826_031615",
                "sample_weight": 5,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/image_fail/blur/20220628_143017/train_CO_OX3GB_D100_L_num612_blur_cls7_night_20220628_143017",
                "sample_weight": 1,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/image_fail/glare/20220629_143232/train_CW_A23GB_A114_L_num221_glare_cls7_night_20220629_143232",
                "sample_weight": 1,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/image_fail/blur/20220629_143637/train_CW_A23GB_A114_L_num2189_blur_cls7_night_20220629_143637",
                "sample_weight": 4,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/image_fail/blockage/20220629_143903/train_CW_A23GB_A114_L_num97_blockage_cls7_day_20220629_143903",
                "sample_weight": 1,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/image_fail/blur/20220629_144057/train_CW_A23GB_A114_L_num988_blur_cls7_day_20220629_144057",
                "sample_weight": 2,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/image_fail/blockage/20220629_144219/train_X3C-OFT224_blockage_cls7_day_num96_20220629_144219",
                "sample_weight": 1,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/image_fail/blur/20220629_151014/train_CO_OX3GB_D100_L_num1834_blur_cls7_day_20220629_151014",
                "sample_weight": 3,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/image_fail/blur/20220629_152225/train_X3C-OFT224_blur_cls7_night_num6068_20220629_152225",
                "sample_weight": 10,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/image_fail/blur/20210627_220752/train_ar0233-RGGB_blur_cls5_num2860_20210627_220752",
                "sample_weight": 4,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/image_fail/blur/20210708_181516/train_ar0233-RGGB_blur_cls7_all_num1531_20210708_181516",
                "sample_weight": 2,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/image_fail/blur/20210708_212718/train_ar0233-RGGB_blur_cls7_day_num2785_20210708_212718",
                "sample_weight": 4,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/image_fail/blur/20210708_214348/train_ar0233-RGGB_blur_cls7_all_num2465_20210708_214348",
                "sample_weight": 4,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/image_fail/blur/20210728_010308/train_ar0233-RGGB_blur_cls7_all_num2120_20210728_010308",
                "sample_weight": 4,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/image_fail/blur/20210728_011114/train_ar0233-RGGB_blur_cls7_day_num1961_20210728_011114",
                "sample_weight": 3,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/image_fail/blur/20210807_215030/train_ar0233-RGGB_blur_cls7_all_num9751_20210807_215030",
                "sample_weight": 16,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/image_fail/blur/20210807_230644/train_ar0233-WISSEN_blur_cls7_day_num2990_20210807_230644",
                "sample_weight": 5,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/image_fail/blur/20210807_231606/train_ar0233-WISSEN_blur_cls7_night_num244_20210807_231606",
                "sample_weight": 1,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/image_fail/blur/20210807_232342/train_ar0233-RGGB_blur_cls7_day_num3753_20210807_232342",
                "sample_weight": 7,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/image_fail/blur/20210807_233108/train_ar0233-RGGB_blur_cls7_night_num533_20210807_233108",
                "sample_weight": 2,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/image_fail/blur/20210816_010056/train_ar0233-RGGB_blur_cls7_day_num5040_20210816_010056",
                "sample_weight": 8,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/image_fail/blur/20211008_100708/train_ar0233-WISSEN_blur_cls7_day_num149_20211008_100708",
                "sample_weight": 1,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/image_fail/blur/20211008_100928/train_ar0233-WISSEN_blur_cls7_night_num3494_20211008_100928",
                "sample_weight": 6,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/image_fail/blur/20211008_101249/train_ar0233-RGGB_blur_cls7_day_num1690_20211008_101249",
                "sample_weight": 3,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/image_fail/blur/20211008_101710/train_ar0233-RGGB_blur_cls7_night_num4470_20211008_101710",
                "sample_weight": 7,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/image_fail/blur/20211009_235508/train_ar0233-RGGB_blur_cls7_day_num1432_20211009_235508",
                "sample_weight": 2,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/image_fail/blur/20211015_184244/train_ar0233-WISSEN_blur_cls7_day_num4551_20211015_184244",
                "sample_weight": 7,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/image_fail/blur/20211015_184719/train_ar0233-WISSEN_blur_cls7_night_num1805_20211015_184719",
                "sample_weight": 3,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/image_fail/blur/20211015_185125/train_ar0233-RGGB_blur_cls7_day_num4455_20211015_185125",
                "sample_weight": 7,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/image_fail/blur/20211026_105958/train_ar0233-WISSEN_blur_cls7_day_num383_20211026_105958",
                "sample_weight": 1,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/image_fail/blur/20211026_110648/train_ar0233-WISSEN_blur_cls7_night_num10117_20211026_110648",
                "sample_weight": 16,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/image_fail/blur/20211110_172204/train_ar0233-WISSEN_blur_cls7_night_num7357_20211110_172204",
                "sample_weight": 12,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/image_fail/blur/20211124_131318/train_ar0233-WISSEN_blur_cls7_night_num7887_20211124_131318",
                "sample_weight": 13,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/image_fail/blur/20211202_125733/train_x3c_blur_cls7_day_num1548_20211202_125733",
                "sample_weight": 2,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/image_fail/blur/20211202_130146/train_ar0233-WISSEN_blur_cls7_night_num4234_20211202_130146",
                "sample_weight": 7,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/image_fail/blur/20211202_130732/train_x3c_blur_cls7_night_num4540_20211202_130732",
                "sample_weight": 7,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/image_fail/blur/20211208_165446/train_x3c_blur_cls7_day_num9921_20211208_165446",
                "sample_weight": 16,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/image_fail/blur/20211208_170626/train_x3c_blur_cls7_night_num7662_20211208_170626",
                "sample_weight": 12,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/image_fail/blur/20211214_110951/train_x3c_blur_cls7_day_num7776_20211214_110951",
                "sample_weight": 12,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/image_fail/blur/20211214_111833/train_x3c_blur_cls7_night_num5942_20211214_111833",
                "sample_weight": 10,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/image_fail/blur/20211222_171010/train_x3c_blur_cls7_day_num4313_20211222_171010",
                "sample_weight": 7,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/image_fail/blur/20211222_171321/train_x3c_blur_cls7_night_num1190_20211222_171321",
                "sample_weight": 2,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/image_fail/blur/20211222_215121/train_x3c_blur_cls7_day_num3988_20211222_215121",
                "sample_weight": 6,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/image_fail/blur/20211228_192820/train_x3c_blur_cls7_day_num3110_20211228_192820",
                "sample_weight": 5,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/image_fail/blur/20211228_193513/train_x3c_blur_cls7_night_num7169_20211228_193513",
                "sample_weight": 11,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/image_fail/blur/20220112_150702/train_x3c_blur_cls7_day_num4961_20220112_150702",
                "sample_weight": 8,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/image_fail/blur/20220112_151404/train_x3c_blur_cls7_night_num3925_20220112_151404",
                "sample_weight": 6,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/image_fail/blur/20220125_102315/train_ar0233-WISSEN_blur_cls7_day_num15758_20220125_102315",
                "sample_weight": 25,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/image_fail/blur/20220125_103635/train_x3c_blur_cls7_day_num1032_20220125_103635",
                "sample_weight": 2,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/image_fail/blur/20220125_104108/train_ar0233-WISSEN_blur_cls7_night_num3794_20220125_104108",
                "sample_weight": 6,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/image_fail/blur/20220125_104722/train_x3c_blur_cls7_night_num2985_20220125_104722",
                "sample_weight": 5,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/image_fail/blur/20220228_221544/train_x3c_blur_cls7_night_num14994_20220228_221544",
                "sample_weight": 24,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/image_fail/blur/20220228_224910/train_x3c_blur_cls7_day_num13360_20220228_224910",
                "sample_weight": 21,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/image_fail/blur/20220228_233335/train_ar0233-WISSEN_blur_cls7_night_num1784_20220228_233335",
                "sample_weight": 3,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/image_fail/blur/20220307_112550/train_ar0233-WISSEN_blur_cls7_day_num6460_20220307_112550",
                "sample_weight": 10,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/image_fail/blur/20220314_213658/train_x3c_blur_cls7_night_num7035_20220314_213658",
                "sample_weight": 11,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/image_fail/blur/20220314_215023/train_x3c_blur_cls7_day_num6820_20220314_215023",
                "sample_weight": 11,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/image_fail/blur/20220328_125658/train_ar0233-WISSEN_blur_cls7_night_num13329_20220328_125658",
                "sample_weight": 21,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/image_fail/blur/20220328_131538/train_ar0233-WISSEN_blur_cls7_day_num13491_20220328_131538",
                "sample_weight": 22,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/image_fail/blur/20220328_122021/train_x3c_blur_cls7_night_num4772_20220328_122021",
                "sample_weight": 8,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/image_fail/blur/20220328_123413/train_x3c_blur_cls7_day_num8963_20220328_123413",
                "sample_weight": 14,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/image_fail/blur/20220417_132204/train_ar0233-306_blur_cls7_night_num480_20220417_132204",
                "sample_weight": 1,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/image_fail/blur/20220417_132318/train_ar0233-306_blur_cls7_day_num643_20220417_132318",
                "sample_weight": 1,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/image_fail/blur/20220417_133307/train_x3c_blur_cls7_night_num5323_20220417_133307",
                "sample_weight": 9,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/image_fail/blur/20220417_134547/train_x3c_blur_cls7_day_num7742_20220417_134547",
                "sample_weight": 12,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/image_fail/blur/20220417_135958/train_ar0233-WISSEN_blur_cls7_night_num6055_20220417_135958",
                "sample_weight": 10,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/image_fail/blur/20220417_140834/train_ar0233-WISSEN_blur_cls7_day_num3828_20220417_140834",
                "sample_weight": 6,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/image_fail/blur/20220503_142704/train_ar0233-306_blur_cls7_night_num3229_20220503_142704",
                "sample_weight": 5,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/image_fail/blur/20220503_143834/train_ar0233-306_blur_cls7_day_num8040_20220503_143834",
                "sample_weight": 13,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/image_fail/blur/20220503_151213/train_x3c_blur_cls7_night_num9563_20220503_151213",
                "sample_weight": 15,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/image_fail/blur/20220503_152032/train_x3c_blur_cls7_day_num1146_20220503_152032",
                "sample_weight": 2,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/image_fail/blur/20220520_110000/train_x3c-CC02_blur_cls7_day_num3905_20220520_110000",
                "sample_weight": 6,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/image_fail/blur/20220520_111817/train_ar0233-306_blur_cls7_night_num4484_20220520_111817",
                "sample_weight": 7,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/image_fail/blur/20220520_112200/train_ar0233-306_blur_cls7_day_num152_20220520_112200",
                "sample_weight": 1,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/image_fail/blur/20220520_112708/train_x3c_blur_cls7_day_num3627_20220520_112708",
                "sample_weight": 6,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/image_fail/blur/20220520_113406/train_x3c_blur_cls7_night_num2176_20220520_113406",
                "sample_weight": 3,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/image_fail/blur/20220520_115011/train_ar0233-WISSEN_blur_cls7_day_num4870_20220520_115011",
                "sample_weight": 8,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/image_fail/blur/20220606_232942/train_x3c-CC02_blur_cls7_night_num7410_20220606_232942",
                "sample_weight": 12,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/image_fail/blur/20220606_233739/train_ar0233-306_blur_cls7_night_num1911_20220606_233739",
                "sample_weight": 3,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/image_fail/blur/20220606_234032/train_x3c-CC02_blur_cls7_day_num502_20220606_234032",
                "sample_weight": 1,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/image_fail/blur/20220621_003203/train_ar0233-306_blur_cls7_night_num2587_20220621_003203",
                "sample_weight": 4,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/image_fail/blur/20220621_003757/train_x3c-CC02_blur_cls7_night_num3079_20220621_003757",
                "sample_weight": 5,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/image_fail/glare/20210618_000000/train_glare_cls5",
                "sample_weight": 1,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/image_fail/glare/20210708_220605/train_ar0233-RGGB_glare_cls7_day_num1468_20210708_220605",
                "sample_weight": 2,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/image_fail/glare/20210708_221637/train_ar0233-RGGB_glare_cls7_night_num248_20210708_221637",
                "sample_weight": 1,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/image_fail/glare/20210728_004701/train_ar0233-RGGB_glare_cls7_night_num4603_20210728_004701",
                "sample_weight": 8,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/image_fail/glare/20210807_224529/train_ar0233-RGGB_glare_cls7_night_num2643_20210807_224529",
                "sample_weight": 4,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/image_fail/glare/20211008_100501/train_ar0233-WISSEN_glare_cls7_night_num1941_20211008_100501",
                "sample_weight": 3,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/image_fail/glare/20211202_131053/train_x3c_glare_cls7_day_num222_20211202_131053",
                "sample_weight": 1,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/image_fail/glare/20211202_131145/train_x3c_glare_cls7_night_num370_20211202_131145",
                "sample_weight": 1,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/image_fail/glare/20211208_171247/train_x3c_glare_cls7_day_num1034_20211208_171247",
                "sample_weight": 2,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/image_fail/glare/20211208_171637/train_x3c_glare_cls7_night_num3412_20211208_171637",
                "sample_weight": 5,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/image_fail/glare/20211222_170642/train_ar0233-WISSEN_glare_cls7_night_num477_20211222_170642",
                "sample_weight": 1,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/image_fail/glare/20211222_171425/train_x3c_glare_cls7_night_num253_20211222_171425",
                "sample_weight": 1,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/image_fail/glare/20211222_215449/train_x3c_glare_cls7_day_num236_20211222_215449",
                "sample_weight": 1,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/image_fail/glare/20211222_215529/train_x3c_glare_cls7_night_num432_20211222_215529",
                "sample_weight": 1,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/image_fail/glare/20211228_194028/train_x3c_glare_cls7_day_num348_20211228_194028",
                "sample_weight": 1,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/image_fail/glare/20211228_194242/train_x3c_glare_cls7_night_num2232_20211228_194242",
                "sample_weight": 4,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/image_fail/glare/20220112_151755/train_x3c_glare_cls7_day_num1188_20220112_151755",
                "sample_weight": 2,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/image_fail/glare/20220112_152113/train_x3c_glare_cls7_night_num2594_20220112_152113",
                "sample_weight": 4,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/image_fail/glare/20220112_190329/train_ar0233-WISSEN_glare_cls7_night_num4395_20220112_190329",
                "sample_weight": 7,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/image_fail/glare/20220125_105151/train_x3c_glare_cls7_day_num500_20220125_105151",
                "sample_weight": 1,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/image_fail/glare/20220125_105304/train_x3c_glare_cls7_night_num321_20220125_105304",
                "sample_weight": 1,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/image_fail/glare/20220228_223110/train_x3c_glare_cls7_night_num2771_20220228_223110",
                "sample_weight": 4,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/image_fail/glare/20220228_233101/train_x3c_glare_cls7_day_num1177_20220228_233101",
                "sample_weight": 2,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/image_fail/glare/20220314_212752/train_x3c_glare_cls7_day_num792_20220314_212752",
                "sample_weight": 1,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/image_fail/glare/20220314_215740/train_x3c_glare_cls7_night_num1391_20220314_215740",
                "sample_weight": 2,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/image_fail/glare/20220314_215947/train_ar0233-WISSEN_glare_cls7_night_num496_20220314_215947",
                "sample_weight": 1,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/image_fail/glare/20220328_113848/train_x3c_glare_cls7_night_num570_20220328_113848",
                "sample_weight": 1,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/image_fail/glare/20220328_121446/train_x3c_glare_cls7_day_num107_20220328_121446",
                "sample_weight": 1,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/image_fail/glare/20220328_124309/train_ar0233-WISSEN_glare_cls7_night_num1321_20220328_124309",
                "sample_weight": 2,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/image_fail/glare/20220328_124626/train_ar0233-WISSEN_glare_cls7_day_num844_20220328_124626",
                "sample_weight": 1,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/image_fail/glare/20220417_132054/train_ar0233-306_glare_cls7_night_num232_20220417_132054",
                "sample_weight": 1,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/image_fail/glare/20220417_132601/train_x3c_glare_cls7_night_num1542_20220417_132601",
                "sample_weight": 2,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/image_fail/glare/20220417_135112/train_x3c_glare_cls7_day_num118_20220417_135112",
                "sample_weight": 1,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/image_fail/glare/20220417_135205/train_ar0233-WISSEN_glare_cls7_night_num419_20220417_135205",
                "sample_weight": 1,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/image_fail/glare/20220417_135318/train_ar0233-WISSEN_glare_cls7_day_num615_20220417_135318",
                "sample_weight": 1,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/image_fail/glare/20220503_144425/train_ar0233-306_glare_cls7_night_num239_20220503_144425",
                "sample_weight": 1,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/image_fail/glare/20220503_144511/train_x3c_glare_cls7_day_num185_20220503_144511",
                "sample_weight": 1,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/image_fail/glare/20220503_145735/train_x3c_glare_cls7_night_num3059_20220503_145735",
                "sample_weight": 5,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/image_fail/glare/20220520_111007/train_ar0233-306_glare_cls7_night_num1494_20220520_111007",
                "sample_weight": 2,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/image_fail/glare/20220520_114019/train_x3c_glare_cls7_night_num2665_20220520_114019",
                "sample_weight": 4,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/image_fail/glare/20220606_233928/train_x3c-CC02_glare_cls7_night_num152_20220606_233928",
                "sample_weight": 1,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/image_fail/glare/20220607_000424/train_x3c_glare_cls7_night_num1579_20220607_000424",
                "sample_weight": 3,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/image_fail/glare/20220621_002814/train_ar0233-306_glare_cls7_night_num308_20220621_002814",
                "sample_weight": 1,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/image_fail/blockage/20210922_100235/train_ar0233-WISSEN_blockage_cls7_day_num1639_20210922_100235",
                "sample_weight": 2,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/image_fail/blockage/20210630_101019/train_ar0233-WISSEN_blockage_cls7_night_num462_20210630_101019",
                "sample_weight": 1,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/image_fail/blockage/20210807_223121/train_ar0233-WISSEN_blockage_cls7_day_num8329_20210807_223121",
                "sample_weight": 16,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/image_fail/blockage/20210807_233439/train_ar0233-RGGB_blockage_cls7_all_num350_20210807_233439",
                "sample_weight": 2,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/image_fail/blockage/20210822_235801/train_ar0233-WISSEN_blockage_cls7_day_num4301_20210822_235801",
                "sample_weight": 7,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/image_fail/blockage/20210823_000353/train_ar0233-WISSEN_blockage_cls7_night_num4581_20210823_000353",
                "sample_weight": 7,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/image_fail/blockage/20210902_200116/train_ar0233-RGGB_blockage_cls7_day_num10852_20210902_200116",
                "sample_weight": 17,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/image_fail/blockage/20211008_102302/train_ar0233-RGGB_blockage_cls7_day_num4240_20211008_102302",
                "sample_weight": 7,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/image_fail/blockage/20211010_001929/train_ar0233-RGGB_blockage_cls7_day_num11365_20211010_001929",
                "sample_weight": 18,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/image_fail/blockage/20211015_183543/train_ar0233-WISSEN_blockage_cls7_day_num3877_20211015_183543",
                "sample_weight": 6,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/image_fail/blockage/20211026_105851/train_ar0233-WISSEN_blockage_cls7_day_num204_20211026_105851",
                "sample_weight": 1,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/image_fail/blockage/20211110_171446/train_ar0233-WISSEN_blockage_cls7_day_num1401_20211110_171446",
                "sample_weight": 2,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/image_fail/blockage/20211124_130539/train_ar0233-WISSEN_blockage_cls7_day_num1143_20211124_130539",
                "sample_weight": 2,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/image_fail/blockage/20211124_130703/train_ar0233-WISSEN_blockage_cls7_night_num115_20211124_130703",
                "sample_weight": 1,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/image_fail/blockage/20211124_135143/train_ar0233-WISSEN_blockage_cls7_day_num4761_20211124_135143",
                "sample_weight": 8,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/image_fail/blockage/20211208_171916/train_x3c_blockage_cls7_day_num498_20211208_171916",
                "sample_weight": 1,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/image_fail/blockage/20211208_172036/train_x3c_blockage_cls7_night_num475_20211208_172036",
                "sample_weight": 1,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/image_fail/blockage/20211222_170428/train_ar0233-WISSEN_blockage_cls7_day_num2650_20211222_170428",
                "sample_weight": 4,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/image_fail/blockage/20211228_194508/train_x3c_blockage_cls7_day_num798_20211228_194508",
                "sample_weight": 1,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/image_fail/blockage/20211228_194648/train_x3c_blockage_cls7_night_num845_20211228_194648",
                "sample_weight": 1,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/image_fail/blockage/20220112_152409/train_x3c_blockage_cls7_day_num791_20220112_152409",
                "sample_weight": 1,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/image_fail/blockage/20220125_105400/train_x3c_blockage_cls7_day_num218_20220125_105400",
                "sample_weight": 1,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/image_fail/blockage/20220228_232649/train_x3c_blockage_cls7_night_num284_20220228_232649",
                "sample_weight": 1,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/image_fail/blockage/20220228_232810/train_x3c_blockage_cls7_day_num469_20220228_232810",
                "sample_weight": 1,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/image_fail/blockage/20220314_215905/train_x3c_blockage_cls7_day_num111_20220314_215905",
                "sample_weight": 1,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/image_fail/blockage/20220328_124130/train_x3c_blockage_cls7_day_num256_20220328_124130",
                "sample_weight": 1,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/image_fail/blockage/20220328_132454/train_ar0233-WISSEN_blockage_cls7_night_num477_20220328_132454",
                "sample_weight": 1,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/image_fail/blockage/20220328_132756/train_ar0233-WISSEN_blockage_cls7_day_num3327_20220328_132756",
                "sample_weight": 5,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/image_fail/blockage/20220503_145007/train_x3c_blockage_cls7_night_num3681_20220503_145007",
                "sample_weight": 6,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/image_fail/blockage/20220503_145311/train_x3c_blockage_cls7_day_num150_20220503_145311",
                "sample_weight": 1,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/image_fail/blockage/20220520_110519/train_x3c-CC02_blockage_cls7_day_num2180_20220520_110519",
                "sample_weight": 3,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/image_fail/blockage/20220520_110714/train_ar0233-306_blockage_cls7_night_num139_20220520_110714",
                "sample_weight": 1,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/image_fail/blockage/20220520_113002/train_x3c_blockage_cls7_day_num284_20220520_113002",
                "sample_weight": 1,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/image_fail/blockage/20220520_113620/train_ar0233-306_blockage_cls7_day_num286_20220520_113620",
                "sample_weight": 1,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/image_fail/blockage/20220520_114251/train_ar0233-WISSEN_blockage_cls7_night_num268_20220520_114251",
                "sample_weight": 1,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/image_fail/blockage/20220520_114419/train_ar0233-WISSEN_blockage_cls7_day_num819_20220520_114419",
                "sample_weight": 1,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/image_fail/blockage/20220606_234152/train_x3c_blockage_cls7_night_num246_20220606_234152",
                "sample_weight": 1,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/image_fail/blockage/20220607_000149/train_x3c_blockage_cls7_day_num280_20220607_000149",
                "sample_weight": 1,
            },
            {
                "data_path": f"{root}/multicam_pilot/data/lmdb_datasets/multicam_pilot/data/train/image_fail/blockage/20220621_003436/train_ar0233-306_blockage_cls7_night_num106_20220621_003436",
                "sample_weight": 1,
            },
        ]
    },
}
datapaths = EasyDict(datapaths)
