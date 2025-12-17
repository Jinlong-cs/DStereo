import torch

NUM_LDMK = 68
LDMK_PAIRS = [
    [0, 16],
    [1, 15],
    [2, 14],
    [3, 13],
    [4, 12],
    [5, 11],
    [6, 10],
    [7, 9],
    [17, 26],
    [18, 25],
    [19, 24],
    [20, 23],
    [21, 22],
    [31, 35],
    [32, 34],
    [36, 45],
    [37, 44],
    [38, 43],
    [39, 42],
    [40, 47],
    [41, 46],
    [48, 54],
    [49, 53],
    [50, 52],
    [61, 63],
    [60, 64],
    [67, 65],
    [58, 56],
    [59, 55],
]

transforms = [
    dict(  # noqa
        type="CropRecROI",
        crop_type="center",
        target_shape=(160, 160, 3),
        base_roi=[51, 51, 205, 205],
        crop_jitter_range=0.0,
    ),
    dict(type="ToTensor"),  # noqa
    dict(type="BgrToYuv444", rgb_input=True),  # noqa
    dict(  # noqa
        type="TorchVisionAdapter",
        interface="Normalize",
        mean=128.0,
        std=128.0,
    ),
]

# data
recs = [
    "/horizon-bucket/interaction/face/landmark/68pts/rec/rec_long_square/train_WFLW.rec",  # noqa
    "/horizon-bucket/interaction/face/landmark/68pts/rec/rec_long_square/train_tianjin201806.rec",  # noqa
]

datasets = [
    dict(  # noqa
        type="LdmkRecDataset",
        filename=rec_file,
        num_ldmk=68,
        use_3d=False,
        task_type="face",
        transforms=transforms,
        ldmk_pairs=LDMK_PAIRS,
    )
    for rec_file in recs
]

data_loader = dict(  # noqa
    type=torch.utils.data.DataLoader,
    dataset=dict(  # noqa
        type=torch.utils.data.ConcatDataset,
        datasets=datasets,
    ),
    batch_size=4,
    shuffle=False,
    num_workers=4,
    pin_memory=True,
)
