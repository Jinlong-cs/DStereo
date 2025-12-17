def get_train_transform(
    data_type,
    preprocess_type,
    img_scale,
    std,
    mean,
    split_transform,
    split_trans_h,
    split_trans_w,
    wfe=False,
    sensor="ovx8b",
    input_bit=12,
    output_bit=20,
):
    if data_type == "raw":
        common_transforms = [
            dict(
                type="RandomFlip",
                px=0.5,
                py=0,
            ),
            dict(
                type="RandomCrop",
                size=img_scale,
                min_area=-1,
                min_iou=-1,
                keep_raw_pattern=False,
            ),
            dict(
                type="Pad",
                size=img_scale,
            ),
            dict(
                type="ToTensor",
                to_yuv=False,
                split_transform=split_transform,
            ),
            dict(
                type="Normalize",
                mean=mean,
                std=std,
                raw_norm=True,
                split_transform=split_transform,
            ),
        ]
        if preprocess_type == "demosaic":
            resize_transforms = [
                dict(
                    type="Resize",
                    img_scale=img_scale,
                    keep_ratio=True,
                    ratio_range=(0.5, 1.5),
                    raw_scaler_enable=True,
                    sample1c_enable=False,
                    split_transform=split_transform,
                    split_trans_h=split_trans_h,
                    split_trans_w=split_trans_w,
                ),
            ]
            transforms = resize_transforms + common_transforms
        elif preprocess_type == "rawpack":
            pack_transforms = [
                dict(
                    type="RawPack",
                    method=preprocess_type,
                    resize_gt=True,
                ),
                dict(
                    type="Resize",
                    img_scale=img_scale,
                    keep_ratio=True,
                    ratio_range=(0.5, 1.5),
                    raw_scaler_enable=False,
                    sample1c_enable=False,
                    split_transform=split_transform,
                    split_trans_h=split_trans_h,
                    split_trans_w=split_trans_w,
                ),
            ]
            transforms = pack_transforms + common_transforms
        else:
            raise NotImplementedError

        if wfe:
            fe_transforms = [
                dict(
                    type="DEC",
                    sensor=sensor,
                    input_bit=input_bit,
                    output_bit=output_bit,
                ),
                dict(
                    type="DGain",
                    output_bit=output_bit,
                ),
                dict(
                    type="BLC",
                    sensor=sensor,
                    output_bit=output_bit,
                ),
            ]
            transforms = fe_transforms + transforms
    elif data_type == "yuv":
        transforms = [
            dict(
                type="Resize",
                img_scale=img_scale,
                keep_ratio=True,
                ratio_range=(0.5, 1.5),
            ),
            dict(
                type="RandomFlip",
                px=0.5,
                py=0,
            ),
            dict(
                type="RandomCrop",
                size=img_scale,
                min_area=-1,
                min_iou=-1,
            ),
            dict(
                type="Pad",
                size=img_scale,
            ),
            dict(type="ToTensor", to_yuv=True),
            dict(
                type="Normalize",
                mean=mean,
                std=std,
            ),
        ]
    else:
        raise NotImplementedError

    return transforms


def get_val_transform(
    data_type,
    preprocess_type,
    img_scale,
    std,
    mean,
    split_transform,
    split_trans_h,
    split_trans_w,
    wfe=False,
    sensor="ovx8b",
    input_bit=12,
    output_bit=20,
):
    if data_type == "raw":
        if preprocess_type == "demosaic":
            transforms = [
                dict(
                    type="Resize",
                    img_scale=img_scale,
                    keep_ratio=True,
                    raw_scaler_enable=True,
                    sample1c_enable=False,
                    split_transform=split_transform,
                    split_trans_h=split_trans_h,
                    split_trans_w=split_trans_w,
                ),
                dict(
                    type="Pad",
                    size=img_scale,
                ),
                dict(
                    type="ToTensor",
                    to_yuv=False,
                    split_transform=split_transform,
                ),
                dict(
                    type="Normalize",
                    mean=mean,
                    std=std,
                    raw_norm=True,
                    split_transform=split_transform,
                ),
            ]
        elif preprocess_type == "rawpack":
            transforms = [
                dict(
                    type="RawPack",
                    method=preprocess_type,
                    resize_gt=True,
                ),
                dict(
                    type="Resize",
                    img_scale=img_scale,
                    keep_ratio=True,
                    raw_scaler_enable=False,
                    sample1c_enable=False,
                    split_transform=split_transform,
                    split_trans_h=split_trans_h,
                    split_trans_w=split_trans_w,
                ),
                dict(
                    type="Pad",
                    size=img_scale,
                ),
                dict(
                    type="ToTensor",
                    to_yuv=False,
                    split_transform=split_transform,
                ),
                dict(
                    type="Normalize",
                    mean=mean,
                    std=std,
                    raw_norm=True,
                    split_transform=split_transform,
                ),
            ]
        else:
            raise NotImplementedError

        if wfe:
            fe_transforms = [
                dict(
                    type="DEC",
                    sensor=sensor,
                    input_bit=input_bit,
                    output_bit=output_bit,
                ),
                dict(
                    type="DGain",
                    output_bit=output_bit,
                ),
                dict(
                    type="BLC",
                    sensor=sensor,
                    output_bit=output_bit,
                ),
            ]
            transforms = fe_transforms + transforms
    elif data_type == "yuv":
        transforms = [
            dict(
                type="Resize",
                img_scale=img_scale,
                keep_ratio=True,
            ),
            dict(
                type="Pad",
                size=img_scale,
            ),
            dict(
                type="ToTensor",
                to_yuv=True,
            ),
            dict(
                type="Normalize",
                mean=mean,
                std=std,
            ),
        ]
    else:
        raise NotImplementedError

    return transforms


def get_nnloglut_eval_transform(
    preprocess_type,
    img_scale,
    std,
    mean,
    lut_x,
    lut_y,
    pregamma=None,
    sensor="ovx8b",
    dec_in_bit=12,
    dec_out_bit=20,
    nnloglut_out_bit=16,
):
    if preprocess_type == "demosaic":
        transforms = [
            dict(
                type="Resize",
                img_scale=img_scale,
                keep_ratio=True,
                raw_scaler_enable=True,
                sample1c_enable=False,
            ),
            dict(
                type="Pad",
                size=img_scale,
            ),
            dict(
                type="ToTensor",
                to_yuv=False,
            ),
            dict(
                type="Normalize",
                mean=mean,
                std=std,
                raw_norm=True,
            ),
        ]
    elif preprocess_type == "rawpack":
        transforms = [
            dict(
                type="RawPack",
                method=preprocess_type,
                resize_gt=True,
            ),
            dict(
                type="Resize",
                img_scale=img_scale,
                keep_ratio=True,
                raw_scaler_enable=False,
                sample1c_enable=False,
            ),
            dict(
                type="Pad",
                size=img_scale,
            ),
            dict(
                type="ToTensor",
                to_yuv=False,
            ),
            dict(
                type="Normalize",
                mean=mean,
                std=std,
                raw_norm=True,
            ),
        ]
    else:
        raise NotImplementedError

    fe_transforms = [
        dict(
            type="DEC",
            sensor=sensor,
            input_bit=dec_in_bit,
            output_bit=dec_out_bit,
        ),
        dict(
            type="DGain",
            output_bit=dec_out_bit,
        ),
        dict(
            type="BLC",
            sensor=sensor,
            output_bit=dec_out_bit,
        ),
        dict(
            type="NNLogLut",
            input_bit=dec_out_bit,
            output_bit=nnloglut_out_bit,
            lut_x=lut_x,
            lut_y=lut_y,
            pregamma=pregamma,
        ),
    ]
    transforms = fe_transforms + transforms
    return transforms
