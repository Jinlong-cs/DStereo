from cluster_train import parse_full_args, run_submit

if __name__ == "__main__":
    config_path = "projects/halo/cv/configs/imagequality/image_fail_segmentation.py"  # noqa
    train_stages = [
        "float",
        "freeze_bn_1",
        "freeze_bn_2",
        "qat",
        # "int_infer"
    ]
    train_stages = ",".join(train_stages)

    args = parse_full_args(config_path, train_stages)

    run_submit(**vars(args))
