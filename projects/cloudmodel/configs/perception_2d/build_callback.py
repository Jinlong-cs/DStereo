def build_callback(
    task_configs,
    num_steps,
    checkpoint_save_root,
    checkpoint_interval_by="epoch",
    checkpoint_save_interval=1,
    log_freq=50,
    do_ema=False,
    do_freeze_bn=False,
):
    """Build callbacks for monitoring training, saving checkpoints and so on.

    Args:
        task_configs: The list of task configs that define all tasks.
        num_steps: The number of total training steps.
        checkpoint_save_root: Where to save the checkpoint.
        checkpoint_interval_by: Save checkpoint by "step" or "epoch".
        checkpoint_save_interval: Interval of saving checkpoint.
        log_freq: Step interval of logging.
        do_ema: Whether to do EMA.
            See hat/callbacks/exponential_moving_average.py
        do_freeze_bn: Whether to freeze bn in specified modules and time. See
            class FreezeModule in hat/callbacks/online_model_trick.py.

    Returns:
        A list of callbacks used in model training.
    """

    callbacks = []

    # status monitor callback
    stat_callback = dict(type="StatsMonitor", log_freq=log_freq)
    callbacks.append(stat_callback)

    # learning rate updater callback, see hat/callbacks/lr_updater.py.
    # Be aware that if stop_by="epoch" in float_trainer, and you want the lr
    # updater to compute the total training steps by itself, max_steps should
    # be set as -1.
    lr_callback = dict(
        type="CosLrUpdater",
        max_steps=num_steps,
        stop_lr=1e-6,
        warmup_by="step",
        warmup_len=2000,
        warmup_begin_lr=1e-6,
        warmup_mode="linear",
        step_log_interval=log_freq,
    )
    callbacks.append(lr_callback)

    # checkpoint callback
    checkpoint_callback = dict(
        type="Checkpoint",
        save_dir=checkpoint_save_root,
        name_prefix="big_model_",
        interval_by=checkpoint_interval_by,
        save_interval=checkpoint_save_interval,
        save_on_train_end=False if num_steps == 0 else True,
        strict_match=False,
        best_refer_metric=None,
        save_hash=False,
    )
    callbacks.append(checkpoint_callback)

    # task's train metric updater callback
    for t in task_configs:
        # for training
        callbacks.append(t.train_metric_updater)

    if do_ema:
        # ema callback
        ema_callback = dict(type="ExponentialMovingAverage")
        callbacks.append(ema_callback)

    if do_freeze_bn:
        # freeze bn callback. If you use this callback, please remember to set
        # the parameters required by this callback carefully.
        # See hat/callbacks/online_model_trick.py
        freeze_bn_callback = dict(
            type="FreezeModule",
            modules=[["backbone"], ["neck"]],
            step_or_epoch=[10000, 15000],
            update_by="step",
            only_batchnorm=True,
        )
        callbacks.append(freeze_bn_callback)

    return callbacks
