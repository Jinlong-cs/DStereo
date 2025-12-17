def build_model(task_configs, common_inputs=None, mode="train"):
    """This function builds a multitask graph model that generally includes a
    shared backbone, a shared neck, and different task heads which are defined
    in specific task configs.

    Args:
        task_configs: A list contains each task definition which includes task
            dataloader, task inputs, task model, and task metrics.
        common_inputs: A dict contains shared inputs to this graph model, used
            to build input nodes in this graph. This argument is typically
            defined in common.py.
        mode: The multitask graph model type. This mode is associated with
            task_configs. Different mode corresponds to different task
            dataloader, inputs, model, and metrics. In general, we have
            "train", "val", "test", and "infer" modes. The "infer" mode is a
            bit special as it normally uses "BasicBatchProcessor" for inference
            , while other modes use "MultiBatchProcessor". In this way, we
            do not have task inputs anymore since the input image to be
            predicted is the only input, and we need put "orig_img=None" into
            the common_inputs for visualization in inference.

    Returns:
        A multitask graph model.

    """
    assert mode in [
        "train",
        "val",
        "test",
        "infer",
    ], f"Mode should be `train`, `val`, `test` or `infer`, but got {mode}."

    if mode == "infer" and "orig_img" not in common_inputs:
        common_inputs.update(orig_img=None)

    task_inputs = {t.task_name: t.inputs[mode] for t in task_configs}
    task_modules = {t.task_name: t.get_model(mode=mode) for t in task_configs}

    model = dict(
        type="MultitaskGraphModel",
        inputs=common_inputs,
        opt_inputs=None,
        task_inputs=task_inputs,
        task_modules=task_modules,
        funnel_modules={},
        flatten_outputs=False,
        lazy_forward=True,
    )

    return model


if __name__ == "__main__":
    from importlib import import_module

    from multitasks import TASKS

    from hat.registry import build_from_registry
    from projects.cloudmodel.configs.perception_2d.common import common_inputs

    TASK_CONFIGS = [import_module(t) for t in TASKS]
    train_model = build_model(
        TASK_CONFIGS,
        common_inputs,
        mode="train",
    )
    built_train_model = build_from_registry(train_model)
    val_model = build_model(
        TASK_CONFIGS,
        common_inputs,
        mode="val",
    )
    built_val_model = build_from_registry(val_model)
    print("Success")
