import os

from aidisdk.utils.env import token_from_env

from hat.callbacks.callbacks import CallbackMixin
from projects.cloudmodel.configs.perception_2d.evaluation import (
    reformat_prediction_fn,
)


# This function is used to build dataloaders and their corresponding callbacks
# for evaluating a model on multiple AIDI eval datasets at a time.
def build_aidi_eval_callback(
    task_configs, save_root, pipeline_test, tags=("resize", "merge")
):
    triplets = []
    for t in task_configs:
        task_name = t.task_name
        loaders = t.test_dataloader
        for loader in loaders:
            dataset_id = loader["dataset"]["data_path"].split("/")[-2]
            prediction_name = "cloudmodel_" + task_name

            project_id = os.getenv("PROJECT_ID", None)
            user_token = token_from_env()

            assert project_id is not None and user_token is not None, (
                "You need to specify your project id (PROJECT_ID) and "
                "user_token (USER_TOKEN) by exporting them as environment "
                "variables or setting them here directly."
            )

            aidi_eval_callback = (
                CallbackMixin()
                if pipeline_test
                else dict(
                    type="AIDIEval",
                    output_root=os.path.join(save_root, "prediction"),
                    prediction_name=prediction_name,
                    prediction_tags=t.name_dict
                    if task_name.endswith("_classification")
                    else list(tags),
                    project_id=project_id,
                    aidi_eval_token=user_token,
                    aidi_eval_dataset_name=None,
                    aidi_eval_dataset_id=[dataset_id],
                    reformat_input_fn=None,
                    reformat_output_fn=reformat_prediction_fn,
                    reformat_out_fn_kwargs=dict(
                        task_name=task_name,
                        name_dict=t.name_dict
                        if task_name.endswith("_classification")
                        else None,
                    ),
                    overwrite=True,
                    cpu=4,
                    cpu_mem_ratio=6,
                )
            )

            task_loader = dict(
                type="MultitaskLoader",
                loaders={task_name: loader},
                return_task=True,
                custom_length=50 if pipeline_test else None,
                wrap_batch=True,
            )

            triplets.append((task_name, task_loader, aidi_eval_callback))

    return triplets


# This function can be used to do online AIDI evaluation. Alternatively, you
# can use projects/cloudmodel/aidi_eval.py to do AIDI evaluation after training
# is completed.
def build_evaluation(
    test_model,
    test_batch_processor,
    task_configs,
    save_root,
    vis_callback=None,
    **kwargs
):
    evaluation_callbacks = []

    triplets = build_aidi_eval_callback(
        task_configs,
        save_root,
    )

    for triplet in triplets:
        task_name, task_loader, aidi_eval_callback = triplet
        callbacks = [aidi_eval_callback]
        if vis_callback is not None:
            callbacks.extend(vis_callback)

        evaluation_callback = dict(
            type="Validation",
            val_model=test_model,
            data_loader=task_loader,
            batch_processor=test_batch_processor,
            callbacks=callbacks,
            val_interval=kwargs.get("val_interval", 1000000),
            interval_by=kwargs.get("interval_by", "epoch"),
            log_interval=kwargs.get("log_interval", 50),
            init_with_train_model=True,
        )
        evaluation_callbacks.append(evaluation_callback)

    return evaluation_callbacks
