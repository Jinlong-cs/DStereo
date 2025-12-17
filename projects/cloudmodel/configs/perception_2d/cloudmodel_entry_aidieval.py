from importlib import import_module

from projects.cloudmodel.configs.perception_2d.build_evaluation import (
    build_aidi_eval_callback,
)
from projects.cloudmodel.configs.perception_2d.build_model import build_model
from projects.cloudmodel.configs.perception_2d.common import (
    checkpoint_path_evaluation,
    common_inputs,
    pipeline_test,
    save_root_evaluation,
)
from projects.cloudmodel.configs.perception_2d.multitasks import TASKS

TASK_CONFIGS = [import_module(t) for t in TASKS]
test_model = build_model(TASK_CONFIGS, common_inputs, mode="test")
test_batch_processor = dict(
    type="MultiBatchProcessor",
    need_grad_update=False,
    enable_amp=False,
)

triplets = build_aidi_eval_callback(
    TASK_CONFIGS, save_root_evaluation, pipeline_test=pipeline_test
)
if pipeline_test:
    triplets = triplets[:1]

loaders_list = []
callbacks_list = []
for triplet in triplets:
    task_name, task_loader, aidi_eval_callback = triplet
    loaders_list.append(task_loader)
    callbacks_list.append(aidi_eval_callback)

# specify the model checkpoint
if checkpoint_path_evaluation is None:
    model_convert_pipeline = None
else:
    model_convert_pipeline = dict(
        type="LoadCheckpoint",
        checkpoint_path=checkpoint_path_evaluation,
        state_dict_update_func=None,
        check_hash=False,
        allow_miss=True,
        ignore_extra=True,
        verbose=True,
    )

float_predictor = dict(
    type="Predictor",
    model=test_model,
    model_convert_pipeline=model_convert_pipeline,
    batch_processor=test_batch_processor,
    data_loader=loaders_list,
    callbacks=callbacks_list,
    device=None,
    metrics=None,
    log_interval=10 if pipeline_test else 50,
    share_callbacks=False,  # share_callbacks should be False
)
