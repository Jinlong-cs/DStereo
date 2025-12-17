from collections import OrderedDict
from copy import deepcopy

import torch
from common import aidi_eval, model_input_size, val_decoders
from horizon_plugin_pytorch.march import March
from schedule import TASK_CONFIGS

march = March.BERNOULLI2

inputs = dict(img=torch.zeros((1, 3, *model_input_size)))

opt_inputs = dict(
    train=dict(),
    val=dict(),
    test=dict(),
)

traced_inputs = dict(
    img=torch.randn((1, 3, model_input_size[0], model_input_size[1])),
)


def get_model(mode):
    converted_decoders = OrderedDict(
        {
            (tuple(task_names), group): decoder
            for group, (task_names, decoder) in val_decoders.items()
        }
    )
    return dict(
        type="MultitaskGraphModel",
        inputs=inputs,
        opt_inputs=opt_inputs[mode],
        task_inputs={T.task_name: T.get_inputs(mode) for T in TASK_CONFIGS},
        task_modules={T.task_name: T.get_model(mode) for T in TASK_CONFIGS},
        funnel_modules=converted_decoders if "val" in mode else None,
        flatten_outputs=not aidi_eval,
        lazy_forward=False,
    )


model = get_model("train")
val_model = deepcopy(get_model("val"))
test_model = deepcopy(get_model("test"))
