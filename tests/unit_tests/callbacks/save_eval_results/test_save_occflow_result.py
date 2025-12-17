from collections import namedtuple

import torch

from hat.callbacks.save_eval_results.save_occflow_result import (
    SaveOccFlowResult,
)


def test_SaveOccFlowResult():

    scenario_id = "12afca4234"
    fake_batch = {"scenario_id": [scenario_id]}

    occ_preds = torch.zeros((1, 16, 256, 256))
    flow_preds = torch.zeros((1, 16, 256, 256))
    model_output1 = namedtuple("output1", ["occ_preds"])(occ_preds)
    model_output2 = namedtuple("output2", ["flow_preds"])(flow_preds)
    fake_model_outs = [model_output1, model_output2]

    callback = SaveOccFlowResult(output_dir="./")
    callback.on_batch_end(fake_batch, fake_model_outs, None)
