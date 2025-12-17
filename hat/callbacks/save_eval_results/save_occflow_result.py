import os
import pickle

from hat.callbacks.callbacks import CallbackMixin
from hat.registry import OBJECT_REGISTRY


@OBJECT_REGISTRY.register
class SaveOccFlowResult(CallbackMixin):
    """
    Save OccFlow result for visualization.

    Args:
        output_dir: Output dir for saving results.
    """

    def __init__(
        self,
        output_dir: str,
    ):
        self.output_dir = output_dir
        os.makedirs(
            self.output_dir,
            exist_ok=True,
        )

    def on_batch_end(self, batch, model_outs, train_metrics, **kwargs):
        preds = {}
        preds["occ_preds"] = model_outs[0].occ_preds.detach()
        preds["flow_preds"] = model_outs[1].flow_preds.detach()
        scenario_ids = batch["scenario_id"]
        self.save_result(preds, scenario_ids)

    def save_result(self, predictions, scenario_ids):
        batch_size = len(scenario_ids)
        predictions_numpy = []
        for i in range(batch_size):
            pred_tmp = {}
            for k in predictions.keys():
                pred_tmp[k] = predictions[k][i].cpu().detach().numpy()
            predictions_numpy.append(pred_tmp)
        for i in range(batch_size):
            scenario_id = scenario_ids[i]
            save_file = os.path.join(self.output_dir, scenario_id + ".pkl")
            pickle.dump(predictions_numpy[i], open(save_file, "wb"))
