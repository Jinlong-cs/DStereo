# Copyright (c) Horizon Robotics. All rights reserved.

from typing import Any, Dict, List, Optional

import torch

from hat.core.traj_pred_typing import ANCHOR_TYPE
from hat.models.task_modules.traj_pred.post_process.multipath_post_process import (  # noqa: E501
    MultiPathMetricAdapter,
    disable_anchors,
    disable_anchors_for_one_class,
    filter_reverse_trajectories,
    sample_trajectories,
    trajectory_nms,
)
from hat.registry import OBJECT_REGISTRY

__all__ = ["VecterNetPostProcessor"]


@OBJECT_REGISTRY.register
class VecterNetPostProcessor(torch.nn.Module):
    """The post processing of the VectorNet trajectory prediction model.

    This processor should be used in the `BasicVectorNet` structure.
    """

    def __init__(
        self,
        filter_reverse: bool = False,
        disable_anchor_list: Optional[List] = None,
        sample_traj: bool = False,
        num_samples_per_anchor: int = 1,
        use_traj_nms: bool = False,
        nms_params: Optional[Dict] = None,
        anchor_type_version: str = "classical",
        anchor_type_dict: Optional[Dict] = None,
        anchor_type_blocklist: Optional[Dict] = None,
        use_adaptor: bool = True,
    ):
        """Initialize method.

        Args:
            filter_reverse (bool, optional): whether to filter the reversed
                trajectories. Defaults to False.
            disable_anchor_list (List, optional): the disabled anchor index
                list. Default to [].
            sample_traj (bool, optional): whether to sample trajectories
                based on the predicted means and convariance matrix. Default
                to False. If this parameter is False, the class will directly
                use the mean as the sampled trajectory.
            num_samples_per_anchor (int, optional): the number of samples
                per anchor. Default to 1.
            use_traj_nms (bool, optional): whether to use trajectory NMS.
                Default to False.
            nms_params (Dict, optional): the parameters for trajectory NMS.
                It should have the following keys:
                'nms_threshold' (float): the similarity threshold that
                    indicates two anchors are similar enough.
                'nms_prob_threshold' (float): the minimum probability of
                    anchors that can be selected by NMS.
                'nms_num_trajs' (int): the number of trajectories selected
                    by NMS.
                'change_probs' (bool): whether to sort the anchor probability
                    after NMS to make the top-k trajectories multi-modal.
            use_adaptor: whether use 'MultiPathMetricAdapter' to refine
                trajectories, 'scale_trils' is needed if the parameter is true.
                Default to true.
        """
        super(VecterNetPostProcessor, self).__init__()
        self.filter_reverse = filter_reverse
        self.disable_anchor_list = disable_anchor_list
        self.disable_anchors = (disable_anchor_list is not None) and len(
            disable_anchor_list
        )
        self.sample_traj = sample_traj
        self.num_samples_per_anchor = num_samples_per_anchor
        assert (
            anchor_type_version in ANCHOR_TYPE
        ), f"Undefined anchor type split {anchor_type_version}."
        self.anchor_type = ANCHOR_TYPE[anchor_type_version]
        self.anchor_type_dict = anchor_type_dict
        self.anchor_type_blocklist = anchor_type_blocklist
        self.disable_anchor_type = (
            anchor_type_dict is not None
        ) and anchor_type_blocklist is not None

        self.use_traj_nms = use_traj_nms
        self.nms_params = nms_params
        if use_traj_nms:
            assert nms_params is not None, "The paramters of NMS is Nonetype."
            required_nms_params = [
                "nms_threshold",
                "nms_prob_threshold",
                "nms_num_trajs",
                "change_probs",
            ]
            for param in required_nms_params:
                assert (
                    param in nms_params
                ), f"Required nms parameter {param} is not found."

        self.use_adaptor = use_adaptor
        if use_adaptor:
            self.metric_adapter = MultiPathMetricAdapter()

    @torch.no_grad()
    def forward(self, model_output, data: Dict[str, Any]):
        """Post processing forward.

        The post processing contains the following steps, the
        "switches" are defined in the initialization method:
        1. Filter reverse trajectories;
        2. Disable some bad anchors;
        3. Sample trajectories;
        4. Trajectory NMS;
        5. Process for metrc calculation.

        Args:
            model_output (Dict): the model output dictionary.
            data (Dict): the model input dictionary.

        Return:
            results (Dict): the dictionary of model trajectory
                for visualization and metric calculation. It contains
                the following keys:
                ['cur_head_mask', 'trajectories',
                'means', 'scale_trils', 'log_anchors_probs', 'gts',
                'masks']
        """
        # Extract data.
        means = model_output["means"]
        scale_trils = model_output["scale_trils"]
        anchor_probs = model_output["log_anchors_probs"]
        gts = model_output["gts"]
        masks = model_output["masks"]
        classes = data["batch_valid_class"]
        if "cur_head_mask" in model_output:
            head_mask = model_output["cur_head_mask"]
        else:
            head_mask = torch.ones([masks.shape[0]], dtype=masks.dtype)
        anchors_sampled_mask = None
        if "sampled_anchors_mask" in data:
            anchors_sampled_mask = data["sampled_anchors_mask"]

        # 1. Filter reverse trajectories.
        if self.filter_reverse:
            anchor_probs = filter_reverse_trajectories(means, anchor_probs)

        # 2. Disable anchors.
        # -- single anchor.
        if self.disable_anchors:
            anchor_probs = disable_anchors(
                self.disable_anchor_list, anchor_probs
            )
        # -- anchor types.
        if self.disable_anchor_type:
            # -- vehicle
            anchor_probs = disable_anchors_for_one_class(
                0,
                classes,
                anchor_probs,
                self.anchor_type,
                self.anchor_type_dict,
                self.anchor_type_blocklist["veh"],
            )
            # -- pedestrain
            anchor_probs = disable_anchors_for_one_class(
                1,
                classes,
                anchor_probs,
                self.anchor_type,
                self.anchor_type_dict,
                self.anchor_type_blocklist["ped"],
            )
            # -- cyclist
            anchor_probs = disable_anchors_for_one_class(
                2,
                classes,
                anchor_probs,
                self.anchor_type,
                self.anchor_type_dict,
                self.anchor_type_blocklist["cyc"],
            )
        # -- scale the probabilities. the `anchor_probs` is actually log-prob,
        # i.e., in (-inf, 0].
        if anchors_sampled_mask is not None and len(anchors_sampled_mask):
            anchor_probs = torch.where(
                anchors_sampled_mask > 0, 0.5 * anchor_probs, anchor_probs
            )

        # 3. Sample trajectories.
        need_sample = self.sample_traj and self.num_samples_per_anchor > 1
        if need_sample:
            trajectories = sample_trajectories(
                means, scale_trils, self.num_samples_per_anchor
            )
        else:
            trajectories = means[:, :, None, :, :]

        # 4. Trajectory NMS.
        if self.use_traj_nms:
            nms_trajs, nms_probs, nms_scale_trils, _ = trajectory_nms(
                trajs=means,
                traj_probs=anchor_probs,
                scale_trils=scale_trils,
                similarity_threshold=self.nms_params["nms_threshold"],
                prob_threshold=self.nms_params["nms_prob_threshold"],
                n_trajs=self.nms_params["nms_num_trajs"],
                change_probs_after_nms=self.nms_params["change_probs"],
            )
            anchor_probs = nms_probs
            means = nms_trajs
            scale_trils = nms_scale_trils
            trajectories = nms_trajs[:, :, None, :, :]

        # 5. Process for metrc calculators.
        if self.use_adaptor:
            metric_input_pack = [
                trajectories[head_mask],
                gts[head_mask],
                anchor_probs[head_mask],
                means[head_mask],
                scale_trils[head_mask],
                masks[head_mask],
                None,  # resize_ratio
                classes[head_mask],
            ]
            metric_info = self.metric_adapter(metric_input_pack)
        else:
            metric_info = {
                "pred_trajs": means[head_mask].cpu().detach(),
                "traj_probs": torch.exp(anchor_probs[head_mask])
                .cpu()
                .detach(),
                "gt_trajs": torch.stack(
                    [gts[head_mask]] * means.shape[1], dim=1
                )
                .cpu()
                .detach(),
                "timestep_masks": masks[head_mask].cpu().detach(),
                "obs_cls": classes[head_mask].cpu().detach(),
            }

        # 6. Bool track valid classification head.

        # 7. Update.
        results = {}
        results["cur_head_mask"] = head_mask
        results["trajectories"] = trajectories
        results["means"] = means
        results["scale_trils"] = scale_trils
        results["log_anchors_probs"] = anchor_probs
        results["gts"] = gts
        results["masks"] = masks
        results.update(metric_info)
        return results
