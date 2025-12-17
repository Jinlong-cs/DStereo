# Copyright (c) Horizon Robotics. All rights reserved.

from collections import OrderedDict
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from torch.distributions import MultivariateNormal

from hat.core.traj_pred_typing import (
    ANCHOR_TYPE,
    StaticSafeArea,
    VehicleSafeArea,
    detect_safe_area_collision,
)
from hat.registry import OBJECT_REGISTRY

__all__ = [
    "MultipathPostProcessor",
    "TrackValidHeadPostProcessor",
    "BehavHeadPostProcessor",
]


def filter_reverse_trajectories(means, traj_probs):  # noqa: D205,D400
    """Filter out the predicted trajectories whose directions are opposite
        to the actual movement directions.

    This postprocess can only be used when `self.mode` is 'rotated_roi' or
    'warping'. Because these modes have a strong priori hypothesis that
    the prediction result (in the centric coordinates) should have positive
    x coordinates. Therefore, we can filter out the invalid trajectory
    results based on this rule.

    Args:
        means (torch.Tensor,[num_obj, num_anchors, traj_len, 2]): means of the
            two dimensional gaussian distributions.
        traj_probs (torch.Tensor, [num_obj, num_anchors]): the log
            probabilities of different anchors.

    Returns:
        traj_probs (torch.Tensor, [num_obj, num_anchors]): the log
            probabilities of different anchors.
    """
    x_means = torch.mean(means[:, :, :, 0], axis=2)
    min_prob = float(torch.min(traj_probs))
    min_prob = torch.full_like(traj_probs, min_prob).to(traj_probs.device)
    traj_probs = torch.where(x_means < 0, min_prob, traj_probs)
    return traj_probs


def disable_anchors(anchor_list, traj_probs):
    """Disable some anchors.

    Args:
        anchor_list (list): the anchor indices to disable.
        traj_probs (torch.Tensor, [num_obj, num_anchors]): the log
            probabilities of different anchors.

    Returns:
        traj_probs (torch.Tensor, [num_obj, num_anchors]): the log
            probabilities of different anchors.
    """
    min_prob = torch.min(traj_probs)
    for anc_idx in anchor_list:
        traj_probs[:, anc_idx] = min_prob
    return traj_probs


def disable_anchors_for_one_class(
    class_type: int,
    batch_class: torch.Tensor,
    traj_probs: torch.Tensor,
    anchor_type: Dict,
    anchor_type_dict: Dict,
    block_anchor_types: List,
    is_lite_model: bool = False,
):
    min_prob = torch.min(traj_probs)
    if is_lite_model:
        for anc_type in block_anchor_types:
            traj_probs[batch_class == class_type, anc_type] = min_prob
    else:
        anchor_list = []
        for anc_type in block_anchor_types:
            anchor_list += anchor_type_dict[anchor_type[anc_type]]
        for anc_idx in anchor_list:
            traj_probs[batch_class == class_type, anc_idx] = min_prob
    return traj_probs


def sample_trajectories(means, scale_trils, sample_times=1):
    """Sample trajectories according to given `means` and `scale_trils`.

    Args:
        means (torch.Tensor, [num_obj, num_anchor, traj_len, 2]): means of
            two dimensional gaussian distributions.
        scale_trils (torch.Tensor, [num_obj, num_anchor, traj_len, 2, 2]):
            covariance matrices of two dimensional gaussian distributions.
        sample_times (int, optional): sample times. Defaults to 1.

    Returns:
        trajectories (torch.Tensor, [num_obj, num_anchor, sample_times,
            traj_len, 2]): sampled trajectories.
    """
    num_obj, num_anchor, traj_len, _ = means.shape
    # move to cpu to avoid gpu memory consumption
    means, scale_trils = [item.cpu() for item in [means, scale_trils]]
    means = means.reshape(-1, 2)
    scale_trils = scale_trils.reshape(-1, 2, 2)
    gmm = MultivariateNormal(means, scale_tril=scale_trils)
    sampled_trajs = gmm.sample(sample_shape=[sample_times])
    sampled_trajs = sampled_trajs.reshape(
        [sample_times, num_obj, num_anchor, traj_len, 2]
    )
    sampled_trajs = sampled_trajs.permute([1, 2, 0, 3, 4])
    # shape [num_obj, num_anchor, sample_times, traj_len, 2]
    return sampled_trajs


def trajectory_nms(
    trajs,
    traj_probs,
    scale_trils,
    similarity_threshold,
    prob_threshold,
    n_trajs,
    change_probs_after_nms=False,
):
    """Perform Non-Maximum Suppression (NMS) on predicted trajectories.

    Args:
        trajs (torch.Tensor, [batch_size, num_anchors, traj_len, 2]):
            the predicted trajectories.
        traj_probs (torch.Tensor, [batch_size, num_anchors]): the log
            probabilities of different anchors.
        scale_trils (torch.Tensor, [batch_size, num_anchors, 2, 2]):
            the lower-triangular factor of covariance.
        similarity_threshold (float): the similarity threshold that indicates
            two anchors are similar enough.
        prob_threshold (float): the minimum probability of anchors that
            can be selected by NMS.
        n_trajs (int): the number of trajectories selected by NMS.
        change_probs_after_nms (bool, optional): whether to sort the anchor
            probability after NMS. Default to False. After NMS, if the high
            probability trajectories are in the same cluster, the top-k
            prediction will not be multi-modal. We can simplely sort the probs
            to ensure the top-k results have as many clusters as possible.

    Returns:
        all_trajs_after_nms (torch.Tensor, [batch_size, n_trajs, traj_len, 2]):
            the selected trajectories by NMS.
        all_trajs_probs_after_nms (torch.Tensor, [batch_size, n_trajs]): the
            probability of the selected anchors by NMS.
        all_scale_trils_after_nms  (torch.Tensor, [batch_size, n_trajs, 2, 2]):
            the lower-triangular factor of covariance of the selected anchors.
        all_indices (list): all the indices of selected anchors.
    """
    batch_size = trajs.shape[0]
    all_trajs = []
    all_trajs_probs = []
    all_scale_trils = []
    all_indices = []
    ranked_trajs_indices = torch.argsort(traj_probs, dim=-1, descending=True)

    for batch_idx in range(batch_size):
        ranked_trajs_indices_dict = OrderedDict(
            {
                int(idx): rank
                for rank, idx in enumerate(ranked_trajs_indices[batch_idx, :])
            }
        )

        nms_clusters = OrderedDict()
        while len(ranked_trajs_indices_dict):
            traj_idx = ranked_trajs_indices_dict.popitem(last=False)[0]
            # If the anchor probability of traj_idx is very small, stop the
            #   clustering.
            if torch.exp(traj_probs[batch_idx, traj_idx]) < prob_threshold:
                break
            cluster_trajs = [traj_idx]
            indices_left = list(ranked_trajs_indices_dict)
            # If no trajectory left, stop the clustering.
            if not len(indices_left):
                break
            indices_left = torch.tensor(indices_left)
            # Calculate the likelihood of different anchors under the GMM of
            #   the selected trajectory (traj_idx).
            cur_gmm = MultivariateNormal(
                loc=trajs[batch_idx, traj_idx : (traj_idx + 1), :, :],
                scale_tril=scale_trils[
                    batch_idx, traj_idx : (traj_idx + 1), :, :, :
                ],
            )
            log_ll = cur_gmm.log_prob(trajs[batch_idx, indices_left, :, :])
            likelihood = torch.exp(torch.mean(log_ll, dim=-1)).to(
                indices_left.device
            )
            similar_list = indices_left[likelihood > similarity_threshold]
            similar_list = similar_list.tolist()
            # Add the anchors that has a larger probability than prob_threshold
            #   to the corresponding cluster.
            for c_idx in similar_list:
                if torch.exp(traj_probs[batch_idx, c_idx]) >= prob_threshold:
                    cluster_trajs.append(c_idx)
            nms_clusters[traj_idx] = cluster_trajs
            for key in cluster_trajs[1:]:
                ranked_trajs_indices_dict.pop(key)

        # Selected trajs from the clusters.
        ordered_nms_indices = []
        while len(nms_clusters):
            cluster_list = list(nms_clusters)
            for idx in cluster_list:
                ordered_nms_indices.append(nms_clusters[idx][0])
                nms_clusters[idx].pop(0)
                if not len(nms_clusters[idx]):
                    nms_clusters.pop(idx)
        nms_proposal = ordered_nms_indices[0:n_trajs]
        idx = 0
        # Make sure enough trajs can be output after NMS.
        while len(nms_proposal) < n_trajs:
            while int(ranked_trajs_indices[batch_idx, idx]) in nms_proposal:
                idx += 1
            nms_proposal.append(int(ranked_trajs_indices[batch_idx, idx]))

        nms_trajs = trajs[batch_idx, nms_proposal, :, :]
        nms_trajs_probs = traj_probs[batch_idx, nms_proposal]
        nms_scale_trils = scale_trils[batch_idx, nms_proposal, :, :]
        all_trajs.append(nms_trajs)
        all_trajs_probs.append(nms_trajs_probs)
        all_scale_trils.append(nms_scale_trils)
        all_indices.append(nms_proposal)
    all_trajs_after_nms = torch.stack(all_trajs)
    all_trajs_probs_after_nms = torch.stack(all_trajs_probs)
    all_scale_trils_after_nms = torch.stack(all_scale_trils)

    if change_probs_after_nms:
        all_trajs_probs_after_nms = torch.sort(
            all_trajs_probs_after_nms, dim=-1, descending=True
        )[0]

    return (
        all_trajs_after_nms,
        all_trajs_probs_after_nms,
        all_scale_trils_after_nms,
        all_indices,
    )


def get_topk_traj_and_indices(trajs, traj_probs, topk=5):
    """Get topk predicted trajectories and their indices.

    Args:
        trajs (torch.Tensor, [batch_size, num_anchors, 1, traj_len, 2]):
            the predicted trajectories.
        traj_probs (torch.Tensor, [batch_size, num_anchors]): the log
            probabilities of different anchors.
        topk (int): the number of pred trajectory to blend.

    Returns:
        topk_traj (torch.Tensor, [batch_size, n_trajs, traj_len, 2]):
            the selected trajectories by probs.
        topk_indices (torch.Tensor, [batch_size, n_trajs]):
            the selected anchor indices.
    """
    bs, num_anchor, _, traj_len, _ = trajs.shape
    top_indices = (-traj_probs).argsort(axis=1)
    indices = top_indices[:, :topk].reshape(-1, topk, 1, 1)
    topk_indices = indices.repeat(1, 1, traj_len, 2)
    topk_traj = trajs.squeeze(2).gather(1, topk_indices).cpu()
    return topk_traj, topk_indices


def cal_expected_next_position(hist_trajs, dt=0.5):
    """Calculate next position.

    Args:
        hist_trajs (torch.Tensor, [batch_size, num_anchors, traj_len, 2]):
            The predicted trajectories.

    Returns:
        exp_next_pos (torch.Tensor, [batch_size, n_trajs, traj_len, 2]):
            The calculated expected next position by linear interpolation .
    """
    lct_pos = hist_trajs[:, -1:]
    his_vel_2d = torch.diff(hist_trajs, dim=1) / dt
    his_acc_2d = torch.diff(his_vel_2d, dim=1) / dt

    # Linear interpolation
    vel_mean = torch.mean(his_vel_2d, dim=1, keepdim=True)
    acc_mean = torch.mean(his_acc_2d, dim=1, keepdim=True)
    lct_vel_2d = vel_mean + acc_mean * dt * 2.0
    lct_vel_2d = torch.where(lct_vel_2d > 0, lct_vel_2d, torch.tensor(0.0))
    next_pos = lct_vel_2d * dt + lct_pos
    return next_pos


def cal_scale(pred_next_position, expected_next_position):
    """Calculate scale for scaling.

    Args:
        pred_next_position (torch.Tensor, [num_obls, num_traj, 1, 2]):
            The predicted next position.
        expected_next_position (torch.Tensor, [num_obls, num_traj, 1, 2]):
            The expected next position.

    Returns:
        scale (torch.Tensor, [num_obls, num_traj, 1, 1]):
           The resulting scale vector.
    """
    eps = 1e-9
    scale = expected_next_position[:, :, :, :1] / (
        pred_next_position[:, :, :, :1] + eps
    )
    return scale


def cal_vel_acc(inputs, dt=0.5, dim=1):
    """Calculate the velocity and acceleration.

    Args:
        inputs (torch.Tensor, [batch_size, num_anchors, traj_len, 2]):
            the predicted trajectories.
        dt (float): time difference.
        dim (int): index of time dimension.

    Returns:
        vel (torch.Tensor, [batch_size, n_trajs, traj_len, 2]):
            the calculated velocity.
        acc (torch.Tensor, [batch_size, n_trajs, traj_len, 2]):
            the calculated acceleration.
    """
    # Cal velocity
    vel = torch.diff(inputs, dim=dim) / dt

    # Cal acceleration
    acc = torch.diff(vel, dim=dim) / dt
    return vel, acc


def trajectory_blend(
    track_ids: List,
    valid_track_ids: List,
    context_states: List,
    trajectories: torch.Tensor,
    traj_probs: torch.Tensor,
):
    """Blend topk trajectory if both history and pred are slowing down.

    Args:
        track_ids (List): track id list.
        valid_track_ids (List): valid track id list.
        context_states (List of np.ndarray): the \
        ground-truth states of objects within the historical context time \
        frames.
        trajectories (torch.Tensor, [batch_size, num_anchors, 1, traj_len, 2]):
            the predicted trajectories.
        traj_probs (torch.Tensor, [batch_size, num_anchors]):
            the predicted anchor probabilities.

    Returns:
        trajectories (torch.Tensor, [batch_size, n_trajs, 1, traj_len, 2]):
            the blended trajectories.
        blend_track_ids (List): track id list of blend trajectories.
    """
    # Get history and pred traj
    his_traj = []
    for ids, valid_idxs, states in zip(
        track_ids, valid_track_ids, context_states
    ):
        ids = np.array(ids)
        valid_indexes = np.array(
            [np.where(ids == track_id)[0] for track_id in valid_idxs]
        ).ravel()
        his_states = torch.from_numpy(states[valid_indexes]).float()
        his_traj.append(his_states)
    his_traj = torch.cat(his_traj, dim=0)
    topk_traj, indices = get_topk_traj_and_indices(trajectories, traj_probs)

    # Cal scale for blending
    exp_next_pos = cal_expected_next_position(his_traj)
    exp_next_pos = exp_next_pos[:, None]
    pred_next_pos = topk_traj[:, :, :1]
    scale = cal_scale(pred_next_pos, exp_next_pos)

    # Scale is valid when pred trajectories are
    # considered over-predicted and non-backward
    invalid_mask = (scale[:, :, :, :1] >= 1.0) | (scale[:, :, :, :1] < 0.0)
    scale[invalid_mask] = 1.0

    # Cal blend traj
    blend_traj = topk_traj * scale

    # Update trajectories
    blend_traj = blend_traj.cuda()
    trajectories = (
        trajectories.squeeze(2).scatter(1, indices, blend_traj).unsqueeze(2)
    )

    # Save blend track ids
    valid_track_ids = np.concatenate(valid_track_ids)
    scale_mask = (~invalid_mask).squeeze(-1).squeeze(-1)
    blend_track_ids_index = torch.any(scale_mask, dim=1)
    blend_track_ids = valid_track_ids[blend_track_ids_index].tolist()
    return trajectories, blend_track_ids


class MultiPathMetricAdapter:  # noqa: D205,D400
    """Adapter for multipath model outputs.

    This class performs pre-processing of the Multipath prediction results.
    The processed result can be directly used in trajectory prediction
    metric calculators.
    """

    def __init__(self):
        pass

    def __call__(self, input_pack: List):  # noqa: D205,D400
        """Transform multipath model output trajectories and ground-truth to
        metric evaluation-friendly trajectories.

        Args:
            input_pack (List): a list containing the following variables:
                trajs (FloatTensor, [num_obj, num_anchor, num_sample,
                    traj_len, 2]): multipath output trajectories tensor.
                gts (FloatTensor, [num_obj, traj_len, 2]): the accompanying
                    ground-truth trajectories for each object.
                anchor_probs (FloatTensor, [num_obj, num_anchor]): a 1-D
                    tensor holding probabilities or each anchor trajectories.
                anchor_means (FloatTensor, [num_obj, num_anchor, traj_len, 2]):
                    means of two dimensional gaussian distributions for each
                    anchor.
                anchor_scale_trils (FloatTensor, [num_obj, num_anchor,
                    traj_len, 2, 2]): covariance matrices of two dimensional
                    gaussian distributions for each anchor.
                timestep_masks: (FloatTensor, [num_obj, traj_len]): timestep
                    boolean masks which indicate the existence of coordinates
                    at each timestep.
                resize_ratio: (FloatTensor, [num_obj, 2]): the resize ratio
                    between raw images and model input images.
        Returns:
            out_sample (Dict): a dict contains the following keys.
                1. '{}_pred_trajs' (FloatTensor, [num_obj, all_samples,
                    traj_len, 2]): the predicted trajectories.
                2. '{}_traj_probs' (FloatTensor, [num_obj, all_samples]): the
                    normalized likelihood of each sample trajectory.
                3. '{}_gt_trajs': (FloatTensor, [num_obj, all_samples,
                    traj_len, 2]): the ground-truth trajectories.
                3. '{}_timestep_masks': (FloatTensor, [num_obj, traj_len]):
                    the masks to indicate whether a time stamp is valid.
        """
        # 0. Move tensors to cpu.
        (
            trajs,
            gts,
            log_anchor_probs,
            anchor_means,
            anchor_scale_trils,
            timestep_masks,
            resize_ratio,
            obs_cls,
        ) = [
            item.to(torch.device("cpu")).detach()
            if isinstance(item, torch.Tensor)
            else item
            for item in input_pack
        ]
        num_obj, num_anchor, num_sample, traj_len, states = trajs.shape
        num_all_samples = num_anchor * num_sample

        if num_obj:
            # 1. Calculate sample probabilities by anchor probabilities and
            # gaussian likelihood based on (anchor_means, anchor_scale_trils).
            # 1.1 Calculate likelihood by multi variate Gaussian distribution.
            # -- `likelihood`: [num_obj, num_anchor, num_sample]
            anchor_means = torch.stack(
                [anchor_means] * num_sample, dim=2
            ).reshape(-1, 2)
            anchor_scale_trils = torch.stack(
                [anchor_scale_trils] * num_sample, dim=2
            ).reshape(-1, 2, 2)
            gmms = MultivariateNormal(
                loc=anchor_means, scale_tril=anchor_scale_trils
            )
            log_likelihood = gmms.log_prob(trajs.reshape(-1, 2))
            log_likelihood = torch.clip(log_likelihood, min=-30, max=0)
            likelihood = torch.exp(log_likelihood)
            likelihood = likelihood.reshape(
                num_obj, num_anchor, num_sample, traj_len
            )
            likelihood = torch.sum(likelihood, dim=-1)
            # 1.2 Calculate sample likelihood cross anchors.
            # -- `sample_trajs_likelihood`: [num_obj, num_all_samples]
            log_anchor_probs = torch.stack(
                [log_anchor_probs] * num_sample, dim=2
            )
            anchor_probs = torch.exp(log_anchor_probs)
            sample_trajs_likelihood = anchor_probs * likelihood
            # 1.3 Squash the `anchor` and `sample` into the same dimension.
            # -- `sample_trajs_likelihood`: [num_obj, num_anchor, num_sample]
            sample_trajs_likelihood = torch.reshape(
                sample_trajs_likelihood, (num_obj, num_all_samples)
            )
            # 1.4 Normalize.
            out_sample_trajs_probs = (
                sample_trajs_likelihood
                / torch.sum(sample_trajs_likelihood, dim=1, keepdim=True)
                + 1e-5
            )
        else:
            out_sample_trajs_probs = torch.exp(log_anchor_probs)

        # 2. Squash the `anchor` and `sample` of other elements.
        out_trajs = trajs.reshape((num_obj, num_all_samples, traj_len, states))
        out_gts = torch.stack([gts] * num_all_samples, dim=1)
        if resize_ratio is not None and len(resize_ratio) > 0:
            out_trajs = out_trajs / resize_ratio[:, None, None, :]
            out_gts = out_gts / resize_ratio[:, None, None, :]

        out_sample = {
            "pred_trajs": out_trajs,
            "traj_probs": out_sample_trajs_probs,
            "gt_trajs": out_gts,
            "timestep_masks": timestep_masks,
            "obs_cls": obs_cls,
        }

        return out_sample


@OBJECT_REGISTRY.register
class MultipathPostProcessor(torch.nn.Module):
    """The post processing of the Multipath trajectory prediction model.

    This processor should be used in the `Multipath` structure.
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
        is_multipath_lite: bool = False,
        use_adaptor: bool = True,
        blend_traj_by_history: bool = False,
        ego_track_id: int = -42,
    ):
        """Initialize method.

        Args:
            filter_reverse: whether to filter the reversed trajectories.
            disable_anchor_list: the disabled anchor index list.
            sample_traj: whether to sample trajectories based on the
                predicted means and convariance matrix. Default to False.
                If this parameter is False, the class will directly
                use the mean as the sampled trajectory.
            num_samples_per_anchor: the number of samples per anchor.
            use_traj_nms: whether to use trajectory NMS.
            nms_params: the parameters for trajectory NMS.
                It should have the following keys:
                'nms_threshold' the similarity threshold that indicates two
                    anchors are similar enough.
                'nms_prob_threshold': the minimum probability of anchors
                    that can be selected by NMS.
                'nms_num_trajs': the number of trajectories selected by NMS.
                'change_probs': whether to sort the anchor probability
                    after NMS to make the top-k trajectories multi-modal.
            use_adaptor: whether use 'MultiPathMetricAdapter' to refine
                trajectories, 'scale_trils' is needed if the parameter is
                true.
            blend_traj_by_history: where use history state to blend predicted
                trajectories.
            ego_track_id: the track id of the ego vehicle.
        """
        super(MultipathPostProcessor, self).__init__()
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
        self.is_multipath_lite = is_multipath_lite
        self.ego_track_id = ego_track_id

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
        self.blend_traj_by_history = blend_traj_by_history

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
        results = {}

        if "cur_head_mask" in model_output:
            head_mask = model_output["cur_head_mask"]
        else:
            head_mask = torch.ones([masks.shape[0]], dtype=masks.dtype)

        if "resize_ratio" in model_output:
            resize_ratio = model_output["resize_ratio"]
            resize_ratio = resize_ratio[head_mask]
        else:
            resize_ratio = None

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
                self.is_multipath_lite,
            )
            # -- pedestrain
            anchor_probs = disable_anchors_for_one_class(
                1,
                classes,
                anchor_probs,
                self.anchor_type,
                self.anchor_type_dict,
                self.anchor_type_blocklist["ped"],
                self.is_multipath_lite,
            )
            # -- cyclist
            anchor_probs = disable_anchors_for_one_class(
                2,
                classes,
                anchor_probs,
                self.anchor_type,
                self.anchor_type_dict,
                self.anchor_type_blocklist["cyc"],
                self.is_multipath_lite,
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

        # Blend trajectory by latest history state.
        if self.blend_traj_by_history:
            track_ids = data["track_ids"]
            valid_track_ids = data["valid_track_ids"]
            context_states = data["context_states"]
            if not isinstance(context_states, List):
                context_states = [context_states]
                track_ids = [track_ids]
            trajectories, blend_track_ids = trajectory_blend(
                track_ids,
                valid_track_ids,
                context_states,
                trajectories,
                anchor_probs,
            )
            results["blend_track_ids"] = blend_track_ids

        # 5. Process for metrc calculators.
        if self.use_adaptor:
            metric_input_pack = [
                trajectories[head_mask],
                gts[head_mask],
                anchor_probs[head_mask],
                means[head_mask],
                scale_trils[head_mask],
                masks[head_mask],
                resize_ratio,
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

        # 6. Find important obstacles by safe area.
        if "safe_area" in data:
            pred_traj = metric_info["pred_trajs"].numpy()
            prob = metric_info["traj_probs"].numpy()
            valid_track_ids = data["valid_track_ids"]
            track_stat_dict = data["track_stat_dict"]
            all_safe_area = data["safe_area"]
            agent_classes = data["agent_classes"]
            prob_max_idx = np.argmax(prob, axis=1)

            start_idx = 0
            all_expand_track_ids = []
            for v_track_ids, safe_area, stat_dict in zip(
                valid_track_ids, all_safe_area, track_stat_dict
            ):
                tmp_ego_idx = v_track_ids.index(self.ego_track_id) + start_idx
                tmp_prob_max_idx = prob_max_idx[tmp_ego_idx]
                tmp_ego_pred_traj = pred_traj[
                    tmp_ego_idx, tmp_prob_max_idx, :, :
                ]
                tmp_ego_safe_area = VehicleSafeArea(tmp_ego_pred_traj)
                if tmp_ego_safe_area.static:
                    tmp_ego_safe_area = StaticSafeArea(
                        point=[0, 0],
                        obs_cls=0,
                    )

                expand_track_ids = []
                for t_id, sf_area in safe_area.items():
                    if t_id not in v_track_ids:
                        if detect_safe_area_collision(
                            tmp_ego_safe_area, sf_area
                        ):
                            expand_track_ids.append(
                                [t_id, stat_dict[t_id], sf_area]
                            )
                all_expand_track_ids.append(expand_track_ids)
                start_idx += len(v_track_ids)
            results["expand_important_obs"] = all_expand_track_ids
            results["all_obs_classes"] = agent_classes

        # 7. Update.
        results["cur_head_mask"] = head_mask
        results["trajectories"] = trajectories
        results["means"] = means
        results["scale_trils"] = scale_trils
        results["log_anchors_probs"] = anchor_probs
        results["gts"] = gts
        results["masks"] = masks
        results.update(metric_info)
        return results


@OBJECT_REGISTRY.register
class TrackValidHeadPostProcessor(torch.nn.Module):
    """The post processing of the Multipath track valid head.

    This processor should be used in the `BasicTrackValidDecoder` head.
    """

    def __init__(
        self,
        valid_head_threshold: List = None,
    ):
        """Initialize method.

        Args:
            valid_head_threshold (List): the different thresholds of valid
                head prob to binary output for diffent obstacle class.
        """
        super(TrackValidHeadPostProcessor, self).__init__()
        self.valid_head_threshold = valid_head_threshold

    @torch.no_grad()
    def forward(self, model_output, data: Dict[str, Any]):
        """Post processing forward.

        Args:
            model_output (Dict): the model output dictionary.
            data (Dict): the model input dictionary.

        Return:
            results (Dict): the dictionary of model trajectory
                for visualization and metric calculation. It contains
                the following keys:
                ["track_valid_cls_binary", "track_valid_cls_metric"]
        """
        # Extract data.
        masks = model_output["masks"]
        classes = data["batch_valid_class"]
        if "cur_head_mask" in model_output:
            head_mask = model_output["cur_head_mask"]
        else:
            head_mask = torch.ones([masks.shape[0]], dtype=masks.dtype)

        # Bool track valid classification head.
        if (
            "track_valid_cls" in model_output
            and not model_output["model_is_training"]
        ):
            track_valid_prob = model_output["track_valid_cls"].to(
                classes.device
            )
            track_index = torch.arange(track_valid_prob.shape[0])
            track_valid_prob = track_valid_prob[track_index, classes.long()]
            track_valid_prob = track_valid_prob.reshape(-1)
            track_valid_cls_binary = torch.tensor(
                [False for _ in range(len(track_valid_prob))]
            ).to(classes.device)
            for i in range(len(self.valid_head_threshold)):
                mask_cls_i = (classes == i) & (
                    track_valid_prob > self.valid_head_threshold[i]
                )
                track_valid_cls_binary = track_valid_cls_binary | mask_cls_i
            track_valid_cls_binary = track_valid_cls_binary.int()
        else:
            track_valid_cls_binary = torch.ones(
                [masks.shape[0]], dtype=masks.dtype
            )
        track_valid_cls_metric = (
            track_valid_cls_binary[head_mask].cpu().detach()
        )

        results = {
            "track_valid_cls_binary": track_valid_cls_binary,
            "track_valid_cls_metric": track_valid_cls_metric,
        }
        return results


@OBJECT_REGISTRY.register
class BehavHeadPostProcessor(torch.nn.Module):
    """The post processing of the Multipath Behavior prediction head.

    This processor should be used in the `BasicBehavDecoder` head.
    """

    def __init__(
        self,
        front_range: float = 50.0,
        lane_width: float = 3.75,
    ):
        """Initialize method.

        Args:
            front_range: the front max distance. If only vcs_x in (0, front)m,
                a cutin behavior is considered as true cutin.
            lane_width: the standard width of a lane, which is used as a basis
                for judging whether the obstacle is in the adjacent lane.
        """
        super(BehavHeadPostProcessor, self).__init__()
        self.front_range = front_range
        self.lane_width = lane_width

    @torch.no_grad()
    def forward(self, model_output, data: Dict[str, Any]):
        """Post processing forward.

        Args:
            model_output (Dict): the model output dictionary.
            data (Dict): the model input dictionary.

        Return:
            results (Dict): the dictionary of model behaviors for
                visualization and metric calculation. It contains the
                following keys: ["pred_lat_behav", "pred_lon_behav"].
        """
        # Extract data.
        lat_behav_probs = model_output["lat_behav_probs"].view(-1, 3)
        lon_behav_probs = model_output["lon_behav_probs"].view(-1, 3)

        # Decode the predict label of lateral and longitudinal bahaviors.
        indices1 = lat_behav_probs.topk(1, dim=1)[1]
        indices2 = lon_behav_probs.topk(1, dim=1)[1]
        pred_lat_behav = indices1.reshape(-1).detach()
        pred_lon_behav = indices2.reshape(-1).detach()

        results = {
            "pred_lat_behav": pred_lat_behav,
            "pred_lon_behav": pred_lon_behav,
        }
        return results
