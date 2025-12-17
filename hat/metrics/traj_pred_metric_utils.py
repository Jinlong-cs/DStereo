# Copyright (c) Horizon Robotics. All rights reserved.

from typing import Callable, Dict, List

import numpy as np


def traj_mean_l2_dist(pred_trajs, gt_trajs, timestep_masks):  # noqa: D205,D400
    """Calculate the mean L2 norm between predicted and ground-truth
    trajectories.

    Args:
        pred_trajs (np.array, [num_obj, num_samples, num_steps, num_states]):
            the predicted trajectories.
        gt_trajs (np.array, [num_obj, num_samples, num_steps, num_states]):
            the ground-truth trajectories.
        timestep_masks (np.array, [num_obj, timesteps]): timestep masks.

    Returns:
        mean_l2 (np.array, [num_obj, num_samples]): the mean L2 distances.
    """
    # Compute distance loss.
    dist = np.linalg.norm(pred_trajs - gt_trajs, axis=-1)
    # Substitute nans with 0, since nan can cause unexpected problems.
    dist[np.isnan(dist)] = 0
    # Calculate average loss using total loss and sum mask.
    # -- add a small value for denom to avoid division by zero.
    l2 = np.sum(dist * (timestep_masks[:, None, :]), axis=-1)
    mean_l2 = l2 / (np.sum(timestep_masks, axis=-1)[:, None] + 1e-10)
    return mean_l2


def traj_final_dist(pred_trajs, gt_trajs, timestep_masks):  # noqa: D205,D400
    """Calculate the final displacement distance between predicted and
    ground-truth trajectories.

    Args:
        pred_trajs (np.array, [num_obj, num_samples, num_steps, num_states]):
            the predicted trajectories.
        gt_trajs (np.array, [num_obj, num_samples, num_steps, num_states]):
            the ground-truth trajectories.
        timestep_masks (np.array, [num_obj, timesteps]): timestep masks.

    Returns:
        final_dist (np.array, [num_obj, num_samples]): the final distances.
    """
    # If there is no objects, return an empty array with the right shape on
    # the second dimension. If we don't do this, the following np.meshgrid
    # function will throw exceptions.
    n_obj, n_sample, n_time, n_states = pred_trajs.shape
    if n_obj == 0:
        return np.zeros(shape=[n_obj, n_sample], dtype=np.float32)

    # Compute distance loss.
    diff = pred_trajs - gt_trajs
    # Get final dist indexes.
    # -- invert timestep_masks array's timestep axis for later index locating.
    inv_timestep_masks = timestep_masks[:, None, ::-1, None]
    # Use argmax to locate the index of the first non-zero element in the
    # timestep dimension of the inverse masked diff, which correspond
    # to the last non-zero element of the masked diff.
    inv_final_dist_time_index = np.argmax(inv_timestep_masks, axis=2)
    final_dist_time_index = n_time - 1 - inv_final_dist_time_index
    obj_mg, sample_mg, states_mg = np.meshgrid(
        range(n_obj), range(n_sample), range(n_states), indexing="ij"
    )
    final_states = diff[obj_mg, sample_mg, final_dist_time_index, states_mg]
    final_dist = np.linalg.norm(final_states, axis=-1)
    return final_dist


def traj_max_l2_dist(pred_trajs, gt_trajs, timestep_masks):  # noqa: D205,D400
    """Calculate the max L2 norm between predicted and ground-truth
    trajectories.

    Args:
        pred_trajs (np.array, [num_obj, num_samples, num_steps, num_states]):
            the predicted trajectories.
        gt_trajs (np.array, [num_obj, num_samples, num_steps, num_states]):
            the ground-truth trajectories.
        timestep_masks (np.array, [num_obj, timesteps]): timestep masks.

    Returns:
        mean_l2 (np.array, [num_obj, num_samples]): the mean L2 distances.
    """
    # Compute distance loss.
    dist = np.linalg.norm(pred_trajs - gt_trajs, axis=-1)
    # Substitute nans with 0, since nan can cause unexpected problems.
    dist[np.isnan(dist)] = 0
    # Calculate average loss using total loss and sum mask.
    # -- add a small value for denom to avoid division by zero.
    max_l2 = np.max(dist * (timestep_masks[:, None, :]), axis=-1)
    return max_l2


def traj_velo_diff(
    pred_trajs,
    gt_trajs,
    timestep_masks,
    delta_t=0.5,
    return_first_only=True,
    return_gt_velo=False,
):
    """Calculate the velocity difference between the preds trajs and the gts.

    Args:
        pred_trajs (np.array, [num_obj, num_samples, num_steps, num_states]):
            the predicted trajectories.
        sample_probs (np.array, [num_obj, num_samples]): sample probabilities.
        gt_trajs (np.array, [num_obj, num_samples, num_steps, num_states]):
            the ground-truth trajectories.
        timestep_masks (np.array, [num_obj, timesteps]): timestep masks.
            delta_t (float): time interval between positions
        return_first_only(bool): set True for return the velo diff of the
            first timestep.
        return_gt_velo(bool): return the velocity sequence of the gts.

    Returns:
        vel_diff_ (np.array, [num_obj, num_samples, num_steps-1]): the
        velocity difference between the preds trajs and the gts.
    """
    # Calculate timestamp masks.
    pred_timestamp_masks = timestep_masks[:, 1:][:, None, :]
    pred_delta_t = delta_t * (np.sum(pred_timestamp_masks, axis=-1) + 1)

    # Calculate mean velocities of the predicted trajectories.
    pred_delta_xy = np.diff(pred_trajs, axis=2)
    pred_delta_s = np.sqrt(
        np.sum(pred_delta_xy ** 2, axis=-1)
    )  # ->(N, num_anchor, len-1)
    pred_velo_s = pred_delta_s / delta_t  # m/s

    # Calculate mean velocities of the gt trajectories.
    gt_delta_xy = np.diff(gt_trajs, axis=2)
    gt_delta_s = np.sqrt(np.sum(gt_delta_xy ** 2, axis=-1))
    gt_velo_s = gt_delta_s / delta_t  # m/s
    gt_sum_s = np.sum(gt_delta_s * pred_timestamp_masks, axis=-1)
    gt_mean_velo = gt_sum_s / pred_delta_t * 3.6  # km/h

    # Calculate velocities between the preds and the gts
    velo_diff = np.abs(pred_velo_s - gt_velo_s)
    if return_first_only:
        vel_diff_ = velo_diff[:, :, 0]
    else:
        avg_non_zero_index = pred_timestamp_masks
        vel_diff_ = np.sum(velo_diff * pred_timestamp_masks, axis=2) / (
            np.sum(avg_non_zero_index, axis=2) + 1e-10
        )

    if return_gt_velo:
        return vel_diff_, gt_mean_velo[:, 0]
    else:
        return vel_diff_


def sort_traj_samples_by_likelyhood(
    metric_results, sample_probs, applied_func: str
):  # noqa: D205,D400
    """Arrange metric results according to sample probabilites and apply either
    'np.accumulative.min' or 'np.accumulative.max' function.

    This function is a helper function for calculating `min_ade_k` and
    `min_fde_k`, It arranges metric results for each object according to
    sample probabilies in descending order, and then applies either 'min' or
    'max' function to calculate min or max metric in a set of k most probable
    results.

    Args:
        metric_results (np.array, [num_obj, num_samples]): metric result.
        sample_probs (np.array, [num_obj, num_samples]): sample probabilities.
        applied_func (str): apply accumulative_max/min to metric values.

    Returns:
        sorted_metrics (np.array, [num_obj, num_samples]): metric result
            after sort and accumulative maximize or minimize. That makes
            it easlier to get the min or max metric with in top-k metrics.
    """
    if applied_func == "accumulative_min":
        func = np.minimum.accumulate
    elif applied_func == "accumulative_max":
        func = np.maximum.accumulate
    else:
        raise ValueError(
            "applied_func must be one of accumulative_min or "
            f"accumulative_min. Received {applied_func}"
        )

    p_sorted = (-sample_probs).argsort(axis=-1)
    indices = np.indices(metric_results.shape)
    sorted_metrics = metric_results[indices[0], p_sorted]
    sorted_metrics = func(sorted_metrics, axis=-1)
    return sorted_metrics


def top_k_min_calculator_template(
    measure_func,
    k_values,
    pred_trajs,
    sample_probs,
    gt_trajs,
    timestep_masks=None,
    absolute_mode=False,
    **kwargs,
):
    # noqa: D205,D400
    """Given a measure function, calculate the minimum metric for a set
    of k most probable predicted results.

    Args:
        measure_func (Callable): the measure function.
        k_values (List): list of top-k values to calculate the metric.
        pred_trajs (np.array, [num_obj, num_samples, num_steps, num_states]):
            the predicted trajectories.
        sample_probs (np.array, [num_obj, num_samples]): sample probabilities.
        gt_trajs (np.array, [num_obj, num_samples, num_steps, num_states]):
            the ground-truth trajectories.
        timestep_masks (np.array, [num_obj, timesteps]): timestep masks.

    Returns:
        topk_min_measure (np.array, [num_obj, len(k_values)]): the min metric
            of the top k samples of each object.
    """
    num_obj, num_sample, num_step, _ = pred_trajs.shape
    if timestep_masks is None:
        timestep_masks = np.ones((num_obj, num_step)).astype(bool)
    obj_mask = np.any(timestep_masks, axis=-1)
    pred_trajs = pred_trajs[obj_mask, :, :, :]
    gt_trajs = gt_trajs[obj_mask, :, :, :]
    timestep_masks = timestep_masks[obj_mask, :]
    sample_probs = sample_probs[obj_mask, :]
    ret_measures = measure_func(pred_trajs, gt_trajs, timestep_masks, **kwargs)
    if type(ret_measures) in [tuple, list]:
        ret_measures = list(ret_measures)
        measures = ret_measures[0]
    else:
        measures = ret_measures

    if absolute_mode:
        apply_func = "accumulative_abs_min"
    else:
        apply_func = "accumulative_min"
    ranked_measures = sort_traj_samples_by_likelyhood(
        measures, sample_probs, apply_func
    )
    topk_min_measure = ranked_measures[
        :, [min(k, num_sample) - 1 for k in k_values]
    ]

    if type(ret_measures) is list:
        return [topk_min_measure] + ret_measures[1:]
    else:
        return topk_min_measure


def top_k_min_ade_calculator(
    k_values,
    pred_trajs,
    sample_probs,
    gt_trajs,
    timestep_masks=None,
):  # noqa: D205,D400
    """Calculate the minimum average displacement error (ADE) metric for a
    set of k most probable predicted results.

    Args:
        k_values (List): list of top-k values to calculate the metric.
        pred_trajs (np.array, [num_obj, num_samples, num_steps, num_states]):
            the predicted trajectories.
        sample_probs (np.array, [num_obj, num_samples]): sample probabilities.
        gt_trajs (np.array, [num_obj, num_samples, num_steps, num_states]):
            the ground-truth trajectories.
        timestep_masks (np.array, [num_obj, timesteps]): timestep masks.

    Returns:
        topk_min_ade (np.array, [num_obj, len(k_values)]): the min ADE metric
            of the top k samples of each object.
    """
    topk_min_ade = top_k_min_calculator_template(
        traj_mean_l2_dist,
        k_values,
        pred_trajs,
        sample_probs,
        gt_trajs,
        timestep_masks,
        absolute_mode=False,
    )
    return topk_min_ade


def top_k_min_fde_calculator(
    k_values,
    pred_trajs,
    sample_probs,
    gt_trajs,
    timestep_masks=None,
):  # noqa: D205,D400
    """Calculate the minimum final displacement error (FDE) metric for a set
    of k most probable predicted results.

    Args:
        k_values (List): list of top-k values to calculate the metric.
        pred_trajs (np.array, [num_obj, num_samples, num_steps, num_states]):
            the predicted trajectories.
        sample_probs (np.array, [num_obj, num_samples]): sample probabilities.
        gt_trajs (np.array, [num_obj, num_samples, num_steps, num_states]):
            the ground-truth trajectories.
        timestep_masks (np.array, [num_obj, timesteps]): timestep masks.

    Returns:
        topk_min_fde (np.array, [num_obj, len(k_values)]): the min FDE metric
            of the top k samples of each object.
    """
    topk_min_fde = top_k_min_calculator_template(
        traj_final_dist,
        k_values,
        pred_trajs,
        sample_probs,
        gt_trajs,
        timestep_masks,
        absolute_mode=False,
    )
    return topk_min_fde


def top_k_miss_rate_calculator(
    k_values,
    miss_rate_dis_thr,
    pred_trajs,
    sample_probs,
    gt_trajs,
    timestep_masks=None,
):  # noqa: D205,D400
    """Calculate the miss rate metric for a set of k most probable
    predicted results. If the maximum pointwise L2 distance between
    the prediction and ground truth is greater than the miss_rate_dis_thr,
    we define the prediction as a miss.

    Args:
        k_values (List): list of top-k values to calculate the metric.
        miss_rate_dis_thr (float): Threshold to consider if a prediction
            is a hit or not.
        pred_trajs (np.array, [num_obj, num_samples, num_steps, num_states]):
            the predicted trajectories.
        sample_probs (np.array, [num_obj, num_samples]): sample probabilities.
        gt_trajs (np.array, [num_obj, num_samples, num_steps, num_states]):
            the ground-truth trajectories.
        timestep_masks (np.array, [num_obj, timesteps]): timestep masks.

    Returns:
        topk_miss_flag (np.array, [num_obj, len(k_values)]): the miss flag
            of the top k samples of each object, the value will be true if
            the prediction is a miss.
    """
    topk_max_dist = top_k_min_calculator_template(
        traj_max_l2_dist,
        k_values,
        pred_trajs,
        sample_probs,
        gt_trajs,
        timestep_masks,
        absolute_mode=False,
    )
    topk_miss_flag = topk_max_dist >= miss_rate_dis_thr
    return topk_miss_flag


def top_k_velo_diff_calculator(
    k_values,
    pred_trajs,
    sample_probs,
    gt_trajs,
    timestep_masks=None,
    return_first_only=False,
    return_gt_velo=False,
):  # noqa: D205,D400
    """Calculate the velocity difference metric for a set of k most probable
    predicted results. If the maximum pointwise L2 distance between
    the prediction and ground truth is greater than the miss_rate_dis_thr,
    we define the prediction as a miss.

    Args:
        k_values (List): list of top-k values to calculate the metric.
        pred_trajs (np.array, [num_obj, num_samples, num_steps, num_states]):
            the predicted trajectories.
        sample_probs (np.array, [num_obj, num_samples]): sample probabilities.
        gt_trajs (np.array, [num_obj, num_samples, num_steps, num_states]):
            the ground-truth trajectories.
        timestep_masks (np.array, [num_obj, timesteps]): timestep masks.
        return_first_only(bool): set True for return the velo diff of the
            first timestep.
        return_gt_velo(bool): return the velocity sequence of the gts.

    Returns:
        vel_diff_ (np.array, [num_obj, num_samples, num_steps-1]): the
        velocity difference between the preds trajs and the gts.
    """
    cal_kwargs = {
        "return_gt_velo": return_gt_velo,
        "return_first_only": return_first_only,
    }
    return top_k_min_calculator_template(
        traj_velo_diff,
        k_values,
        pred_trajs,
        sample_probs,
        gt_trajs,
        timestep_masks,
        absolute_mode=True,
        **cal_kwargs,
    )


def fpvped_get_top_k_template(
    measure_func: Callable,
    target_traj: np.array,
    pred_traj: np.array,
    prob: np.array,
    top_k_values: List,
    apply_func: str = "accumulative_min",
):  # noqa: D205,D400
    """Given a measure function, calculate the minimum metric for a set
    of k most probable predicted results.

    Args:
        measure_func: the measure function.
        target_traj ([batch_size, K, step, dim]): the ground-truth
            trajectories.
        pred_trajs ([[batch_size, K, step, dim]]): the predicted trajectories.
        prob ([batch_size, K]): the predict probabilities of K trajectories.
        top_k_values: list of the top-k values to calculate the metric.
        applied_func: apply accumulative_max/min to metric values.

    Returns:
        topk_min_measure ([batch_size, len(top_k_values)]):
            the min metric of the top k samples of each object.
    """
    # [batch_size, k_values]
    ret_measures = measure_func(target_traj, pred_traj)
    k_value = pred_traj.shape[1]
    # [batch_size, k_values]
    ranked_measures = sort_traj_samples_by_likelyhood(
        ret_measures, prob, apply_func
    )

    # [batch_size, len(top_k_values)]
    topk_min_measure = ranked_measures[
        :, [min(k, k_value) - 1 for k in top_k_values]
    ]

    return topk_min_measure


def fpvped_rmse_dist(target_traj: np.array, pred_traj: np.array):
    """Calculate the rmse between predicted and ground-truth trajectories.

    Args:
        target_traj ([batch_size, K, step, dim]): the ground-truth
            trajectories.
        pred_traj ([batch_size, K, step, dim]): the predicted trajectories.

    Returns:
        rmse: (np.array, [batch_size, K]): the batch rmse distances.
    """
    # [batch_size, k_value, dec_step, dim]
    diff = target_traj - pred_traj
    # [batch_size, k_value, dec_step, dim] -> [batch_size, k_value, dec_step]
    rmse = np.square(diff).mean(axis=(-1))
    rmse = np.sqrt(rmse)
    # [batch_size, k_value, dec_step] -> [batch_size, k_value]
    rmse = np.mean(rmse, axis=2)

    return rmse


def fpvped_ade_dist(target_traj: np.array, pred_traj: np.array):
    """Calculate the ade between predicted and ground-truth trajectories.

    Args:
        target_traj ([batch_size, K, step, dim]): the ground-truth
            trajectories.
        pred_traj ([batch_size, K, step, dim]): the predicted trajectories.

    Returns:
        ade: ([batch_size, K]): the batch ade distances.
    """
    # [batch_size, k_value, dec_step, dim]
    diff = target_traj - pred_traj
    # [batch_size, k_value, dec_step, dim] -> [batch_size, k_value]
    ade = np.linalg.norm(diff, axis=-1).mean(axis=2)
    return ade


def fpvped_fde_dist(target_traj: np.array, pred_traj: np.array):
    """Calculate the fde between predicted and ground-truth trajectories.

    Args:
        target_traj ([batch_size, K, step, dim]): the ground-truth
            trajectories.
        pred_traj ([batch_size, K, step, dim]): the predicted trajectories.

    Returns:
        fde: ([batch_size, K]): the batch fde distances.
    """
    # [batch_size, k_value, dec_step, dim]
    diff = target_traj - pred_traj
    # [batch_size, k_value, dec_step, dim] -> [batch_size, k_value]
    fde = np.linalg.norm(diff[:, :, -1, :], axis=-1)
    return fde


def top_k_fpvped_min_rmse_calculator(
    target_traj: np.array,
    pred_traj: np.array,
    prob: np.array,
    top_k_values: List,
    metrics_type: Dict,
):  # noqa: D205,D400
    """Calculate the fpvped_rmse metric for a set of k most probable
    predicted results.

    Args:
        target_traj ([batch_size, K, step, dim]):
            the ground-truth trajectories and target_traj is in x1y1x2y2
            style in its last dimension.
        pred_traj ([batch_size, K, step, dim]):
            the predicted trajectories and pred_traj is in x1y1x2y2
            style in its last dimension.
        prob ([batch_size, K]): sample probabilities.
        prob ([batch_size, K]): sample probabilities.
        top_k_values: list of top-k values to calculate the metric.
        metrics_type: the metrics will be calculated, the keys of the
            dict represent the trajectory length used to calculate the
            metric, "05" is 0.5s, "10" is 1s and "15" is 1.5s.

    Returns:
        rmse: the batch top-k rmse result, the keys of rmse is the
            same as metrics_type.
    """
    rmse = {}
    for metric in metrics_type.keys():
        end = metrics_type[metric]
        rmse[metric] = fpvped_get_top_k_template(
            fpvped_rmse_dist,
            target_traj[:, :, :end, :],
            pred_traj[:, :, :end, :],
            prob,
            top_k_values,
        )
    return rmse


def top_k_fpvped_min_ade_calculator(
    target_traj: np.array,
    pred_traj: np.array,
    prob: np.array,
    top_k_values: List,
    metrics_type: Dict,
):  # noqa: D205,D400
    """Calculate the fpvped_min_ade metric for a set of k most probable
    predicted results.

    Args:
        target_traj ([batch_size, K, step, dim]):
            the ground-truth trajectories and target_traj is in cxcympb
            style in its last dimension where mpb represents the middle
            point of the bottom of a bbox, we can know that:
            target_traj[:, :, :, :2]: x and y coords of the center point.
            target_traj[:, :, :, 2:4]: x and y coords of mpb.
        pred_traj ([batch_size, K, step, dim]):
            the predicted trajectories and pred_traj is in cxcympb
            style in its last dimension where mpb represents the middle
            point of the bottom of a bbox, we can know that:
            pred_traj[:, :, :, :2]: x and y coords of the center point.
            pred_traj[:, :, :, 2:4]: x and y coords of mpb.
        prob ([batch_size, K]): sample probabilities.
        top_k_values: list of top-k values to calculate the metric.
        metrics_type: the metrics will be calculated, the keys of the dict
            represent the trajectory length used to calculate the metric,
            "05" is 0.5s, "10" is 1s and "15" is 1.5s.

    Returns:
        ade_center: the batch top-k ade_center result, the keys of
            rmse is the same as metrics_type.
        ade_mpb:  the batch top-k ade_mpb result, the keys of rmse is
            the same as metrics_type.
    """
    ade_center = {}
    for metric in metrics_type.keys():
        end = metrics_type[metric]
        target_center = target_traj[:, :, :end, :2]
        pred_center = pred_traj[:, :, :end, :2]
        ade_center[metric] = fpvped_get_top_k_template(
            fpvped_ade_dist,
            target_center,
            pred_center,
            prob,
            top_k_values,
        )

    ade_mpb = {}
    for metric in metrics_type.keys():
        end = metrics_type[metric]
        target_mpb = target_traj[:, :, :end, 2:4]
        pred_mpb = pred_traj[:, :, :end, 2:4]
        ade_mpb[metric] = fpvped_get_top_k_template(
            fpvped_ade_dist,
            target_mpb,
            pred_mpb,
            prob,
            top_k_values,
        )
    return ade_center, ade_mpb


def top_k_fpvped_min_fde_calculator(
    target_traj: np.array,
    pred_traj: np.array,
    prob: np.array,
    top_k_values: List,
    metrics_type: Dict,
):  # noqa: D205,D400
    """Calculate the fpvped_min_fde metric for a set of k most probable
    predicted results.

    Args:
        target_traj ([batch_size, K, step, dim]):
            the ground-truth trajectories and target_traj is in cxcympb
            style in its last dimension where mpb represents the middle
            point of the bottom of a bbox, we can know that:
            target_traj[:, :, :, :2]: x and y coords of the center point.
            target_traj[:, :, :, 2:4]: x and y coords of mpb.
        pred_traj ([batch_size, K, step, dim]):
            the predicted trajectories and pred_traj is in cxcympb
            style in its last dimension where mpb represents the middle
            point of the bottom of a bbox, we can know that:
            pred_traj[:, :, :, :2]: x and y coords of the center point.
            pred_traj[:, :, :, 2:4]: x and y coords of mpb.
        prob ([batch_size, K]): sample probabilities.
        top_k_values: list of top-k values to calculate the metric.
        metrics_type: the metrics will be calculated, the keys of the dict
            represent the trajectory length used to calculate the metric,
            "05" is 0.5s, "10" is 1s and "15" is 1.5s.

    Returns:
        fde_center: the batch top-k fde_center result, the keys of
            rmse is the same as metrics_type.
        fde_mpb: the batch top-k fde_mpb result, the keys of rmse is
            the same as metrics_type.
    """
    fde_center = {}
    for metric in metrics_type.keys():
        end = metrics_type[metric]
        target_center = target_traj[:, :, :end, :2]
        pred_center = pred_traj[:, :, :end, :2]
        fde_center[metric] = fpvped_get_top_k_template(
            fpvped_fde_dist,
            target_center,
            pred_center,
            prob,
            top_k_values,
        )

    fde_mpb = {}
    for metric in metrics_type.keys():
        end = metrics_type[metric]
        target_mpb = target_traj[:, :, :end, 2:4]
        pred_mpb = pred_traj[:, :, :end, 2:4]
        fde_mpb[metric] = fpvped_get_top_k_template(
            fpvped_fde_dist,
            target_mpb,
            pred_mpb,
            prob,
            top_k_values,
        )
    return fde_center, fde_mpb
