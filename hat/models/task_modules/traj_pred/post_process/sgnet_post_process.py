# Copyright (c) Horizon Robotics. All rights reserved.

import os
from typing import Callable

import numpy as np
import pandas as pd
import torch

from hat.core.traj_pred_utils import (
    bbox_denormalize,
    cxcywh_to_x1y1x2y2,
    x1y1x2y2_to_cxcympb,
)
from hat.registry import OBJECT_REGISTRY

__all__ = ["SGNetPostProcessor"]


@OBJECT_REGISTRY.register
class SGNetPostProcessor(torch.nn.Module):
    """SGNet post process.

    The following operations are performed in the post-processing:
    1.Convert the model output tensors to [batch_size, K, step, dim] in shape.
    2.Convert the target and predicted trajectories to absolute coords from
    relative coords relative to the input trajectories.
    3.Denormalize the model output result.
    4.Save the model output result to csv file.
    """

    def __init__(
        self,
        bbox_type: str = "cxcywh",
        normalize_type: str = "zero-one",
        save_res: bool = False,
        save_res_path: str = None,
        dir_name: str = None,
    ):
        """Initialize method.

        Args:
            bbox_type: the type of bboxes in model output, only cxcywh
                is supported now.
            normalize_type: the normalize type used in dataset, only
                "zero-one" is supported now.
            save_res: save the test result in predict if it is True.
            save_res_path: the path to save the model output result.
            dir_name: the current local time, used as the name of the
                result save file.
        """
        super(SGNetPostProcessor, self).__init__()
        assert (
            bbox_type == "cxcywh"
        ), f"Only cxcywh is supported now. But found: {bbox_type}."
        assert (
            normalize_type == "zero-one"
        ), f"Only zero-one normalization is supported now: {normalize_type}."
        self.normalize_type = normalize_type
        self.save_res = save_res
        self.save_res_path = save_res_path
        self.dir_name = dir_name

    def get_max_prob_index(self, probabilities: torch.Tensor):
        """Get the index of the trajectory with max probability.

        Args:
            probabilities ([batch_size, 1, 1, K]): the model predicted
                probabilities.

        Returns:
            index ([2, batch_size]): the index of the max probability
                trajectories.
        """
        probabilities = probabilities.squeeze(1).squeeze(1)
        # [batch_size, 1]
        index = probabilities.argmax(dim=1).tolist()

        return index

    def save_test_res(
        self,
        save_path: str,
        dir_name: str,
        pred_traj: torch.Tensor,
        target_traj: torch.Tensor,
        input_traj: torch.Tensor,
        stamp: list,
        ped_id: list,
        date_token: list,
        prob: torch.Tensor,
        prob_func: Callable,
        position: np.ndarray,
        height: np.ndarray,
        vcs_vel: np.ndarray,
        global_vel: np.ndarray,
    ):
        """Save the model output result for visualization.

        Args:
            save_path: the absolute path to save the model output result.
            dir_name: the directory name of the result save file.
            pred_traj ([batch_size, K, step, dim]):
                the model output trajectories.
            target_traj ([batch_size, 1, dec_step, dim]):
                the groundtruth trajectories.
            input_traj ([batch_size, 1, enc_step, dim]):
                the model input trajectories.
            stamp: the stamps of a batch data.
            date_token: the date_token of a batch data.
            prob ([batch_size, 1, 1, K]):
                the model predicted probabilities.
            prob_func: the function that get the index of the trajectory
                with max probability in model output.
            position ([batch_size, 1, enc_step, 2]): current position of
                the obstacles.
            height ([batch_size, enc_step, 1)]): height of obstacles.
            vcs_vel ([batch_size, enc_step, 2)]):
                the relative velocity of obstacle in VCS.
            global_vel ([batch_size, enc_step, 2)]):
                the absolute velocity of obstacle in VCS.
        """
        batch_size = pred_traj.shape[0]
        # list, [batch_size]
        idx = prob_func(prob)
        pred_traj = pred_traj.cpu().detach().numpy().astype(np.int64)
        target_traj = target_traj.cpu().detach().numpy().astype(np.int64)
        input_traj = input_traj.cpu().detach().numpy().astype(np.int64)
        data = []
        cols = ["date_token", "stamp", "id", "input_last"]
        for i in range(pred_traj.shape[2]):
            cols.append("pred_traj_step_" + str(i + 1))
        for i in range(pred_traj.shape[2]):
            cols.append("target_traj_step_" + str(i + 1))
        cols += [
            "vcs_x",
            "vcs_y",
            "height",
            "vcs_vx",
            "vcs_vy",
            "global_vx",
            "global_vy",
        ]
        for step in range(batch_size):
            # [k_value, dec_step, dim]
            pred_step = pred_traj[step, :, :, :]
            # [dec_step, dim]
            pred_step = pred_step[idx[step]]
            # [dec_step, dim]
            target_step = target_traj[step, -1, :, :]
            # [batch_size, dim, k_value, 1] -> [1, dim]
            input_last_step = input_traj[step, -1, -1:, :]
            stamp_step = str(stamp[step][-1])
            id_step = str(ped_id[step][-1])
            date_token_step = str(date_token[step][-1])
            vcs_x = position[step, 0, -1, 0]
            vcs_y = position[step, 0, -1, 1]
            ped_height = height[step, 0, -1, 0]
            vcs_vx = vcs_vel[step, 0, -1, 0]
            vcs_vy = vcs_vel[step, 0, -1, 1]
            global_vx = global_vel[step, 0, -1, 0]
            global_vy = global_vel[step, 0, -1, 1]
            line = []
            line.extend([date_token_step, stamp_step, id_step])
            line.append(list(input_last_step[0]))
            for point in pred_step:
                line.append(list(point))
            if target_step.mean() == 0:
                target_step = pred_step
            for point in target_step:
                line.append(list(point))
            line += [
                vcs_x,
                vcs_y,
                ped_height,
                vcs_vx,
                vcs_vy,
                global_vx,
                global_vy,
            ]
            data.append(line)

        df = pd.DataFrame(data, columns=cols)
        res_path = os.path.join(save_path, dir_name)
        os.makedirs(res_path, exist_ok=True)
        res_csv = os.path.join(res_path, dir_name + "_test_result.csv")

        if os.path.exists(res_csv):
            df.to_csv(res_csv, mode="a", header=False, index=None)
        else:
            df.to_csv(res_csv, index=None)

    def forward(self, model_output: dict):
        """Post processing forward.

        Args:
            model_output: A dict of model output which includes the predicted
            and ground-truth trajectories. The following keys of output will
            be used in post process:
            1. 'pred_traj' ([batch_size, K, dim, dec_step]): the predicted
            trajectories in cxcywh style.
            2. 'input_traj' ([batch_size, dim, 1, enc_step]): the model input
            trajectories in cxcywh style.
            3. 'target_traj' ([batch_size, dim ,1, dec_step]): the ground-truth
            trajectories in cxcywh style.
            4. 'all_goal_trajs' ([batch_size, dim ,1, dec_step]): the goal
            trajectories generated in SGE module in cxcywh style.
            5. 'enable_relative': the target_traj will be relative coordinates
            relative to the last step of input_traj if it is True.
            6. stamp: the stamp of a batch data.
            7. date_token: the date_token of a batch data.
            8. stage: the model stage: "train", "val", "test".
            9. position ([batch_size, enc_step, 2)]):
            historical position of obstacle in VCS.
            10. height ([batch_size, enc_step]): height of obstacles.
            11. vcs_vel ([batch_size, enc_step, 2)]):
            the relative velocity of obstacle in VCS.
            12. global_vel ([batch_size, enc_step, 2)]):
            the absolute velocity of obstacle in VCS.

        Returns:
            results: the postprocessed model output, the following items are
            included:
            1. "pred_traj" ([batch_size, K, dec_step, dim]): the predicted
            trajectories in x1x2y1y2 style.
            2. "input_traj" ([batch_size, 1, enc_step, dim]): the model input
            trajectories in x1x2y1y2 style.
            3. "all_goal_trajs" ([batch_size, 1, dec_step, dim]): the goal
            trajectories generated in SGE module in x1x2y1y2 style.
            4. "target_traj" ([batch_size, 1, dec_step, dim]): the ground-truth
            trajectories in x1x2y1y2 style.
            5. "pred_cxcympb" ([atch_size, K, dec_step, dim]): the predicted
            trajectories in cxcympb style.
            6. "input_cxcympb" ([batch_size, 1, enc_step, dim]): the model
            input trajectories in cxcympb style.
            7. "goal_cxcympb" ([batch_size, 1, dec_step, dim]): the goal
            trajectories generated in SGE module in cxcympb style.
            8. "target_cxcympb" ([batch_size, 1, dec_step, dim]): the
            ground-truth trajectories in cxcympb style.
            9. position ([batch_size, 1, enc_step, 2)]):
            historical position of obstacle in VCS.
            10. height ([batch_size, 1, enc_step, 1)]): height of obstacles.
            11. vcs_vel ([batch_size, 1, enc_step, 2)]):
            the relative velocity of obstacle in VCS.
            12. global_vel ([batch_size, 1, enc_step, 2)]):
            the absolute velocity of obstacle in VCS.
        """
        # 1.Extract model output data.
        # [batch_size, K, dim, dec_step]
        pred_traj = model_output["pred_traj"]
        # [batch_size, dim, 1, enc_step]
        input_traj = model_output["input_traj"]
        # [batch_size, dim ,1, dec_step]
        all_goal_trajs = model_output["all_goal_trajs"]
        # [batch_size, dim ,1, dec_step]
        target_traj = model_output["target_traj"]
        enable_relative = model_output["enable_relative"]
        stamp = model_output["stamp"]
        ped_id = model_output["id"]
        date_token = model_output["date_token"]
        stage = model_output["stage"]
        # [batch_size, enc_step, 2]
        position = model_output["position"]
        # [batch_size, enc_step, 1]
        height = model_output["height"]
        # [batch_size, enc_step, 2]
        vcs_vel = model_output["vcs_vel"]
        # [batch_size, enc_step, 2]
        global_vel = model_output["global_vel"]
        # [batch_size, 1, 1, K]
        k_value = pred_traj.shape[1]
        batch_size = pred_traj.shape[0]
        if k_value > 1 and "probabilities" in model_output:
            prob = model_output["probabilities"]
        else:
            prob = torch.ones((batch_size, 1, 1, k_value))

        # 2.Transform to [batch_size, K, step, dim].
        pred_traj = pred_traj.permute(0, 1, 3, 2)
        # [batch_size, 1, enc_step, dim]
        input_traj = input_traj.permute(0, 2, 3, 1)
        # [batch_size, 1, dec_step, dim]
        all_goal_trajs = all_goal_trajs.permute(0, 2, 3, 1)
        target_traj = target_traj.permute(0, 2, 3, 1)
        # [batch_size, 1, enc_step, dim]
        position = np.expand_dims(position, 3)
        position = position.transpose([0, 3, 1, 2])
        height = np.expand_dims(height, (2, 3))
        height = height.transpose([0, 3, 1, 2])
        vcs_vel = np.expand_dims(vcs_vel, 3)
        vcs_vel = vcs_vel.transpose([0, 3, 1, 2])
        global_vel = np.expand_dims(global_vel, 3)
        global_vel = global_vel.transpose([0, 3, 1, 2])
        # 3. Recover from relative coordinates.
        if enable_relative[-1]:
            last_input = input_traj[:, :, -1:, :]
            target_traj = last_input + target_traj
            all_goal_trajs = last_input + all_goal_trajs
            last_input = last_input.repeat(1, k_value, 1, 1)
            pred_traj = last_input + pred_traj

        # 4.Denormalize and convert to x1y1x2y2 from cxcywh.
        pred_traj = bbox_denormalize(pred_traj, self.normalize_type)
        pred_traj = cxcywh_to_x1y1x2y2(pred_traj)
        input_traj = bbox_denormalize(input_traj, self.normalize_type)
        input_traj = cxcywh_to_x1y1x2y2(input_traj)
        all_goal_trajs = bbox_denormalize(all_goal_trajs, self.normalize_type)
        all_goal_trajs = cxcywh_to_x1y1x2y2(all_goal_trajs)
        target_traj = bbox_denormalize(target_traj, self.normalize_type)
        target_traj = cxcywh_to_x1y1x2y2(target_traj)
        goal_cxcympb = x1y1x2y2_to_cxcympb(all_goal_trajs)
        target_cxcympb = x1y1x2y2_to_cxcympb(target_traj)
        pred_cxcympb = x1y1x2y2_to_cxcympb(pred_traj)
        input_cxcympb = x1y1x2y2_to_cxcympb(input_traj)

        # 5.Update the result dict.
        results = {
            "pred_traj": pred_traj,
            "input_traj": input_traj,
            "all_goal_trajs": all_goal_trajs,
            "target_traj": target_traj,
            "pred_cxcympb": pred_cxcympb,
            "input_cxcympb": input_cxcympb,
            "goal_cxcympb": goal_cxcympb,
            "target_cxcympb": target_cxcympb,
            "position": position,
            "height": height,
            "vcs_vel": vcs_vel,
            "global_vel": global_vel,
        }

        # 6.save the model results for visualization.
        if stage[-1] == "test" and self.save_res:
            self.save_test_res(
                self.save_res_path,
                self.dir_name,
                pred_traj,
                target_traj,
                input_traj,
                stamp,
                ped_id,
                date_token,
                prob,
                self.get_max_prob_index,
                position,
                height,
                vcs_vel,
                global_vel,
            )

        return results
