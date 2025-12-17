# Copyright (c) Horizon Robotics. All rights reserved.
import logging
from typing import Dict, List, Tuple, Union

import numpy as np
from scipy import optimize
from scipy.ndimage.interpolation import rotate, shift
from skimage.draw import polygon

from hat.core.traj_pred_typing import SeqCenter
from hat.core.traj_pred_utils import Affine2D
from hat.core.traj_pred_utils import TdtCoordHelper as TCH
from hat.registry import OBJECT_REGISTRY

logger = logging.getLogger(__name__)


__all__ = [
    "PlanGaussianRandom",
    "PlanEgoMotion",
    "PlanAgentMotion",
    "PlanPerturbation",
    "PlanEgoGridGenerate",
    "PlanAgentsGridGenerate",
    "PlanGridAction",
]


@OBJECT_REGISTRY.register
class PlanGaussianRandom:
    """Draws samples from a normal distribution with specified \
    mean and standard deviation. This gaussian can be multidimensional.

    Args:
        mean: mean noise value of longitudinal distance(m),
              lateral distance(m) and angular(rad)
        std: stadard deviation noise of longitudinal distance(m),
              lateral distance(m) and angular(rad)
        random_seed: default 0

    """

    def __init__(
        self,
        std: List,
        mean: List = (0, 0, 0),
        random_seed: int = 0,
    ):

        self.mean = np.array(mean)
        self.std = np.array(std)
        self.rng = np.random.default_rng(random_seed)

    def __call__(self, sample: Dict) -> Dict:
        """Callable function.

        Generate the gaussian noise for feature perturbation

        Returns:
            sample (Dict): the updated sample as described above.
        """
        (lon_offset_m, lat_offset_m, yaw_offset_rad) = self._sample()

        perturb_center = SeqCenter(lon_offset_m, lat_offset_m, yaw_offset_rad)

        sample["perturb_center"] = perturb_center

        return sample

    def _sample(self) -> np.ndarray:
        return self.rng.normal(self.mean, self.std)


@OBJECT_REGISTRY.register
class PlanEgoMotion:
    """Generate a expert routing and ego history/future state in detail. \
    Routing info is a long term global coordinate from expert trajectory. \
    All state signal are interpolated from the global coordinate.

    Args:
        context_frames: number of context frames,
        seq_length: length of the entire sequence.

    """

    def __init__(
        self,
        context_frames: int = 4,
        seq_length: int = 16,
    ) -> None:

        self.context_frames = context_frames
        self.traj_len = seq_length - context_frames

    def __call__(self, sample: Dict) -> Dict:
        """Callable function.

        The following keys are required:
            1. `traj_navi`.
            2. `seq_center`.
        traj_navi is a dataframe sample with folling keys:
            ["timestamp", "frame_id",
            "pos_x", "pos_y", "yaw"]

        Args:
            sample (dict): input original sample.

        Returns:
            sample (dict): the sample with added
                plan_navi_traj, plan_his_state, plan_fut_state
        """

        traj_navi = sample["traj_navi"]
        seq_center = sample["seq_center"]

        # navi traj
        phy_car_array = traj_navi[["pos_x", "pos_y"]].values

        if np.all(np.isnan(phy_car_array)):
            route_traj = np.zeros_like(phy_car_array)
        else:
            # trnasform global coordinates to local coordinates
            route_traj = TCH.global_phy_to_agent_centric_phy(
                phy_car_array, seq_center[:2], seq_center[-1]
            ).astype(
                "float64"
            )  # noqa

        route_yaw = traj_navi["yaw"].values - seq_center.yaw
        route_yaw_unwrap = np.unwrap(route_yaw)

        state_stamp = traj_navi["timestamp"].values / 1000
        time_diff = state_stamp[1:] - state_stamp[:-1]
        dist = np.sqrt(np.sum((route_traj[1:] - route_traj[:-1]) ** 2, axis=1))
        state_velo = dist / time_diff

        # calculate 1 order diff
        state_yaw_rate = (
            route_yaw_unwrap[1:] - route_yaw_unwrap[:-1]
        ) / time_diff[-1]
        state_acc = (state_velo[1:] - state_velo[:-1]) / time_diff[-1]

        state_yaw_rate = np.insert(
            state_yaw_rate, -1, state_yaw_rate[-1], axis=0
        )
        state_velo = np.insert(state_velo, -1, state_velo[-1], axis=0)
        state_acc = np.insert(state_acc, -2, 0.0, axis=0)
        state_acc = np.insert(state_acc, -1, 0.0, axis=0)

        # state_yaw_rate = np.insert(
        #     state_yaw_rate, 0, state_yaw_rate[-1], axis=0)
        # state_velo = np.insert(state_velo, 0, state_velo[0], axis=0)
        # state_acc = np.insert(state_acc, 0, 0.0, axis=0)
        # state_acc = np.insert(state_acc, 1, 0.0, axis=0)

        state_cur = state_yaw_rate / (state_velo + 1e-8)
        state_cur[state_velo == 0] = 0

        # get current s0 frame id
        fut_start = (
            traj_navi["frame_id"]
            .values.tolist()
            .index(sample["last_context_frame_id"])
            + 1
        )
        his_start = fut_start - self.context_frames
        fut_end = fut_start + self.traj_len

        state_yaw = route_yaw_unwrap - route_yaw_unwrap[fut_start - 1]

        # TODO (bikun)
        # history : x y velocity yaw accelerate yaw_rate
        # future : x y velocity yaw accelerate yaw_rate
        his_state = np.zeros((self.context_frames, 6))
        fut_state = np.zeros((self.traj_len, 6))

        his_state[:, :2] = route_traj[his_start:fut_start]
        his_state[:, 2] = state_velo[his_start:fut_start]
        his_state[:, 3] = state_yaw[his_start:fut_start]
        his_state[:, 4] = state_acc[his_start:fut_start]
        his_state[:, 5] = state_cur[his_start:fut_start]

        fut_state[:, :2] = route_traj[fut_start:fut_end]
        fut_state[:, 2] = state_velo[fut_start:fut_end]
        fut_state[:, 3] = state_yaw[fut_start:fut_end]
        fut_state[:, 4] = state_acc[fut_start:fut_end]
        fut_state[:, 5] = state_cur[fut_start:fut_end]

        sample["plan_navi_traj"] = route_traj[fut_start:]
        sample["plan_his_state"] = his_state
        sample["plan_fut_state"] = fut_state

        return sample


@OBJECT_REGISTRY.register
class PlanEgoState:
    """Generate ego state at last context frame including ego speed \
    and road speed limits. The state vector is rasterized in grid map.

    Args:
        max_speed: max speed value in data inputs.
        max_speed_limit: max road speed limits in data inputs.

    """

    def __init__(
        self,
        grid_dim: int,
        max_speed: float = 20.0,
        max_speed_limit: float = 60.0,
        ego_track_id: int = -42,
    ) -> None:

        self.grid_dim = grid_dim
        self.max_speed = max_speed
        self.max_speed_lmt = max_speed_limit
        self.ego_track_id = ego_track_id

    def __call__(self, sample: Dict) -> Dict:
        """Callable function.

        `seq_df` is a dataframe requred in sample with following columns:
        'ego_speed', 'speedlimit',

        Args:
            sample (dict): input original sample.

        Returns:
            sample (dict): the sample with added
                plan_ego_motion
        """

        seq_df = sample["seq_df"]
        lcf_mask = (seq_df["timestamp"] == sample["lcf_timestamp"]) & (
            seq_df["track_id"] == self.ego_track_id
        )
        seq_df_vals = seq_df.values
        seq_df_cols = seq_df.columns

        ego_motion = np.zeros((2, self.grid_dim, self.grid_dim))

        if hasattr(seq_df, "ego_speed"):
            speed_col = seq_df_cols.get_loc("ego_speed")
            state_speed = seq_df_vals[lcf_mask, speed_col]
            norm_spd = (
                0.0
                if state_speed.shape[0] == 0
                else state_speed.item() / self.max_speed
            )
            ego_motion[0] = norm_spd

        if hasattr(seq_df, "speedlimit"):
            speed_lim_col = seq_df_cols.get_loc("speedlimit")
            state_limit = seq_df_vals[lcf_mask, speed_lim_col]
            norm_spdlmt = (
                0.0
                if state_limit.shape[0] == 0
                else state_limit.item() / self.max_speed_lmt
            )
            ego_motion[1] = norm_spdlmt

        ego_motion = np.clip(ego_motion, 0, 1) * 2 - 1
        sample["plan_ego_motion"] = ego_motion

        return sample


@OBJECT_REGISTRY.register
class PlanAgentMotion:
    """Collect agent state and transform to ego vcs coordinates. \
    the information is for agents occupancy grid prediction.

    Args:
        ego_track_id: the ego vehicle track id.
        ego_center: ego coordinate in BEV image coordinates.
        res: pixel resolution of the BEV image.
    """

    def __init__(
        self,
        ego_track_id: int,
        ego_center: Tuple,
        res: int,
    ) -> None:

        self.ego_track_id = ego_track_id
        self.center = ego_center
        self.res = res

    def __call__(self, sample: Dict) -> Dict:
        """Callable function.

        The following keys are required:
            [`valid_track_ids`, `track_ids`, `valid_img_coords`,
            `future_trajectories`,`context_states`,
            `last_context_frame_id`]

        traj_navi_obs is a dataframe sample with folling keys:
            ["timestamp", "frame_id", "track_id"
            "x", "y", "yaw"]

        Args:
            sample (dict): input original sample.

        Returns:
            sample (dict): the sample with added
                plan_agents_traj_vcs, plan_agents_his_traj_vcs,
                plan_agents_state_vector
        """
        valid_track_ids = sample["valid_track_ids"]
        track_ids = sample["track_ids"]
        coords = sample["valid_img_coords"]
        fut_traj = sample["future_trajectories"]
        his_traj = sample["context_states"]
        lcf = sample["last_context_frame_id"]

        OBS_COLS = ["timestamp", "frame_id", "track_id", "x", "y", "obs_yaw"]
        traj_navi_obs = sample["seq_df"][OBS_COLS]

        fut_agents_vcs = []
        hist_agents_vcs = []
        state_agents = []

        for t_idx, track_id in enumerate(valid_track_ids):
            if t_idx == valid_track_ids.index(self.ego_track_id):
                continue

            traj_mask = traj_navi_obs.values[:, 2] == track_id
            lcf_mask = traj_navi_obs.values[:, 1] == lcf
            phy_obs_center = traj_navi_obs.values[traj_mask & lcf_mask, 3:][0]

            #  ## agent future traj in vcs
            agent_traj_vcs = self._agents_global_phy_to_vcs(
                fut_traj[t_idx], phy_obs_center, sample["seq_center"]
            )

            agent_traj_hist_vcs = self._agents_global_phy_to_vcs(
                his_traj[track_ids.index(track_id)],
                phy_obs_center,
                sample["seq_center"],
            )

            # sped,acc,yawrate,yaw
            agent_yaw = coords[t_idx][2]
            agent_motion = sample["state_vectors"][t_idx]
            agent_motion = np.hstack([agent_motion, agent_yaw])

            # generate long-term horizon traj of agents
            # if np.all(np.isnan(phy_obs_array)):
            #     route_traj = np.zeros_like(phy_obs_array)
            # else:
            #     route_traj = TCH.global_phy_to_agent_centric_phy(
            #         phy_obs_array[:,:2], seq_center[:2],
            #         seq_center[-1]).astype("float64")

            fut_agents_vcs.append(agent_traj_vcs)
            hist_agents_vcs.append(agent_traj_hist_vcs)
            state_agents.append(agent_motion)

        sample["plan_agents_traj_vcs"] = fut_agents_vcs
        sample["plan_agents_his_traj_vcs"] = hist_agents_vcs
        sample["plan_agents_state_vector"] = state_agents

        return sample

    def _agents_global_phy_to_vcs(self, fut_traj, phy_obs_center, seq_center):
        """Transform the agents-centric trajectory into global coordinate \
        and transform the global coord into ego vcs coordinate.

        Args:
            fut_traj : agents-centric trajectory
            phy_obs_center : global center coord.
            seq_center : global ego center coord.

        Returns:
            agent_traj_vcs: ego-centric agents trajectory
        """
        agent_traj = TCH.agent_centric_phy_to_global_phy(
            fut_traj, phy_obs_center[:2], phy_obs_center[2]
        ).astype("float64")
        agent_traj_vcs = TCH.global_phy_to_agent_centric_phy(
            agent_traj, seq_center[:2], seq_center[2]
        ).astype("float64")

        # agent_traj_vcs_img = -agent_traj_vcs / self.res + (362, 256)

        return agent_traj_vcs


@OBJECT_REGISTRY.register
class PlanPerturbation:
    """Apply perturbation of the current ego state as data augmentation. \
    Ackerman model optimizer is used to get a feasible trajectory.

    Args:
        perturb_prob: probability between 0 and 1 of
                        applying the perturbation.
        ego_center: ego coordinate in BEV image coordinates.
        map_height: BEV image height.
        map_width: BEV image width.
        context_frames: number of context frames,
        seq_length:  length of the entire sequence,
        res: pixel resolution of the BEV image,
        min_displacement: minimum displacement required to
                            apply lateral & yaw perturbation
        min_acc: min acceleration constraint,
        max_acc: max acceleration constraint,
        min_steer: max curvature constraint,
        max_steer: max curvature constraint.
    """

    def __init__(
        self,
        perturb_prob: float,
        ego_center: Tuple,
        map_height: int,
        map_width: int,
        context_frames: int,
        seq_length: int,
        res: int,
        min_displacement: float = 4,
        min_acc: float = -0.3,  # min acceleration: -3 mps2
        max_acc: float = 0.15,  # max acceleration: 3 mps2
        min_steer: float = -0.2,  # max cur: 1/5 m
        max_steer: float = 0.2,  # max cur: 1/5 m
    ) -> None:
        self.perturb_prob = perturb_prob
        self.map_height = map_height
        self.map_width = map_width
        self.context_frames = context_frames
        self.center = ego_center
        self.res = res
        self.min_displacement = min_displacement
        # lateral, longitudinal and angular
        future_len = seq_length - context_frames
        self.wgx = np.ones(future_len)
        self.wgy = np.ones(future_len)
        self.wgr = np.ones(future_len)
        self.wgv = np.zeros(future_len)
        self.wsteer_acc = np.hstack(
            [3.0 * np.ones(future_len), 3.0 * np.ones(future_len)]
        )
        self.min_acc = min_acc
        self.max_acc = max_acc
        self.min_steer = min_steer
        self.max_steer = max_steer

    def __call__(self, sample: Dict) -> Dict:
        """Callable function.

        Transform the features with pertubated ego center inlcuding
        bev road map, bev obstacles, ego future gt trajectory, ego
        future optimize trajectory and optional agents gt trajectory

        Args:
            sample (dict): input original sample.

        Returns:
            sample (dict): the sample with pertubated features.
        """
        if np.random.rand() >= self.perturb_prob:
            return sample

        perturb_center = sample["perturb_center"]
        fut_state = sample["plan_fut_state"]
        navi_traj = sample["plan_navi_traj"]
        bev_road = sample["road_map"]
        rendered_obs_fut = sample["rendered_obs_fut"]
        rendered_obs = sample["rendered_obs"]

        displacements = np.linalg.norm(
            np.diff(fut_state[:, :2], axis=0), axis=1
        )

        if np.sum(displacements) < self.min_displacement:
            return sample

        solved_state_perturb = self.solve_optimize_traj(
            fut_state, perturb_center
        )
        fut_state_perturb = solved_state_perturb.copy()

        # Render fut frames as polygons
        collision = self.collision_detection(
            fut_state_perturb, rendered_obs_fut
        )

        if collision > 0:
            return sample

        # feature update
        road_perturb = self.feature_transform(perturb_center, bev_road, cval=0)
        rendered_obs_pt = self.feature_transform(
            perturb_center, rendered_obs, cval=-1
        )
        rendered_obs_fut_pt = self.feature_transform(
            perturb_center, rendered_obs_fut, cval=-1
        )

        # # traj axis transfer to pertubate agent center
        fut_state_perturb[:, :2] = TCH.global_phy_to_agent_centric_phy(
            fut_state_perturb[:, :2], perturb_center[:2], perturb_center[-1]
        ).astype(
            "float64"
        )  # noqa
        fut_state_perturb[:, 2] -= perturb_center[2]

        navi_traj_perturb = TCH.global_phy_to_agent_centric_phy(
            navi_traj, perturb_center[:2], perturb_center[-1]
        ).astype(
            "float64"
        )  # noqa

        # traj axis transfer to pertubate agent center
        if hasattr(sample, "plan_agents_traj_vcs"):
            sample["plan_agents_traj_vcs"] = self.coord_trans_agent_traj(
                sample["plan_agents_traj_vcs"], perturb_center
            )

        if hasattr(sample, "plan_agents_his_traj_vcs"):
            sample["plan_agents_his_traj_vcs"] = self.coord_trans_agent_traj(
                sample["plan_agents_his_traj_vcs"], perturb_center
            )
        """
        TODO debug info to be delete
        import matplotlib.pyplot as plt
        gx_img, gy_img = self.bev_2_img(fut_state[:,0], fut_state[:,1])
        nx_img, ny_img = self.bev_2_img(solved_state_perturb[:,0],
                                solved_state_perturb[:,1])
        gx_img_p_center, gy_img_p_center = \
                  self.bev_2_img(navi_traj_perturb[:12,0],
                  navi_traj_perturb[:12,1])
        new_xs_img_p_center, new_ys_img_p_center = \
                  self.bev_2_img(fut_state_perturb[:,0],
                  fut_state_perturb[:,1])

        fig, ax = plt.subplots(2, 2)
        ax[0,0].imshow(bev_road)
        ax[0,0].plot(gy_img, gx_img, color='y', lw=1, marker='o',
                    markeredgecolor='y', markersize=2, alpha=1)
        ax[0,0].plot(ny_img, nx_img, color='g',
          lw=1, marker='o',markeredgecolor='g', markersize=2, alpha=1)
        ax[0,1].imshow(road_perturb)
        ax[0,1].plot(gy_img_p_center, gx_img_p_center, color='y', lw=1,
              marker='o', markeredgecolor='y', markersize=2, alpha=1)
        ax[0,1].plot(new_ys_img_p_center, new_xs_img_p_center, color='g',
          lw=1, marker='o',markeredgecolor='g', markersize=2, alpha=1)
        ax[1,0].imshow(rendered_obs_pt.sum(axis=2))
        ax[1,1].imshow(rendered_obs_fut_pt.sum(axis=2))
        plt.savefig('test.png')
        """

        sample["rendered_obs"] = rendered_obs_pt
        sample["rendered_obs_fut"] = rendered_obs_fut_pt
        sample["road_map"] = road_perturb
        sample["plan_fut_state"] = fut_state_perturb
        sample["plan_navi_traj"] = navi_traj_perturb
        return sample

    def solve_optimize_traj(self, target_state, perturb_center):
        """Wrap for traj optimization.

        Args:
            target_state : expert gt trajectory
            perturb_center : perturbation coord(lon,lat,anglar)

        Returns:
            traj_optimize: solved optimize trajectory
        """
        gx = target_state[:, 0]
        gy = target_state[:, 1]
        gr = target_state[:, 3]
        gv = target_state[:, 2]

        x0 = perturb_center[0]
        y0 = perturb_center[1]
        r0 = perturb_center[2]
        v0 = gv[0]

        #  perform ackerman steering model fitting
        (
            new_xs,
            new_ys,
            new_yaws,
            new_vs,
            new_acc,
            new_steer,
        ) = self.fit_ackerman_model_exact(
            x0,
            y0,
            r0,
            v0,
            gx,
            gy,
            gr,
            gv,
        )

        traj_optimize = np.concatenate(
            [
                new_xs.reshape(-1, 1),
                new_ys.reshape(-1, 1),
                new_yaws.reshape(-1, 1),
                new_vs.reshape(-1, 1),
                new_acc.reshape(-1, 1),
                new_steer.reshape(-1, 1),
            ],
            axis=1,
        )  # noqa

        return traj_optimize

    def fit_ackerman_model_exact(
        self,
        x0: np.ndarray,
        y0: np.ndarray,
        r0: np.ndarray,
        v0: np.ndarray,
        gx: np.ndarray,
        gy: np.ndarray,
        gr: np.ndarray,
        gv: np.ndarray,
    ) -> Tuple[
        np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray
    ]:  # noqa
        """
        Fits feasible ackerman-steering trajectory to groundtruth control points.  # noqa
        Groundtruth is represented as 4 numpy arrays ``(gx, gy, gr, gv)``
        each of shape ``(N,)`` representing position, rotation and velocity at time i.
        Returns 4 arrays ``(x, y, r, v)`` each of shape ``(N,)`` - the optimal trajectory.

        The solution is found as minimisation of the following non-linear least squares problem:
        ::
        minimize F(steer, acc) = 0.5 * sum(
        (wgx[i] * (x[i] - gx[i])) ** 2 +
        (wgy[i] * (y[i] - gy[i])) ** 2 +
        (wgr[i] * (r[i] - gr[i])) ** 2 +
        (wgv[i] * (v[i] - gv[i])) ** 2 +
        (ws * steer[i]) ** 2 +
        (wa * acc[i]) ** 2)
        i = 1 ... N)
        subject to following unicycle motion model equations:
        x[i+1] = x[i] + cos(r[i]) * v[i]
        y[i+1] = y[i] + sin(r[i]) * v[i]
        r[i+1] = r[i] + steer[i]
        v[i+1] = v[i] + acc[i]
        min_steer < steer[i] < max_steer
        min_acc < acc[i] < max_acc
        for i = 0 .. N
        Weights ``wg*`` control adherence to the control points
        In a typical usecase ``wgx = wgy = 1`` and ``wgr = wgv = 0``

        Return:
            4 arrays ``(x, y, r, v)`` each of shape ``(N,)``- the optimal trajectory.
        """
        N = len(gx)

        def control2position(
            steer_acc: np.ndarray,
        ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:  # noqa
            steer, acc = np.split(steer_acc, 2)
            r = r0 + np.cumsum(steer * 0.5)
            v = v0 + np.cumsum(acc * 0.5)
            x = x0 + np.cumsum(np.cos(r) * 0.5 * v)
            y = y0 + np.cumsum(np.sin(r) * 0.5 * v)
            return x, y, r, v

        def residuals(steer_acc: np.ndarray) -> np.ndarray:
            x, y, r, v = control2position(steer_acc)
            return np.hstack(
                [
                    self.wgx * (x - gx),
                    self.wgy * (y - gy),
                    self.wgr * self.angular_distance(r, gr),
                    self.wgv * (v - gv),
                    self.wsteer_acc * steer_acc,
                ]
            )

        def jacobian(steer_acc: np.ndarray) -> np.ndarray:
            x, y, r, v = control2position(steer_acc)

            Jr1, Jr2, Jr3, Jr4 = (
                np.zeros(N),
                np.zeros(N),
                np.zeros(N),
                np.zeros(N),
            )
            Jv1, Jv2, Jv3, Jv4 = (
                np.zeros(N),
                np.zeros(N),
                np.zeros(N),
                np.zeros(N),
            )
            J = np.zeros((2 * N, 4 * N))

            for i in range(N - 1, -1, -1):
                Jr1[i:] = Jr1[i:] - np.sin(r[i]) * v[i] * self.wgx[i:]
                Jr2[i:] = Jr2[i:] + np.cos(r[i]) * v[i] * self.wgy[i:]
                Jv1[i:] = Jv1[i:] + np.cos(r[i]) * self.wgx[i:]
                Jv2[i:] = Jv2[i:] + np.sin(r[i]) * self.wgy[i:]
                Jr3[i:] = self.wgr[i:]
                Jv4[i:] = self.wgv[i:]

                J[i, :] = np.hstack([Jr1, Jr2, Jr3, Jr4])
                J[N + i, :] = np.hstack([Jv1, Jv2, Jv3, Jv4])

            return np.vstack([J.T, np.eye(N + N) * self.wsteer_acc])

        min_bound = np.concatenate(
            (self.min_steer * np.ones(N), self.min_acc * np.ones(N))
        )
        max_bound = np.concatenate(
            (self.max_steer * np.ones(N), self.max_acc * np.ones(N))
        )
        result = optimize.least_squares(
            residuals, np.zeros(2 * N), jacobian, (min_bound, max_bound)
        )

        x, y, r, v = control2position(result["x"])
        steer, acc = result["x"][:N], result["x"][N:]
        return x, y, r, v, acc, steer

    def angular_distance(
        self,
        angle_a: Union[float, np.ndarray],
        angle_b: Union[float, np.ndarray],
    ) -> Union[float, np.ndarray]:
        """Take two arrays of angles in radian \
        and compute the angular distance, wrap the angular \
        distance such that they are always in the [-pi, pi) range.

        Args:
            angle_a (np.ndarray, float): first array of angles in radians
            angle_b (np.ndarray, float): second array of angles in radians

        Returns:
            angular distance in radians between two arrays of angles
        """

        return (angle_a - angle_b + np.pi) % (2 * np.pi) - np.pi

    def collision_detection(self, fut_state_perturb, rendered_obs_fut):
        """Calculate the rendered occupancy between obstacles and ego \
        in related frames to check the collision risk.

        Args:
            fut_state_perturb : ego trajectory in vcs
            rendered_obs_fut : rendered obstacle agents occupancy

        Returns:
            mask_collision: collision flag
        """

        # ego HW
        offsets = (
            np.array(
                [
                    [2, 2, -2, -2],  # x0, x1, x2, x3
                    [0.75, -0.75, -0.75, 0.75],  # y0, y1, y2, y3
                ]
            )
            * 1.5
        )

        # generate pertubated ego bounding boxes
        # -- caluclate rotate matrix: [df_length, 4] -> [df_length, 2, 2]
        yaw = fut_state_perturb[:, 2, None]
        rotate = np.concatenate(
            (np.cos(yaw), -np.sin(yaw), np.sin(yaw), np.cos(yaw)), axis=-1
        )
        rotate = rotate.reshape([-1, 2, 2])
        # -- caluclate corners: [df_length, 2, 4]
        coors = fut_state_perturb[:, :2, None]
        coors = coors.reshape(-1, 2)[:, :, None]
        corners = np.matmul(rotate, offsets) + coors

        # render ego bounding boxes
        rendered_ego_fut = np.zeros(
            [3 * self.map_height, 3 * self.map_width, coors.shape[0]]
        )
        for ind in range(coors.shape[0]):

            v_x, v_y = corners[ind, 0, :], corners[ind, 1, :]
            v_x, v_y = self._bev_2_img(v_x, v_y)

            if v_x.size == 0 or v_y.size == 0:
                continue
            elif (
                np.min(v_x) < -self.map_height
                or np.max(v_x) >= 2 * self.map_height
                or np.min(v_y) < -self.map_width
                or np.max(v_y) >= 2 * self.map_width
            ):
                continue
            else:
                xx, yy = polygon(v_x, v_y)
                rendered_ego_fut[
                    xx + self.map_height, yy + self.map_width, ind
                ] = 1

        rendered_ego_fut = rendered_ego_fut[
            self.map_height : 2 * self.map_height,
            self.map_width : 2 * self.map_width,
            :,
        ]  # noqa

        # collision mask detection
        mask_ego_fut = rendered_ego_fut > 0
        mask_obs_fut = rendered_obs_fut > 0
        mask_collision = mask_obs_fut * mask_ego_fut

        return mask_collision.any()

    def feature_transform(self, perturb_center, feature, cval):
        """Images transform with shift offset and yaw angle \
        for data agumentation.

        Args:
            feature : BEV images to be transformed
            perturb_center : shift offset and yaw angle
            cval: padding value

        Returns:
            feature_trans: transformed features
        """

        rotation_angle_in_degrees = -perturb_center[2] * 180 / np.pi
        shift_offset = (
            perturb_center[0] / self.res,
            perturb_center[1] / self.res,
            0,
        )

        """
        TODO: cv2 transform same as scipy for double check.
        M = np.float32([[1, 0, shift_offset[1]], [0, 1, shift_offset[0]]])
        R = cv2.getRotationMatrix2D(
            (self.center[1], self.center[0]), rotation_angle_in_degrees, 1)
        bev_pertubate = cv2.warpAffine(
            feature, M, (self.map_width, self.map_height))
        bev_pertubate = cv2.warpAffine(
            bev_pertubate, R, (self.map_width, self.map_height))
        """

        # padding to keep the ego in the center of the image
        padding_height = 2 * self.center[0] - feature.shape[0]
        padding_compensation = (
            np.zeros((padding_height, self.map_width, feature.shape[-1]))
            + cval
        )
        feature_padding = np.concatenate(
            [feature, padding_compensation], axis=0
        )

        # images agumentation
        feature_trans = shift(
            feature_padding, shift_offset, order=0, cval=cval
        )
        feature_trans = rotate(
            feature_trans,
            rotation_angle_in_degrees,
            order=0,
            reshape=False,
            cval=cval,
        )

        # eliminate padding to recover the original image size
        feature_trans = feature_trans[: self.map_height, :, :]

        return feature_trans

    def coord_trans_agent_traj(self, agent_traj, perturb_center):
        """Translate agents trajectory with new ego offset coord.

        Args:
            agent_traj : agents obstacle gt trajectory
            perturb_center : perturbation coord(lon,lat,anglar)

        Returns:
            agent_traj_perturb: agents obstacle gt trajectory in new vcs
        """
        x0 = perturb_center[0]
        y0 = perturb_center[1]
        r0 = perturb_center[2]

        agent_traj_perturb = []
        for ind in range(len(agent_traj)):

            agent_traj_single = agent_traj[ind]

            agentx_p_center, agenty_p_center = Affine2D.coord_translate(
                agent_traj_single[:, 0], agent_traj_single[:, 1], x0, y0
            )
            agentx_p_center, agenty_p_center = Affine2D.coord_rotate(
                agentx_p_center, agenty_p_center, r0
            )

            agent_traj_new = np.concatenate(
                [
                    agentx_p_center.reshape(-1, 1),
                    agenty_p_center.reshape(-1, 1),
                ],
                axis=1,
            )  # noqa
            agent_traj_perturb.append(agent_traj_new)

        return agent_traj_perturb

    def _bev_2_img(self, x, y):

        new_x = -x / self.res + self.center[0]
        new_y = -y / self.res + self.center[1]

        return new_x, new_y


@OBJECT_REGISTRY.register
class PlanEgoGridGenerate:
    """Function to get the expert's state visitation frequencies \
    This transform method used for imitation planning.

    Args:
        interpolate_freq: trajectory interplolate frequency.
        grid_dim: max grid dim of BEV.
        horizon: max mdp grid solve steps.
        grid_extent: BEV image width.

    """

    def __init__(
        self,
        interpolate_freq: int = 10,
        grid_dim: int = 65,
        horizon: int = 50,
        grid_extent: Tuple[int, int, int, int] = (-25, 25, -10, 40),
    ) -> None:

        self.grid_dim = grid_dim
        self.horizon = horizon
        self.ext = grid_extent
        self.interpolate_freq = interpolate_freq
        grid_size_m = self.ext[1] - self.ext[0]
        # calculate Coordinates(x,y) of every gride's center
        self.row_centers = np.linspace(
            self.ext[3] - grid_size_m / (self.grid_dim * 2),  # noqa
            self.ext[2] + grid_size_m / (self.grid_dim * 2),  # noqa
            self.grid_dim,
        )

        self.col_centers = np.linspace(
            self.ext[1] - grid_size_m / (self.grid_dim * 2),  # noqa
            self.ext[0] + grid_size_m / (self.grid_dim * 2),  # noqa
            self.grid_dim,
        )

    def __call__(self, sample: Dict) -> Dict:
        """Callable function.

        Generate long-term expert goal and path inside the grid map.
        The trajectory is interpolated to keep the consistency of the
        grid path.

        Args:
            sample (dict): input original sample.

        Returns:
            sample (dict): the sample with added grid features.
        """

        # generate temporal grid occupancy
        fut_traj = sample["plan_fut_state"][:, :2]
        fut_traj = np.insert(fut_traj, 0, 0, axis=0)

        svf_seg = []
        for i in range(len(fut_traj) - 1):
            traj_seg = self.traj_interpolated(
                fut_traj[i + 1].reshape(1, 2), fut_traj[i]
            )
            svf_e_seg, _, _ = self.traj_to_grid_path(traj_seg)
            svf_seg.append(svf_e_seg[0])

        svf_seg = np.stack(svf_seg)
        sample["plan_svf_segment"] = svf_seg

        # generate spatial grid path and grid goal
        navi_traj = sample["plan_navi_traj"]
        fut_interpolated = self.traj_interpolated(navi_traj, [0, 0])
        svf_e, waypts_e, grid_idcs = self.traj_to_grid_path(fut_interpolated)

        sample["plan_svf_path"] = svf_e[:1, :, :]
        sample["plan_svf_goal"] = svf_e[1:, :, :]
        sample["plan_waypts_expert"] = waypts_e
        sample["plan_grid_idcs"] = grid_idcs

        return sample

    def traj_interpolated(self, navi_traj, start):
        """Trajectory linear interpolation.

        Args:
            navi_traj : long term expert gt trajectory
            start : start point of the trajectory

        Returns:
            fut_interpolated: interpolated trajectory
        """
        fut_interpolated = np.zeros(
            (navi_traj.shape[0] * self.interpolate_freq + 1, 2)
        )
        param_query = np.linspace(
            0,
            navi_traj.shape[0],
            navi_traj.shape[0] * self.interpolate_freq + 1,
        )
        param_given = np.linspace(
            0, navi_traj.shape[0], navi_traj.shape[0] + 1
        )
        val_given_x = np.concatenate(([start[0]], navi_traj[:, 0]))
        val_given_y = np.concatenate(([start[1]], navi_traj[:, 1]))
        fut_interpolated[:, 0] = np.interp(
            param_query, param_given, val_given_x
        )
        fut_interpolated[:, 1] = np.interp(
            param_query, param_given, val_given_y
        )

        return fut_interpolated

    def traj_to_grid_path(self, fut_interpolated, gamma=1):
        """Expert state visitation frequencies for training reward model, \
        Also waypoints in meters and grid indices will be calculated.

        Args:
            fut_interpolated : long term expert interplolated trajectory
            gamma : path faded parameter

        Returns:
            (state visitation frequencies results)
            svf_e:  raster path and goal
            waypts_e: path waypoints in meters
            grid_idcs: path grid center indices

        """
        svf_e = np.zeros((2, self.grid_dim, self.grid_dim))

        waypts_e = np.zeros((self.horizon, 2))
        grid_idcs = np.zeros((self.horizon, 2))

        count = 0
        row_prev = np.nan
        column_prev = np.nan
        for k in range(fut_interpolated.shape[0]):

            # Convert trajectory (x,y) co-ordinates to grid locations:
            column = np.argmin(
                np.absolute(fut_interpolated[k, 1] - self.col_centers)
            )
            row = np.argmin(
                np.absolute(fut_interpolated[k, 0] - self.row_centers)
            )

            # Demonstration ends when expert leaves the image crop
            # corresponding to the grid:
            if (
                count < self.horizon
                and self.ext[0] <= fut_interpolated[k, 1] <= self.ext[1]
                and self.ext[2] <= fut_interpolated[k, 0] <= self.ext[3]
                and (
                    count == 0
                    or abs(row - row_prev) < 2
                    and abs(column - column_prev) < 2
                )
            ):  # noqa

                # Check if cell location has changed
                if row != row_prev or column != column_prev:
                    # Add cell location to path states of expert
                    svf_e[0, row.astype(int), column.astype(int)] = gamma
                    gamma = gamma * gamma

                    # Get BEV coordinates corresponding to cell locations
                    waypts_e[count, 0] = self.row_centers[row]
                    waypts_e[count, 1] = self.col_centers[column]
                    grid_idcs[count, 0] = row
                    grid_idcs[count, 1] = column
                    count += 1
            else:
                break
            column_prev = column
            row_prev = row

        # Last cell location where demonstration terminates is the goal state:
        if not (np.isnan(column_prev) or np.isnan(column_prev)):
            svf_e[1, row_prev.astype(int), column_prev.astype(int)] = 1

        return svf_e, waypts_e, grid_idcs


@OBJECT_REGISTRY.register
class PlanAgentsGridGenerate(PlanEgoGridGenerate):
    """Function to get the agents obstacles' state visitation frequencies."""

    def __init__(
        self,
        interpolate_freq: int = 10,
        grid_dim: int = 65,
        horizon: int = 50,
        grid_extent: Tuple[int, int, int, int] = (-25, 25, -10, 40),
    ) -> None:

        kwargs = {
            "interpolate_freq": interpolate_freq,
            "grid_dim": grid_dim,
            "horizon": horizon,
            "grid_extent": grid_extent,
        }
        super(PlanAgentsGridGenerate, self).__init__(**kwargs)

    def __call__(self, sample: Dict) -> Dict:
        """Callable function.

        Generate future and history agents consictent grid occupancy,
        raster svf and actions grid steps.

        Args:
            sample (dict): input original sample.

        Returns:
            sample (dict): the sample with added grid features.
        """

        agent_traj = sample["plan_agents_traj_vcs"]
        agent_his = sample["plan_agents_his_traj_vcs"]
        agent_state = sample["plan_agents_state_vector"]

        assert len(agent_traj) == len(agent_his)

        motion_feats_agents = self.agents_motion_generate(
            agent_his, agent_state
        )

        waypts_e_agents = []
        grid_idcs_agents = []
        svf_agents_plus = np.zeros((2, self.grid_dim, self.grid_dim))
        svf_agents_his_plus = np.zeros((2, self.grid_dim, self.grid_dim))

        for ind in range(len(agent_traj)):

            fut_inter_agent = self.traj_interpolated(
                agent_traj[ind], agent_his[ind][-1, :]
            )

            his_inter_agent = self.traj_interpolated(
                agent_his[ind][1:, :], agent_his[ind][0, :]
            )

            svf_e_agents, waypts_e_a, grid_idcs_a = self.traj_to_grid_path(
                fut_inter_agent
            )
            svf_his_agents, _, _ = self.traj_to_grid_path(
                his_inter_agent, gamma=0.9
            )

            waypts_e_agents.append(waypts_e_a)
            grid_idcs_agents.append(grid_idcs_a)
            svf_agents_plus[svf_e_agents > 0] = 1
            svf_agents_his_plus = svf_agents_his_plus + svf_his_agents

        sample["plan_agents_svf"] = svf_agents_plus
        sample["plan_agents_state_vector"] = motion_feats_agents
        sample["plan_agents_svf_hist"] = svf_agents_his_plus
        sample["plan_agents_waypts"] = waypts_e_agents
        sample["plan_agents_grid_idcs"] = grid_idcs_agents

        return sample

    def agents_motion_generate(self, agent_his, agent_state):
        """Generate agents motion history feauture.

        Args:
            agent_his : agents history states
            agent_state : agents current state

        Returns:
            motion_feats_agents: agents grid motion features
        """
        motion_feats_agents = np.zeros((4, self.grid_dim, self.grid_dim))
        for ind in range(len(agent_his)):
            states = agent_state[ind]
            current_state = agent_his[ind][-1, :]

            column_start = np.argmin(
                np.absolute(current_state[1] - self.col_centers)
            )
            row_start = np.argmin(
                np.absolute(current_state[0] - self.row_centers)
            )
            motion_feats_agents[
                0, row_start.astype(int), column_start.astype(int)
            ] = states[0]
            motion_feats_agents[
                1, row_start.astype(int), column_start.astype(int)
            ] = states[1]
            motion_feats_agents[
                2, row_start.astype(int), column_start.astype(int)
            ] = states[2]
            motion_feats_agents[
                3, row_start.astype(int), column_start.astype(int)
            ] = states[3]

        return motion_feats_agents


@OBJECT_REGISTRY.register
class PlanGridAction:
    """Generate expert's action sequence which are grid direction \
    movements. the action distribution is groud truth for behavior \
    clone method.

    Args:
        grid_dim: max grid dim of BEV.
        horizon: max mdp grid solve steps.
        action_num: max number of actions.
    """

    def __init__(
        self, grid_dim: int = 65, horizon: int = 50, action_num: int = 9
    ) -> None:

        self.grid_dim = grid_dim
        self.horizon = horizon
        self.action_num = action_num
        if self.action_num == 9:
            # Actions: [D, R, U, L, DR, UR, DL, UL, end]
            self.actions_list = [
                (1, 0, 0),
                (0, 1, 0),
                (-1, 0, 0),
                (0, -1, 0),
                (1, 1, 0),
                (-1, 1, 0),
                (1, -1, 0),
                (-1, -1, 0),
                (0, 0, 1),
            ]
        elif self.action_num == 5:
            # Actions: [D, R, U, L, end]
            self.actions_list = [
                (1, 0, 0),
                (0, 1, 0),
                (-1, 0, 0),
                (0, -1, 0),
                (0, 0, 1),
            ]

    def __call__(self, sample: Dict) -> Dict:
        """Callable function.

        Transform the grid waypoints index into action movements.
        Agents action will be generated if 'plan_agents_grid_idcs'
        is already in input sample' key.

        Args:
            sample (dict): input original sample.

        Returns:
            sample (dict): the sample with action targets.
        """
        waypts_e = sample["plan_waypts_expert"]
        grid_idcs = sample["plan_grid_idcs"]

        # get action series
        bc_targets = self._calc_bc_target(waypts_e, grid_idcs)
        sample["plan_bc_targets"] = bc_targets

        if hasattr(sample, "plan_agents_grid_idcs"):
            waypts_agents = sample["plan_agents_waypts"]
            grid_idcs_agents = sample["plan_agents_grid_idcs"]

            bc_targets_agents = np.zeros(
                (self.horizon, self.action_num, self.grid_dim, self.grid_dim)
            )
            for ind in range(len(waypts_agents)):
                waypts_e = waypts_agents[ind]
                grid_idcs = grid_idcs_agents[ind]
                tar_agent = self._calc_bc_target(waypts_e, grid_idcs)

                bc_targets_agents[tar_agent == 1] = 1

            sample["plan_agents_bc_targets"] = bc_targets_agents

        return sample

    def _calc_bc_target(self, waypts_e, grid_idcs):
        """
        Clculate action targets using grid waypoints.

        Args:
            waypts_e : waypoints in meters.
            grid_idcs : grid center index.

        Returns:
            bc_targets: action ground truth
        """

        actions = np.zeros((self.horizon, 3))
        bc_targets = np.zeros(
            (self.horizon, self.action_num, self.grid_dim, self.grid_dim)
        )

        # get waypts length
        tmp = np.sum(np.abs(waypts_e), axis=1)
        tmp = tmp != 0
        tmp[0] = True
        waypt_len = np.sum(tmp, axis=0)

        actions[:-1, :2] = grid_idcs[1:, :] - grid_idcs[:-1, :]
        actions[waypt_len - 1] = np.array([0, 0, 1])

        actions_labels = np.zeros(self.horizon)
        actions_labels[:] = -1

        # get action sequence
        for i in range(waypt_len):
            for j in range(self.action_num):
                if (actions[i] == self.actions_list[j]).all():
                    actions_labels[i] = j

        for i in range(waypt_len):
            bc_targets[
                i,
                int(actions_labels[i]),
                int(grid_idcs[i, 0]),
                int(grid_idcs[i, 1]),
            ] = 1.0

        return bc_targets
