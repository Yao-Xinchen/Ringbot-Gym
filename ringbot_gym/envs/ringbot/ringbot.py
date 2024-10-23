from ringbot_gym import GYM_ROOT_DIR, envs
from time import time
from warnings import WarningMessage
import numpy as np
import os

from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil

import torch
from torch import Tensor
from typing import Tuple, Dict, Optional, Union, Any

from ringbot_gym.envs.base.legged_robot import LeggedRobot
from ringbot_gym.utils.terrain import Terrain
from ringbot_gym.utils.math import (
    quat_apply_yaw,
    wrap_to_pi,
    torch_rand_sqrt_float,
)
from ringbot_gym.utils.helpers import class_to_dict
from .ringbot_config import RingbotCfg, RingbotCfgPPO


class Ringbot(LeggedRobot):
    def __init__(self, cfg: RingbotCfg, sim_params, physics_engine, sim_device, headless):
        self.cfg = cfg
        self.sim_params = sim_params
        self.height_samples = None
        self.debug_viz = False
        self.init_done = False
        self._parse_cfg(cfg)
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)
        self.pi = torch.tensor(np.pi, device=self.device)

        if not self.headless:
            self.set_camera(self.cfg.viewer.pos, self.cfg.viewer.lookat)
        self._init_buffers()
        self._prepare_reward_function()
        self.init_done = True

    def step(self, actions):
        clip_actions = self.cfg.normalization.clip_actions
        self.actions = torch.clip(actions, -clip_actions, clip_actions).to(self.device)
        self.render()
        # self.pre_physics_step()

        # physics step
        for _ in range(self.cfg.control.decimation):
            self.envs_steps_buf += 1  # TODO: define this
            self.action_ringbuf = torch.cat(
                (self.actions.unsqueeze(1), self.action_ringbuf[:, :-1]),
                dim=1
            )
            self.torques = self._compute_torques(
                self.action_ringbuf[torch.arange(self.num_envs), self.action_delay_idx, :]
            )
            self.gym.set_dof_actuation_force_tensor(
                self.sim, gymtorch.unwrap_tensor(self.torques)
            )
            # TODO: other physics steps

        self.post_physics_step()

        # observation
        clip_obs = self.cfg.normalization.clip_observations
        self.obs_buf = torch.clip(self.obs_buf, -clip_obs, clip_obs)
        if self.privileged_obs_buf is not None:
            self.privileged_obs_buf = torch.clip(self.privileged_obs_buf, -clip_obs, clip_obs)
        return (
            self.obs_buf,
            self.privileged_obs_buf,  # TODO: check this
            self.rew_buf,
            self.reset_buf,
            self.extras,
            self.obs_history,  # TODO: define this
        )

    def _compute_torques(self, actions):
        num_torques = self.cfg.env.num_actions  # TODO: define this
        torques = torch.zeros(
            self.num_envs, num_torques, device=self.device
        )
        # TODO: define this
        return torques

    def post_physics_step(self):
        pass

    def _init_buffers(self):
        super()._init_buffers()
        delay_max = np.int64(
            np.ceil(self.cfg.domain_rand.delay_ms_range[1] / 1000 / self.sim_params.dt)
        )
        self.action_ringbuf = torch.zeros(
            (self.num_envs, delay_max, self.cfg.env.num_actions),
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        self.action_delay_idx = torch.zeros(
            self.num_envs,
            dtype=torch.long,
            device=self.device,
            requires_grad=False,
        )  # TODO: update this
