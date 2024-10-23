from ringbot_gym.envs.base.legged_robot_config import (
    LeggedRobotCfg,
    LeggedRobotCfgPPO,
)

class RingbotCfg(LeggedRobotCfg):
    class domain_rand:
        randomize_friction = True
        friction_range = [0.5, 1.25]
        randomize_base_mass = False
        added_mass_range = [-1., 1.]
        push_robots = True
        push_interval_s = 15
        max_push_vel_xy = 1.
        delay_ms_range = [0, 10]

class RingbotCfgPPO(LeggedRobotCfgPPO):
    pass
