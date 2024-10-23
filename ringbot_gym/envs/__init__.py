from .base.legged_robot import LeggedRobot
from .ringbot.ringbot import Ringbot
from .ringbot.ringbot_config import RingbotCfg, RingbotCfgPPO

import os
from ringbot_gym.utils.task_registry import task_registry

task_registry.register(
    "ringbot", Ringbot, RingbotCfg(), RingbotCfgPPO()
)