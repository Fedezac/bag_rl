"""Reward-shaping transforms, grouped by concept.

base      the Transform plumbing and the weighted-sum combinator
gait      contact-pattern shaping, driven by layout.gait_pairs
twist     body-frame (vx, vy, wz) command tracking
combined  twist plus gait, gated and additive
effort    torque and torque-rate penalties
posture   posture terms
registry  name -> shaper, and transform-tree search
"""

from src.env.rewards.base import CompositeReward, RewardShapingBase
from src.env.rewards.base import CompositeReward
from src.env.rewards.combined import (
    TrackingGatedGait,
    gait_twist,
    gait_twist_sum,
)
from src.env.rewards.effort import ActionCostReward, with_action_cost
from src.env.rewards.gait import AntGaitReward, GaitReward
from src.env.rewards.posture import HumanoidUprightReward
from src.env.rewards.registry import REWARD_SHAPERS, find_twist_shaper
from src.env.rewards.twist import TwistTrackingReward

__all__ = [
    "REWARD_SHAPERS",
    "ActionCostReward",
    "AntGaitReward",
    "CompositeReward",
    "GaitReward",
    "HumanoidUprightReward",
    "RewardShapingBase",
    "TrackingGatedGait",
    "TwistTrackingReward",
    "find_twist_shaper",
    "gait_twist",
    "gait_twist_sum",
    "with_action_cost",
]
