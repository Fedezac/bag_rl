"""Shaper lookup by name, and search within an env's transform tree."""

from src.env.rewards.base import CompositeReward
from src.env.rewards.combined import (
    TrackingGatedGait,
    gait_twist,
    gait_twist_sum,
)
from src.env.rewards.gait import GaitReward
from src.env.rewards.posture import HumanoidUprightReward
from src.env.rewards.twist import TwistTrackingReward


# Keyed by name so a term written for one robot never silently lands on
# another:
REWARD_SHAPERS = {
    "ant_gait": GaitReward,
    "gait": GaitReward,
    "humanoid_upright": HumanoidUprightReward,
    "twist": TwistTrackingReward,
    "gait_twist": gait_twist,
    "gait_twist_sum": gait_twist_sum,
}


def find_twist_shaper(transform):
    """The :class:`TwistTrackingReward` inside an env's transform, or ``None``."""
    if isinstance(transform, TwistTrackingReward):
        return transform
    if isinstance(transform, TrackingGatedGait):
        return transform.twist
    if isinstance(transform, CompositeReward):
        for _, member in transform.members:
            found = find_twist_shaper(member)
            if found is not None:
                return found
    for child in getattr(transform, "transforms", []):
        found = find_twist_shaper(child)
        if found is not None:
            return found
    return None
