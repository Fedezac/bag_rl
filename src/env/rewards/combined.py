"""Twist tracking combined with gait quality."""

from src.env.rewards.base import CompositeReward, RewardShapingBase
from src.env.rewards.gait import GaitReward
from src.env.rewards.twist import TwistTrackingReward


class TrackingGatedGait(RewardShapingBase):
    """Twist tracking, with gait quality as a bonus GATED on the tracking"""

    replaces_task_reward = True

    def __init__(self, env_name="Ant-v5", w_gait=0.5, **twist_kwargs):
        super().__init__()
        self.twist = TwistTrackingReward(env_name=env_name, **twist_kwargs)
        # Speed / lateral / yaw are the twist term's job; leaving them on would
        # be scoring the same quantity twice with two different kernels.
        self.gait = GaitReward(env_name=env_name, w_speed=0.0, w_lateral=0.0, w_yaw=0.0)
        self.w_gait = w_gait
        # Normalises the gate to [0, 1] so w_gait keeps its meaning: the value
        # of a perfect gait relative to perfect tracking.
        self.track_max = self.twist.track_max

    def shaping(self, tensordict, next_tensordict):
        track = self.twist.shaping(tensordict, next_tensordict)
        gait = self.gait.shaping(tensordict, next_tensordict)
        return track + self.w_gait * (track / self.track_max) * gait

    # The command plumbing is the twist shaper's; the gait term is stateless.
    def after_step(self, next_tensordict):
        return self.twist.after_step(next_tensordict)

    def _reset(self, tensordict, tensordict_reset):
        return self.twist._reset(tensordict, tensordict_reset)

    def transform_observation_spec(self, observation_spec):
        return self.twist.transform_observation_spec(observation_spec)


def gait_twist(env_name="Ant-v5", w_gait=0.5, **twist_kwargs):
    """Track a commanded twist while keeping a clean gait."""
    return TrackingGatedGait(env_name=env_name, w_gait=w_gait, **twist_kwargs)


def gait_twist_sum(env_name="Ant-v5", w_gait=0.5, **twist_kwargs):
    """The additive predecessor of :func:`gait_twist`"""
    return CompositeReward(
        [
            (1.0, TwistTrackingReward(env_name=env_name, **twist_kwargs)),
            (
                w_gait,
                GaitReward(env_name=env_name, w_speed=0.0, w_lateral=0.0, w_yaw=0.0),
            ),
        ]
    )
