"""Twist tracking combined with gait quality."""

from src.env.rewards.base import CompositeReward, RewardShapingBase
from src.env.rewards.gait import GaitReward, PhaseGaitReward
from src.env.rewards.twist import TwistTrackingReward

#: Default per-term gait weights, as ``(height, trot, stance)``.
GAIT_WEIGHTS = (1.0, 0.5, 0.3)
#: Default ``(walk, trot, blend)`` speeds for the phase clock, m/s.
GAIT_SPEEDS = (0.10, 1.00, 0.30)
#: Default ``(min, max)`` gait frequency, Hz.
GAIT_FREQ = (1.2, 2.6)


def build_gait(env_name, twist, gait_mode, gait_weights, gait_speeds, gait_freq):
    """The gait shaper for a twist-tracking combination.

    Speed / lateral / yaw are the twist term's job; leaving them on would be
    scoring the same quantity twice with two different kernels.

    ``phase`` hands the gait a clock and the live command, so the pattern it
    asks for follows what was commanded. It also widens the observation, which
    the twist shaper has to know about: its own command must stay last.
    """
    height, trot, stance = gait_weights or GAIT_WEIGHTS
    common = dict(
        env_name=env_name,
        w_speed=0.0,
        w_lateral=0.0,
        w_yaw=0.0,
        w_height=height,
        w_trot=trot,
        w_stance=stance,
    )
    if gait_mode == "static":
        return GaitReward(**common)
    if gait_mode != "phase":
        raise ValueError(f"unknown gait mode {gait_mode!r}; expected static or phase")
    walk_speed, trot_speed, blend = gait_speeds or GAIT_SPEEDS
    gait = PhaseGaitReward(
        command_source=twist,
        walk_speed=walk_speed,
        trot_speed=trot_speed,
        gait_blend=blend,
        freq=gait_freq or GAIT_FREQ,
        **common,
    )
    twist.obs_offset = gait.PHASE_DIM
    return gait


class TrackingGatedGait(RewardShapingBase):
    """Twist tracking, with gait quality as a bonus GATED on the tracking"""

    replaces_task_reward = True

    def __init__(
        self,
        env_name,
        w_gait=0.5,
        gait_weights=None,
        gait_mode="static",
        gait_speeds=None,
        gait_freq=None,
        **twist_kwargs,
    ):
        super().__init__()
        self.twist = TwistTrackingReward(env_name=env_name, **twist_kwargs)
        self.gait = build_gait(
            env_name, self.twist, gait_mode, gait_weights, gait_speeds, gait_freq
        )
        self.gait_mode = gait_mode
        self.w_gait = w_gait
        # Normalises the gate to [0, 1] so w_gait keeps its meaning: the value
        # of a perfect gait relative to perfect tracking.
        self.track_max = self.twist.track_max

    def shaping(self, tensordict, next_tensordict):
        track = self.twist.shaping(tensordict, next_tensordict)
        gait = self.gait.shaping(tensordict, next_tensordict)
        return track + self.w_gait * (track / self.track_max) * gait

    # The gait goes first everywhere it appends: the trainer reads the command
    # back off the end of the observation, so it has to stay there.
    def after_step(self, next_tensordict):
        return self.twist.after_step(self.gait.after_step(next_tensordict))

    def _reset(self, tensordict, tensordict_reset):
        tensordict_reset = self.gait._reset(tensordict, tensordict_reset)
        return self.twist._reset(tensordict, tensordict_reset)

    def transform_observation_spec(self, observation_spec):
        observation_spec = self.gait.transform_observation_spec(observation_spec)
        return self.twist.transform_observation_spec(observation_spec)


def gait_twist(
    env_name,
    w_gait=0.5,
    gait_weights=None,
    gait_mode="static",
    gait_speeds=None,
    gait_freq=None,
    **twist_kwargs,
):
    """Track a commanded twist while keeping a clean gait."""
    return TrackingGatedGait(
        env_name=env_name,
        w_gait=w_gait,
        gait_weights=gait_weights,
        gait_mode=gait_mode,
        gait_speeds=gait_speeds,
        gait_freq=gait_freq,
        **twist_kwargs,
    )


def gait_twist_sum(
    env_name,
    w_gait=0.5,
    gait_weights=None,
    gait_mode="static",
    gait_speeds=None,
    gait_freq=None,
    **twist_kwargs,
):
    """Twist plus gait as a plain weighted sum, with no tracking gate."""
    twist = TwistTrackingReward(env_name=env_name, **twist_kwargs)
    gait = build_gait(
        env_name, twist, gait_mode, gait_weights, gait_speeds, gait_freq
    )
    # Gait first: CompositeReward runs its members in order, and the twist
    # command has to be the last thing appended to the observation.
    return CompositeReward([(w_gait, gait), (1.0, twist)])
