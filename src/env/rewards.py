"""Reward-shaping transforms.

Each shaper is a ``Transform`` that adds terms to the env reward while keeping
the untouched task reward under ``"task_reward"``, so training curves stay
comparable across shaping variants.
"""

import torch
from torchrl.data import Unbounded
from torchrl.envs import Transform

from src.env.layouts import get_layout


class RewardShapingBase(Transform):
    """Adds shaping terms to the env reward, preserving the true task reward.

    ``task_reward`` MUST be declared in the reward spec
    """

    #: When True the shaping term *replaces* the env reward instead of adding
    #: to it. The original is still kept under ``task_reward``, so logging and
    #: cross-run comparison keep working even when the policy never sees it.
    replaces_task_reward = False

    def __init__(self):
        super().__init__(in_keys=[], out_keys=[])

    def shaping(self, tensordict, next_tensordict):
        raise NotImplementedError

    def after_step(self, next_tensordict):
        """Hook for shapers that also modify the observation, run post-reward."""
        return next_tensordict

    def _step(self, tensordict, next_tensordict):
        reward = next_tensordict["reward"]
        next_tensordict["task_reward"] = reward.clone()
        term = self.shaping(tensordict, next_tensordict).unsqueeze(-1).to(reward.dtype)
        next_tensordict["reward"] = term if self.replaces_task_reward else reward + term
        return self.after_step(next_tensordict)

    def transform_reward_spec(self, reward_spec):
        reward_spec["task_reward"] = Unbounded(
            shape=reward_spec["reward"].shape, device=reward_spec.device
        )
        return reward_spec


class HumanoidUprightReward(RewardShapingBase):
    """Torso-height shaping for Humanoid. ``obs[0]`` is qpos[2] (torso z)."""

    def __init__(self, upright_weight=0.5, target_height=1.4):
        super().__init__()
        self.upright_weight = upright_weight
        self.target_height = target_height

    def shaping(self, tensordict, next_tensordict):
        height = next_tensordict["observation"][..., 0]
        return -self.upright_weight * (height - self.target_height).abs()


class GaitReward(RewardShapingBase):
    """Shapes a legged robot toward a clean, phase-correct gait.

    Every term is derived from the observation alone, so the transform stays
    stateless -- nothing to reset, nothing to special-case for batched envs.

    Observation layout (Ant-v5, each slice verified against ``mujoco`` directly):
        [0]      torso z
        [1:5]    torso quaternion (w, x, y, z)
        [5:13]   8 joint angles
        [13:16]  torso linear velocity (x, y, z)
        [16:19]  torso angular velocity
        [19:27]  8 joint velocities
        [27:105] cfrc_ext[1:].flatten() -- 13 bodies x 6

    The two gait terms are the point of the exercise:
      * ``trot``   -- diagonal feet should share a phase, and the two diagonal
        pairs should be in antiphase. That is precisely a trot.
      * ``stance`` -- peaks at exactly two feet planted, penalising both the
        four-down shuffle and the zero-down bound.
    """

    #: Height tolerance as a fraction of nominal standing height. 0.2/0.55 is
    #: chosen so Ant reproduces the 0.2 it was originally tuned with.
    HEIGHT_SIGMA_FRAC = 0.2 / 0.55

    def __init__(
        self,
        env_name="Ant-v5",
        target_speed=2.0,
        target_height=None,
        contact_threshold=0.3,
        w_speed=1.0,
        w_height=1.0,
        w_trot=0.5,
        w_stance=0.3,
        w_lateral=0.2,
        w_yaw=0.1,
        speed_sigma=0.5,
        height_sigma=None,
    ):
        super().__init__()
        self.layout = get_layout(env_name)
        self.target_speed = target_speed
        self.target_height = (
            self.layout.nominal_height if target_height is None else target_height
        )
        self.height_sigma = (
            self.HEIGHT_SIGMA_FRAC * self.layout.nominal_height
            if height_sigma is None
            else height_sigma
        )
        self.contact_threshold = contact_threshold
        self.w_speed = w_speed
        self.w_height = w_height
        self.w_trot = w_trot
        self.w_stance = w_stance
        self.w_lateral = w_lateral
        self.w_yaw = w_yaw
        self.speed_sigma = speed_sigma

    def _gait_terms(self, obs):
        """``(phase, stance)``, derived from ``layout.gait_pairs``.

        A gait is a statement about which feet move together and which move
        opposite. ``gait_pairs`` groups the feet into couplets that should share
        a phase, and the two groups should be in antiphase -- which for the
        quadruped grouping ((0, 2), (1, 3)) is exactly a trot, and for the
        biped grouping ((0,), (1,)) is exactly alternating steps. The same
        arithmetic covers a hexapod tripod without changing here.

        ``stance`` peaks at one group's worth of feet planted -- two for a
        quadruped trot, one for a biped -- penalising both the all-down shuffle
        and the all-airborne bound.

        Returns zeros for robots whose observation carries no contact forces
        (the planar walkers), so those fall back to posture terms alone rather
        than indexing into something that is not there.
        """
        L = self.layout
        foot_f = L.foot_forces(obs)
        groups = L.gait_pairs
        if foot_f is None or len(groups) < 2:
            zero = torch.zeros_like(obs[..., 0])
            return zero, zero

        contact = foot_f > self.contact_threshold
        dtype = obs.dtype

        # Within-group agreement: every foot in a group matches the group's
        # first foot. A single-foot group agrees with itself trivially.
        syncs = []
        for g in groups:
            if len(g) == 1:
                syncs.append(torch.ones_like(obs[..., 0]))
            else:
                ref = contact[..., g[0]]
                syncs.append(
                    torch.stack(
                        [(contact[..., i] == ref).to(dtype) for i in g[1:]], dim=-1
                    ).mean(-1)
                )
        sync = torch.stack(syncs, dim=-1).mean(-1)

        if len(groups) == 2:
            antiphase = (contact[..., groups[0][0]] != contact[..., groups[1][0]]).to(
                dtype
            )
            phase = 0.5 * sync + 0.5 * antiphase
        else:
            # Antiphase is only well defined between two groups; with more, the
            # ordering is a sequence rather than an alternation, so score the
            # within-group agreement alone rather than inventing a criterion.
            phase = sync

        n_feet = len(L.foot_rows)
        ideal = n_feet / len(groups)
        span = max(ideal, n_feet - ideal)
        n_contact = contact.sum(-1).to(dtype)
        stance = 1.0 - (n_contact - ideal).abs() / span
        return phase, stance

    def terms(self, obs):
        """Individual reward terms, kept separate so they can be logged.

        Velocities stay in the WORLD frame here, as they were, so the numbers
        this produces are identical to the runs already recorded against it.
        """
        L = self.layout
        z = L.torso_height(obs)
        vx = obs[..., L.linear_velocity[0]]
        vy = L.lateral_velocity(obs)
        wz = L.yaw_rate(obs)
        up_z = L.upright(obs)

        trot, stance = self._gait_terms(obs)

        return {
            "speed": torch.exp(-(((vx - self.target_speed) / self.speed_sigma) ** 2)),
            "upright": up_z.clamp(0.0, 1.0),
            "height": torch.exp(-(((z - self.target_height) / self.height_sigma) ** 2)),
            "trot": trot,
            "stance": stance,
            "lateral": -vy.abs(),
            "yaw": -wz.abs(),
        }

    def shaping(self, tensordict, next_tensordict):
        t = self.terms(next_tensordict["observation"])
        # ``upright`` to penalize if ant is trotting on its "back"
        earned = (
            self.w_speed * t["speed"]
            + self.w_height * t["height"]
            + self.w_trot * t["trot"]
            + self.w_stance * t["stance"]
        )
        return (
            t["upright"] * earned
            + self.w_lateral * t["lateral"]
            + self.w_yaw * t["yaw"]
        )


#: The gait shaper was quadruped-only when it was written; the name is kept so
#: existing configs and the recorded ant_gait runs still resolve.
AntGaitReward = GaitReward


class TwistTrackingReward(RewardShapingBase):
    """Track a commanded body-frame twist, *replacing* the env task reward.

    The command ``(vx, vy, wz)`` is body-fixed at the centre of mass: vx is
    forward along the robot's own heading, vy sideways, wz the yaw rate about
    its own vertical. See :meth:`ObservationLayout.body_twist` for why the
    observation's world-frame velocities have to be rotated to compare against
    it.

    Both tracking terms are Gaussian kernels on the error, the standard
    formulation for velocity-command locomotion
    ``upright`` gates multiplicatively rather than adding, the same lesson
    :class:`AntGaitReward` produced: as an additive term a flipped robot with a
    good velocity trace scored the same as an upright one, because no single
    term could outweigh the rest. Gated, tracking credit is unearnable while
    inverted.
    """

    replaces_task_reward = True

    COMMAND_DIM = 3

    def __init__(
        self,
        env_name="Ant-v5",
        vx=1.0,
        vy=0.0,
        wz=0.0,
        command_ranges=None,
        lin_sigma=0.25,
        ang_sigma=0.25,
        w_lin=1.0,
        w_ang=0.5,
    ):
        super().__init__()
        self.layout = get_layout(env_name)
        self.base_obs_dim = self.layout.obs_dim
        self.command_ranges = command_ranges
        self.lin_sigma = lin_sigma
        self.ang_sigma = ang_sigma
        self.w_lin = w_lin
        self.w_ang = w_ang
        # Live command, replaced on every reset when ranges are configured.
        self.command = torch.tensor([float(vx), float(vy), float(wz)])

    # -- command plumbing ---------------------------------------------------

    def _resample(self, reference):
        """Draw a new command for the episode, if randomisation is enabled."""
        if self.command_ranges is None:
            return
        lo = torch.tensor([r[0] for r in self.command_ranges])
        hi = torch.tensor([r[1] for r in self.command_ranges])
        self.command = lo + (hi - lo) * torch.rand(self.COMMAND_DIM)

    def _append_command(self, tensordict):
        """Concatenate the command onto the observation."""
        obs = tensordict["observation"]
        if obs.shape[-1] != self.base_obs_dim:
            return tensordict
        cmd = self.command.to(obs.device, obs.dtype).expand(
            *obs.shape[:-1], self.COMMAND_DIM
        )
        tensordict["observation"] = torch.cat([obs, cmd], dim=-1)
        return tensordict

    def _reset(self, tensordict, tensordict_reset):
        self._resample(tensordict_reset)
        return self._append_command(tensordict_reset)

    def transform_observation_spec(self, observation_spec):
        spec = observation_spec["observation"]
        observation_spec["observation"] = Unbounded(
            shape=(*spec.shape[:-1], spec.shape[-1] + self.COMMAND_DIM),
            dtype=spec.dtype,
            device=spec.device,
        )
        return observation_spec

    # -- reward -------------------------------------------------------------

    def terms(self, obs):
        vx, vy, wz = self.layout.body_twist(obs)
        cmd = self.command.to(obs.device, obs.dtype)
        cx, cy, cw = cmd[0], cmd[1], cmd[2]
        lin_err2 = (vx - cx) ** 2 + (vy - cy) ** 2
        ang_err2 = (wz - cw) ** 2
        return {
            "lin": torch.exp(-lin_err2 / self.lin_sigma),
            "ang": torch.exp(-ang_err2 / self.ang_sigma),
            "upright": self.layout.upright(obs).clamp(0.0, 1.0),
            "vx": vx,
            "vy": vy,
            "wz": wz,
        }

    def shaping(self, tensordict, next_tensordict):
        t = self.terms(next_tensordict["observation"])
        return t["upright"] * (self.w_lin * t["lin"] + self.w_ang * t["ang"])

    def after_step(self, next_tensordict):
        # Reward first, from the raw observation, then widen it. Order is not
        # actually load-bearing -- every accessor indexes from the front -- but
        # computing the reward before mutating what it read keeps it obvious.
        return self._append_command(next_tensordict)


class CompositeReward(RewardShapingBase):
    """Weighted sum of several shapers, applied as ONE transform.

    ``replaces_task_reward`` is true if ANY member replaces: mixing a
    replacing shaper with an additive one and still adding the env reward
    would reintroduce the unbounded term the replacing shaper existed to
    remove.
    """

    def __init__(self, members):
        super().__init__()
        # (weight, shaper) pairs; a bare shaper is weight 1.0.
        self.members = [m if isinstance(m, tuple) else (1.0, m) for m in members]
        self.replaces_task_reward = any(s.replaces_task_reward for _, s in self.members)

    def shaping(self, tensordict, next_tensordict):
        total = None
        for weight, shaper in self.members:
            term = weight * shaper.shaping(tensordict, next_tensordict)
            total = term if total is None else total + term
        return total

    def after_step(self, next_tensordict):
        for _, shaper in self.members:
            next_tensordict = shaper.after_step(next_tensordict)
        return next_tensordict

    def _reset(self, tensordict, tensordict_reset):
        for _, shaper in self.members:
            tensordict_reset = shaper._reset(tensordict, tensordict_reset)
        return tensordict_reset

    def transform_observation_spec(self, observation_spec):
        for _, shaper in self.members:
            observation_spec = shaper.transform_observation_spec(observation_spec)
        return observation_spec


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
        self.track_max = self.twist.w_lin + self.twist.w_ang

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
