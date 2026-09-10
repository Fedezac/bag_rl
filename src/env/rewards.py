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

    Each axis gets its OWN Gaussian kernel on its own error. Sharing one kernel
    between vx and vy couples their gradients: while vx is badly wrong the
    shared exponential is near zero, and vy is invisible to the optimiser
    regardless of its own error.

    Each axis is scored against what standing still would earn for its own
    command, so a stationary robot scores ~0 on any axis it is asked to move
    and the reward measures progress rather than proximity. Without this,
    per-axis kernels pay free credit whenever a command happens to be near
    zero, and standing becomes a strong local optimum.

    Each axis has its own kernel width, sized to the range it is commanded
    over: ``lin_sigma`` for vx, ``lat_sigma`` for vy, ``ang_sigma`` for wz. A
    width borrowed from a wider axis leaves standing still paying too well on
    the narrow one. Matching the credit a stationary robot earns across axes
    means sigma ~ c**2 / 4.79, for a typical command ``c``.

    ``upright`` gates multiplicatively, so tracking credit is unearnable while
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
        command_deadzone=0.5,
        command_zero_prob=0.1,
        command_stop_prob=0.15,
        command_straight_prob=0.15,
        command_strafe_prob=0.0,
        lin_sigma=0.25,
        lat_sigma=None,
        ang_sigma=0.4,
        w_vx=1.0,
        w_vy=1.0,
        w_wz=1.0,
        idle_weight=1.0,
        rest_eps=0.05,
    ):
        super().__init__()
        self.layout = get_layout(env_name)
        self.base_obs_dim = self.layout.obs_dim
        self.command_ranges = command_ranges
        # Fraction of each axis's extent kept clear of zero when sampling.
        # A scalar applies to all three; a triple sets them per axis.
        self.command_deadzone = (
            (float(command_deadzone),) * self.COMMAND_DIM
            if isinstance(command_deadzone, (int, float))
            else tuple(float(d) for d in command_deadzone)
        )
        self.command_zero_prob = command_zero_prob
        self.command_stop_prob = command_stop_prob
        self.command_straight_prob = command_straight_prob
        self.command_strafe_prob = command_strafe_prob
        self.lin_sigma = lin_sigma
        # vy gets its own width: a kernel is only as sharp as the range it
        # was sized for, and sharing lin_sigma across a 3x narrower axis
        # paid a motionless robot 0.11 there against 0.05 on vx.
        self.lat_sigma = lin_sigma if lat_sigma is None else lat_sigma
        self.ang_sigma = ang_sigma
        # Per-axis weights. Equal by default: all three axes are commanded, so
        # none is a second-class objective.
        self.w_vx = w_vx
        self.w_vy = w_vy
        self.w_wz = w_wz
        # Weight an axis keeps when commanded to zero, as a fraction of its
        # full weight. At 1.0 every axis counts the same whatever it was asked
        # for, which pays a motionless robot for the axes that happen to be
        # zero -- 71% of maximum on a straight-line command, where two of the
        # three are.
        self.idle_weight = idle_weight
        spans = (
            [max(abs(lo), abs(hi)) for lo, hi in command_ranges]
            if command_ranges is not None
            else [1.0] * self.COMMAND_DIM
        )
        self.command_span = torch.tensor(spans).clamp_min(1e-6)
        # Keeps the normalisation finite when the command is zero, where
        # standing IS the goal and there is no improvement to normalise by.
        self.rest_eps = rest_eps
        # Live command, replaced on every reset when ranges are configured.
        self.command = torch.tensor([float(vx), float(vy), float(wz)])
        # Set by use_fixed_commands(); overrides the random draw when present.
        self._fixed_commands = None
        self._fixed_index = 0

    # -- command plumbing ---------------------------------------------------

    def _sample_axis(self, lo, hi, deadzone, generator=None, allow_zero=True):
        """One axis: exactly zero, or outside the dead zone around zero.

        A command drawn just above zero is nearly satisfied by standing still,
        so a range full of them makes standing a strong optimum however the
        kernel is normalised. Excluding the band leaves commands the robot has
        to move to earn. Exact zero is kept at ``command_zero_prob`` because it
        is a real command, and one standing already scores correctly.
        """

        def _u(a, b):
            return a + (b - a) * float(torch.rand(1, generator=generator))

        if (
            allow_zero
            and lo <= 0.0 <= hi
            and float(torch.rand(1, generator=generator)) < self.command_zero_prob
        ):
            return 0.0
        dead = deadzone * max(abs(lo), abs(hi))
        # The parts of [lo, hi] left once the dead zone is removed.
        bands = [(a, b) for a, b in ((lo, min(hi, -dead)), (max(lo, dead), hi)) if b > a]
        if not bands:
            return _u(lo, hi)
        widths = [b - a for a, b in bands]
        pick = float(torch.rand(1, generator=generator)) * sum(widths)
        for (a, b), w in zip(bands, widths):
            if pick <= w:
                return _u(a, b)
            pick -= w
        return _u(*bands[-1])

    def sample_command(self, generator=None):
        """Draw one command from the configured ranges.

        Stops are drawn jointly. Zeroing axes independently makes a full stop
        vanishingly rare -- at a 0.3 per-axis rate only 2.8% of episodes are
        one -- while every single-axis zero still pays a stationary robot, so
        the cost of teaching the robot to stop lands almost entirely on the
        reward floor rather than on the behaviour.
        """
        draw = float(torch.rand(1, generator=generator))
        if draw < self.command_stop_prob:
            return torch.zeros(self.COMMAND_DIM)
        draw -= self.command_stop_prob
        # Reserved episodes, drawn at a fixed rate the way stops are, because
        # independent sampling almost never isolates an axis. Zeroing every
        # other axis can ask for a twist the robot cannot produce: with the
        # heading pinned, Ant reaches 0.07 m/s sideways against a commanded
        # 0.4, so lateral episodes keep yaw non-zero, which is the regime where
        # sideways motion exists. Either sign of yaw works.
        for axis, prob, paired in (
            (0, self.command_straight_prob, ()),
            (1, self.command_strafe_prob, (2,)),
        ):
            if draw < prob:
                command = torch.zeros(self.COMMAND_DIM)
                for i in (axis, *paired):
                    lo, hi = self.command_ranges[i]
                    command[i] = self._sample_axis(
                        lo, hi, self.command_deadzone[i], generator, allow_zero=False
                    )
                return command
            draw -= prob
        return torch.tensor(
            [
                self._sample_axis(lo, hi, dead, generator)
                for (lo, hi), dead in zip(self.command_ranges, self.command_deadzone)
            ]
        )

    # Evaluation probes as (fraction, bound) per axis; "hi"/"lo" picks a sign.
    _EVAL_PROBES = (
        (None, None, None),                  # stop
        ((0.90, "hi"), None, None),          # forward, fast
        ((0.35, "hi"), None, None),          # forward, slow
        ((0.90, "lo"), None, None),          # reverse
        ((0.75, "hi"), None, (0.75, "hi")),  # turn left under way
        ((0.75, "hi"), None, (0.75, "lo")),  # turn right under way
        (None, None, (0.90, "hi")),          # spin left in place
        (None, None, (0.90, "lo")),          # spin right in place
        (None, (0.90, "hi"), (0.25, "hi")),  # crab left, yaw left free
        (None, (0.90, "lo"), (0.25, "lo")),  # crab right
        ((0.60, "hi"), (0.60, "hi"), None),  # diagonal
        ((0.50, "hi"), None, (0.40, "hi")),  # gentle arc
    )

    def command_set(self, n, seed=0):
        """``n`` evaluation commands, mixed the way the policy is judged.

        One command per axis left 10 of 12 probes at ``vx = 0``, so the score
        hardly moved when forward tracking did. These combine axes -- a turn
        carries forward speed -- and give vx the share the training draw does.
        Probes that collapse onto an earlier one, because an axis has zero
        span, are dropped rather than repeated.
        """
        probes = []
        for spec in self._EVAL_PROBES:
            command = [0.0] * self.COMMAND_DIM
            for i, axis in enumerate(spec):
                if axis is None:
                    continue
                frac, bound = axis
                lo, hi = self.command_ranges[i]
                command[i] = frac * (hi if bound == "hi" else lo)
            if command not in probes:
                probes.append(command)
        g = torch.Generator().manual_seed(seed)
        # Anything past the probes is a plain draw, so the set stays
        # representative of what the policy is actually trained on.
        fill = [self.sample_command(g) for _ in range(max(0, n - len(probes)))]
        return torch.stack([torch.tensor(c) for c in probes[:n]] + fill)

    def use_fixed_commands(self, commands):
        """Cycle a fixed list of commands instead of drawing at random.

        Evaluation redrew its commands every time, so the eval number moved
        with the draw as much as with the policy -- a single episode of a
        motionless policy spans 749-4242 on this reward. A fixed set makes
        successive evals comparable, which is what checkpoint selection needs.
        """
        self._fixed_commands = commands
        self._fixed_index = 0

    def rewind_commands(self):
        """Restart the fixed set, so every eval sees the same commands."""
        self._fixed_index = 0

    def _resample(self, reference):
        """Draw a new command for the episode, if randomisation is enabled."""
        if self._fixed_commands is not None:
            n = len(self._fixed_commands)
            self.command = self._fixed_commands[self._fixed_index % n].clone()
            self._fixed_index += 1
            return
        if self.command_ranges is None:
            return
        self.command = self.sample_command()

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

    def _axis(self, achieved, command, sigma):
        """One axis, scored 0 at standstill and 1 on target.

        ``rest`` is the raw kernel's value for a stationary robot under this
        command; subtracting it and rescaling removes the credit that would
        otherwise be earned by not moving. At ``command == 0`` the two collapse
        and ``rest_eps`` leaves the raw kernel, which correctly rewards
        holding still.
        """
        k = torch.exp(-((achieved - command) ** 2) / sigma)
        rest = torch.exp(-(command**2) / sigma)
        return ((k - rest + self.rest_eps) / (1 - rest + self.rest_eps)).clamp(0.0, 1.0)

    def terms(self, obs):
        vx, vy, wz = self.layout.body_twist(obs)
        cmd = self.command.to(obs.device, obs.dtype)
        cx, cy, cw = cmd[0], cmd[1], cmd[2]
        return {
            "vx_track": self._axis(vx, cx, self.lin_sigma),
            "vy_track": self._axis(vy, cy, self.lat_sigma),
            "wz_track": self._axis(wz, cw, self.ang_sigma),
            "upright": self.layout.upright(obs).clamp(0.0, 1.0),
            "vx": vx,
            "vy": vy,
            "wz": wz,
        }

    @property
    def track_max(self):
        """Reward at perfect tracking, for callers that normalise by it."""
        return self.w_vx + self.w_vy + self.w_wz

    def shaping(self, tensordict, next_tensordict):
        obs = next_tensordict["observation"]
        t = self.terms(obs)
        weights = self._axis_weights(obs)
        tracks = torch.stack([t["vx_track"], t["vy_track"], t["wz_track"]], dim=-1)
        base = obs.new_tensor([self.w_vx, self.w_vy, self.w_wz])
        # Renormalised, so perfect tracking is still worth track_max whatever
        # the command asked for.
        scaled = (weights * tracks).sum(-1) / weights.sum(-1) * base.sum()
        return t["upright"] * scaled

    def _axis_weights(self, obs):
        """Per-axis weight, scaled by how much motion the command demands."""
        base = obs.new_tensor([self.w_vx, self.w_vy, self.w_wz])
        if self.idle_weight >= 1.0:
            return base
        command = self.command.to(obs.device, obs.dtype)
        span = self.command_span.to(obs.device, obs.dtype)
        demand = (command.abs() / span).clamp(0.0, 1.0)
        return (base * (self.idle_weight + demand)).clamp_min(1e-6)

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
