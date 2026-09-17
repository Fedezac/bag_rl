"""Body-frame twist-command tracking."""

import torch
from torchrl.data import Unbounded

from src.env.layouts import get_layout
from src.env.rewards.base import RewardShapingBase


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
        # three are. A scalar applies to all three; a triple sets them per
        # axis, so an axis commanded to zero can stay expensive to violate
        # while the others relax.
        self.idle_weight = (
            (float(idle_weight),) * self.COMMAND_DIM
            if isinstance(idle_weight, (int, float))
            else tuple(float(w) for w in idle_weight)
        )
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
        # A fraction of each side's own extent, so an asymmetric range keeps
        # both signs: at 0.5 on -0.2:0.4 the bands are [-0.2,-0.1], [0.2,0.4].
        # Taken off the span instead, the shorter side vanishes entirely.
        dead_lo = deadzone * abs(min(lo, 0.0))
        dead_hi = deadzone * max(hi, 0.0)
        # The parts of [lo, hi] left once the dead zone is removed.
        bands = [
            (a, b)
            for a, b in ((lo, min(hi, -dead_lo)), (max(lo, dead_hi), hi))
            if b > a
        ]
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
        (None, None, None),  # stop
        ((0.90, "hi"), None, None),  # forward, fast
        ((0.35, "hi"), None, None),  # forward, slow
        ((0.90, "lo"), None, None),  # reverse
        ((0.75, "hi"), None, (0.75, "hi")),  # turn left under way
        ((0.75, "hi"), None, (0.75, "lo")),  # turn right under way
        (None, None, (0.90, "hi")),  # spin left in place
        (None, None, (0.90, "lo")),  # spin right in place
        (None, (0.90, "hi"), (0.25, "hi")),  # crab left, yaw left free
        # C2 maps (vx, vy, wz) to (-vx, -vy, wz), so the twin of crab left
        # keeps the yaw sign. Flipping it too asks for a different manoeuvre.
        (None, (0.90, "lo"), (0.25, "hi")),  # crab right
        ((0.60, "hi"), (0.60, "hi"), None),  # diagonal
        ((0.50, "hi"), None, (0.40, "hi")),  # gentle arc
    )

    def command_set(self, n):
        """``n`` evaluation commands, mixed the way the policy is judged.

        One command per axis left 10 of 12 probes at ``vx = 0``, so the score
        hardly moved when forward tracking did. These combine axes -- a turn
        carries forward speed -- and give vx the share the training draw does.
        Probes that collapse onto an earlier one, because an axis has zero
        span, are dropped rather than repeated.

        Past one pass the probes repeat. Extra episodes are there to average
        over reset states -- the crab commands reach two different attractors
        from different starts -- which a random tail would not do.
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
        return torch.stack(
            [torch.tensor(probes[i % len(probes)]) for i in range(n)]
        )

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
        idle = obs.new_tensor(self.idle_weight)
        command = self.command.to(obs.device, obs.dtype)
        span = self.command_span.to(obs.device, obs.dtype)
        demand = (command.abs() / span).clamp(0.0, 1.0)
        # Interpolate rather than add, so an axis at full demand is worth its
        # base weight whatever its idle value. Adding the two coupled them:
        # raising the floor on an axis also raised what it earned when
        # commanded, which tilted the whole task toward that axis.
        return (base * (idle + (1.0 - idle) * demand)).clamp_min(1e-6)

    def after_step(self, next_tensordict):
        # Reward first, from the raw observation, then widen it. Order is not
        # actually load-bearing -- every accessor indexes from the front -- but
        # computing the reward before mutating what it read keeps it obvious.
        return self._append_command(next_tensordict)
