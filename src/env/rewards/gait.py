"""Contact-pattern shaping, driven by ``layout.gait_pairs``."""

import math

import torch
from torchrl.data import Unbounded

from src.env.layouts import get_layout
from src.env.rewards.base import RewardShapingBase


class GaitReward(RewardShapingBase):
    """Shapes a legged robot toward a clean, phase-correct gait.

    Every term reads the observation through the robot's
    :class:`ObservationLayout`, so the transform is stateless and robot-
    agnostic: nothing to reset, nothing to special-case for batched envs.

    The two gait terms are the point of the exercise:
      * ``trot``   -- feet within a ``gait_pairs`` group share a phase, and the
        two groups run in antiphase. For a quadruped that is exactly a trot.
      * ``stance`` -- peaks at one group's worth of feet planted, penalising
        both the all-down shuffle and the airborne bound.

    One gait at every speed. :class:`PhaseGaitReward` is the version that
    schedules the pattern on the commanded twist.
    """

    name = "gait"

    #: Height tolerance as a fraction of nominal standing height. Kept as a
    #: ratio, not folded to a constant: the numerator is the absolute tolerance
    #: the term was tuned with, the denominator the height it was tuned at.
    HEIGHT_SIGMA_FRAC = 0.2 / 0.55

    def __init__(
        self,
        env_name,
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

    # -- contact pattern ----------------------------------------------------

    def contacts(self, obs):
        """Boolean foot contacts, or ``None`` where the robot reports none."""
        foot_f = self.layout.foot_forces(obs)
        if foot_f is None:
            return None
        return foot_f > self.contact_threshold

    def stance_term(self, contact, ideal, dtype):
        """Score the number of planted feet against ``ideal``.

        ``ideal`` may be fractional -- the phase gait schedules it between a
        walk's three feet and a trot's two -- so the normaliser is the larger
        of the two distances to the ends of the range.
        """
        n_feet = contact.shape[-1]
        span = max(float(ideal), n_feet - float(ideal))
        n_contact = contact.sum(-1).to(dtype)
        return 1.0 - (n_contact - ideal).abs() / span

    def gait_terms(self, obs):
        """``(phase, stance)``, derived from ``layout.gait_pairs``.

        ``gait_pairs`` groups feet that should share a phase; the groups run in
        antiphase. ((0, 2), (1, 3)) is a quadruped trot, ((0,), (1,)) biped
        alternation, and a hexapod tripod needs no change here.

        ``stance`` peaks at one group's worth of feet planted -- two for a
        quadruped trot, one for a biped -- penalising both the all-down shuffle
        and the all-airborne bound.

        Returns zeros for robots whose observation carries no contact forces
        (the planar walkers), so those fall back to posture terms alone rather
        than indexing into something that is not there.
        """
        L = self.layout
        contact = self.contacts(obs)
        groups = L.gait_pairs
        if contact is None or len(groups) < 2:
            zero = torch.zeros_like(obs[..., 0])
            return zero, zero

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
        stance = self.stance_term(contact, n_feet / len(groups), dtype)
        return phase, stance

    def terms(self, obs):
        """Individual reward terms, kept separate so they can be logged.

        Velocities are WORLD-frame, unlike :class:`TwistTrackingReward`'s: the
        posture terms here are direction-agnostic, so no rotation is needed.
        """
        L = self.layout
        z = L.torso_height(obs)
        vx = obs[..., L.linear_velocity[0]]
        vy = L.lateral_velocity(obs)
        wz = L.yaw_rate(obs)
        up_z = L.upright(obs)

        trot, stance = self.gait_terms(obs)

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
        # Gate on ``upright``: an inverted robot earns none of the posture or
        # gait credit, however good its contact pattern looks.
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


class PhaseGaitReward(GaitReward):
    """A gait clock whose pattern is scheduled on the commanded twist.

    The robot carries a phase that advances at a commanded-speed-dependent
    frequency and is handed to the policy as ``(sin, cos)``. Each foot is
    scored against a target contact window: a duty factor, and an offset into
    the cycle. Both interpolate with the command, from

      * a stand -- every foot planted, the clock irrelevant,
      * through a four-beat lateral-sequence walk, one foot in swing at a time,
      * to a two-beat trot, the diagonal couplets landing together.

    The command comes from ``command_source`` -- the
    :class:`TwistTrackingReward` this shaper is composed with -- and NOT from
    the observation: the command is appended in ``after_step``, after the
    reward has been computed, so at scoring time it is not there yet.

    Scheduling on the *commanded* twist rather than the achieved one is what
    makes the gait an instruction instead of a measurement: a robot cannot earn
    the walk pattern by failing to reach the speed it was asked for.

    The clock also closes the hole a duty-factor target leaves open. "Three
    feet down" is satisfied by parking one foot in the air and shuffling on the
    other three; a foot that never lands disagrees with its target for three
    quarters of every cycle.
    """

    name = "phase_gait"

    #: ``(sin, cos)`` of the phase. Two, rather than the angle itself, so the
    #: wrap from 1 back to 0 is continuous for the network.
    PHASE_DIM = 2

    def __init__(
        self,
        env_name,
        command_source=None,
        walk_speed=0.10,
        trot_speed=1.00,
        gait_blend=0.30,
        yaw_radius=0.45,
        freq=(1.2, 2.6),
        freq_ref=None,
        **kwargs,
    ):
        super().__init__(env_name=env_name, **kwargs)
        self.command_source = command_source
        self.base_obs_dim = self.layout.obs_dim
        # Effective commanded speed, m/s, at which a foot must start leaving
        # the ground and at which the trot is required in full.
        self.walk_speed = walk_speed
        self.trot_speed = trot_speed
        # Width of each transition. Wide enough that the duty target and the
        # offsets slide rather than snap.
        self.gait_blend = gait_blend
        # Turns a commanded yaw rate into the rim speed of a foot, so spinning
        # in place asks for stepping the way translating does. Kyon's feet sit
        # 0.49 m from the stance centre at the home pose.
        self.yaw_radius = yaw_radius
        self.freq_min, self.freq_max = (float(f) for f in freq)
        self.freq_ref = freq_ref
        # Live phase in [0, 1). A plain float, deliberately: EnvCreator
        # snapshots state_dict() from a shadow env and share_memory_()s its
        # tensors across ParallelEnv workers, so a registered buffer here would
        # have every worker step one another's clock.
        self._phase = 0.0

    # -- the schedule -------------------------------------------------------

    @property
    def command(self):
        """The live twist command, or ``None`` when nothing supplies one."""
        return None if self.command_source is None else self.command_source.command

    def effective_speed(self, command=None):
        """How much motion the command asks for, as one speed in m/s.

        The linear command plus what the yaw command demands of the feet. A
        fast spin in place is as much a gait as translation is, and scoring it
        as a stand would ask the robot to pivot with four feet planted.

        ``command`` overrides the live one, for scoring a finished rollout
        against the command it was actually driven with.
        """
        cmd = self.command if command is None else command
        if cmd is None:
            return float(self.trot_speed + self.gait_blend)  # no command: trot
        cmd = cmd.detach().flatten()
        linear = float(cmd[:2].norm())
        return linear + self.yaw_radius * abs(float(cmd[2]))

    def _ramp(self, speed, onset):
        """0 below ``onset``, 1 a blend width above it."""
        if self.gait_blend <= 0.0:
            return 1.0 if speed >= onset else 0.0
        return min(max((speed - onset) / self.gait_blend, 0.0), 1.0)

    def duty(self, speed=None):
        """Fraction of the cycle a foot should spend on the ground.

        1 standing, ``1 - 1/n_feet`` walking (one foot in swing at a time),
        ``1/len(gait_pairs)`` trotting. All three come from the layout, so a
        biped or a hexapod gets its own numbers rather than a quadruped's.
        """
        speed = self.effective_speed() if speed is None else speed
        n_feet = len(self.layout.foot_rows)
        groups = self.layout.gait_pairs
        stand, walk = 1.0, 1.0 - 1.0 / n_feet
        trot = 1.0 / len(groups) if groups else walk
        return (
            stand
            - (stand - walk) * self._ramp(speed, self.walk_speed)
            - (walk - trot) * self._ramp(speed, self.trot_speed)
        )

    def offsets(self, speed=None):
        """Where in the cycle each foot lands, as a fraction, per ``foot_rows``.

        A group of ``gait_pairs`` shares a base offset, evenly spaced around
        the cycle -- the two diagonals of a quadruped half a cycle apart. What
        separates the gaits is the lag between a fore foot and the hind foot
        diagonal to it: a quarter cycle in a lateral-sequence walk, which makes
        the four footfalls even, and nothing at all in a trot, which collapses
        each couplet onto one beat.
        """
        L = self.layout
        n_feet = len(L.foot_rows)
        groups = L.gait_pairs or ((i,) for i in range(n_feet))
        lag = (1.0 / n_feet) * (1.0 - self._ramp(
            self.effective_speed() if speed is None else speed, self.trot_speed
        ))
        out = [0.0] * n_feet
        groups = list(groups)
        for g, members in enumerate(groups):
            base = g / len(groups)
            for foot in members:
                out[foot] = (base + (0.0 if foot in L.fore_feet else lag)) % 1.0
        return out

    def frequency(self, speed=None):
        """Cycles per second, rising with the command."""
        speed = self.effective_speed() if speed is None else speed
        ref = self.freq_ref
        if ref is None:
            ref = (
                float(self.command_source.command_span[0])
                if self.command_source is not None
                else 1.0
            )
        reach = min(max(speed / max(ref, 1e-6), 0.0), 1.0)
        return self.freq_min + (self.freq_max - self.freq_min) * reach

    def target_contact_at(self, phase, speed=None):
        """``(..., n_feet)`` booleans for arbitrary phases.

        Takes the phase rather than reading the clock, so a whole rollout can
        be scored after the fact -- the phase is in the observation, which is
        what the eval diagnostics read it back out of.
        """
        speed = self.effective_speed() if speed is None else speed
        duty = self.duty(speed)
        offsets = torch.as_tensor(
            self.offsets(speed), dtype=phase.dtype, device=phase.device
        )
        return ((phase.unsqueeze(-1) - offsets) % 1.0) < duty

    def target_contact(self, obs, speed=None):
        """``(n_feet,)`` booleans: which feet the clock wants on the ground."""
        phase = obs.new_tensor(self._phase)
        return self.target_contact_at(phase, speed)

    # -- terms --------------------------------------------------------------

    def gait_terms(self, obs):
        """``(phase match, stance)``.

        ``phase match`` replaces the base class's trot criterion: the fraction
        of feet whose contact agrees with the clock. Binary agreement, like the
        sync and antiphase terms it stands in for.
        """
        contact = self.contacts(obs)
        if contact is None:
            zero = torch.zeros_like(obs[..., 0])
            return zero, zero
        speed = self.effective_speed()
        target = self.target_contact(obs, speed)
        match = (contact == target).to(obs.dtype).mean(-1)
        n_feet = len(self.layout.foot_rows)
        stance = self.stance_term(contact, n_feet * self.duty(speed), obs.dtype)
        return match, stance

    # -- the clock ----------------------------------------------------------

    def _phase_features(self, obs):
        angle = 2.0 * math.pi * self._phase
        return obs.new_tensor([math.sin(angle), math.cos(angle)])

    def _append_phase(self, tensordict):
        obs = tensordict["observation"]
        if obs.shape[-1] != self.base_obs_dim:
            return tensordict
        phase = self._phase_features(obs).expand(*obs.shape[:-1], self.PHASE_DIM)
        tensordict["observation"] = torch.cat([obs, phase], dim=-1)
        return tensordict

    def advance(self):
        self._phase = (self._phase + self.frequency() * self.layout.control_dt) % 1.0

    def _reset(self, tensordict, tensordict_reset):
        # Random, not zero: from a fixed start the policy can count steps
        # instead of reading the clock, and the two are indistinguishable until
        # the command changes mid-episode.
        self._phase = float(torch.rand(1))
        return self._append_phase(tensordict_reset)

    def after_step(self, next_tensordict):
        # Score first, against the phase the policy was shown, then move the
        # clock on and hand the new one to the next action.
        self.advance()
        return self._append_phase(next_tensordict)

    def transform_observation_spec(self, observation_spec):
        spec = observation_spec["observation"]
        observation_spec["observation"] = Unbounded(
            shape=(*spec.shape[:-1], spec.shape[-1] + self.PHASE_DIM),
            dtype=spec.dtype,
            device=spec.device,
        )
        return observation_spec


#: Alias kept so recorded configs naming ``ant_gait`` still resolve. The class
#: itself is robot-agnostic.
AntGaitReward = GaitReward
