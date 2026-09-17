"""Contact-pattern shaping, driven by ``layout.gait_pairs``."""

import torch

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


#: Alias kept so recorded configs naming ``ant_gait`` still resolve. The class
#: itself is robot-agnostic.
AntGaitReward = GaitReward
