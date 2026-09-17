"""Contact-pattern shaping, driven by ``layout.gait_pairs``."""

import torch

from src.env.layouts import get_layout
from src.env.rewards.base import RewardShapingBase


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
