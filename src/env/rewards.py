"""Reward-shaping transforms.

Each shaper is a ``Transform`` that adds terms to the env reward while keeping
the untouched task reward under ``"task_reward"``, so training curves stay
comparable across shaping variants.
"""

import torch
from torchrl.data import Unbounded
from torchrl.envs import Transform


class RewardShapingBase(Transform):
    """Adds shaping terms to the env reward, preserving the true task reward.

    ``task_reward`` MUST be declared in the reward spec
    """

    def __init__(self):
        super().__init__(in_keys=[], out_keys=[])

    def shaping(self, tensordict, next_tensordict):
        raise NotImplementedError

    def _step(self, tensordict, next_tensordict):
        reward = next_tensordict["reward"]
        next_tensordict["task_reward"] = reward.clone()
        bonus = self.shaping(tensordict, next_tensordict)
        next_tensordict["reward"] = reward + bonus.unsqueeze(-1).to(reward.dtype)
        return next_tensordict

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


class AntGaitReward(RewardShapingBase):
    """Shapes Ant-v5 toward a clean quadrupedal trot.

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

    FOOT_ROWS = (3, 6, 9, 12)  # cfrc_ext[1:] rows of the four ankle bodies
    N_CFRC_BODIES = 13

    def __init__(
        self,
        target_speed=2.0,
        target_height=0.55,
        contact_threshold=0.3,
        w_speed=1.0,
        w_height=1.0,
        w_trot=0.5,
        w_stance=0.3,
        w_lateral=0.2,
        w_yaw=0.1,
        speed_sigma=0.5,
        height_sigma=0.2,
    ):
        super().__init__()
        self.target_speed = target_speed
        self.target_height = target_height
        self.contact_threshold = contact_threshold
        self.w_speed = w_speed
        self.w_height = w_height
        self.w_trot = w_trot
        self.w_stance = w_stance
        self.w_lateral = w_lateral
        self.w_yaw = w_yaw
        self.speed_sigma = speed_sigma
        self.height_sigma = height_sigma

    def terms(self, obs):
        """Individual reward terms, kept separate so they can be logged."""
        z = obs[..., 0]
        qx, qy = obs[..., 2], obs[..., 3]
        vx, vy = obs[..., 13], obs[..., 14]
        wz = obs[..., 18]

        # Torso z-axis projected onto world up: 1 upright, 0 on its side.
        up_z = 1.0 - 2.0 * (qx * qx + qy * qy)

        cfrc = obs[..., 27:].unflatten(-1, (self.N_CFRC_BODIES, 6))
        foot_f = cfrc[..., self.FOOT_ROWS, :3].norm(dim=-1)
        contact = foot_f > self.contact_threshold

        c0, c1, c2, c3 = (contact[..., i] for i in range(4))
        sync = 0.5 * ((c0 == c2).to(obs.dtype) + (c1 == c3).to(obs.dtype))
        antiphase = (c0 != c1).to(obs.dtype)
        trot = 0.5 * sync + 0.5 * antiphase

        n_contact = contact.sum(-1).to(obs.dtype)
        stance = 1.0 - (n_contact - 2.0).abs() / 2.0

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


# Keyed by name so a term written for one robot never silently lands on
# another:
REWARD_SHAPERS = {
    "ant_gait": AntGaitReward,
    "humanoid_upright": HumanoidUprightReward,
}
