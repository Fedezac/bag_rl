"""Posture shaping."""

from src.env.rewards.base import RewardShapingBase


class HumanoidUprightReward(RewardShapingBase):
    """Torso-height shaping for Humanoid. ``obs[0]`` is qpos[2] (torso z)."""

    def __init__(self, upright_weight=0.5, target_height=1.4):
        super().__init__()
        self.upright_weight = upright_weight
        self.target_height = target_height

    def shaping(self, tensordict, next_tensordict):
        height = next_tensordict["observation"][..., 0]
        return -self.upright_weight * (height - self.target_height).abs()
