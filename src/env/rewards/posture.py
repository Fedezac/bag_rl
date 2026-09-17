"""Posture shaping."""

from src.env.layouts import get_layout
from src.env.rewards.base import RewardShapingBase


class HumanoidUprightReward(RewardShapingBase):
    """Penalise torso height away from a standing target.

    ``target_height`` defaults to the layout's ``nominal_height``, so the term
    is only Humanoid-specific in its registry name.
    """

    name = "humanoid_upright"

    def __init__(self, env_name, upright_weight=0.5, target_height=None):
        super().__init__()
        self.layout = get_layout(env_name)
        self.upright_weight = upright_weight
        self.target_height = (
            self.layout.nominal_height if target_height is None else target_height
        )

    def shaping(self, tensordict, next_tensordict):
        height = self.layout.torso_height(next_tensordict["observation"])
        return -self.upright_weight * (height - self.target_height).abs()
