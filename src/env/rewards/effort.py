"""Torque magnitude and torque-rate penalties."""

from src.env.rewards.base import CompositeReward, RewardShapingBase


class ActionCostReward(RewardShapingBase):
    """Penalises torque magnitude and torque CHANGE between steps.

    Nothing else charges for effort: a shaper with ``replaces_task_reward``
    discards the env reward and its ``ctrl_cost`` with it, so without this term
    torque and chatter are free.

    Both terms are means over joints rather than sums, so a weight keeps its
    meaning on a robot with a different joint count.

    The rate term needs the previous action -- the only state in this file. It
    is cleared on reset, so the first step of an episode is never charged for
    the jump from the last step of the one before. One transform serves one
    env (``make_single_env`` builds a fresh shaper per instance), so a single
    buffer is enough; there are no partial resets to mask.
    """

    def __init__(self, w_torque=0.0, w_action_rate=0.0):
        super().__init__()
        self.w_torque = w_torque
        self.w_action_rate = w_action_rate
        self._prev_action = None

    def shaping(self, tensordict, next_tensordict):
        action = tensordict["action"]
        cost = self.w_torque * action.pow(2).mean(-1)
        if self.w_action_rate:
            prev = self._prev_action
            if prev is not None and prev.shape == action.shape:
                cost = cost + self.w_action_rate * (action - prev).pow(2).mean(-1)
            self._prev_action = action.detach().clone()
        return -cost

    def _reset(self, tensordict, tensordict_reset):
        self._prev_action = None
        return tensordict_reset


def with_action_cost(base, env_name="Ant-v5", w_torque=0.0, w_action_rate=0.0):
    """``base`` plus effort penalties, or ``base`` untouched if both are zero.

    The penalties are added, never gated on tracking the way the gait bonus is:
    scaling them by tracking quality would hand a thrashing policy a discount
    exactly when it is tracking worst.
    """
    shaper = base(env_name=env_name)
    if not (w_torque or w_action_rate):
        return shaper
    return CompositeReward(
        [shaper, ActionCostReward(w_torque=w_torque, w_action_rate=w_action_rate)]
    )
