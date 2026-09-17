"""Transform plumbing shared by every shaper."""

from torchrl.data import Unbounded
from torchrl.envs import Transform


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
