"""The transform that sums constraint terms into a ``cost`` key."""

import torch
from torchrl.data import Unbounded
from torchrl.envs import Transform

from src.env.constraints.terms import CONSTRAINT_TERMS
from src.env.layouts import get_layout


class CostTransform(Transform):
    """Writes a per-step ``cost`` alongside the reward.

    ``cost`` MUST be declared in the reward spec. The multiprocessed collector
    preallocates its buffers from the specs, so an undeclared extra key is
    silently dropped on the way back to the trainer -- no error, no warning,
    just a missing key and a cost critic quietly training on zeros.
    """

    def __init__(self, env_name, terms):
        super().__init__(in_keys=[], out_keys=[])
        self.layout = get_layout(env_name)
        self.terms = list(terms)

    def _step(self, tensordict, next_tensordict):
        obs = next_tensordict["observation"]
        # The action that produced this transition lives on the *input*
        # tensordict; the output only carries where it landed.
        action = tensordict.get("action", None)
        total = torch.zeros_like(obs[..., 0])
        for term in self.terms:
            total = total + term.cost(self.layout, obs, action)
        # Match the reward's dtype, not the observation's
        reward = next_tensordict["reward"]
        next_tensordict["cost"] = total.unsqueeze(-1).to(reward.dtype)
        return next_tensordict

    def transform_reward_spec(self, reward_spec):
        reward_spec["cost"] = Unbounded(
            shape=reward_spec["reward"].shape,
            dtype=reward_spec["reward"].dtype,
            device=reward_spec.device,
        )
        return reward_spec


def build_cost_transform(env_name, spec):
    """Build a :class:`CostTransform` from a comma-separated term list.

    ``spec`` is e.g. ``"tilt,height"``, a sequence of names, or ``None``.
    """
    if not spec:
        return None
    names = spec.split(",") if isinstance(spec, str) else list(spec)
    terms = []
    for raw in names:
        name = raw.strip()
        if name not in CONSTRAINT_TERMS:
            raise KeyError(
                f"unknown constraint {name!r}. Known: {sorted(CONSTRAINT_TERMS)}"
            )
        terms.append(CONSTRAINT_TERMS[name]())
    return CostTransform(env_name, terms)
