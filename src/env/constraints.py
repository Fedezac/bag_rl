"""Constraint costs: a separate channel from the reward.

A constraint is not a reward term with a big negative weight.

A cost channel instead carries its own signal to its own critic, and the
policy is bounded by ``E[cost] <= limit`` through a Lagrange multiplier that
*rises until the constraint holds*. There is no weight to tune and no price at
which the constraint can be bought.

Costs are non-negative by convention, and per-step. A ``cost_limit`` is
therefore a per-step rate: 0.02 means "violate on at most 2% of steps".
"""

import torch
from torchrl.data import Unbounded
from torchrl.envs import Transform

from src.env.layouts import get_layout


class ConstraintTerm:
    """One constraint. Returns a non-negative per-step cost.

    Plain object rather than a ``Transform``: several terms are summed by a
    single :class:`CostTransform`, so they never bind to an env themselves.
    """

    name = "constraint"

    def cost(self, layout, obs, action):
        raise NotImplementedError


class TiltCost(ConstraintTerm):
    """Torso tilted past ``max_tilt`` from vertical.

    Posture is the canonical hard constraint for a legged robot: a machine
    that has rolled onto its side has failed, however much distance it covered
    getting there.
    """

    name = "tilt"

    def __init__(self, max_tilt=0.4):
        # cos of the maximum allowed tilt angle; 0.4 rad ~= 23 degrees.
        self.min_upright = float(torch.cos(torch.tensor(max_tilt)))

    def cost(self, layout, obs, action):
        return (self.min_upright - layout.upright(obs)).clamp_min(0.0)


class HeightCost(ConstraintTerm):
    """Torso below ``fraction`` of its nominal standing height.

    Scaled by the robot's own nominal height rather than an absolute metre
    value
    """

    name = "height"

    def __init__(self, fraction=0.6):
        self.fraction = fraction

    def cost(self, layout, obs, action):
        if not layout.nominal_height:
            return torch.zeros_like(obs[..., 0])
        floor = self.fraction * layout.nominal_height
        return (floor - layout.torso_height(obs)).clamp_min(0.0) / floor


class NonFootContactCost(ConstraintTerm):
    """Any body other than a foot pushing against the world.

    A knee, elbow or torso registering contact force means the robot is down,
    dragging, or catching itself on something it should not be using. Silently
    inactive on robots whose observation omits ``cfrc_ext`` (the planar
    walkers) -- see :meth:`ObservationLayout.non_foot_forces`.
    """

    name = "non_foot_contact"

    def __init__(self, threshold=1.0):
        self.threshold = threshold

    def cost(self, layout, obs, action):
        forces = layout.non_foot_forces(obs)
        if forces is None:
            return torch.zeros_like(obs[..., 0])
        return (forces.max(dim=-1).values - self.threshold).clamp_min(0.0)


class JointVelocityCost(ConstraintTerm):
    """Any joint spinning faster than ``max_speed``.

    Stands in for actuator limits and for the flailing that PPO finds
    attractive early on
    """

    name = "joint_velocity"

    def __init__(self, max_speed=8.0):
        self.max_speed = max_speed

    def cost(self, layout, obs, action):
        jv = layout.joint_velocities(obs).abs()
        return (jv.max(dim=-1).values - self.max_speed).clamp_min(0.0)


class OverSpeedCost(ConstraintTerm):
    """Forward velocity above ``max_speed``, in m/s of excess.

    One-sided by design. Going slower than target is already handled by the
    shaping reward's speed Gaussian; the constraint exists solely to close off
    the unbounded direction.

    ``max_speed`` should sit slightly above the shaping reward's
    ``target_speed`` so the two are not fighting over the same band -- the reward shapes behaviour
    inside the budget, the constraint caps the top.
    """

    name = "over_speed"

    def __init__(self, max_speed=2.5):
        self.max_speed = max_speed

    def cost(self, layout, obs, action):
        return (layout.forward_velocity(obs) - self.max_speed).clamp_min(0.0)


class FlightPhaseCost(ConstraintTerm):
    """No foot carrying load -- the robot is airborne"""

    name = "flight_phase"

    def __init__(self, threshold=0.3):
        self.threshold = threshold

    def cost(self, layout, obs, action):
        forces = layout.foot_forces(obs)
        if forces is None:
            # Planar walkers carry no cfrc_ext
            return torch.zeros_like(obs[..., 0])
        airborne = forces.max(dim=-1).values <= self.threshold
        return airborne.to(obs.dtype)


class ActionMagnitudeCost(ConstraintTerm):
    """Actuation beyond ``limit`` of the available range.

    A budget on effort. Unlike the usual quadratic control cost folded into
    the reward, this cannot be outbid by moving faster.
    """

    name = "action_magnitude"

    def __init__(self, limit=0.9):
        self.limit = limit

    def cost(self, layout, obs, action):
        if action is None:
            return torch.zeros_like(obs[..., 0])
        return (action.abs().max(dim=-1).values - self.limit).clamp_min(0.0)


CONSTRAINT_TERMS = {
    cls.name: cls
    for cls in (
        TiltCost,
        HeightCost,
        NonFootContactCost,
        JointVelocityCost,
        OverSpeedCost,
        FlightPhaseCost,
        ActionMagnitudeCost,
    )
}


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
