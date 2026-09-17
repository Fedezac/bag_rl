"""Constraint costs: a separate channel from the reward.

A constraint is not a reward term with a big negative weight.

A cost channel instead carries its own signal to its own critic, and the
policy is bounded by ``E[cost] <= limit`` through a Lagrange multiplier that
*rises until the constraint holds*. There is no weight to tune and no price at
which the constraint can be bought.

Costs are non-negative by convention, and per-step. A ``cost_limit`` is
therefore a per-step rate: 0.02 means "violate on at most 2% of steps".
"""

from src.env.constraints.base import ConstraintTerm
from src.env.constraints.terms import (
    CONSTRAINT_TERMS,
    ActionMagnitudeCost,
    FlightPhaseCost,
    HeightCost,
    JointVelocityCost,
    NonFootContactCost,
    OverSpeedCost,
    TiltCost,
)
from src.env.constraints.transform import CostTransform, build_cost_transform

__all__ = [
    "CONSTRAINT_TERMS",
    "ActionMagnitudeCost",
    "ConstraintTerm",
    "CostTransform",
    "FlightPhaseCost",
    "HeightCost",
    "JointVelocityCost",
    "NonFootContactCost",
    "OverSpeedCost",
    "TiltCost",
    "build_cost_transform",
]
