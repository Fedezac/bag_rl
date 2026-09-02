"""The dual variable that turns a cost channel into an actual constraint."""

import math

import torch
import torch.nn.functional as F
from torch import nn


def _inverse_softplus(value, floor=-5.0):
    """``nu`` such that ``softplus(nu) == value``; ``floor`` for value <= 0.

    The floor is not arbitrary. Dual ascent moves ``nu`` by roughly the
    learning rate per iteration, so starting too far below zero costs hundreds
    of iterations before the multiplier reaches a magnitude that influences the
    policy at all -- on a few-hundred-iteration run the constraint would look
    like it was being ignored.
    """
    value = float(value)
    if value <= 0.0:
        return floor
    return max(floor, math.log(math.expm1(value)))


class LagrangeMultiplier(nn.Module):
    """Enforces ``E[cost] <= cost_limit`` by dual ascent on a multiplier.

    The multiplier is what makes this a constraint rather than another reward
    term. A fixed penalty weight has a fixed exchange rate -- the policy can
    always buy the violation by earning more reward elsewhere. Here the
    price rises on its own whenever the constraint is violated and keeps
    rising until it is not, so no amount of reward makes violation profitable
    at equilibrium.

    Parameterised through ``softplus`` rather than a raw clamp at zero: the
    multiplier must stay non-negative (a negative one would *reward* violating
    the constraint), and a clamped parameter sitting at the boundary receives
    no gradient and can get stuck there.

    ``cost_limit`` is a per-step rate, matching the per-step costs emitted by
    :class:`~src.env.constraints.CostTransform`. 0.02 reads as "violate on at
    most 2% of steps" and is independent of episode length, which sidesteps
    tracking episode boundaries in a collector running with
    ``split_trajs=False``.
    """

    def __init__(self, cost_limit, lr=0.035, init_value=0.0, max_value=None):
        super().__init__()
        # ``init_value`` is the initial MULTIPLIER, not the raw parameter, so
        # it has to be pushed through the inverse of softplus. Passing it
        # straight in would make init_value=0.0 mean lambda=softplus(0)=0.69 --
        # a run asked to start unconstrained would start with real constraint
        # pressure instead. log(exp(0) - 1) is -inf, so the floor stands in for
        # "effectively zero".
        self.nu = nn.Parameter(torch.tensor(_inverse_softplus(init_value)))
        self.cost_limit = float(cost_limit)
        self.max_value = max_value
        self.optim = torch.optim.Adam([self.nu], lr=lr)

    @property
    def value(self):
        """Current multiplier, always >= 0."""
        lam = F.softplus(self.nu)
        if self.max_value is not None:
            lam = lam.clamp_max(self.max_value)
        return lam

    def update(self, mean_cost):
        """One dual-ascent step on the observed mean per-step cost.

        Minimising ``-lambda * (J_c - d)`` with respect to ``nu`` raises the
        multiplier while the constraint is violated (``J_c > d``) and lets it
        decay back toward zero once it holds.
        """
        loss = -self.value * (float(mean_cost) - self.cost_limit)
        self.optim.zero_grad(set_to_none=True)
        loss.backward()
        self.optim.step()
        return float(self.value.detach())

    def combine(self, reward_advantage, cost_advantage):
        """Fold the cost advantage into the reward advantage.

        ``(A_r - lambda * A_c) / (1 + lambda)`` -- the standard Lagrangian
        surrogate. The division keeps the advantage scale roughly constant as
        the multiplier grows, so the effective learning rate of the policy does
        not silently balloon along with it.
        """
        lam = self.value.detach()
        return (reward_advantage - lam * cost_advantage) / (1.0 + lam)
