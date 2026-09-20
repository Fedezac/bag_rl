"""What locomotion costs: torque, torque rate, and dragging the feet."""

import torch

from src.env.layouts import get_layout
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


class FootDragReward(RewardShapingBase):
    """Penalises a SETTLED foot sliding across the ground.

    Two failure modes, one signal. A stance foot that slides wastes the contact
    it has; a swing foot that scuffs forward through the ground does the same
    with the leg it should have lifted. It is also what keeps a walk honest:
    the cheapest way to hold three feet down is to park one in the air and
    shuffle on the others, and a shuffle drags.

    "Settled" is load-bearing, not decoration. A foot is moving at over 1 m/s
    at the instant its contact force crosses the threshold, so charging every
    foot that merely *touches* prices a normal footfall as a drag -- measured
    over the 60M policy, that is 0.46 per step in a trot against 0.15 for the
    same gait with the touchdown step dropped, and 0.10 for a robot standing
    perfectly still. Requiring contact on two consecutive steps drops the
    touchdown transient, and takes standing to 0.001.

    The previous observation is already to hand -- torchrl passes the step's
    input tensordict -- so this needs no state and no reset.

    A penalty rather than a constraint: the constraint machinery sums every
    term into one ``cost`` against one ``--cost-limit``, and it is built for
    violation-only budgets like ``tilt``. Some slip is inherent to walking, so
    a drag budget can never be met, the multiplier ramps without bound, and it
    takes the posture constraints sharing that budget down with it.
    """

    def __init__(self, env_name, w_drag=0.0, contact_threshold=0.3, tolerance=0.05):
        super().__init__()
        self.layout = get_layout(env_name)
        self.w_drag = w_drag
        self.contact_threshold = contact_threshold
        # Below this a planted foot is planted: solver jitter under load, not
        # motion. A robot standing still measures 0.02 m/s.
        self.tolerance = tolerance

    def loaded(self, obs):
        forces = self.layout.foot_forces(obs)
        return None if forces is None else forces > self.contact_threshold

    def shaping(self, tensordict, next_tensordict):
        obs = next_tensordict["observation"]
        velocities = self.layout.foot_velocities(obs)
        now = self.loaded(obs)
        # The observation the action was chosen from. Wider than ``obs`` by
        # whatever the other shapers appended, but every accessor indexes from
        # the front, so the foot rows are in the same place.
        before = self.loaded(tensordict["observation"])
        if velocities is None or now is None or before is None:
            return torch.zeros_like(obs[..., 0])
        settled = (now & before).to(obs.dtype)
        slip = velocities[..., :2].norm(dim=-1)
        return -self.w_drag * (settled * (slip - self.tolerance).clamp_min(0.0)).sum(-1)


def with_action_cost(base, env_name, w_torque=0.0, w_action_rate=0.0, w_drag=0.0):
    """``base`` plus the effort penalties, or ``base`` untouched if all are zero.

    The penalties are added, never gated on tracking the way the gait bonus is:
    scaling them by tracking quality would hand a thrashing policy a discount
    exactly when it is tracking worst.
    """
    shaper = base(env_name=env_name)
    if not (w_torque or w_action_rate or w_drag):
        return shaper
    members = [shaper]
    if w_torque or w_action_rate:
        members.append(
            ActionCostReward(w_torque=w_torque, w_action_rate=w_action_rate)
        )
    if w_drag:
        members.append(FootDragReward(env_name=env_name, w_drag=w_drag))
    return CompositeReward(members)
