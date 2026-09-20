"""Shaper lookup by name, and search within an env's transform tree."""

from src.env.rewards.base import CompositeReward
from src.env.rewards.combined import (
    TrackingGatedGait,
    gait_twist,
    gait_twist_sum,
)
from src.env.rewards.gait import GaitReward, PhaseGaitReward
from src.env.rewards.posture import HumanoidUprightReward
from src.env.rewards.twist import TwistTrackingReward


def _key(shaper):
    """Registry key: a class's explicit ``name``, else the callable's own."""
    return getattr(shaper, "name", None) or shaper.__name__


#: Keyed by name so a term written for one robot never silently lands on
#: another. Derived from the shapers themselves, so a new one is added here and
#: nowhere else.
REWARD_SHAPERS = {
    _key(shaper): shaper
    for shaper in (
        GaitReward,
        PhaseGaitReward,
        HumanoidUprightReward,
        TwistTrackingReward,
        gait_twist,
        gait_twist_sum,
    )
}

#: Retired name, kept so recorded configs still resolve.
REWARD_SHAPERS["ant_gait"] = GaitReward


def find_shaper(transform, kind):
    """The first shaper of type ``kind`` in an env's transform tree, or ``None``.

    Covers the two ways shapers nest -- ``CompositeReward.members`` and the
    twist/gait pair a :class:`TrackingGatedGait` holds -- plus any ``Compose``.
    """
    if isinstance(transform, kind):
        return transform
    if isinstance(transform, TrackingGatedGait):
        for member in (transform.twist, transform.gait):
            if isinstance(member, kind):
                return member
    if isinstance(transform, CompositeReward):
        for _, member in transform.members:
            found = find_shaper(member, kind)
            if found is not None:
                return found
    for child in getattr(transform, "transforms", []):
        found = find_shaper(child, kind)
        if found is not None:
            return found
    return None


def find_twist_shaper(transform):
    """The :class:`TwistTrackingReward` inside an env's transform, or ``None``."""
    return find_shaper(transform, TwistTrackingReward)


def find_gait_shaper(transform):
    """The :class:`GaitReward` inside an env's transform, or ``None``."""
    return find_shaper(transform, GaitReward)
