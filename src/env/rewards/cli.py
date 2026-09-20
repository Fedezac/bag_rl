"""Build a shaping spec from parsed CLI arguments.

Kept beside the shapers rather than in ``main``, so the knob names live next to
the signatures they feed.
"""

from functools import partial

from src.env.rewards.combined import gait_twist, gait_twist_sum
from src.env.rewards.effort import with_action_cost
from src.env.rewards.twist import TwistTrackingReward

#: Shapings that take a twist command; the rest resolve by name alone.
TWIST_SHAPINGS = ("twist", "gait_twist", "gait_twist_sum")


def axis_values(spec):
    """``'0.5'`` -> 0.5; ``'0.2,0.5,0.5'`` -> a per-axis triple."""
    parts = [float(v) for v in str(spec).split(",")]
    if len(parts) == 1:
        return parts[0]
    if len(parts) != 3:
        raise SystemExit(f"expected one value or three comma-separated: {spec!r}")
    return tuple(parts)


def parse_ranges(spec):
    """``'-0.5:1.5,-0.5:0.5,-1:1'`` -> three ``(lo, hi)`` pairs, or ``None``."""
    if not spec:
        return None
    ranges = tuple(
        tuple(float(x) for x in part.split(":")) for part in spec.split(",")
    )
    if len(ranges) != 3 or any(len(r) != 2 for r in ranges):
        raise SystemExit(
            "--twist-range needs three lo:hi pairs, e.g. '-0.5:1.5,-0.5:0.5,-1:1'"
        )
    return ranges


def gait_weights(spec):
    """``'1.0,0.5,0.3'`` -> ``(height, trot, stance)``, or ``None`` for defaults."""
    if not spec:
        return None
    parts = tuple(float(v) for v in str(spec).split(","))
    if len(parts) != 3:
        raise SystemExit("--gait-weights needs three values: height,trot,stance")
    return parts


def gait_speeds(spec):
    """``'0.1,1.0,0.3'`` -> ``(walk, trot, blend)`` in m/s, or ``None``."""
    if not spec:
        return None
    parts = tuple(float(v) for v in str(spec).split(","))
    if len(parts) != 3:
        raise SystemExit("--gait-speeds needs three values: walk,trot,blend")
    return parts


def gait_freq(spec):
    """``'1.2,2.6'`` -> ``(min, max)`` in Hz, or ``None`` for defaults."""
    if not spec:
        return None
    parts = tuple(float(v) for v in str(spec).split(","))
    if len(parts) != 2:
        raise SystemExit("--gait-freq needs two values: min,max")
    return parts


def build_shaping(args):
    """The ``custom_reward_functions`` spec for ``args``.

    A twist shaper needs the robot's layout and the command bound in, so it
    comes back as a partial rather than a name: that stays picklable for the
    collector workers, which rebuild their envs in their own processes.
    """
    shaping = args.shaping
    if shaping not in TWIST_SHAPINGS:
        return shaping

    vx, vy, wz = (float(v) for v in args.twist.split(","))
    w_vx, w_vy, w_wz = (float(v) for v in args.twist_weights.split(","))
    kw = dict(
        env_name=args.env_name,
        vx=vx,
        vy=vy,
        wz=wz,
        command_ranges=parse_ranges(args.twist_range),
        command_deadzone=axis_values(args.twist_deadzone),
        command_zero_prob=args.twist_zero_prob,
        command_stop_prob=args.twist_stop_prob,
        command_straight_prob=args.twist_straight_prob,
        command_strafe_prob=args.twist_strafe_prob,
        lat_sigma=args.twist_lat_sigma,
        idle_weight=axis_values(args.twist_idle_weight),
        w_vx=w_vx,
        w_vy=w_vy,
        w_wz=w_wz,
    )

    gait_kw = dict(
        w_gait=args.gait_weight,
        gait_mode=getattr(args, "gait_mode", "static"),
        gait_weights=gait_weights(getattr(args, "gait_weights", None)),
        gait_speeds=gait_speeds(getattr(args, "gait_speeds", None)),
        gait_freq=gait_freq(getattr(args, "gait_freq", None)),
    )

    if shaping == "gait_twist":
        spec = partial(gait_twist, **gait_kw, **kw)
    elif shaping == "gait_twist_sum":
        spec = partial(gait_twist_sum, **gait_kw, **kw)
    else:
        spec = partial(TwistTrackingReward, **kw)

    drag_weight = getattr(args, "drag_weight", 0.0)
    if args.torque_weight or args.action_rate_weight or drag_weight:
        spec = partial(
            with_action_cost,
            spec,
            w_torque=args.torque_weight,
            w_action_rate=args.action_rate_weight,
            w_drag=drag_weight,
        )
    return spec
