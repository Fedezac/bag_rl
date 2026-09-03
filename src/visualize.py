#!/usr/bin/env python
"""Render a trained twist-tracking policy following a scripted command sequence.

An eval number says the policy scores well; it does not say whether the robot
actually goes where it was told. This drives one continuous episode through a
series of commanded twists -- forward, turn, strafe, reverse -- without
resetting between them, so what you see is the policy switching gaits on
command rather than four separate lucky rollouts stitched together.

The per-step trace is written alongside the video: commanded vs achieved
(vx, vy, wz) is the actual result, and the video is how you check it is not
being achieved by falling over in the right direction.
"""

import argparse
import json
import sys
from functools import partial
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torchrl.envs.utils import (  # noqa: E402
    ExplorationType,
    set_exploration_type,
    step_mdp,
)

from src.env.rewards import (  # noqa: E402
    CompositeReward,
    GaitReward,
    TrackingGatedGait,
    TwistTrackingReward,
    gait_twist,
    gait_twist_sum,
)
from src.env.utils import make_single_env  # noqa: E402
from src.ppo import PPO  # noqa: E402

#: (label, vx, vy, wz). Chosen so each segment is visually distinguishable from
#: the last -- a sequence of near-identical commands proves nothing.
DEFAULT_SCRIPT = [
    ("forward", 1.2, 0.0, 0.0),
    ("turn left", 0.6, 0.0, 0.8),
    ("turn right", 0.6, 0.0, -0.8),
    ("strafe left", 0.0, 0.4, 0.0),
    ("stand still", 0.0, 0.0, 0.0),
    ("reverse", -0.4, 0.0, 0.0),
]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--env-name", default="Ant-v5")
    p.add_argument(
        "--shaping",
        choices=["gait_twist", "gait_twist_sum", "twist"],
        default="gait_twist",
    )
    p.add_argument("--gait-weight", type=float, default=0.5)
    p.add_argument("--num-cells", type=int, default=256)
    p.add_argument(
        "--policy-std", choices=["dependent", "independent"], default="independent"
    )
    p.add_argument("--steps-per-command", type=int, default=200)
    p.add_argument("--out", default="./videos/twist_demo.mp4")
    p.add_argument("--trace-out", default=None, help="per-step trace, as JSON")
    p.add_argument("--fps", type=int, default=50)
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--device", default=None)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def find_twist_shaper(transform):
    """Dig the twist shaper out of whatever the env was built with."""
    if isinstance(transform, TwistTrackingReward):
        return transform
    if isinstance(transform, TrackingGatedGait):
        return transform.twist
    if isinstance(transform, CompositeReward):
        for _, member in transform.members:
            found = find_twist_shaper(member)
            if found is not None:
                return found
    for child in getattr(transform, "transforms", []):
        found = find_twist_shaper(child)
        if found is not None:
            return found
    return None


def find_gait_shaper(transform):
    """The gait member, for its contact threshold. ``None`` if twist-only."""
    if isinstance(transform, GaitReward):
        return transform
    if isinstance(transform, TrackingGatedGait):
        return transform.gait
    if isinstance(transform, CompositeReward):
        for _, member in transform.members:
            found = find_gait_shaper(member)
            if found is not None:
                return found
    for child in getattr(transform, "transforms", []):
        found = find_gait_shaper(child)
        if found is not None:
            return found
    return None


def load_font(size):
    from PIL import ImageFont

    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    ):
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def annotate(frame, label, command, achieved, contacts, font, small):
    """Burn the command, the achieved twist and the footfall state into a frame."""
    from PIL import Image, ImageDraw

    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img, "RGBA")
    draw.rectangle([0, 0, img.width, 112], fill=(16, 18, 24, 205))
    draw.text((12, 6), label.upper(), font=font, fill=(255, 255, 255))
    draw.text((12, 38), "commanded -> achieved", font=small, fill=(150, 156, 168))

    # One axis per LINE. Three side-by-side columns overrun 640px and collide.
    rows = [
        ("vx", command[0], achieved[0], "m/s"),
        ("vy", command[1], achieved[1], "m/s"),
        ("wz", command[2], achieved[2], "rad/s"),
    ]
    for i, (name, cmd, got, unit) in enumerate(rows):
        y = 60 + i * 18
        err = abs(got - cmd)
        # Green / amber / red on the tracking error, so a bad axis is obvious
        # without reading the numbers.
        colour = (
            (120, 230, 150)
            if err < 0.15
            else (240, 205, 110) if err < 0.4 else (240, 130, 130)
        )
        draw.text(
            (12, y), f"{name} {cmd:+.2f} -> {got:+.2f} {unit}", font=small, fill=colour
        )

    if contacts is not None:
        # One dot per foot, filled while that foot carries load. On a trotting
        # Ant the diagonal pairs light up in alternation.
        base_x = img.width - 18 - len(contacts) * 24
        for i, down in enumerate(contacts):
            cx = base_x + i * 24
            draw.ellipse(
                [cx, 40, cx + 16, 56],
                fill=(120, 230, 150) if down else (58, 62, 72),
                outline=(190, 195, 205),
            )
        # Right-anchored: left-anchoring runs the label off the frame.
        draw.text(
            (img.width - 18, 16),
            "foot contact",
            font=small,
            fill=(150, 156, 168),
            anchor="rt",
        )
    return np.asarray(img)


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    builder = {
        "gait_twist": partial(gait_twist, w_gait=args.gait_weight),
        "gait_twist_sum": partial(gait_twist_sum, w_gait=args.gait_weight),
        "twist": TwistTrackingReward,
    }[args.shaping]
    shaping = partial(builder, env_name=args.env_name)

    # Warmup off: the observation statistics come from the checkpoint, and
    # 2000 throwaway random steps would only overwrite what we are about to
    # load.
    algorithm = PPO(
        args.env_name,
        hidden_layers_size=args.num_cells,
        custom_reward_functions=shaping,
        observations_warmup_steps=0,
        policy_std=args.policy_std,
        device=args.device,
    )
    algorithm.load_state_dict(
        torch.load(args.checkpoint, map_location=algorithm.device, weights_only=False)
    )

    env = make_single_env(
        args.env_name,
        algorithm.device,
        custom_reward_functions=shaping,
        render_mode="rgb_array",
        width=args.width,
        height=args.height,
    )
    env.set_seed(args.seed)
    shaper = find_twist_shaper(env.transform)
    if shaper is None:
        raise SystemExit("no twist shaper on the env -- nothing to command")
    # The script drives the command; per-episode randomisation would fight it.
    shaper.command_ranges = None
    layout = shaper.layout
    # Match the threshold the gait reward itself scores contacts with, so the
    # dots in the video agree with the number the policy was trained on.
    gait = find_gait_shaper(env.transform)
    contact_threshold = 0.3 if gait is None else gait.contact_threshold

    import imageio

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(args.out, fps=args.fps)

    font, small = load_font(26), load_font(17)
    trace = []
    resets = 0

    td = env.reset()
    with set_exploration_type(ExplorationType.DETERMINISTIC), torch.no_grad():
        for label, vx, vy, wz in DEFAULT_SCRIPT:
            command = (vx, vy, wz)
            shaper.command = torch.tensor([vx, vy, wz], dtype=torch.float32)
            for _ in range(args.steps_per_command):
                td = env.step(algorithm.policy(td))
                obs = td["next", "observation"]
                achieved = tuple(float(v) for v in layout.body_twist(obs))
                forces = layout.foot_forces(obs)
                contacts = (
                    None
                    if forces is None
                    else [bool(f > contact_threshold) for f in forces]
                )
                writer.append_data(
                    annotate(env.render(), label, command, achieved, contacts, font, small)
                )
                row = {
                    "label": label,
                    "cmd": list(command),
                    "got": list(achieved),
                    "height": float(layout.torso_height(obs)),
                    "upright": float(layout.upright(obs)),
                    "contacts": contacts,
                }
                if gait is not None:
                    # The other half of the objective. Tracking numbers alone
                    # cannot tell a trot from a controlled belly-slide at the
                    # right velocity.
                    phase, stance = gait._gait_terms(obs)
                    row["phase"] = float(phase)
                    row["stance"] = float(stance)
                trace.append(row)
                if bool(td["next", "done"].any()):
                    # A fall mid-script is a result, not an error -- record it
                    # and carry on so the remaining commands still get shown.
                    resets += 1
                    td = env.reset()
                else:
                    td = step_mdp(td)

    writer.close()
    env.close()

    if args.trace_out:
        Path(args.trace_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.trace_out).write_text(json.dumps({"resets": resets, "trace": trace}))

    print(f"wrote {args.out} ({len(trace)} frames, {resets} resets)")
    for label, vx, vy, wz in DEFAULT_SCRIPT:
        rows = [t for t in trace if t["label"] == label]
        if not rows:
            continue
        # Skip the first 25% of each segment: the robot needs a moment to
        # transition, and averaging that in measures the switch, not the hold.
        held = rows[len(rows) // 4 :]
        got = np.array([r["got"] for r in held])
        cmd = np.array([vx, vy, wz])
        mean = got.mean(axis=0)
        err = np.abs(got - cmd).mean(axis=0)
        gait_bits = ""
        if "phase" in held[0]:
            gait_bits = (
                f" phase={np.mean([r['phase'] for r in held]):.2f}"
                f" stance={np.mean([r['stance'] for r in held]):.2f}"
            )
        print(
            f"  {label:<12} cmd=({vx:+.2f},{vy:+.2f},{wz:+.2f}) "
            f"got=({mean[0]:+.2f},{mean[1]:+.2f},{mean[2]:+.2f}) "
            f"mae=({err[0]:.2f},{err[1]:.2f},{err[2]:.2f}){gait_bits}"
        )


if __name__ == "__main__":
    main()
