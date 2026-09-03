#!/usr/bin/env python
"""PPO on a MuJoCo Gym task, parallelised across many env instances."""

import argparse
import sys
from functools import partial
from pathlib import Path

# In order to have this file run both as python main.py and python -m main.py
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402
from torch import multiprocessing  # noqa: E402

from src.env.constraints import CONSTRAINT_TERMS  # noqa: E402
from src.env.rewards import (  # noqa: E402
    REWARD_SHAPERS,
    TwistTrackingReward,
    gait_twist,
    gait_twist_sum,
)
from src.ppo import PPO  # noqa: E402
from src.trainer import Trainer  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--env-name", default="InvertedDoublePendulum-v5")
    p.add_argument("--num-workers", type=int, default=8, help="collector processes")
    p.add_argument(
        "--envs-per-worker",
        type=int,
        default=8,
        help="gym instances batched inside each collector process",
    )
    p.add_argument(
        "--env-batch-mode",
        choices=["parallel", "serial"],
        default="serial",
        help=(
            "how each worker batches its envs. 'serial' steps them in-process "
            "(cheap, best for fast MuJoCo envs); 'parallel' gives each env its "
            "own subprocess (nested under the collector workers)."
        ),
    )
    p.add_argument("--frames-per-batch", type=int, default=8192)
    p.add_argument("--total-frames", type=int, default=204_800)
    p.add_argument("--sub-batch-size", type=int, default=256)
    p.add_argument("--num-epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--num-cells", type=int, default=256)
    p.add_argument("--entropy-eps", type=float, default=1e-4)
    p.add_argument(
        "--shaping",
        choices=sorted(REWARD_SHAPERS),
        default=None,
        help=(
            "reward-shaping term to add. Env-specific: 'ant_gait' expects "
            "Ant-v5, 'humanoid_upright' expects Humanoid-v5. Off by default, "
            "so the true task reward is used."
        ),
    )
    p.add_argument(
        "--twist",
        default="1.0,0,0",
        help=(
            "desired body-frame twist at the CoM as 'vx,vy,wz' (m/s, m/s, "
            "rad/s), used by --shaping twist. Body-fixed: vx is forward along "
            "the robot's own heading, not world +x. Use --twist=-1,0,0 (equals "
            "sign) for a negative component."
        ),
    )
    p.add_argument(
        "--twist-range",
        default=None,
        help=(
            "randomise the twist per episode, as 'vxlo:vxhi,vylo:vyhi,wzlo:wzhi'. "
            "Use an EQUALS sign when any bound is negative, or argparse reads "
            "the value as a flag: --twist-range=-0.5:1.5,-0.5:0.5,-1:1 . "
            "The command is appended to the "
            "observation, so the policy can learn to follow ANY twist in the "
            "range rather than the single one it was trained on. Without this "
            "the command is fixed at --twist."
        ),
    )
    p.add_argument(
        "--gait-weight",
        type=float,
        default=0.5,
        help=(
            "weight of the gait-quality terms in --shaping gait_twist. The "
            "twist tracking term is fixed at 1.0, so this sets how much clean "
            "footfall is worth relative to following the command."
        ),
    )
    p.add_argument(
        "--obs-clip",
        type=float,
        default=10.0,
        help="clamp on normalized observations; guards near-zero-variance channels",
    )
    p.add_argument(
        "--obs-warmup",
        type=int,
        default=2000,
        help="random steps used to seed the observation stats; 0 disables",
    )
    p.add_argument(
        "--value-norm",
        choices=["running", "popart", "none"],
        default="running",
        help=(
            "value-target normalisation. 'running' = exact running mean/var, "
            "'popart' = EMA (better under reward-scale drift), 'none' = train "
            "the critic directly on raw returns."
        ),
    )
    p.add_argument(
        "--eval-every", type=int, default=5, help="iterations between evals; 0 disables"
    )
    p.add_argument(
        "--eval-episodes",
        type=int,
        default=1,
        help=(
            "rollouts averaged per eval."
        ),
    )
    p.add_argument(
        "--render-every",
        type=int,
        default=5,
        help="iterations between video renders; 0 disables",
    )
    p.add_argument("--video-folder", default="./videos")
    p.add_argument(
        "--checkpoint-dir",
        default=None,
        help=(
            "directory for best.pt / final.pt policy checkpoints, written at "
            "each eval. Off by default; without it a finished run leaves no "
            "policy behind to inspect."
        ),
    )
    p.add_argument(
        "--policy-std",
        choices=["dependent", "independent"],
        default="dependent",
        help=(
            "how the policy standard deviation is produced. 'dependent' reads "
            "it off the network output (NormalParamExtractor); 'independent' "
            "learns one free parameter per action dim (PPO/SB3 convention)."
        ),
    )
    p.add_argument(
        "--constraints",
        default=None,
        help=(
            "comma-separated constraint costs, e.g. 'tilt,height'. These go to "
            "a SEPARATE cost channel bounded by a Lagrange multiplier, not "
            "into the reward -- a constraint cannot be bought by earning more "
            "reward elsewhere. Available: " + ", ".join(sorted(CONSTRAINT_TERMS))
        ),
    )
    p.add_argument(
        "--cost-limit",
        type=float,
        default=0.02,
        help=(
            "per-step cost budget. 0.02 reads as 'violate on at most 2%% of "
            "steps'; independent of episode length."
        ),
    )
    p.add_argument(
        "--lagrange-lr",
        type=float,
        default=0.035,
        help="dual ascent rate for the multiplier; higher enforces faster but oscillates",
    )
    p.add_argument(
        "--lagrange-init",
        type=float,
        default=0.01,
        help=(
            "initial multiplier. Non-zero by default: dual ascent moves the "
            "underlying parameter by ~lagrange-lr per iteration, so starting "
            "at exactly 0 leaves the constraint effectively inert for the "
            "first few hundred iterations of a short run."
        ),
    )
    p.add_argument(
        "--lagrange-max",
        type=float,
        default=None,
        help="optional ceiling on the multiplier, to stop it swamping the reward",
    )
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--device", default=None, help="e.g. cuda:0 or cpu")
    p.add_argument("--no-plot", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()

    # Seeded here rather than in the Trainer: network initialisation happens in
    # the PPO constructor, which runs before any trainer exists.
    if args.seed is not None:
        torch.manual_seed(args.seed)

    # A twist shaper needs the robot's layout and the command bound in. Passed
    # as a partial rather than a name so it stays picklable for the collector
    # workers, which re-create their envs in their own processes.
    shaping = args.shaping
    if shaping in ("twist", "gait_twist", "gait_twist_sum"):
        vx, vy, wz = (float(v) for v in args.twist.split(","))
        ranges = None
        if args.twist_range:
            ranges = tuple(
                tuple(float(x) for x in part.split(":"))
                for part in args.twist_range.split(",")
            )
            if len(ranges) != 3 or any(len(r) != 2 for r in ranges):
                raise SystemExit(
                    "--twist-range needs three lo:hi pairs, e.g. "
                    "'-0.5:1.5,-0.5:0.5,-1:1'"
                )
        kw = dict(env_name=args.env_name, vx=vx, vy=vy, wz=wz, command_ranges=ranges)
        if shaping == "gait_twist":
            shaping = partial(gait_twist, w_gait=args.gait_weight, **kw)
        elif shaping == "gait_twist_sum":
            shaping = partial(gait_twist_sum, w_gait=args.gait_weight, **kw)
        else:
            shaping = partial(TwistTrackingReward, **kw)

    algorithm = PPO(
        args.env_name,
        hidden_layers_size=args.num_cells,
        custom_reward_functions=shaping,
        normalized_observation_clip=args.obs_clip,
        observations_warmup_steps=args.obs_warmup,
        value_target_normalizer=args.value_norm,
        policy_std=args.policy_std,
        lr=args.lr,
        num_epochs=args.num_epochs,
        sub_batch_size=args.sub_batch_size,
        entropy_eps=args.entropy_eps,
        constraints=args.constraints,
        cost_limit=args.cost_limit,
        lagrange_lr=args.lagrange_lr,
        lagrange_init=args.lagrange_init,
        lagrange_max=args.lagrange_max,
        device=args.device,
    )
    trainer = Trainer(
        algorithm,
        num_workers=args.num_workers,
        envs_per_worker=args.envs_per_worker,
        env_batch_mode=args.env_batch_mode,
        frames_per_batch=args.frames_per_batch,
        total_frames=args.total_frames,
        eval_every=args.eval_every,
        eval_episodes=args.eval_episodes,
        render_every=args.render_every,
        video_folder=args.video_folder,
        checkpoint_dir=args.checkpoint_dir,
        seed=args.seed,
    )

    try:
        trainer.train()
        print(trainer.result_line())
    finally:
        trainer.close()

    if not args.no_plot:
        trainer.plot()


if __name__ == "__main__":
    # "spawn" is required to hand CUDA tensors to the collector workers, and it
    # is also what makes the env factories safe to ship across processes
    try:
        multiprocessing.set_start_method("spawn")
    except RuntimeError:
        pass
    main()
