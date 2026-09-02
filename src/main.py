#!/usr/bin/env python
"""PPO on a MuJoCo Gym task, parallelised across many env instances."""

import argparse
import sys
from pathlib import Path

# In order to have this file run both as python main.py and python -m main.py
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402
from torch import multiprocessing  # noqa: E402

from src.env.rewards import REWARD_SHAPERS  # noqa: E402
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
        "--render-every",
        type=int,
        default=5,
        help="iterations between video renders; 0 disables",
    )
    p.add_argument("--video-folder", default="./videos")
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

    algorithm = PPO(
        args.env_name,
        hidden_layers_size=args.num_cells,
        custom_reward_functions=args.shaping,
        normalized_observation_clip=args.obs_clip,
        observations_warmup_steps=args.obs_warmup,
        value_target_normalizer=args.value_norm,
        policy_std=args.policy_std,
        lr=args.lr,
        num_epochs=args.num_epochs,
        sub_batch_size=args.sub_batch_size,
        entropy_eps=args.entropy_eps,
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
        render_every=args.render_every,
        video_folder=args.video_folder,
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
