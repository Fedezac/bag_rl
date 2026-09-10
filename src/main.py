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


def _axis_values(spec):
    """``'0.5'`` -> 0.5; ``'0.2,0.5,0.5'`` -> a per-axis triple."""
    parts = [float(v) for v in str(spec).split(",")]
    if len(parts) == 1:
        return parts[0]
    if len(parts) != 3:
        raise SystemExit("expected one value or three, as 'vx,vy,wz'")
    return tuple(parts)


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
    p.add_argument(
        "--num-epochs", type=int, default=3, help="passes over each collected batch"
    )
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument(
        "--lr-schedule",
        choices=["linear", "cosine", "constant"],
        default="linear",
        help="decay of the Adam learning rate over the run",
    )
    p.add_argument(
        "--actor-num-cells",
        type=int,
        default=64,
        help=(
            "policy hidden width. Deliberately narrower than the critic: the "
            "policy needs much less capacity than the value function."
        ),
    )
    p.add_argument(
        "--critic-num-cells",
        type=int,
        default=256,
        help="value (and cost) hidden width",
    )
    p.add_argument("--actor-layers", type=int, default=2, help="policy hidden layers")
    p.add_argument(
        "--critic-layers", type=int, default=2, help="value hidden layers"
    )
    p.add_argument(
        "--clip-epsilon", type=float, default=0.25, help="PPO ratio clip range"
    )
    p.add_argument("--gamma", type=float, default=0.99, help="discount")
    p.add_argument("--gae-lambda", type=float, default=0.9, help="GAE lambda")
    p.add_argument(
        "--max-grad-norm", type=float, default=0.5, help="gradient-norm clip"
    )
    p.add_argument(
        "--entropy-eps",
        type=float,
        default=0.0,
        help=(
            "entropy bonus coefficient. Off by default: a large sweep found no "
            "task where it helped. Exploration comes from the policy std."
        ),
    )
    p.add_argument(
        "--policy-init-std",
        type=float,
        default=0.5,
        help="initial action standard deviation (--policy-std independent)",
    )
    p.add_argument(
        "--policy-std-parametrization",
        choices=["softplus", "exp"],
        default="softplus",
        help=(
            "map from the learned parameter to the std. Softplus decays more "
            "slowly than the classic log-std form, which keeps exploration alive."
        ),
    )
    p.add_argument(
        "--policy-final-layer-scale",
        type=float,
        default=0.01,
        help=(
            "multiplier on the final policy layer's weights at init. Starts the "
            "agent near zero mean action so early exploration is set by the std."
        ),
    )
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
        "--twist-deadzone",
        default="0.5",
        help=(
            "fraction of each axis's range kept clear of zero when sampling "
            "commands. A command just above zero is nearly satisfied by "
            "standing still, so a range full of them makes standing optimal. "
            "One value for all axes, or 'vx,vy,wz' to set them separately."
        ),
    )
    p.add_argument(
        "--twist-stop-prob",
        type=float,
        default=0.15,
        help=(
            "probability that a command is a full stop, all three axes zero "
            "at once. Zeroing axes independently almost never produces one, "
            "so stopping goes untrained while the reward floor still rises."
        ),
    )
    p.add_argument(
        "--twist-idle-weight",
        type=float,
        default=1.0,
        help=(
            "weight an axis keeps when commanded to zero, as a fraction of its "
            "full weight. 1.0 counts every axis equally regardless of the "
            "command, which pays a motionless robot for whichever axes are "
            "zero. Lower values make the commanded axes carry the score."
        ),
    )
    p.add_argument(
        "--twist-straight-prob",
        type=float,
        default=0.15,
        help=(
            "probability that a command is straight-line travel: vx non-zero, "
            "every other axis exactly zero. Independent sampling produces this "
            "rarely, so forward motion survives only as part of a turn."
        ),
    )
    p.add_argument(
        "--twist-strafe-prob",
        type=float,
        default=0.0,
        help=(
            "share of episodes commanding vy alone, with vx and wz zero. "
            "Lateral demand otherwise arrives on top of a forward command on "
            "90%% of episodes, where satisfying vx alone costs little."
        ),
    )
    p.add_argument(
        "--twist-lat-sigma",
        type=float,
        default=None,
        help=(
            "tracking kernel width for vy; defaults to the vx width. Size it "
            "to the vy range: sigma = c**2 / 4.79 for a typical command c "
            "matches the credit standing still earns on the other axes."
        ),
    )
    p.add_argument(
        "--twist-zero-prob",
        type=float,
        default=0.1,
        help=(
            "probability that a sampled axis is exactly zero. Standing is a "
            "real command and is scored correctly, so it is kept in the mix."
        ),
    )
    p.add_argument(
        "--twist-weights",
        default="1.0,1.0,1.0",
        help=(
            "per-axis tracking weights as 'vx,vy,wz'. Each axis has its own "
            "Gaussian kernel, so these set how much each is worth relative to "
            "the others."
        ),
    )
    p.add_argument(
        "--gait-weight",
        type=float,
        default=0.5,
        help=(
            "weight of the gait-quality terms in --shaping gait_twist, "
            "relative to perfect tracking. Gait credit is scaled by tracking "
            "quality, so it cannot be earned by standing still."
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
        help=("rollouts averaged per eval."),
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
        default="independent",
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
    p.add_argument(
        "--init-from",
        default=None,
        help=(
            "checkpoint to seed the networks from (best.pt / final.pt). The "
            "optimiser and LR schedule start fresh, so this is fine-tuning, "
            "not resuming. The observation statistics come with the "
            "checkpoint and carry its sample count, so they move slowly if "
            "the new run's observation distribution differs."
        ),
    )
    p.add_argument(
        "--init-reset-std",
        action="store_true",
        help=(
            "put the policy std back to --policy-init-std after loading. A "
            "checkpoint carries the exploration it finished on, which is "
            "usually collapsed; without this there is little left to explore "
            "with. Recommended whenever the objective has changed."
        ),
    )
    p.add_argument(
        "--wandb",
        action="store_true",
        help=(
            "mirror the training metrics to Weights & Biases. Needs "
            "credentials: 'wandb login', or WANDB_API_KEY in the environment."
        ),
    )
    p.add_argument("--wandb-project", default="crawler-rl")
    p.add_argument(
        "--wandb-name", default=None, help="run name; defaults to wandb's own"
    )
    p.add_argument("--wandb-entity", default=None, help="team or user to log under")
    p.add_argument(
        "--wandb-host",
        default=None,
        help=(
            "base URL of a self-hosted W&B server, e.g. http://localhost:8080. "
            "Log in against it once with 'wandb login --host <url>'."
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
        w_vx, w_vy, w_wz = (float(v) for v in args.twist_weights.split(","))
        kw = dict(
            env_name=args.env_name,
            vx=vx,
            vy=vy,
            wz=wz,
            command_ranges=ranges,
            command_deadzone=_axis_values(args.twist_deadzone),
            command_zero_prob=args.twist_zero_prob,
            command_stop_prob=args.twist_stop_prob,
            command_straight_prob=args.twist_straight_prob,
            command_strafe_prob=args.twist_strafe_prob,
            lat_sigma=args.twist_lat_sigma,
            idle_weight=args.twist_idle_weight,
            w_vx=w_vx,
            w_vy=w_vy,
            w_wz=w_wz,
        )
        if shaping == "gait_twist":
            shaping = partial(gait_twist, w_gait=args.gait_weight, **kw)
        elif shaping == "gait_twist_sum":
            shaping = partial(gait_twist_sum, w_gait=args.gait_weight, **kw)
        else:
            shaping = partial(TwistTrackingReward, **kw)

    algorithm = PPO(
        args.env_name,
        actor_hidden_layers=args.actor_layers,
        value_hidden_layers=args.critic_layers,
        actor_hidden_layers_size=args.actor_num_cells,
        critic_hidden_layers_size=args.critic_num_cells,
        custom_reward_functions=shaping,
        normalized_observation_clip=args.obs_clip,
        observations_warmup_steps=args.obs_warmup,
        value_target_normalizer=args.value_norm,
        policy_std=args.policy_std,
        policy_init_std=args.policy_init_std,
        policy_std_parametrization=args.policy_std_parametrization,
        policy_final_layer_scale=args.policy_final_layer_scale,
        lr=args.lr,
        lr_schedule=args.lr_schedule,
        num_epochs=args.num_epochs,
        sub_batch_size=args.sub_batch_size,
        clip_epsilon=args.clip_epsilon,
        gamma=args.gamma,
        lmbda=args.gae_lambda,
        entropy_eps=args.entropy_eps,
        max_grad_norm=args.max_grad_norm,
        constraints=args.constraints,
        cost_limit=args.cost_limit,
        lagrange_lr=args.lagrange_lr,
        lagrange_init=args.lagrange_init,
        lagrange_max=args.lagrange_max,
        device=args.device,
    )
    if args.init_from:
        state = torch.load(
            args.init_from, map_location=algorithm.device, weights_only=False
        )
        loaded, fresh = algorithm.load_pretrained(state)
        note = f"init from {args.init_from}: loaded {', '.join(loaded)}"
        if fresh:
            note += f"; fresh {', '.join(fresh)}"
        if args.init_reset_std:
            note += (
                f"; std reset to {args.policy_init_std}"
                if algorithm.reset_policy_std()
                else "; std is a network output, nothing to reset"
            )
        print(note)

    run = None
    if args.wandb:
        import os

        import wandb

        if args.wandb_host:
            # Read by the client at init; the same variable 'wandb login
            # --host' writes into the local settings.
            os.environ["WANDB_BASE_URL"] = args.wandb_host
        run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_name,
            config=vars(args),
        )
        if run.url:
            print(f"wandb: {run.url}")

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
        wandb_run=run,
    )

    try:
        trainer.train()
        line = trainer.result_line()
        print(line)
        if run is not None:
            # The RESULT line as summary fields, so runs can be sorted on them.
            run.summary.update(
                dict(
                    part.split("=", 1)
                    for part in line.split()
                    if "=" in part and part != "RESULT"
                )
            )
    finally:
        trainer.close()
        if run is not None:
            run.finish()

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
