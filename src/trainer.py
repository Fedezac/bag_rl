"""Training harness: collection, evaluation, rendering, logging."""

import json
import statistics
from collections import defaultdict
from functools import partial
from pathlib import Path

import torch
from torchrl.collectors import MultiSyncCollector
from torchrl.envs.utils import ExplorationType, set_exploration_type
from tqdm import tqdm

from src.env.utils import make_batched_env, make_single_env


def _spec_name(spec):
    """Readable name for a shaping spec that may be a string or a partial."""
    if spec is None or isinstance(spec, str):
        return spec
    func = getattr(spec, "func", spec)
    return getattr(func, "__name__", type(func).__name__)


class Trainer:
    """Drives an :class:`~src.algorithm.Algorithm` against a parallel collector.

    Everything here is algorithm-agnostic: the trainer collects data across
    ``num_workers`` processes, hands each batch to ``algorithm.update()``,
    pushes the updated weights back out to the workers, and takes care of
    evaluation, video capture, the progress bar and the plots. It never reaches
    into the algorithm's modules, so swapping PPO for something else changes
    nothing in this file.
    """

    def __init__(
        self,
        algorithm,
        num_workers=8,
        envs_per_worker=8,
        env_batch_mode="serial",
        frames_per_batch=8192,
        total_frames=204_800,
        eval_every=5,
        eval_steps=1000,
        eval_episodes=1,
        render_every=5,
        render_steps=500,
        video_folder="./videos",
        checkpoint_dir=None,
        seed=None,
        progress=True,
    ):
        total_envs = num_workers * envs_per_worker
        if frames_per_batch % total_envs:
            raise ValueError(
                f"frames_per_batch ({frames_per_batch}) must be divisible by "
                f"num_workers * envs_per_worker ({total_envs})"
            )

        self.algorithm = algorithm
        self.env_name = algorithm.env_name
        self.device = algorithm.device
        self.custom_reward_functions = algorithm.custom_reward_functions
        # Read off the algorithm: the collector envs must emit the cost key the
        # algorithm's cost critic expects, and a mismatch is silent -- the
        # collector drops undeclared keys without warning.
        self.constraints = getattr(algorithm, "constraints", None)

        self.num_workers = num_workers
        self.envs_per_worker = envs_per_worker
        self.env_batch_mode = env_batch_mode
        self.frames_per_batch = frames_per_batch
        self.total_frames = total_frames
        self.num_iterations = max(1, total_frames // frames_per_batch)
        self.frames_per_env = frames_per_batch // total_envs

        self.eval_every = eval_every
        self.eval_steps = eval_steps
        self.eval_episodes = eval_episodes
        self.render_every = render_every
        self.render_steps = render_steps
        self.video_folder = video_folder
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None
        if self.checkpoint_dir is not None:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self._best_eval = None
        self.seed = seed
        self.progress = progress

        # With shaping on, the untouched task reward is kept under
        # 'task_reward'; log THAT so curves stay comparable across variants.
        self.reward_key = (
            "task_reward" if self.custom_reward_functions is not None else "reward"
        )
        self.shaped_key = "reward" if self.custom_reward_functions is not None else None

        self.logs = defaultdict(list)
        self.collector = None
        self.eval_env = None
        self.render_env = None
        self.video_writer = None

    # -- setup / teardown ----------------------------------------------------

    def setup(self):
        # Each worker process gets one torch thread
        torch.set_num_threads(
            max(1, torch.get_num_threads() // max(1, self.num_workers))
        )

        env_fn = partial(
            make_batched_env,
            self.env_name,
            self.envs_per_worker,
            device="cpu",
            mode=self.env_batch_mode,
            custom_reward_functions=self.custom_reward_functions,
            constraints=self.constraints,
        )
        self.collector = MultiSyncCollector(
            create_env_fn=[env_fn] * self.num_workers,
            policy=self.algorithm.policy,
            frames_per_batch=self.frames_per_batch,
            total_frames=self.total_frames,
            split_trajs=False,
            # Envs stay on CPU (MuJoCo is a CPU simulator)
            env_device="cpu",
            policy_device=self.device,
            storing_device=self.device,
            num_sub_threads=1,
            cat_results="stack",
            update_at_each_batch=True,
        )
        if self.seed is not None:
            self.collector.set_seed(self.seed)

        # Evaluation env
        self.eval_env = make_single_env(
            self.env_name,
            self.device,
            custom_reward_functions=self.custom_reward_functions,
            constraints=self.constraints,
        )

        if self.render_every:
            # Imported lazily
            import imageio

            # Same pipeline as eval env
            self.render_env = make_single_env(
                self.env_name,
                self.device,
                custom_reward_functions=self.custom_reward_functions,
                constraints=self.constraints,
                render_mode="rgb_array",
            )
            Path(self.video_folder).mkdir(parents=True, exist_ok=True)
            self.video_writer = imageio.get_writer(
                f"{self.video_folder}/training_full.mp4", fps=30
            )

        print(
            f"{self.num_workers} workers x {self.envs_per_worker} envs = "
            f"{self.num_workers * self.envs_per_worker} gym instances, "
            f"{self.frames_per_env} steps/env per batch, device={self.device}"
        )

    def close(self):
        if self.collector is not None:
            self.collector.shutdown()
            self.collector = None
        if self.eval_env is not None:
            self.eval_env.close()
            self.eval_env = None
        if self.render_env is not None:
            self.render_env.close()
            self.render_env = None
        if self.video_writer is not None:
            self.video_writer.close()
            self.video_writer = None

    # -- training ------------------------------------------------------------

    def train(self):
        """Run the full schedule and return the collected logs."""
        if self.collector is None:
            self.setup()
        self.algorithm.on_training_start(self.frames_per_batch, self.num_iterations)

        pbar = tqdm(total=self.total_frames, disable=not self.progress)
        try:
            for i, batch in enumerate(self.collector):
                for key, value in self.algorithm.update(batch).items():
                    self.logs[key].append(value)

                # Push the freshly-updated weights (and normaliser buffers) out
                # to the workers.
                self.collector.update_policy_weights_()

                self.logs["reward"].append(batch["next", self.reward_key].mean().item())
                if self.shaped_key is not None:
                    self.logs["shaped_reward"].append(
                        batch["next", self.shaped_key].mean().item()
                    )
                self.logs["step_count"].append(batch["step_count"].max().item())

                if self.eval_every and i % self.eval_every == 0:
                    self.evaluate()
                if self.render_env is not None and i % self.render_every == 0:
                    self.render()

                self.algorithm.on_iteration_end()
                pbar.update(batch.numel())
                pbar.set_description(self._describe())
        finally:
            pbar.close()

        return self.logs

    def evaluate(self):
        """Roll the policy out without exploration and log"""
        returns, shaped, per_step, steps, costs = [], [], [], [], []
        with set_exploration_type(ExplorationType.DETERMINISTIC), torch.no_grad():
            for _ in range(self.eval_episodes):
                rollout = self.eval_env.rollout(self.eval_steps, self.algorithm.policy)
                returns.append(rollout["next", self.reward_key].sum().item())
                per_step.append(rollout["next", self.reward_key].mean().item())
                if self.shaped_key is not None:
                    shaped.append(rollout["next", self.shaped_key].sum().item())
                if self.constraints:
                    costs.append(rollout["next", "cost"].mean().item())
                steps.append(rollout["step_count"].max().item())
                del rollout

        self.logs["eval reward"].append(statistics.fmean(per_step))
        self.logs["eval reward (sum)"].append(statistics.fmean(returns))
        # Spread across episodes, so a curve can be read against its own noise.
        self.logs["eval reward (sd)"].append(
            statistics.stdev(returns) if len(returns) > 1 else 0.0
        )
        if shaped:
            self.logs["eval shaped (sum)"].append(statistics.fmean(shaped))
        self.logs["eval step_count"].append(max(steps))
        self.logs["eval step_count (mean)"].append(statistics.fmean(steps))
        if costs:
            self.logs["eval cost"].append(statistics.fmean(costs))

        self._maybe_checkpoint()

    def _maybe_checkpoint(self):
        """Save the latest policy, and separately the best one seen so far"""
        if self.checkpoint_dir is None:
            return
        import torch as _torch

        state = self.algorithm.state_dict()
        _torch.save(state, self.checkpoint_dir / "final.pt")

        # Score on the objective the policy is actually maximising
        current = self.logs[
            "eval shaped (sum)" if self.shaped_key else "eval reward (sum)"
        ][-1]
        if self._best_eval is None or current > self._best_eval:
            self._best_eval = current
            _torch.save(state, self.checkpoint_dir / "best.pt")

        # The curves, alongside the weights
        (self.checkpoint_dir / "logs.json").write_text(
            json.dumps({k: list(v) for k, v in self.logs.items()})
        )

    def render(self):
        """Record one greedy episode into the training video."""

        def grab(env, _td):
            self.video_writer.append_data(env.render())

        with set_exploration_type(ExplorationType.DETERMINISTIC), torch.no_grad():
            self.render_env.rollout(
                self.render_steps, self.algorithm.policy, callback=grab
            )

    # -- reporting -----------------------------------------------------------

    def _describe(self):
        """Progress-bar line, built from whatever keys the logs happen to hold."""
        parts = []
        if self.logs["eval reward (sum)"]:
            parts.append(
                f"eval cumulative reward: {self.logs['eval reward (sum)'][-1]: 4.4f} "
                f"+/- {self.logs['eval reward (sd)'][-1]:.1f} "
                f"(init: {self.logs['eval reward (sum)'][0]: 4.4f}), "
                f"eval step-count: {self.logs['eval step_count'][-1]}"
            )
        if self.logs["reward"]:
            parts.append(
                f"average reward={self.logs['reward'][-1]: 4.4f} "
                f"(init={self.logs['reward'][0]: 4.4f})"
            )
        if self.logs["shaped_reward"]:
            parts.append(
                f"shaped={self.logs['shaped_reward'][-1]: 4.4f} "
                f"(init={self.logs['shaped_reward'][0]: 4.4f})"
            )
        if self.logs["cost"]:
            cost = f"cost={self.logs['cost'][-1]:.4f}"
            if self.logs["eval cost"]:
                cost += f" (eval {self.logs['eval cost'][-1]:.4f})"
            parts.append(f"{cost}, lambda={self.logs['lagrange'][-1]:.3f}")
        if self.logs["step_count"]:
            parts.append(f"step count (max): {self.logs['step_count'][-1]}")
        if self.logs["lr"]:
            parts.append(f"lr policy: {self.logs['lr'][-1]: 4.4f}")
        if self.logs["value_scale"]:
            parts.append(
                f"vnorm scale: {self.logs['value_scale'][-1]:.1f}, "
                f"critic loss: {self.logs['loss_critic'][-1]:.3f}"
            )
        return ", ".join(parts)

    def result_line(self):
        """Compact machine-readable summary, for sweeps / A-B comparisons."""
        evals = self.logs["eval reward (sum)"]
        shaped_evals = self.logs["eval shaped (sum)"]
        tail = self.logs["reward"][-10:] or [float("nan")]
        shaped_tail = self.logs["shaped_reward"][-10:] or [float("nan")]
        scale = self.logs["policy_scale"]
        fields = {
            "env": self.env_name,
            "shaping": _spec_name(self.custom_reward_functions),
            **self.algorithm.summary(),
            "seed": self.seed,
            "train_reward_last10": f"{sum(tail) / len(tail):.4f}",
            # The shaped objective PPO is actually maximising. Without it a
            # falling task-reward curve is uninterpretable.
            "train_shaped_last10": f"{sum(shaped_tail) / len(shaped_tail):.4f}",
            "eval_return_final": f"{evals[-1] if evals else float('nan'):.4f}",
            "eval_return_best": f"{max(evals) if evals else float('nan'):.4f}",
            "eval_return_sd": (
                f"{self.logs['eval reward (sd)'][-1]:.4f}"
                if self.logs["eval reward (sd)"]
                else float("nan")
            ),
            "eval_shaped_final": (
                f"{shaped_evals[-1]:.4f}" if shaped_evals else float("nan")
            ),
            "eval_shaped_best": (
                f"{max(shaped_evals):.4f}" if shaped_evals else float("nan")
            ),
            "eval_steps_final": (
                self.logs["eval step_count"][-1] if self.logs["eval step_count"] else -1
            ),
            # Needed to turn the return sums above into per-step rates
            "eval_steps_mean": (
                f"{self.logs['eval step_count (mean)'][-1]:.1f}"
                if self.logs["eval step_count (mean)"]
                else -1
            ),
            "train_cost_last10": (
                f"{sum(self.logs['cost'][-10:]) / len(self.logs['cost'][-10:]):.4f}"
                if self.logs["cost"]
                else float("nan")
            ),
            "eval_cost_final": (
                f"{self.logs['eval cost'][-1]:.4f}"
                if self.logs["eval cost"]
                else float("nan")
            ),
            "lagrange_final": (
                f"{self.logs['lagrange'][-1]:.4f}"
                if self.logs["lagrange"]
                else float("nan")
            ),
            "policy_scale_final": f"{scale[-1] if scale else float('nan'):.4f}",
            "policy_scale_init": f"{scale[0] if scale else float('nan'):.4f}",
        }
        return "RESULT " + " ".join(f"{k}={v}" for k, v in fields.items())

    def plot(self):
        import matplotlib.pyplot as plt

        plt.figure(figsize=(10, 10))
        panels = [
            ("reward", "training task reward (average)"),
            ("step_count", "Max step count (training)"),
            ("eval reward (sum)", "Task return (test)"),
            ("eval step_count", "Max step count (test)"),
        ]
        if self.logs["shaped_reward"]:
            # With shaping on, the two objectives can move in opposite
            # directions
            panels[1] = ("shaped_reward", "training shaped reward (average)")
            panels[3] = ("eval shaped (sum)", "Shaped return (test)")
        for idx, (key, title) in enumerate(panels, start=1):
            plt.subplot(2, 2, idx)
            plt.plot(self.logs[key])
            plt.title(title)
        plt.show()
