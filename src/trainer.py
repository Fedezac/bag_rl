"""Training harness: collection, evaluation, rendering, logging."""

from collections import defaultdict
from functools import partial

import torch
from tensordict import TensorDict
from torchrl.collectors import MultiSyncCollector
from torchrl.envs.utils import ExplorationType, set_exploration_type
from tqdm import tqdm

from src.env.utils import make_batched_env, make_single_env


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
        render_every=5,
        render_steps=500,
        video_folder="./videos",
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

        self.num_workers = num_workers
        self.envs_per_worker = envs_per_worker
        self.env_batch_mode = env_batch_mode
        self.frames_per_batch = frames_per_batch
        self.total_frames = total_frames
        self.num_iterations = max(1, total_frames // frames_per_batch)
        self.frames_per_env = frames_per_batch // total_envs

        self.eval_every = eval_every
        self.eval_steps = eval_steps
        self.render_every = render_every
        self.render_steps = render_steps
        self.video_folder = video_folder
        self.seed = seed
        self.progress = progress

        # With shaping on, the untouched task reward is kept under
        # 'task_reward'; log THAT so curves stay comparable across variants.
        self.reward_key = (
            "task_reward" if self.custom_reward_functions is not None else "reward"
        )

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
        )

        if self.render_every:
            # Imported lazily
            import gymnasium as gym
            import imageio
            from gymnasium.wrappers import RecordVideo

            self.render_env = RecordVideo(
                gym.make(self.env_name, render_mode="rgb_array"),
                video_folder=self.video_folder,
            )
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
        """Roll the policy out without exploration and log the return."""
        # Execute the policy without exploration for a given
        # number of steps -- ``eval_steps``, the env horizon.
        with set_exploration_type(ExplorationType.DETERMINISTIC), torch.no_grad():
            rollout = self.eval_env.rollout(self.eval_steps, self.algorithm.policy)
            self.logs["eval reward"].append(
                rollout["next", self.reward_key].mean().item()
            )
            self.logs["eval reward (sum)"].append(
                rollout["next", self.reward_key].sum().item()
            )
            self.logs["eval step_count"].append(rollout["step_count"].max().item())
            del rollout

    def render(self):
        """Record one greedy episode into the training video."""
        obs, _ = self.render_env.reset()
        with torch.no_grad():
            for _ in range(self.render_steps):
                # Observations are fed in raw
                td = TensorDict(
                    {
                        "observation": torch.as_tensor(
                            obs, dtype=torch.float32, device=self.device
                        ).unsqueeze(0)
                    },
                    batch_size=[1],
                )
                action = self.algorithm.policy(td)["loc"].squeeze(0).cpu().numpy()

                frame = self.render_env.render()
                if self.video_writer is not None:
                    self.video_writer.append_data(frame)

                obs, _, terminated, truncated, _ = self.render_env.step(action)
                if terminated or truncated:
                    break

    # -- reporting -----------------------------------------------------------

    def _describe(self):
        """Progress-bar line, built from whatever keys the logs happen to hold."""
        parts = []
        if self.logs["eval reward (sum)"]:
            parts.append(
                f"eval cumulative reward: {self.logs['eval reward (sum)'][-1]: 4.4f} "
                f"(init: {self.logs['eval reward (sum)'][0]: 4.4f}), "
                f"eval step-count: {self.logs['eval step_count'][-1]}"
            )
        if self.logs["reward"]:
            parts.append(
                f"average reward={self.logs['reward'][-1]: 4.4f} "
                f"(init={self.logs['reward'][0]: 4.4f})"
            )
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
        tail = self.logs["reward"][-10:] or [float("nan")]
        scale = self.logs["policy_scale"]
        fields = {
            "env": self.env_name,
            "shaping": self.custom_reward_functions,
            **self.algorithm.summary(),
            "seed": self.seed,
            "train_reward_last10": f"{sum(tail) / len(tail):.4f}",
            "eval_return_final": f"{evals[-1] if evals else float('nan'):.4f}",
            "eval_return_best": f"{max(evals) if evals else float('nan'):.4f}",
            "eval_steps_final": (
                self.logs["eval step_count"][-1] if self.logs["eval step_count"] else -1
            ),
            "policy_scale_final": f"{scale[-1] if scale else float('nan'):.4f}",
            "policy_scale_init": f"{scale[0] if scale else float('nan'):.4f}",
        }
        return "RESULT " + " ".join(f"{k}={v}" for k, v in fields.items())

    def plot(self):
        import matplotlib.pyplot as plt

        plt.figure(figsize=(10, 10))
        panels = [
            ("reward", "training rewards (average)"),
            ("step_count", "Max step count (training)"),
            ("eval reward (sum)", "Return (test)"),
            ("eval step_count", "Max step count (test)"),
        ]
        for idx, (key, title) in enumerate(panels, start=1):
            plt.subplot(2, 2, idx)
            plt.plot(self.logs[key])
            plt.title(title)
        plt.show()
