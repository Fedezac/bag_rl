#!/usr/bin/env python
"""Time collection and update separately, per config, to find the real limit.

The GPU sits near 20% because a 64-wide MLP is launch-bound, not math-bound,
so the question is not "how do we feed the GPU" but "where does the wall clock
actually go". Reports frames/s, which is what a run is paid in.
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from src.ppo import PPO  # noqa: E402
from src.trainer import Trainer  # noqa: E402


def bench(env_name, shaping, workers, envs_per_worker, frames_per_batch,
          sub_batch, epochs, device, policy_device, iterations=3):
    algorithm = PPO(
        env_name, custom_reward_functions=shaping, observations_warmup_steps=0,
        num_epochs=epochs, sub_batch_size=sub_batch, device=device,
    )
    trainer = Trainer(
        algorithm, num_workers=workers, envs_per_worker=envs_per_worker,
        frames_per_batch=frames_per_batch,
        total_frames=frames_per_batch * (iterations + 1),
        eval_every=0, render_every=0, checkpoint_dir=None,
    )
    if policy_device is not None:
        trainer.device = torch.device(policy_device)
    trainer.setup()
    algorithm.on_training_start(frames_per_batch, iterations + 1)

    collect = update = 0.0
    t_prev = time.perf_counter()
    for i, batch in enumerate(trainer.collector):
        collect += time.perf_counter() - t_prev
        t = time.perf_counter()
        trainer.algorithm.update(batch)
        trainer.collector.update_policy_weights_()
        update += time.perf_counter() - t
        if i + 1 >= iterations:
            break
        t_prev = time.perf_counter()

    trainer.close()
    total = collect + update
    frames = frames_per_batch * iterations
    return frames / total, collect / total, update / total


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--env-name", default="Kyon-v1")
    p.add_argument("--iterations", type=int, default=3)
    args = p.parse_args()

    configs = [
        ("baseline (smoke settings)", 8, 4, 8192, 256, 3, "cuda:0", None),
        ("+ sub_batch 4096", 8, 4, 8192, 4096, 3, "cuda:0", None),
        ("+ 128 envs, batch 32768", 16, 8, 32768, 4096, 3, "cuda:0", None),
        ("128 envs, cpu policy collect", 16, 8, 32768, 4096, 3, "cuda:0", "cpu"),
        ("128 envs, all cpu", 16, 8, 32768, 4096, 3, "cpu", "cpu"),
    ]
    print(f"{'config':<32} {'frames/s':>10} {'collect':>9} {'update':>8}")
    for name, w, e, fpb, sb, ep, dev, pdev in configs:
        try:
            fps, c, u = bench(args.env_name, "gait_twist", w, e, fpb, sb, ep,
                              dev, pdev, args.iterations)
            print(f"{name:<32} {fps:>10,.0f} {c:>8.0%} {u:>7.0%}", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"{name:<32} {'FAILED':>10}  {type(exc).__name__}: {exc}", flush=True)


if __name__ == "__main__":
    from torch import multiprocessing
    try:
        multiprocessing.set_start_method("spawn")
    except RuntimeError:
        pass
    main()
