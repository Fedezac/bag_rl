#!/usr/bin/env python
"""Check the Kyon model and its observation layout before spending GPU hours.

A wrong layout offset is silent -- the reward term still returns a number,
just a number about the wrong thing -- so every field is cross-checked against
the simulator state it is supposed to name.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mujoco  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from src.env.layouts import get_layout  # noqa: E402
from src.env.robots.kyon_env import CONTACT_BODIES, LEG_JOINTS, KyonEnv  # noqa: E402

FAILURES = []


def check(name, ok, detail=""):
    print(f"  [{'ok' if ok else 'FAIL'}] {name}{f' -- {detail}' if detail else ''}")
    if not ok:
        FAILURES.append(name)


def main():
    print("== model")
    env = KyonEnv()
    model, data = env.model, env.data
    check("nq/nv/nu", (model.nq, model.nv, model.nu) == (31, 30, 24),
          f"nq={model.nq} nv={model.nv} nu={model.nu} nbody={model.nbody}")
    check("actuators are legs-first",
          [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, j) for j in LEG_JOINTS]
          == list(range(12)))
    print(f"  nominal height {env.nominal_height:.4f} m, dt {env.dt:.4f} s "
          f"({1 / env.dt:.0f} Hz control)")

    # The xbot2 gains assume the gravity-compensation plugin the real stack
    # runs, so a bare PD sags a few cm under 82 kg and settles there. What
    # matters is that it converges rather than collapsing; closing the last
    # centimetres is the policy's job.
    print("== standing (5 s holding the home pose)")
    env.reset(seed=0)
    for _ in range(200):
        env.step(np.zeros(12))
    z_mid = float(data.qpos[2])
    for _ in range(50):
        env.step(np.zeros(12))
    z_end = float(data.qpos[2])
    upright = 1 - 2 * (data.qpos[4] ** 2 + data.qpos[5] ** 2)
    check("settles above the healthy floor", z_end > 0.85 * env.nominal_height,
          f"{z_end:.3f} m vs nominal {env.nominal_height:.3f} m")
    check("converged, not still sinking", abs(z_end - z_mid) < 0.005,
          f"{z_mid:.4f} -> {z_end:.4f} m over the last second")
    check("stays upright", upright > 0.95, f"upright={upright:.3f}")

    print("== leg quadrants (pelvis frame, home pose)")
    quad = {}
    for i in (1, 2, 3, 4):
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"contact_{i}")
        x, y = data.xpos[bid][:2] - data.xpos[1][:2]
        quad[i] = ("front" if x > 0 else "rear") + ("-left" if y > 0 else "-right")
        print(f"  contact_{i}: x={x:+.3f} y={y:+.3f}  {quad[i]}")
    layout = get_layout("Kyon-v1")
    diagonals = {frozenset(p) for p in layout.gait_pairs}
    expected = {frozenset({0, 3}), frozenset({1, 2})} if (
        quad[1] == "front-left" and quad[4] == "rear-right") else None
    check("gait_pairs are the trot diagonals", diagonals == expected, str(layout.gait_pairs))

    print("== layout offsets")
    obs = torch.as_tensor(env._get_obs())
    check("obs_dim", layout.obs_dim == obs.shape[-1], f"{layout.obs_dim} vs {obs.shape[-1]}")
    check("height", np.isclose(float(layout.torso_height(obs)), data.qpos[2]))
    check("upright ~ 1 while standing", abs(float(layout.upright(obs)) - 1) < 0.05)
    lo, hi = layout.joint_position
    check("joint_position", np.allclose(obs[lo:hi].numpy(), data.qpos[env._leg_qpos_adr]))
    check("joint_velocity", np.allclose(
        layout.joint_velocities(obs).numpy(), data.qvel[env._leg_qvel_adr]))

    contacts = layout.contact_matrix(obs)
    check("contact_matrix shape", tuple(contacts.shape) == (len(CONTACT_BODIES), 6))
    feet = layout.foot_forces(obs)
    check("all four feet loaded while standing", bool((feet > 0.3).all()),
          f"foot forces {np.round(feet.numpy(), 3)}")
    non_foot = layout.non_foot_forces(obs)
    check("nothing but feet is loaded", bool((non_foot < 1.0).all()),
          f"max non-foot {float(non_foot.max()):.3f}")

    print("== twist frame (base pushed forward at 1 m/s)")
    data.qvel[:6] = 0
    data.qvel[0] = 1.0
    mujoco.mj_forward(model, data)
    vx, vy, wz = (float(v) for v in layout.body_twist(torch.as_tensor(env._get_obs())))
    check("body_twist reads forward motion", abs(vx - 1.0) < 0.05 and abs(vy) < 0.05,
          f"vx={vx:+.3f} vy={vy:+.3f} wz={wz:+.3f}")

    data.qvel[:6] = 0
    data.qvel[5] = 1.0
    mujoco.mj_forward(model, data)
    _, _, wz = (float(v) for v in layout.body_twist(torch.as_tensor(env._get_obs())))
    check("body_twist reads yaw rate", abs(wz - 1.0) < 0.05, f"wz={wz:+.3f}")
    env.close()

    print("== framework")
    from src.env.utils import env_specs
    obs_dim, action_spec = env_specs("Kyon-v1")
    check("check_env_specs + obs dim", obs_dim == 137, f"obs_dim={obs_dim}")
    print(f"  action spec {action_spec.shape} in "
          f"[{float(action_spec.low.min()):.1f}, {float(action_spec.high.max()):.1f}]")

    print()
    if FAILURES:
        print(f"FAILED: {', '.join(FAILURES)}")
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
