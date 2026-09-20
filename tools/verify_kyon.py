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
    front = tuple(i - 1 for i in (1, 2, 3, 4) if quad[i].startswith("front"))
    # The walk is a lateral sequence, so the clock has to know which end of the
    # robot each foot is on. Measured, not read off the leg numbering.
    check("fore_feet are the front pair", layout.fore_feet == front,
          f"{layout.fore_feet} vs measured {front}")
    check("control_dt matches the env", abs(layout.control_dt - env.dt) < 1e-9,
          f"{layout.control_dt} vs {env.dt}")

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

    print("== Kyon-v2 (foot velocities, force half of cfrc_ext)")
    v2 = get_layout("Kyon-v2")
    env2 = KyonEnv(foot_velocities=True)
    env2.reset(seed=0)
    for _ in range(200):
        env2.step(np.zeros(12))
    obs2 = torch.as_tensor(env2._get_obs())
    check("obs_dim", v2.obs_dim == obs2.shape[-1], f"{v2.obs_dim} vs {obs2.shape[-1]}")
    check("v1 offsets still read the same", bool(
        torch.allclose(obs2[: layout.obs_dim], torch.as_tensor(env2._get_obs())[:137])
    ))
    res = np.zeros(6)
    truth = []
    for gid in env2._foot_geom_ids:
        mujoco.mj_objectVelocity(
            env2.model, env2.data, mujoco.mjtObj.mjOBJ_GEOM, gid, res, 0
        )
        truth.append(res[3:].copy())
    check("foot_velocities are the contact spheres\' world velocities",
          np.allclose(v2.foot_velocities(obs2).numpy(), np.array(truth)))
    check("a standing foot is not moving",
          float(v2.foot_velocities(obs2)[:, :2].norm(dim=-1).max()) < 0.05)

    # The force half is a force: the four vertical components have to add up to
    # the robot's weight before the observation clips them.
    weight = float(env2.model.body_mass.sum() * 9.81)
    fz = float(env2.data.cfrc_ext[env2._contact_body_ids][13:, 5].sum())
    check("cfrc_ext[3:6] is the force", abs(fz - weight) < 0.02 * weight,
          f"sum of foot Fz {fz:.0f} N vs weight {weight:.0f} N")
    feet2 = v2.foot_forces(obs2)
    check("standing feet clear the contact threshold", bool((feet2 > 0.3).all()),
          f"{np.round(feet2.numpy(), 3)}")
    check("nothing but feet is loaded", bool((v2.non_foot_forces(obs2) < 1.0).all()),
          f"max non-foot {float(v2.non_foot_forces(obs2).max()):.3f}")
    # Airborne: the same threshold has to read zero, or every body is always
    # in contact and the gait terms score noise.
    qpos = env2.data.qpos.copy()
    qpos[2] = 1.5
    env2.set_state(qpos, np.zeros(env2.model.nv))
    airborne = v2.foot_forces(torch.as_tensor(env2._get_obs()))
    check("airborne feet read as free", bool((airborne < 0.3).all()),
          f"{np.round(airborne.numpy(), 3)}")
    env2.close()

    print("== gait clock")
    from src.env.rewards.gait import PhaseGaitReward

    class _Command:
        command = torch.zeros(3)
        command_span = torch.tensor([1.8, 0.3, 0.8])

    clock = PhaseGaitReward("Kyon-v2", command_source=_Command())
    n_feet = len(v2.foot_rows)
    duties = [clock.duty(v / 100.0) for v in range(0, 200)]
    check("standing still wants every foot down", duties[0] == 1.0, f"{duties[0]}")
    check("duty never rises with speed",
          all(b <= a + 1e-9 for a, b in zip(duties, duties[1:])))
    check("full speed wants the trot duty",
          abs(clock.duty(clock.trot_speed + clock.gait_blend) - 1 / len(v2.gait_pairs))
          < 1e-9)
    walk_duty = clock.duty(clock.walk_speed + clock.gait_blend)
    check("mid speed wants a walk duty", abs(walk_duty - (1 - 1 / n_feet)) < 1e-9,
          f"{walk_duty:.3f}")

    trot_offsets = clock.offsets(clock.trot_speed + clock.gait_blend)
    couplets = {frozenset(g) for g in v2.gait_pairs}
    check("trot offsets collapse onto the diagonals",
          all(abs(trot_offsets[a] - trot_offsets[b]) < 1e-9 for a, b in couplets)
          and abs(trot_offsets[v2.gait_pairs[0][0]] - trot_offsets[v2.gait_pairs[1][0]])
          - 0.5 < 1e-9,
          str([round(o, 3) for o in trot_offsets]))

    # A walk is one foot in swing at a time, in a lateral sequence: every hind
    # foot is followed by the fore foot on its own side.
    walk_speed = clock.walk_speed + clock.gait_blend
    clock._phase = 0.0
    steps = int(round(1.0 / (clock.frequency(walk_speed) * v2.control_dt)))
    swinging, order = [], []
    for _ in range(steps):
        wanted = clock.target_contact_at(torch.tensor(clock._phase), walk_speed)
        up = [i for i, w in enumerate(wanted.tolist()) if not w]
        swinging.append(len(up))
        if up and (not order or order[-1] != up[0]):
            order.append(up[0])
        clock._phase = (clock._phase + clock.frequency(walk_speed) * v2.control_dt) % 1.0
    check("a walk swings exactly one foot at a time", set(swinging) == {1},
          f"feet in swing per step: {sorted(set(swinging))}")
    check("every foot takes a turn", sorted(order) == list(range(n_feet)),
          f"swing order {order}")
    ipsilateral = all(
        quad[order[i] + 1].split("-")[1] == quad[order[(i + 1) % len(order)] + 1].split("-")[1]
        for i in range(len(order))
        if quad[order[i] + 1].startswith("rear")
    )
    check("the walk is a lateral sequence (hind, then the fore on its side)",
          ipsilateral, " -> ".join(quad[i + 1] for i in order))

    print("== framework")
    from src.env.utils import env_specs
    obs_dim, action_spec = env_specs("Kyon-v1")
    check("check_env_specs + obs dim (v1)", obs_dim == 137, f"obs_dim={obs_dim}")
    check("check_env_specs + obs dim (v2)", env_specs("Kyon-v2")[0] == 149)
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
