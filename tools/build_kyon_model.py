#!/usr/bin/env python
"""Turn the ROS-staged Kyon MJCF into one an RL env can actually drive.

The model that ``mujoco_simulator_wrapper.py`` stages has no ``<actuator>`` at
all -- the real stack does its PD outside MuJoCo -- so nothing can be
commanded. It also carries a 5 kHz timestep, a floor a metre below the origin
and ``<size njmax>``, which MuJoCo 3 rejects.

Run once; re-run when the URDF changes. The output (``src/env/assets/kyon``,
meshes included) is gitignored, so this is the only way to get it.
"""

import argparse
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

# SRDF group_state "home" (kyon_srdf/srdf/kyon.srdf). The legs mirror in pairs:
# 1/3 take +hip_pitch, 2/4 take -hip_pitch.
HOME = {
    "hip_roll_1": 0.0, "hip_pitch_1": 0.7, "knee_pitch_1": -1.4,
    "hip_roll_2": 0.0, "hip_pitch_2": -0.7, "knee_pitch_2": 1.4,
    "hip_roll_3": 0.0, "hip_pitch_3": 0.7, "knee_pitch_3": -1.4,
    "hip_roll_4": 0.0, "hip_pitch_4": -0.7, "knee_pitch_4": 1.4,
    "shoulder_yaw_1": 0.0, "shoulder_pitch_1": 1.7, "elbow_pitch_1": -2.5,
    "wrist_pitch_1": 0.75, "wrist_yaw_1": 0.0, "dagana_1_clamp_joint": 0.0,
    "shoulder_yaw_2": 0.0, "shoulder_pitch_2": -1.7, "elbow_pitch_2": 2.5,
    "wrist_pitch_2": -0.75, "wrist_yaw_2": 0.0, "dagana_2_clamp_joint": 0.0,
}

LEG_JOINTS = [f"{j}_{i}" for i in (1, 2, 3, 4) for j in ("hip_roll", "hip_pitch", "knee_pitch")]
ARM_JOINTS = [
    f"{j}_{i}" for i in (1, 2)
    for j in ("shoulder_yaw", "shoulder_pitch", "elbow_pitch", "wrist_pitch", "wrist_yaw")
]
GRIPPER_JOINTS = ["dagana_1_clamp_joint", "dagana_2_clamp_joint"]

# xbot2 decentralised PD gains (kyon_mujoco/config/kyon.yaml). The grippers are
# not listed there; they only have to hold their own weight.
GAINS = {**{j: (500.0, 10.0) for j in LEG_JOINTS},
         **{j: (200.0, 10.0) for j in ARM_JOINTS},
         **{j: (100.0, 5.0) for j in GRIPPER_JOINTS}}

# Reflected rotor inertia and gearbox losses, from the (unreferenced, and so
# inert) "kyon_all" default class in the staged model. Applied per joint here
# rather than as a global default, which would also load the floating base.
ARMATURE, DAMPING, FRICTIONLOSS = 0.234, 1.7, 4.68

# Arms are pinned and never touch anything that matters. Leg links carry full
# collision meshes that reach *below* the foot sphere at the end of the shank,
# so leaving them on has the robot stand on its shins and the feet report no
# contact at all. Kyon has point feet; the contact_* spheres are the feet.
NO_COLLISION_PREFIXES = (
    "shoulder_", "elbow_", "wrist_", "dagana_",
    "hip_roll_", "hip_pitch_", "knee_pitch_",
)


def strip_sdf_plugin(root):
    for ext in root.findall("extension"):
        root.remove(ext)
    for mesh in root.iter("mesh"):
        for plugin in mesh.findall("plugin"):
            mesh.remove(plugin)


def set_compiler_and_options(root):
    for tag in ("compiler", "option", "size", "default"):
        for el in root.findall(tag):
            root.remove(el)
    # The mesh file attributes already carry the "./assets/" prefix.
    ET.SubElement(root, "compiler", angle="radian", autolimits="true", meshdir=".")
    ET.SubElement(root, "option", timestep="0.002", cone="pyramidal",
                  solver="Newton", iterations="10", ls_iterations="5")


def replace_floor(root):
    """One plane at z=0. The staged pair sits at z=-1 so the robot free-falls in."""
    world = root.find("worldbody")
    for geom in list(world.findall("geom")):
        if geom.get("name", "").startswith("floor"):
            world.remove(geom)
    ET.SubElement(world, "geom", name="floor", type="plane", size="0 0 0.05",
                  pos="0 0 0", material="grid", condim="3", friction="0.9 0.1 0.1")


def tune_joints(root):
    for joint in root.iter("joint"):
        if joint.get("type") == "free":
            continue
        joint.set("armature", str(ARMATURE))
        joint.set("damping", str(DAMPING))
        joint.set("frictionloss", str(FRICTIONLOSS))


def restrict_collisions(root):
    """Leave the pelvis and the four foot spheres as the only colliding geoms."""
    for body in root.iter("body"):
        if not body.get("name", "").startswith(NO_COLLISION_PREFIXES):
            continue
        for geom in body.findall("geom"):
            geom.set("contype", "0")
            geom.set("conaffinity", "0")


def add_tracking_camera(root):
    """A camera that follows the robot, defined in the model.

    Gymnasium's renderer rewrites ``cam.type`` on every render() call from the
    camera_id it was given, so a ``trackbodyid`` passed through
    ``default_camera_config`` is silently clobbered and the robot walks out of
    frame. A named MJCF camera survives that, because it is selected *by* id.
    """
    pelvis = root.find("worldbody/body[@name='pelvis']")
    ET.SubElement(pelvis, "camera", name="track", mode="trackcom",
                  pos="0 -3.5 1.4", xyaxes="1 0 0 0 0.35 0.94")


def joint_ranges(root):
    return {j.get("name"): (j.get("range"), j.get("actuatorfrcrange"))
            for j in root.iter("joint") if j.get("type") != "free"}


def add_actuators(root):
    """Position servos, legs first: ``ctrl[:12]`` is what the policy writes."""
    ranges = joint_ranges(root)
    actuator = ET.SubElement(root, "actuator")
    for name in LEG_JOINTS + ARM_JOINTS + GRIPPER_JOINTS:
        rng, frc = ranges[name]
        kp, kv = GAINS[name]
        attrs = {"name": name, "joint": name, "kp": str(kp), "kv": str(kv),
                 "ctrlrange": rng.replace(",", " ")}
        if frc:
            attrs["forcerange"] = frc.replace(",", " ")
        ET.SubElement(actuator, "position", **attrs)


def home_qpos(model, base_z):
    import numpy as np
    qpos = np.zeros(model.nq)
    qpos[:3] = [0.0, 0.0, base_z]
    qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
    for name, angle in HOME.items():
        import mujoco
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        qpos[model.jnt_qposadr[jid]] = angle
    return qpos


def standing_height(xml_path):
    """Drop the home pose onto the floor: base z that puts the feet at z=0."""
    import mujoco
    import numpy as np

    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)
    data.qpos[:] = home_qpos(model, 1.0)
    mujoco.mj_forward(model, data)

    lowest = np.inf
    for i in range(1, 5):
        gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"contact_{i}")
        lowest = min(lowest, data.geom_xpos[gid][2] - model.geom_size[gid][0])
    # qpos[2] was 1.0, so the clearance to close is exactly `lowest`.
    return 1.0 - lowest + 0.005


def add_keyframe(root, qpos):
    key = ET.SubElement(ET.SubElement(root, "keyframe"), "key")
    key.set("name", "home")
    key.set("qpos", " ".join(f"{v:.6g}" for v in qpos))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", default="/tmp/kyon_mujoco/kyon.xml")
    p.add_argument("--out-dir", default="src/env/assets/kyon")
    args = p.parse_args()

    source = Path(args.source)
    out_dir = Path(args.out_dir)
    (out_dir / "assets").mkdir(parents=True, exist_ok=True)

    # The staged assets are symlinks into the ROS package; dereference them so
    # the model stands on its own.
    for stl in (source.parent / "assets").glob("*.stl"):
        shutil.copyfile(stl, out_dir / "assets" / stl.name, follow_symlinks=True)

    tree = ET.parse(source)
    root = tree.getroot()
    strip_sdf_plugin(root)
    set_compiler_and_options(root)
    replace_floor(root)
    tune_joints(root)
    restrict_collisions(root)
    add_tracking_camera(root)
    add_actuators(root)

    out = out_dir / "kyon.xml"
    ET.indent(tree, space="  ")
    tree.write(out, encoding="unicode")

    base_z = standing_height(out)
    import mujoco
    model = mujoco.MjModel.from_xml_path(str(out))
    add_keyframe(root, home_qpos(model, base_z))
    ET.indent(tree, space="  ")
    tree.write(out, encoding="unicode")

    model = mujoco.MjModel.from_xml_path(str(out))
    print(f"wrote {out}: nq={model.nq} nv={model.nv} nu={model.nu} "
          f"nbody={model.nbody} standing z={base_z:.4f}")


if __name__ == "__main__":
    main()
