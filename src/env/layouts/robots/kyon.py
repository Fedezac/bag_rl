from gymnasium.envs.registration import register

from src.env.layouts.layout import ObservationLayout
from src.env.layouts.registry import register_layout

# Kyon is not a gymnasium built-in, so the id has to be created here rather
# than merely described. This module is imported eagerly by
# ``layouts.robots.__init__``, which is what every spawned collector worker
# re-imports -- registering anywhere else would leave the workers unable to
# build the env they were asked for.
register(
    id="Kyon-v1",
    entry_point="src.env.robots.kyon_env:KyonEnv",
    max_episode_steps=1000,
)

# v2 is v1 plus the four feet's world-frame linear velocities. Nothing else
# moves: the block is appended, so every v1 offset still reads the same
# quantity, and a term that does not ask for foot velocities cannot tell the
# two apart.
register(
    id="Kyon-v2",
    entry_point="src.env.robots.kyon_env:KyonEnv",
    max_episode_steps=1000,
    kwargs={"foot_velocities": True},
)

# frame_skip 10 x the model's 2 ms timestep.
CONTROL_DT = 0.02

# Kyon-v1 (obs 137): qpos[2:7] -> [0:5], the 12 leg joints -> [5:17],
# qvel[:6] -> [17:23], the 12 leg joint velocities -> [23:35], then
# cfrc_ext over 17 selected bodies -> [35:137] as 17 x 6. The arms are pinned
# at the home pose and appear in neither the action nor the observation.
#
# Contact rows run [pelvis, hip_roll_1..4, hip_pitch_1..4, knee_pitch_1..4,
# contact_1..4], so the feet are rows 13..16. Leg numbering is 1 front-left,
# 2 front-right, 3 rear-left, 4 rear-right -- measured from the foot bodies'
# world xy at the home keyframe, not read off the names -- so the trot
# diagonals are (FL, RR) = (0, 3) and (FR, RL) = (1, 2).
KYON_V1 = ObservationLayout(
    obs_dim=137,
    height=0,
    quaternion=(1, 5),
    joint_position=(5, 17),
    linear_velocity=(17, 20),
    angular_velocity=(20, 23),
    joint_velocity=(23, 35),
    contact_forces=(35, 137),
    n_contact_bodies=17,
    foot_rows=(13, 14, 15, 16),
    gait_pairs=((0, 3), (1, 2)),
    fore_feet=(0, 1),
    control_dt=CONTROL_DT,
    # Pelvis height with the SRDF home pose resting on the floor, measured by
    # forward kinematics in tools/build_kyon_model.py.
    nominal_height=0.538,
)

# Kyon-v2 (obs 149): v1, then the four contact spheres' world-frame linear
# velocities -> [137:149] as 4 x 3, in foot_rows order.
#
# The contact-force slice moves to the force half of cfrc_ext. v1 reads the
# torque half, which the gymnasium layouts were calibrated against and which
# works as a detector because both halves saturate against the [-1, 1] clip
# under load -- but the drag term needs a real force, and a layout should not
# call a torque one.
KYON_V2 = ObservationLayout(
    obs_dim=149,
    height=0,
    quaternion=(1, 5),
    joint_position=(5, 17),
    linear_velocity=(17, 20),
    angular_velocity=(20, 23),
    joint_velocity=(23, 35),
    contact_forces=(35, 137),
    n_contact_bodies=17,
    foot_rows=(13, 14, 15, 16),
    contact_force_components=(3, 6),
    foot_velocity=(137, 149),
    gait_pairs=((0, 3), (1, 2)),
    fore_feet=(0, 1),
    control_dt=CONTROL_DT,
    nominal_height=0.538,
)

register_layout(KYON_V1, "Kyon-v1")
register_layout(KYON_V2, "Kyon-v2")
