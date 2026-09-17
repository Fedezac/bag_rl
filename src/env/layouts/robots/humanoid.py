from src.env.layouts.layout import ObservationLayout
from src.env.layouts.registry import register_layout

# Humanoid-v5 (obs 348): qpos[2:] -> [0:22], qvel -> [22:45], then cinert (130),
# cvel (78) and qfrc_actuator (17) before cfrc_ext[1:] -> [270:348].
# Feet are rows 5 (right_foot) and 8 (left_foot).
HUMANOID_V5 = ObservationLayout(
    obs_dim=348,
    height=0,
    quaternion=(1, 5),
    joint_position=(5, 22),
    linear_velocity=(22, 25),
    angular_velocity=(25, 28),
    joint_velocity=(28, 45),
    contact_forces=(270, 348),
    n_contact_bodies=13,
    foot_rows=(5, 8),
    gait_pairs=((0,), (1,)),  # bipedal alternation, not a diagonal couplet
    nominal_height=1.4,
)

register_layout(HUMANOID_V5, "Humanoid-v5", "HumanoidStandup-v5")
