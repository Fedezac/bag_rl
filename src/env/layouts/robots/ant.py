from src.env.layouts.layout import ObservationLayout
from src.env.layouts.registry import register_layout

# Ant-v5 (obs 105): qpos[2:] -> [0:13], qvel -> [13:27],
# cfrc_ext[1:] -> [27:105] as 13 bodies x 6.
# Foot rows (3, 6, 9, 12) are the four *unnamed* ankle bodies. The diagonal
# pairs are (0, 2) and (1, 3), derived by measuring ankle world-xy quadrants --
# NOT from the body names, which do not match the geometry: ``back_leg`` sits
# at x = -0.37 and ``front_right_leg`` at x = -0.40.
ANT_V5 = ObservationLayout(
    obs_dim=105,
    height=0,
    quaternion=(1, 5),
    joint_position=(5, 13),
    linear_velocity=(13, 16),
    angular_velocity=(16, 19),
    joint_velocity=(19, 27),
    contact_forces=(27, 105),
    n_contact_bodies=13,
    foot_rows=(3, 6, 9, 12),
    gait_pairs=((0, 2), (1, 3)),
    nominal_height=0.55,
)

register_layout(ANT_V5, "Ant-v5")
