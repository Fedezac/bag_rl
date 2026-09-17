from src.env.layouts.layout import ObservationLayout
from src.env.layouts.registry import register_layout

# Walker2d-v5 (obs 17): qpos[1:] -> [0:8], qvel -> [8:17].
# The observed velocities are CLIPPED to +/-10, so a term reading them is
# reading the clipped value, not true qvel. No cfrc_ext in the observation.
WALKER2D_V5 = ObservationLayout(
    obs_dim=17,
    height=0,
    pitch=1,
    joint_position=(2, 8),
    linear_velocity=(8, 10),
    angular_velocity=(10, 11),
    joint_velocity=(11, 17),
    nominal_height=1.25,
    planar=True,
)

register_layout(WALKER2D_V5, "Walker2d-v5")
