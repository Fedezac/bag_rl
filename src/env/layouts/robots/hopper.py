from src.env.layouts.layout import ObservationLayout
from src.env.layouts.registry import register_layout

HOPPER_V5 = ObservationLayout(
    obs_dim=11,
    height=0,
    pitch=1,
    joint_position=(2, 5),
    linear_velocity=(5, 7),
    angular_velocity=(7, 8),
    joint_velocity=(8, 11),
    nominal_height=1.25,
    planar=True,
)

register_layout(HOPPER_V5, "Hopper-v5")
