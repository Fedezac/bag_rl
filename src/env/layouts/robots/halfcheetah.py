from src.env.layouts.layout import ObservationLayout
from src.env.layouts.registry import register_layout

# HalfCheetah has no healthy-termination and runs nose-down by design, so its
# "nominal height" is descriptive rather than a posture target.
HALFCHEETAH_V5 = ObservationLayout(
    obs_dim=17,
    height=0,
    pitch=1,
    joint_position=(2, 8),
    linear_velocity=(8, 10),
    angular_velocity=(10, 11),
    joint_velocity=(11, 17),
    nominal_height=0.0,
    planar=True,
)

register_layout(HALFCHEETAH_V5, "HalfCheetah-v5")
