"""Per-robot observation layouts.

Reward and constraint terms need to read physical quantities -- torso height,
tilt, forward speed, foot contact -- out of a flat observation vector. Those
offsets differ per robot, and getting them wrong is silent: the term still
computes a number, it is just a number about the wrong thing.
"""

from dataclasses import dataclass, field

import torch


@dataclass(frozen=True)
class ObservationLayout:
    """Where physical quantities sit in one robot's observation vector.

    ``quaternion`` and ``pitch`` are alternatives, not both: 3D robots carry a
    full torso quaternion, planar ones carry a
    single rotation angle. Accessors below paper over the difference so terms
    do not have to branch.

    ``contact_forces`` is ``None`` for robots whose observation omits
    ``cfrc_ext`` -- the planar walkers -- and any contact-based term must
    degrade gracefully rather than index into nothing.
    """

    obs_dim: int
    height: int
    linear_velocity: tuple[int, int]
    angular_velocity: tuple[int, int]
    joint_position: tuple[int, int]
    joint_velocity: tuple[int, int]
    nominal_height: float
    quaternion: tuple[int, int] | None = None
    pitch: int | None = None
    contact_forces: tuple[int, int] | None = None
    n_contact_bodies: int = 0
    foot_rows: tuple[int, ...] = ()
    # Diagonal gait couplets, as indices into ``foot_rows``. A quadruped trot
    # pairs (0, 2) and (1, 3); a biped simply alternates its two feet.
    gait_pairs: tuple[tuple[int, ...], ...] = field(default_factory=tuple)
    planar: bool = False

    # -- accessors ----------------------------------------------------------

    def torso_height(self, obs):
        return obs[..., self.height]

    def upright(self, obs):
        """Torso up-axis projected on world up: 1 upright, 0 on its side, <0 inverted.

        For a quaternion this is the (2, 2) entry of the rotation matrix,
        which reduces to ``1 - 2(x^2 + y^2)``. For a planar robot the same
        quantity is just the cosine of the pitch angle.
        """
        if self.quaternion is not None:
            lo = self.quaternion[0]
            qx, qy = obs[..., lo + 1], obs[..., lo + 2]
            return 1.0 - 2.0 * (qx * qx + qy * qy)
        return torch.cos(obs[..., self.pitch])

    def forward_velocity(self, obs):
        return obs[..., self.linear_velocity[0]]

    def lateral_velocity(self, obs):
        """Sideways velocity; identically zero for a planar robot."""
        if self.planar:
            return torch.zeros_like(obs[..., 0])
        return obs[..., self.linear_velocity[0] + 1]

    def yaw_rate(self, obs):
        """Rotation about world up. Planar robots have only a pitch rate."""
        if self.planar:
            return torch.zeros_like(obs[..., 0])
        return obs[..., self.angular_velocity[0] + 2]

    def joint_velocities(self, obs):
        lo, hi = self.joint_velocity
        return obs[..., lo:hi]

    def contact_matrix(self, obs):
        """``(..., n_contact_bodies, 6)`` external forces, or ``None``."""
        if self.contact_forces is None:
            return None
        lo, hi = self.contact_forces
        return obs[..., lo:hi].unflatten(-1, (self.n_contact_bodies, 6))

    def foot_forces(self, obs):
        """Linear contact-force magnitude per foot, or ``None``."""
        cfrc = self.contact_matrix(obs)
        if cfrc is None or not self.foot_rows:
            return None
        return cfrc[..., self.foot_rows, :3].norm(dim=-1)

    def non_foot_forces(self, obs):
        """Contact-force magnitude on every body that is NOT a foot.

        The basis of a "only feet touch the ground" constraint: a knee, elbow
        or torso registering force means the robot is down, dragging, or
        catching itself on something it should not be using.
        """
        cfrc = self.contact_matrix(obs)
        if cfrc is None:
            return None
        rows = [i for i in range(self.n_contact_bodies) if i not in self.foot_rows]
        if not rows:
            return None
        return cfrc[..., rows, :3].norm(dim=-1)


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

LAYOUTS = {
    "Ant-v5": ANT_V5,
    "Humanoid-v5": HUMANOID_V5,
    "HumanoidStandup-v5": HUMANOID_V5,
    "Walker2d-v5": WALKER2D_V5,
    "Hopper-v5": HOPPER_V5,
    "HalfCheetah-v5": HALFCHEETAH_V5,
}


def get_layout(env_name):
    """Look up a robot's layout, failing loudly on an unknown one"""
    try:
        return LAYOUTS[env_name]
    except KeyError:
        raise KeyError(
            f"no observation layout registered for {env_name!r}. "
            f"Known: {sorted(LAYOUTS)}. Add one to src/env/layouts.py -- "
            f"verify the offsets against mujoco state, do not guess them."
        ) from None
