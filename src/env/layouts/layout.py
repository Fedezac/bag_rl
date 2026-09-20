"""The observation-layout record and its accessors.

Terms read physical quantities -- torso height, tilt, forward speed, foot
contact -- out of a flat observation vector. Offsets differ per robot and a
wrong one is silent: the term still returns a number, just a number about
the wrong thing.
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
    # Which half of each ``cfrc_ext`` row carries the force. MuJoCo spatial
    # vectors are rotational-first, so a row is [torque(3), force(3)] and (3, 6)
    # is the physical contact force. (0, 3) -- the torque about the body CoM --
    # is what the gymnasium-derived layouts were calibrated against: clipped to
    # [-1, 1] both halves saturate under load and vanish in the air, so either
    # works as a contact *detector*, but only the force half is a force.
    contact_force_components: tuple[int, int] = (0, 3)
    # World-frame linear velocity per foot, laid out as ``n_feet x 3``. ``None``
    # for robots whose observation omits it, which is every gymnasium built-in.
    foot_velocity: tuple[int, int] | None = None
    # Diagonal gait couplets, as indices into ``foot_rows``. A quadruped trot
    # pairs (0, 2) and (1, 3); a biped simply alternates its two feet.
    gait_pairs: tuple[tuple[int, ...], ...] = field(default_factory=tuple)
    # Which of ``foot_rows`` are front feet. A walk is a lateral sequence --
    # each hind foot lands a quarter cycle before the fore foot diagonal to it
    # -- so the gait clock has to know which end of the robot a foot is on.
    # Reading it off the pairing instead would only work if every layout wrote
    # its couplets fore-first, which is a convention nothing enforces.
    fore_feet: tuple[int, ...] = ()
    # Seconds of wall-clock per policy step (MuJoCo timestep x frame_skip). Only
    # a term that integrates -- the gait clock -- needs it, and a wrong value is
    # silent, so tools/verify_kyon.py checks it against the env's own ``dt``.
    control_dt: float = 0.05
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

    def body_twist(self, obs):
        """Twist in the body frame at the CoM: ``(vx, vy, wz)``.

        A twist command is body-fixed -- "forward at 1 m/s" means along the
        robot's own heading,
        """
        lo = self.linear_velocity[0]
        alo = self.angular_velocity[0]
        if self.quaternion is None:
            # Planar robot: one pitch angle, and no yaw degree of freedom
            th = obs[..., self.pitch]
            c, s = torch.cos(th), torch.sin(th)
            vx, vz = obs[..., lo], obs[..., lo + 1]
            return c * vx + s * vz, torch.zeros_like(vx), torch.zeros_like(vx)

        qlo = self.quaternion[0]
        w = obs[..., qlo]
        x, y, z = obs[..., qlo + 1], obs[..., qlo + 2], obs[..., qlo + 3]
        vx, vy, vz = obs[..., lo], obs[..., lo + 1], obs[..., lo + 2]
        wx, wy, wz = obs[..., alo], obs[..., alo + 1], obs[..., alo + 2]

        r00, r10, r20 = 1 - 2 * (y * y + z * z), 2 * (x * y + w * z), 2 * (x * z - w * y)
        r01, r11, r21 = 2 * (x * y - w * z), 1 - 2 * (x * x + z * z), 2 * (y * z + w * x)
        r02, r12, r22 = 2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y)

        vx_b = r00 * vx + r10 * vy + r20 * vz
        vy_b = r01 * vx + r11 * vy + r21 * vz
        wz_b = r02 * wx + r12 * wy + r22 * wz
        return vx_b, vy_b, wz_b

    def joint_velocities(self, obs):
        lo, hi = self.joint_velocity
        return obs[..., lo:hi]

    def contact_matrix(self, obs):
        """``(..., n_contact_bodies, 6)`` external forces, or ``None``."""
        if self.contact_forces is None:
            return None
        lo, hi = self.contact_forces
        return obs[..., lo:hi].unflatten(-1, (self.n_contact_bodies, 6))

    def _force_magnitude(self, cfrc, rows):
        lo, hi = self.contact_force_components
        return cfrc[..., rows, lo:hi].norm(dim=-1)

    def foot_forces(self, obs):
        """Contact-force magnitude per foot, or ``None``."""
        cfrc = self.contact_matrix(obs)
        if cfrc is None or not self.foot_rows:
            return None
        return self._force_magnitude(cfrc, list(self.foot_rows))

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
        return self._force_magnitude(cfrc, rows)

    def foot_velocities(self, obs):
        """``(..., n_feet, 3)`` world-frame foot velocities, or ``None``.

        What a slip or drag term needs and cannot reconstruct: the observation
        carries joint velocities, not Cartesian ones, and turning one into the
        other takes the leg's Jacobian.
        """
        if self.foot_velocity is None:
            return None
        lo, hi = self.foot_velocity
        return obs[..., lo:hi].unflatten(-1, (len(self.foot_rows), 3))
