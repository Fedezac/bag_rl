"""Kyon quadruped locomotion env.

Kyon carries two arms and a pair of grippers on top of its four legs. They are
held at the SRDF home pose by their own position servos and kept out of both
the action and the observation: their mass and inertia still move the robot,
but the policy does not steer them.

The observation follows gymnasium's own MuJoCo convention -- ``qpos[2:]``,
``qvel``, then a block of ``cfrc_ext`` -- because that is what
:class:`ObservationLayout` indexes into.
"""

from pathlib import Path

import mujoco
import numpy as np
from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box

DEFAULT_XML = str(Path(__file__).resolve().parents[1] / "assets" / "kyon" / "kyon.xml")

LEG_JOINTS = [
    f"{j}_{i}" for i in (1, 2, 3, 4)
    for j in ("hip_roll", "hip_pitch", "knee_pitch")
]
FOOT_BODIES = [f"contact_{i}" for i in (1, 2, 3, 4)]
# Bodies whose external forces enter the observation. The feet come last, so
# the layout's foot_rows are 13..16. Everything else here is a body that should
# never be carrying load -- which is exactly the non_foot_contact signal.
CONTACT_BODIES = (
    ["pelvis"]
    + [f"hip_roll_{i}_link" for i in (1, 2, 3, 4)]
    + [f"hip_pitch_{i}_link" for i in (1, 2, 3, 4)]
    + [f"knee_pitch_{i}_link" for i in (1, 2, 3, 4)]
    + FOOT_BODIES
)

# Tracking is done by the model's own "track" camera, not by
# default_camera_config: gymnasium rewrites cam.type on every render() call
# from the camera id, which silently undoes a trackbodyid set here.
TRACKING_CAMERA = "track"


class KyonEnv(MujocoEnv, utils.EzPickle):
    metadata = {
        "render_modes": ["human", "rgb_array", "depth_array"],
        "render_fps": 50,
    }

    def __init__(
        self,
        xml_file=DEFAULT_XML,
        frame_skip=10,
        action_scale=0.5,
        healthy_upright=0.4,
        healthy_z_fraction=(0.6, 1.5),
        reset_noise_scale=0.02,
        # Ant's range, and the gait reward's 0.3 contact threshold and the
        # non_foot_contact cost's 1.0 limit are both calibrated to it. Raw
        # Newtons here would read as permanent contact on every body.
        contact_force_range=(-1.0, 1.0),
        forward_reward_weight=1.0,
        healthy_reward=1.0,
        ctrl_cost_weight=0.005,
        **kwargs,
    ):
        utils.EzPickle.__init__(
            self, xml_file, frame_skip, action_scale, healthy_upright,
            healthy_z_fraction, reset_noise_scale, contact_force_range,
            forward_reward_weight, healthy_reward, ctrl_cost_weight, **kwargs,
        )
        self._action_scale = action_scale
        self._healthy_upright = healthy_upright
        self._reset_noise_scale = reset_noise_scale
        self._contact_force_range = contact_force_range
        self._forward_reward_weight = forward_reward_weight
        self._healthy_reward = healthy_reward
        self._ctrl_cost_weight = ctrl_cost_weight

        kwargs.setdefault("camera_name", TRACKING_CAMERA)
        MujocoEnv.__init__(
            self, xml_file, frame_skip, observation_space=None, **kwargs,
        )

        model = self.model
        name2id = mujoco.mj_name2id

        leg_jids = [name2id(model, mujoco.mjtObj.mjOBJ_JOINT, j) for j in LEG_JOINTS]
        self._leg_qpos_adr = np.array([model.jnt_qposadr[j] for j in leg_jids])
        self._leg_qvel_adr = np.array([model.jnt_dofadr[j] for j in leg_jids])
        self._leg_range = model.jnt_range[leg_jids]
        self._contact_body_ids = np.array(
            [name2id(model, mujoco.mjtObj.mjOBJ_BODY, b) for b in CONTACT_BODIES]
        )

        # Home pose, and the servo targets that hold it.
        self._home_qpos = model.key_qpos[0].copy()
        self._home_ctrl = np.array([
            self._home_qpos[model.jnt_qposadr[model.actuator_trnid[i, 0]]]
            for i in range(model.nu)
        ])
        self._home_legs = self._home_qpos[self._leg_qpos_adr]
        self._nominal_height = float(self._home_qpos[2])
        self._healthy_z_range = (
            healthy_z_fraction[0] * self._nominal_height,
            healthy_z_fraction[1] * self._nominal_height,
        )

        obs_size = 1 + 4 + len(LEG_JOINTS) + 6 + len(LEG_JOINTS) + 6 * len(CONTACT_BODIES)
        self.observation_space = Box(-np.inf, np.inf, (obs_size,), dtype=np.float64)
        # Residual around the home stance, not absolute joint targets: it puts
        # the policy's zero action on a pose that already stands.
        self.action_space = Box(-1.0, 1.0, (len(LEG_JOINTS),), dtype=np.float32)

    @property
    def nominal_height(self):
        return self._nominal_height

    def _contact_forces(self):
        lo, hi = self._contact_force_range
        return np.clip(self.data.cfrc_ext[self._contact_body_ids], lo, hi)

    def _get_obs(self):
        qpos, qvel = self.data.qpos, self.data.qvel
        return np.concatenate([
            qpos[2:7],                      # height, then the base quaternion
            qpos[self._leg_qpos_adr],
            qvel[:6],                       # base linear, then angular velocity
            qvel[self._leg_qvel_adr],
            self._contact_forces().ravel(),
        ])

    @property
    def is_healthy(self):
        z = self.data.qpos[2]
        lo, hi = self._healthy_z_range
        qx, qy = self.data.qpos[4], self.data.qpos[5]
        upright = 1.0 - 2.0 * (qx * qx + qy * qy)
        return bool(lo < z < hi and upright > self._healthy_upright)

    def step(self, action):
        action = np.clip(action, -1.0, 1.0)
        ctrl = self._home_ctrl.copy()
        ctrl[: len(LEG_JOINTS)] = np.clip(
            self._home_legs + self._action_scale * action,
            self._leg_range[:, 0], self._leg_range[:, 1],
        )

        xy_before = self.data.body("pelvis").xpos[:2].copy()
        self.do_simulation(ctrl, self.frame_skip)
        xy_after = self.data.body("pelvis").xpos[:2].copy()
        x_velocity, y_velocity = (xy_after - xy_before) / self.dt

        # The real objective arrives as a reward-shaping transform, which
        # replaces this one; it is here so the env is usable on its own.
        reward = (
            self._forward_reward_weight * x_velocity
            + self._healthy_reward * self.is_healthy
            - self._ctrl_cost_weight * np.sum(np.square(action))
        )
        terminated = not self.is_healthy
        info = {"x_velocity": x_velocity, "y_velocity": y_velocity}

        if self.render_mode == "human":
            self.render()
        return self._get_obs(), reward, terminated, False, info

    def reset_model(self):
        n = self._reset_noise_scale
        qpos = self._home_qpos.copy()
        qpos[self._leg_qpos_adr] += self.np_random.uniform(-n, n, len(LEG_JOINTS))
        qpos[2] += self.np_random.uniform(-n, n)
        qvel = self.np_random.uniform(-n, n, self.model.nv)
        self.set_state(qpos, qvel)
        return self._get_obs()
