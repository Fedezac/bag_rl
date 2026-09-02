"""Environment factories.

These are module-level (and only ever closed over via ``functools.partial``) so
that they stay picklable: every collector worker re-creates its own envs in its
own process rather than receiving a live env object.
"""

import copy
from functools import partial

from torchrl.envs import (
    Compose,
    DoubleToFloat,
    EnvCreator,
    ParallelEnv,
    SerialEnv,
    StepCounter,
    Transform,
    TransformedEnv,
)
from torchrl.envs.libs.gym import GymEnv
from torchrl.envs.utils import check_env_specs

from src.env.rewards import REWARD_SHAPERS


def build_reward_transform(spec):
    """Turn a shaping spec into a *fresh* transform instance, or ``None``.

    Accepts a name from :data:`REWARD_SHAPERS`, a transform class, a zero-arg
    callable, or an already-built transform. A fresh instance per env is not
    cosmetic: a ``Transform`` binds to the env it is attached to, so handing the
    same object to the several envs a ``SerialEnv``/``ParallelEnv`` builds from
    one factory would have them all share -- and fight over -- one parent.
    """
    if spec is None:
        return None
    if isinstance(spec, str):
        return REWARD_SHAPERS[spec]()
    if isinstance(spec, Transform):
        return copy.deepcopy(spec)
    return spec()


def make_single_env(env_name, device="cpu", custom_reward_functions=None, **gym_kwargs):
    """Build one gym env. Observations stay *raw* -- the networks normalise."""
    transforms = []
    shaper = build_reward_transform(custom_reward_functions)
    if shaper is not None:
        transforms.append(shaper)
    transforms += [DoubleToFloat(), StepCounter()]
    return TransformedEnv(
        GymEnv(env_name=env_name, device=device, **gym_kwargs),
        Compose(*transforms),
    )


def make_batched_env(
    env_name,
    num_envs,
    device="cpu",
    mode="serial",
    custom_reward_functions=None,
    **gym_kwargs,
):
    """Build a batch of ``num_envs`` gym instances stepped as one env."""
    creator = EnvCreator(
        partial(
            make_single_env,
            env_name,
            device=device,
            custom_reward_functions=custom_reward_functions,
            **gym_kwargs,
        )
    )
    if num_envs == 1:
        return creator()
    env_cls = ParallelEnv if mode == "parallel" else SerialEnv
    return env_cls(num_envs, creator, device=device)


def env_specs(env_name, custom_reward_functions=None, **gym_kwargs):
    """Read obs/action specs off a throwaway env."""
    env = make_single_env(
        env_name, custom_reward_functions=custom_reward_functions, **gym_kwargs
    )
    check_env_specs(env)
    specs = (env.observation_spec["observation"].shape[-1], env.action_spec)
    env.close()
    return specs


def warmup_obs_norm(env_name, obs_norm, num_steps=2000, custom_reward_functions=None):
    """Seed the observation stats from a short random rollout.

    Without this the first collected batch would be gathered under an identity
    normalisation, which for Humanoid means feeding the policy raw channels
    whose std spans six orders of magnitude.
    """
    env = make_single_env(env_name, custom_reward_functions=custom_reward_functions)
    collected = 0
    while collected < num_steps:
        rollout = env.rollout(min(500, num_steps - collected))
        obs_norm.update(rollout["observation"].to(obs_norm.mean.device))
        collected += rollout.shape[-1]
    env.close()
