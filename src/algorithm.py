"""The interface a :class:`~src.trainer.Trainer` drives."""

import abc

import torch

from src.env.utils import env_specs


class Algorithm(abc.ABC):
    """Base class for a learning algorithm.

    An ``Algorithm`` owns everything that touches
    parameters -- the networks, the losses, the optimiser, the normalisers --
    and knows how to turn one collected batch into one round of updates
    """

    def __init__(self, env_name, custom_reward_functions=None, device=None):
        self.env_name = env_name
        self.custom_reward_functions = custom_reward_functions
        self.device = (
            torch.device(device)
            if device is not None
            else (torch.device(0) if torch.cuda.is_available() else torch.device("cpu"))
        )
        self.obs_dim, self.action_spec = env_specs(
            env_name, custom_reward_functions=custom_reward_functions
        )
        self.action_dim = self.action_spec.shape[-1]

    # -- required -----------------------------------------------------------

    @property
    @abc.abstractmethod
    def policy(self):
        """The policy module, as a ``TensorDictModule``.."""

    @abc.abstractmethod
    def update(self, batch):
        """Consume one collected batch and return a ``{name: float}`` log dict."""

    # -- optional hooks -----------------------------------------------------

    def on_training_start(self, frames_per_batch, num_iterations):
        """Called once before the first batch, with the trainer's own sizing."""

    def on_iteration_end(self):
        """Called after each collector iteration"""

    def summary(self):
        """Algorithm-specific fields for the trainer's one-line RESULT."""
        return {}

    # -- checkpointing ------------------------------------------------------

    def checkpoint_objects(self):
        """``{name: object}`` of things carrying a ``state_dict``."""
        return {}

    def state_dict(self):
        return {k: v.state_dict() for k, v in self.checkpoint_objects().items()}

    def load_state_dict(self, state_dict):
        objects = self.checkpoint_objects()
        missing = set(objects) - set(state_dict)
        if missing:
            raise KeyError(f"checkpoint is missing {sorted(missing)}")
        for name, obj in objects.items():
            obj.load_state_dict(state_dict[name])
