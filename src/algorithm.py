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

    def __init__(
        self, env_name, custom_reward_functions=None, constraints=None, device=None
    ):
        self.env_name = env_name
        self.custom_reward_functions = custom_reward_functions
        # Constraint spec (e.g. "tilt,height")
        self.constraints = constraints
        self.device = (
            torch.device(device)
            if device is not None
            else (torch.device(0) if torch.cuda.is_available() else torch.device("cpu"))
        )
        self.obs_dim, self.action_spec = env_specs(
            env_name,
            custom_reward_functions=custom_reward_functions,
            constraints=constraints,
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

    def load_pretrained(self, state_dict, skip=("optim", "scheduler")):
        """Load the networks from a checkpoint, leaving the optimiser fresh.

        Fine-tuning runs a new schedule against a new objective, so the saved
        optimiser state is not what the new run wants -- and its param groups
        differ outright whenever one of the two has a cost critic.

        Anything the checkpoint does not carry is left at its fresh
        initialisation rather than raising, so a policy trained without
        constraints can seed a constrained run. Returns
        ``(loaded, initialised)``.
        """
        objects = {
            k: v for k, v in self.checkpoint_objects().items() if k not in skip
        }
        loaded, fresh = [], []
        for name, obj in objects.items():
            if name in state_dict:
                obj.load_state_dict(state_dict[name])
                loaded.append(name)
            else:
                fresh.append(name)
        return loaded, fresh
