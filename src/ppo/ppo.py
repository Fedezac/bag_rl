import copy

import torch
import torch.nn.functional as F
from tensordict.nn import TensorDictModule
from tensordict.nn.distributions import NormalParamExtractor
from torch import nn
from torchrl.data.replay_buffers import ReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.data.replay_buffers.storages import LazyTensorStorage
from torchrl.modules import (
    PopArtValueNorm,
    ProbabilisticActor,
    RunningValueNorm,
    TanhNormal,
    ValueOperator,
)
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value import GAE

from src.algorithm import Algorithm
from src.env.utils import warmup_obs_norm
from src.ppo.denormalized_value_head import DenormalizedValueHead
from src.ppo.running_observation_normalizer import RunningObsNorm
from src.ppo.state_independent_normal_params import StateIndependentNormalParams


def _activation(spec):
    """Build a fresh activation from a class or a template instance"""
    return spec() if isinstance(spec, type) else copy.deepcopy(spec)


class PPO(Algorithm):
    """Clipped-objective PPO with running observation and value normalisation."""

    def __init__(
        self,
        env_name,
        actor_hidden_layers=2,
        value_hidden_layers=2,
        hidden_layers_size=256,
        activation_function=nn.Tanh,
        custom_reward_functions=None,
        normalized_observation_clip=10.0,
        observations_warmup_steps=2000,
        value_target_normalizer="running",
        policy_std="dependent",
        lr=3e-4,
        num_epochs=10,
        sub_batch_size=256,
        clip_epsilon=0.2,
        gamma=0.99,
        lmbda=0.95,
        entropy_eps=1e-4,
        critic_coeff=1.0,
        max_grad_norm=1.0,
        in_keys=None,
        out_keys=None,
        device=None,
    ):
        super().__init__(
            env_name,
            custom_reward_functions=custom_reward_functions,
            device=device,
        )
        in_keys = list(in_keys) if in_keys is not None else ["observation"]
        out_keys = list(out_keys) if out_keys is not None else ["loc", "scale"]

        self.num_epochs = num_epochs
        self.sub_batch_size = sub_batch_size
        self.critic_coeff = critic_coeff
        self.max_grad_norm = max_grad_norm
        self.policy_std = policy_std

        # First layer common to actor and value nets
        self.obs_norm = RunningObsNorm(
            self.obs_dim, clip=normalized_observation_clip
        ).to(self.device)
        if observations_warmup_steps:
            warmup_obs_norm(
                env_name,
                self.obs_norm,
                observations_warmup_steps,
                custom_reward_functions=custom_reward_functions,
            )

        # Actor net
        actor_layers = [
            self.obs_norm,
            nn.Linear(self.obs_dim, hidden_layers_size),
            _activation(activation_function),
        ]
        for _ in range(actor_hidden_layers):
            actor_layers.extend(
                [
                    nn.Linear(hidden_layers_size, hidden_layers_size),
                    _activation(activation_function),
                ]
            )
        if policy_std == "independent":
            actor_layers.extend(
                [
                    nn.Linear(hidden_layers_size, self.action_dim),
                    StateIndependentNormalParams(self.action_dim),
                ]
            )
        else:
            actor_layers.extend(
                [
                    nn.Linear(hidden_layers_size, 2 * self.action_dim),
                    NormalParamExtractor(),
                ]
            )
        self.actor_net = nn.Sequential(*actor_layers).to(self.device)
        self.policy_module = ProbabilisticActor(
            module=TensorDictModule(self.actor_net, in_keys=in_keys, out_keys=out_keys),
            spec=self.action_spec,
            in_keys=["loc", "scale"],
            distribution_class=TanhNormal,
            distribution_kwargs={
                "low": self.action_spec.space.low,
                "high": self.action_spec.space.high,
            },
            return_log_prob=True,
        )

        # Value net. The trunk predicts a normalised value; the head rescales it
        # so everything downstream (GAE, logging) sees true-scale returns
        if value_target_normalizer == "running":
            self.value_norm = RunningValueNorm(shape=1, device=self.device)
        elif value_target_normalizer == "popart":
            self.value_norm = PopArtValueNorm(shape=1, device=self.device)
        else:
            self.value_norm = None

        value_layers = [
            self.obs_norm,
            nn.Linear(self.obs_dim, hidden_layers_size),
            _activation(activation_function),
        ]
        for _ in range(value_hidden_layers):
            value_layers.extend(
                [
                    nn.Linear(hidden_layers_size, hidden_layers_size),
                    _activation(activation_function),
                ]
            )
        value_layers.append(nn.Linear(hidden_layers_size, 1))
        value_trunk = nn.Sequential(*value_layers).to(self.device)
        self.value_net = (
            value_trunk
            if self.value_norm is None
            else DenormalizedValueHead(value_trunk, self.value_norm).to(self.device)
        )
        self.value_module = ValueOperator(module=self.value_net, in_keys=in_keys)

        # Losses
        self.advantage_module = GAE(
            gamma=gamma,
            lmbda=lmbda,
            value_network=self.value_module,
            average_gae=True,
            device=self.device,
        )
        self.loss_module = ClipPPOLoss(
            actor_network=self.policy_module,
            critic_network=self.value_module,
            clip_epsilon=clip_epsilon,
            entropy_bonus=bool(entropy_eps),
            entropy_coeff=entropy_eps,
            # With a value normaliser we compute the critic loss ourselves, in
            # normalised space
            critic_coeff=0.0 if self.value_norm is not None else 1.0,
            loss_critic_type="smooth_l1",
        )
        self.optim = torch.optim.Adam(self.loss_module.parameters(), lr)
        self.scheduler = None
        self.replay_buffer = None

    # -- Algorithm interface -------------------------------------------------

    @property
    def policy(self):
        return self.policy_module

    def on_training_start(self, frames_per_batch, num_iterations):
        """Size the replay buffer and the LR schedule against the data schedule."""
        self.replay_buffer = ReplayBuffer(
            # Kept on the training device so the inner loop never round-trips
            # the batch through host memory
            storage=LazyTensorStorage(max_size=frames_per_batch, device=self.device),
            sampler=SamplerWithoutReplacement(),
        )
        if num_iterations:
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optim, max(1, num_iterations), 0.0
            )

    def update(self, batch):
        if self.replay_buffer is None:
            # Standalone use, without a Trainer
            self.on_training_start(batch.numel(), num_iterations=None)

        frames_per_batch = batch.numel()
        logs = {}
        for epoch in range(self.num_epochs):
            # The advantage is recomputed each epoch because it depends on the
            # value network, which the inner loop is updating.
            with torch.no_grad():
                self.advantage_module(batch)

            if self.value_norm is not None and epoch == 0:
                # Fold this batch's targets into the running value stats once
                # per batch -- doing it every epoch would count the same
                # (highly correlated) returns ``num_epochs`` times over.
                self.value_norm.update(batch["value_target"])

            self.replay_buffer.extend(batch.reshape(-1))
            for _ in range(frames_per_batch // self.sub_batch_size):
                logs = self._optim_step(self.replay_buffer.sample(self.sub_batch_size))

        # Refresh the observation stats after the epochs
        self.obs_norm.update(batch["observation"])

        logs["lr"] = self.optim.param_groups[0]["lr"]
        if "scale" in batch.keys():
            # Exploration magnitude actually used during collection
            logs["policy_scale"] = batch["scale"].mean().item()
        if self.value_norm is not None:
            # If value normalisation is doing its job, value_scale tracks the
            # true return magnitude while critic loss stays O(1)
            ones = torch.ones(1, device=self.device)
            logs["value_scale"] = float(
                self.value_norm.denormalize(ones)
                - self.value_norm.denormalize(torch.zeros_like(ones))
            )
        return logs

    def _optim_step(self, subdata):
        loss_vals = self.loss_module(subdata)
        loss_value = loss_vals["loss_objective"] + loss_vals["loss_entropy"]
        logs = {
            "loss_objective": loss_vals["loss_objective"].item(),
            "loss_entropy": loss_vals["loss_entropy"].item(),
        }

        if self.value_norm is None:
            loss_value = loss_value + loss_vals["loss_critic"]
            logs["loss_critic"] = loss_vals["loss_critic"].item()
        else:
            # Critic regression in normalised space
            self.value_module(subdata)
            loss_critic = F.smooth_l1_loss(
                self.value_norm.normalize(subdata["state_value"]),
                self.value_norm.normalize(subdata["value_target"]),
            )
            loss_value = loss_value + self.critic_coeff * loss_critic
            logs["loss_critic"] = loss_critic.item()

        loss_value.backward()
        # Good practice to keep the gradient norm bounded.
        nn.utils.clip_grad_norm_(self.loss_module.parameters(), self.max_grad_norm)
        self.optim.step()
        self.optim.zero_grad(set_to_none=True)
        return logs

    def on_iteration_end(self):
        if self.scheduler is not None:
            self.scheduler.step()

    def summary(self):
        return {"policy_std": self.policy_std}

    def checkpoint_objects(self):
        objects = {
            "actor": self.actor_net,
            "value": self.value_net,
            "optim": self.optim,
        }
        if self.scheduler is not None:
            objects["scheduler"] = self.scheduler
        return objects
