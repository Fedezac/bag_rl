import torch
from torch import nn


class StateIndependentNormalParams(nn.Module):
    """Emit ``(loc, scale)`` where the scale is a learned per-action parameter.

    Drop-in replacement for ``NormalParamExtractor``. The difference is where
    the exploration magnitude comes from: ``NormalParamExtractor`` reads it off
    the network output, so it varies with the observation, whereas here it is a
    single free parameter per action dimension.

    State-independent is what the original PPO paper uses for continous control
    """

    def __init__(self, action_dim, init_log_std=0.0, min_std=1e-4):
        super().__init__()
        self.log_std = nn.Parameter(torch.full((action_dim,), float(init_log_std)))
        self.min_std = min_std

    def forward(self, loc):
        scale = self.log_std.exp().clamp_min(self.min_std).expand(loc.shape)
        return loc, scale
