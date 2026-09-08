import math

import torch
import torch.nn.functional as F
from torch import nn


class StateIndependentNormalParams(nn.Module):
    """Emit ``(loc, scale)`` where the scale is a learned per-action parameter.

    Drop-in replacement for ``NormalParamExtractor``. The difference is where
    the exploration magnitude comes from: ``NormalParamExtractor`` reads it off
    the network output, so it varies with the observation, whereas here it is a
    single free parameter per action dimension.

    State-independent is what the original PPO paper uses for continuous
    control, and what Andrychowicz et al. (2021) measure as the better of the
    two on MuJoCo.

    ``parametrization`` selects how the free parameter maps to the scale.
    ``softplus`` is the paper's recommendation; ``exp`` is the classic
    log-std form, kept for comparison. Softplus is close to linear in the
    parameter near the initial value, so the scale decays more slowly than the
    exponential form -- which is what keeps exploration alive.
    """

    def __init__(
        self, action_dim, init_std=0.5, parametrization="softplus", min_std=1e-4
    ):
        super().__init__()
        if parametrization not in ("softplus", "exp"):
            raise ValueError(f"unknown std parametrization {parametrization!r}")
        self.parametrization = parametrization
        # Solve for the raw value that yields ``init_std`` under this mapping.
        raw = (
            math.log(math.expm1(init_std))
            if parametrization == "softplus"
            else math.log(init_std)
        )
        self.raw_std = nn.Parameter(torch.full((action_dim,), float(raw)))
        self.min_std = min_std

    def forward(self, loc):
        raw = (
            F.softplus(self.raw_std)
            if self.parametrization == "softplus"
            else self.raw_std.exp()
        )
        return loc, raw.clamp_min(self.min_std).expand(loc.shape)

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        # Checkpoints predating the softplus option store ``log_std``. Convert
        # rather than reject, so older policies stay loadable.
        legacy = state_dict.pop(prefix + "log_std", None)
        if legacy is not None:
            std = legacy.exp()
            state_dict[prefix + "raw_std"] = (
                std.expm1().log() if self.parametrization == "softplus" else legacy
            )
        return super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)
