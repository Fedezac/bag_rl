import torch
from torch import nn


class RunningObsNorm(nn.Module):
    """Running observation normaliser"""

    def __init__(self, dim, eps=1e-4, clip=10.0):
        super().__init__()
        self.register_buffer("mean", torch.zeros(dim))
        self.register_buffer("m2", torch.zeros(dim))
        self.register_buffer("count", torch.zeros(()))
        self.eps = eps
        self.clip = clip

    @torch.no_grad()
    def update(self, obs):
        """Fold a batch of raw observations in (Chan et al. parallel variance)."""
        obs = obs.reshape(-1, obs.shape[-1]).to(self.mean.dtype)
        n = obs.shape[0]
        if n == 0:
            return
        batch_mean = obs.mean(0)
        batch_var = obs.var(0, unbiased=False)
        delta = batch_mean - self.mean
        total = self.count + n
        self.mean.add_(delta * (n / total))
        self.m2.add_(batch_var * n + delta.pow(2) * (self.count * n / total))
        self.count.fill_(total)

    @property
    def std(self):
        var = self.m2 / self.count.clamp_min(1.0)
        return var.clamp_min(self.eps).sqrt()

    def forward(self, obs):
        if self.count.item() < 1:
            return obs
        return ((obs - self.mean) / self.std).clamp(-self.clip, self.clip)
