from torch import nn


class DenormalizedValueHead(nn.Module):
    """Critic trunk whose O(1) output is rescaled to true return magnitude.

    The trunk predicts a *normalised* value; this wrapper maps it back to the
    real reward scale so GAE bootstraps with correctly-scaled values. The
    critic loss is then taken in normalised space (see ``critic_loss``), so the
    trunk's final layer only ever has to produce O(1) numbers -- it never has
    to learn to emit the ~5000-magnitude returns a healthy Humanoid earns.
    """

    def __init__(self, trunk, value_norm):
        super().__init__()
        self.trunk = trunk
        self.value_norm = value_norm

    def forward(self, obs):
        return self.value_norm.denormalize(self.trunk(obs))
