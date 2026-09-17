"""The constraint-term interface."""


class ConstraintTerm:
    """One constraint. Returns a non-negative per-step cost.

    Plain object rather than a ``Transform``: several terms are summed by a
    single :class:`CostTransform`, so they never bind to an env themselves.
    """

    name = "constraint"

    def cost(self, layout, obs, action):
        raise NotImplementedError
