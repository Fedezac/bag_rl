"""Per-robot observation layouts.

To add a robot: write ``robots/<robot>.py`` with an :class:`ObservationLayout`
and a :func:`register_layout` call, then import it from ``robots/__init__``.
Nothing in ``rewards`` or ``constraints`` needs to change.
"""

from src.env.layouts.layout import ObservationLayout
from src.env.layouts.registry import LAYOUTS, get_layout, register_layout

# Eager: a spawned collector worker re-imports this package from scratch and
# must find the registry already populated.
from src.env.layouts import robots  # noqa: E402,F401
from src.env.layouts.robots.ant import ANT_V5  # noqa: E402
from src.env.layouts.robots.halfcheetah import HALFCHEETAH_V5  # noqa: E402
from src.env.layouts.robots.hopper import HOPPER_V5  # noqa: E402
from src.env.layouts.robots.humanoid import HUMANOID_V5  # noqa: E402
from src.env.layouts.robots.kyon import KYON_V1  # noqa: E402
from src.env.layouts.robots.walker2d import WALKER2D_V5  # noqa: E402

__all__ = [
    "ANT_V5",
    "HALFCHEETAH_V5",
    "HOPPER_V5",
    "HUMANOID_V5",
    "KYON_V1",
    "LAYOUTS",
    "ObservationLayout",
    "WALKER2D_V5",
    "get_layout",
    "register_layout",
]
