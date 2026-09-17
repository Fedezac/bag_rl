"""Gym env id -> :class:`ObservationLayout`."""

LAYOUTS = {}


def register_layout(layout, *env_names):
    """Bind ``layout`` to one or more gym env ids.

    Re-registering a different layout under a taken name raises: two robots
    silently sharing offsets is the failure this module exists to prevent.
    """
    for name in env_names:
        if LAYOUTS.get(name, layout) is not layout:
            raise KeyError(
                f"a different layout is already registered for {name!r}"
            )
        LAYOUTS[name] = layout
    return layout


def get_layout(env_name):
    """Look up a robot's layout, failing loudly on an unknown one"""
    try:
        return LAYOUTS[env_name]
    except KeyError:
        raise KeyError(
            f"no observation layout registered for {env_name!r}. "
            f"Known: {sorted(LAYOUTS)}. Add one under "
            f"src/env/layouts/robots/ -- verify the offsets against mujoco "
            f"state, do not guess them."
        ) from None
