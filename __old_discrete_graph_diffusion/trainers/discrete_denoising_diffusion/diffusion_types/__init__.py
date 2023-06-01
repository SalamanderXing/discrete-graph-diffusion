from jaxtyping import install_import_hook
from typing import Callable

with install_import_hook("diffusion_types", "typeguard.typechecked"):
    from .graph_distribution import (
        GraphDistribution,
        XDistType,
        EDistType,
        MaskType,
    )
    from .q import Q
    from .graph import Graph
    from .distribution import Distribution
    from .transition_model import TransitionModel

    # Forward =
