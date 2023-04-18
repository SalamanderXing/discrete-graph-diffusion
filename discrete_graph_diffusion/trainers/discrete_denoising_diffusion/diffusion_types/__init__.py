from jaxtyping import install_import_hook

with install_import_hook("diffusion_types", "typeguard.typechecked"):
    from .embedded_graph import EmbeddedGraph
    from .noisy_data import NoisyData
    from .q import Q
    from .graph import Graph
