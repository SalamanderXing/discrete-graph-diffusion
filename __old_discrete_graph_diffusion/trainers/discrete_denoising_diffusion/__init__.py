from jaxtyping import install_import_hook

with install_import_hook("discrete_denoising_diffusion", "typeguard.typechecked"):
    from .discrete_denoising_diffusion import run_model
    from .config import TrainingConfig
