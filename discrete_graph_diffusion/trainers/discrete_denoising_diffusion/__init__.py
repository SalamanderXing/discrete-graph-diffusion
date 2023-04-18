from jaxtyping import install_import_hook

with install_import_hook("discrete_denoising_diffusion", "typeguard.typechecked"):
    from .discrete_denoising_diffusion import train_model
    from .config import TrainingConfig 
