import jax.numpy as np


def cosine_beta_schedule_discrete(timesteps, s=0.008):
    """Cosine schedule as proposed in https://openreview.net/forum?id=-NEXDKk8gZ."""
    steps = timesteps + 2
    x = np.linspace(0, steps, steps)

    alphas_cumprod = np.cos(0.5 * np.pi * ((x / steps) + s) / (1 + s)) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    alphas = alphas_cumprod[1:] / alphas_cumprod[:-1]
    betas = 1 - alphas
    return betas.squeeze()


def custom_beta_schedule_discrete(timesteps, average_num_nodes=50, s=0.008):
    """Cosine schedule as proposed in https://openreview.net/forum?id=-NEXDKk8gZ."""
    steps = timesteps + 2
    x = np.linspace(0, steps, steps)

    alphas_cumprod = np.cos(0.5 * np.pi * ((x / steps) + s) / (1 + s)) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    alphas = alphas_cumprod[1:] / alphas_cumprod[:-1]
    betas = 1 - alphas

    assert timesteps >= 100

    p = 4 / 5  # 1 - 1 / num_edge_classes
    num_edges = average_num_nodes * (average_num_nodes - 1) / 2

    # First 100 steps: only a few updates per graph
    updates_per_graph = 1.2
    beta_first = updates_per_graph / (p * num_edges)

    betas[betas < beta_first] = beta_first
    return np.array(betas)


class PredefinedNoiseScheduleDiscrete:
    def __init__(self, noise_schedule, timesteps):
        self.timesteps = timesteps

        if noise_schedule == "cosine":
            betas = cosine_beta_schedule_discrete(timesteps)
        elif noise_schedule == "custom":
            betas = custom_beta_schedule_discrete(timesteps)
        else:
            raise NotImplementedError(noise_schedule)

        self.betas = np.array(betas, dtype=np.float32)
        self.alphas = 1 - np.clip(self.betas, a_min=0, a_max=0.9999)

        log_alpha = np.log(self.alphas)
        log_alpha_bar = np.cumsum(log_alpha, axis=0)
        self.alphas_bar = np.exp(log_alpha_bar)

    def __call__(self, t_normalized=None, t_int=None):
        assert int(t_normalized is None) + int(t_int is None) == 1
        t_int = t_int if t_int is not None else np.round(t_normalized * self.timesteps)
        return self.betas[np.array(t_int, dtype=np.int32)]

    def get_alpha_bar(self, t_normalized=None, t_int=None):
        assert int(t_normalized is None) + int(t_int is None) == 1
        t_int = t_int if t_int is not None else np.round(t_normalized * self.timesteps)
        return self.alphas_bar[np.array(t_int, dtype=np.int32)]
