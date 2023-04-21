import jax.numpy as np
from .q import Q
from abc import ABC, abstractmethod
from jax import Array
from jaxtyping import Float
from mate.jax import SFloat, SInt, typed
import jax_dataclasses as jdc
import ipdb
from .noise_schedule import NoiseSchedule
from .q import Q

'''
class TransitionModel(ABC):
    @abstractmethod
    def get_Qt(self, beta_t: Array) -> Q:
        """Returns one-step transition matrices for X and E, from step t - 1 to step t.
        Qt = (1 - beta_t) * I + beta_t / K

        beta_t: (bs)                         noise level between 0 and 1
        returns: qx (bs, dx, dx), qe (bs, de, de), qy (bs, dy, dy).
        """
        pass

    @abstractmethod
    def get_Qt_bar(self, alpha_bar_t: Array) -> Q:
        """Returns t-step transition matrices for X and E, from step 0 to step t.
        Qt = prod(1 - beta_t) * I + (1 - prod(1 - beta_t)) / K

        alpha_bar_t: (bs)         Product of the (1 - beta_t) for each time step from 0 to t.
        returns: qx (bs, dx, dx), qe (bs, de, de), qy (bs, dy, dy).
        """
        pass

    X_classes: int = 0
    E_classes: int = 0
    y_classes: int = 0


class DiscreteUniformTransition(TransitionModel):
    def __init__(self, x_classes: int, e_classes: int, y_classes: int):
        self.X_classes = x_classes
        self.E_classes = e_classes
        self.y_classes = y_classes
        self.u_x = np.ones((1, self.X_classes, self.X_classes))
        if self.X_classes > 0:
            self.u_x = self.u_x / self.X_classes

        self.u_e = np.ones((1, self.E_classes, self.E_classes))
        if self.E_classes > 0:
            self.u_e = self.u_e / self.E_classes

        self.u_y = np.ones((1, self.y_classes, self.y_classes))
        if self.y_classes > 0:
            self.u_y = self.u_y / self.y_classes


        

    def get_Qt(self, beta_t: SFloat):
        """Returns one-step transition matrices for X and E, from step t - 1 to step t.
        $Q_t = (1 - \beta_t) * I + \beta_t / K$

        beta_t: (bs)                         noise level between 0 and 1
        returns: qx (bs, dx, dx), qe (bs, de, de), qy (bs, dy, dy).
        """
        beta_t = beta_t[:, None]

        q_x = beta_t * self.u_x + (1 - beta_t) * np.eye(self.X_classes)[None]
        q_e = beta_t * self.u_e + (1 - beta_t) * np.eye(self.E_classes)[None]
        q_y = beta_t * self.u_y + (1 - beta_t) * np.eye(self.y_classes)[None]

        return Q(x=q_x, e=q_e, y=q_y)

    def get_Qt_bar(self, alpha_bar_t: SFloat):
        """Returns t-step transition matrices for X and E, from step 0 to step t.
        Qt = prod(1 - beta_t) * I + (1 - prod(1 - beta_t)) / K

        alpha_bar_t: (bs)         Product of the (1 - beta_t) for each time step from 0 to t.
        returns: qx (bs, dx, dx), qe (bs, de, de), qy (bs, dy, dy).
        """
        alpha_bar_t = alpha_bar_t[:, None]

        q_x = alpha_bar_t * np.eye(self.X_classes)[None] + (1 - alpha_bar_t) * self.u_x
        q_e = alpha_bar_t * np.eye(self.E_classes)[None] + (1 - alpha_bar_t) * self.u_e
        q_y = alpha_bar_t * np.eye(self.y_classes)[None] + (1 - alpha_bar_t) * self.u_y
        ipdb.set_trace()
        return Q(x=q_x, e=q_e, y=q_y)
'''


@jdc.pytree_dataclass
class TransitionModel(jdc.EnforcedAnnotationsMixin):
    x_marginals: Float[Array, "n"]
    e_marginals: Float[Array, "m"]
    y_classes: int
    diffusion_steps: int
    qs: Q
    q_bars: Q

    @classmethod
    @typed
    def create(
        cls,
        x_marginals: Float[Array, "n"],
        e_marginals: Float[Array, "m"],
        y_classes: int,
        diffusion_steps: int,
    ) -> "TransitionModel":
        x_classes = len(x_marginals)
        e_classes = len(e_marginals)
        u_x = np.broadcast_to(
            x_marginals[None, None], (1, x_classes, x_marginals.shape[0])
        )
        u_e = np.broadcast_to(
            e_marginals[None, None], (1, e_classes, e_marginals.shape[0])
        )
        u_y = np.ones((1, y_classes, y_classes)) / (y_classes if y_classes > 0 else 1)
        noise_schedule = NoiseSchedule.create("cosine", diffusion_steps)
        betas = noise_schedule.betas[:, None, None]
        q_xs = betas * u_x + (1 - betas) * np.eye(x_classes)[None]
        q_es = (
            betas * u_e
            + (1 - betas)
            * np.eye(
                e_classes,
            )[None]
        )
        q_ys = betas * u_y + (1 - betas) * np.eye(y_classes)[None]
        qs = Q(x=q_xs, e=q_es, y=q_ys)

        alpha_bars = noise_schedule.alphas_bar[:, None, None]
        q_bar_xs = alpha_bars * np.eye(x_classes)[None] + (1 - alpha_bars) * u_x
        q_bar_es = alpha_bars * np.eye(e_classes)[None] + (1 - alpha_bars) * u_e
        q_bar_ys = alpha_bars * np.eye(y_classes)[None] + (1 - alpha_bars) * u_y
        q_bars = Q(x=q_bar_xs, e=q_bar_es, y=q_bar_ys)

        return cls(
            x_marginals=x_marginals,
            e_marginals=e_marginals,
            y_classes=y_classes,
            diffusion_steps=diffusion_steps,
            qs=qs,
            q_bars=q_bars,
        )

    '''
    def get_Qt(self, beta_t: SFloat) -> Q:
        """Returns one-step transition matrices for X and E, from step t - 1 to step t.
        Qt = (1 - beta_t) * I + beta_t / K

        beta_t: (bs)                         noise level between 0 and 1
        returns: qx (bs, dx, dx), qe (bs, de, de), qy (bs, dy, dy)."""
        beta_t = beta_t[:, None]
        beta_t = beta_t

        q_x = beta_t * self.u_x + (1 - beta_t) * np.eye(self.X_classes)[None]
        q_e = (
            beta_t * self.u_e
            + (1 - beta_t)
            * np.eye(
                self.E_classes,
            )[None]
        )
        q_y = beta_t * self.u_y + (1 - beta_t) * np.eye(self.y_classes)[None]

        return Q(x=q_x, e=q_e, y=q_y)

    def get_Qt_bar(self, alpha_bar_t: Array):
        """Returns t-step transition matrices for X and E, from step 0 to step t.
        Qt = prod(1 - beta_t) * I + (1 - prod(1 - beta_t)) * K

        alpha_bar_t: (bs)         Product of the (1 - beta_t) for each time step from 0 to t.
        returns: qx (bs, dx, dx), qe (bs, de, de), qy (bs, dy, dy).
        """
        alpha_bar_t = alpha_bar_t[:, None]
        alpha_bar_t = alpha_bar_t

        q_x = alpha_bar_t * np.eye(self.X_classes)[None] + (1 - alpha_bar_t) * self.u_x
        q_e = alpha_bar_t * np.eye(self.E_classes)[None] + (1 - alpha_bar_t) * self.u_e
        q_y = alpha_bar_t * np.eye(self.y_classes)[None] + (1 - alpha_bar_t) * self.u_y

        return Q(x=q_x, e=q_e, y=q_y)
    '''
