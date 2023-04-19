import jax.numpy as np
from .diffusion_types import Q
from abc import ABC, abstractmethod
from jax import Array
import ipdb


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

    def get_Qt(self, beta_t: Array):
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

    def get_Qt_bar(self, alpha_bar_t: Array):
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


class MarginalUniformTransition(TransitionModel):
    def __init__(self, x_marginals: Array, e_marginals: Array, y_classes: int):
        self.X_classes = len(x_marginals)
        self.E_classes = len(e_marginals)
        self.y_classes = y_classes
        self.x_marginals = x_marginals
        self.e_marginals = e_marginals

        self.u_x = np.broadcast_to(
            x_marginals[None, None], (1, self.X_classes, x_marginals.shape[0])
        )
        self.u_e = np.broadcast_to(
            e_marginals[None, None], (1, self.E_classes, e_marginals.shape[0])
        )

        self.u_y = np.ones((1, self.y_classes, self.y_classes))
        if self.y_classes > 0:
            self.u_y = self.u_y / self.y_classes

    def get_Qt(self, beta_t: Array):
        """Returns one-step transition matrices for X and E, from step t - 1 to step t.
        Qt = (1 - beta_t) * I + beta_t / K

        beta_t: (bs)                         noise level between 0 and 1
        returns: qx (bs, dx, dx), qe (bs, de, de), qy (bs, dy, dy)."""
        beta_t = beta_t[:, None]
        beta_t = beta_t
        self.u_x = self.u_x
        self.u_e = self.u_e
        self.u_y = self.u_y

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
        self.u_x = self.u_x
        self.u_e = self.u_e
        self.u_y = self.u_y

        q_x = alpha_bar_t * np.eye(self.X_classes)[None] + (1 - alpha_bar_t) * self.u_x
        q_e = alpha_bar_t * np.eye(self.E_classes)[None] + (1 - alpha_bar_t) * self.u_e
        q_y = alpha_bar_t * np.eye(self.y_classes)[None] + (1 - alpha_bar_t) * self.u_y

        return Q(x=q_x, e=q_e, y=q_y)
