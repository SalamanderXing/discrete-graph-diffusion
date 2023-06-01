from dataclasses import dataclass
from jax import Array
import jax
from jax import numpy as np
import ipdb
import jax_dataclasses as jdc
from jaxtyping import Float, Bool, Int, jaxtyped


def safe_div(a: Array, b: Array) -> Array:
    mask = b == 0
    return np.where(mask, 0, a / np.where(mask, 1, b))


@jdc.pytree_dataclass
class Q(jdc.EnforcedAnnotationsMixin):
    x: Float[Array, "b n"]
    e: Float[Array, "b n n"]

    # overrides the square bracket indexing
    def __getitem__(self, key: Int[Array, "n"]) -> "Q":
        return Q(x=self.x[key], e=self.e[key])

    @property
    def shape(self) -> dict[str, tuple[int, ...]]:
        return dict(x=self.x.shape, e=self.e.shape)

    def __truediv__(self, other: "Float[Array, ' '] | Q") -> "Q":
        if isinstance(other, Q):
            return Q(x=safe_div(self.x, other.x), e=safe_div(self.e, other.e))
        else:
            return Q(x=self.x / other, e=self.e / other)

    def __matmul__(self, other: "Q") -> "Q":
        return Q(x=self.x @ other.x, e=self.e @ other.e)

    def __len__(self) -> int:
        return self.x.shape[0]

    def cumulative_matmul(self) -> "Q":
        def f(a, b):
            result = a @ b
            return result, result

        res_x = jax.lax.scan(f, self.x, self.x[0][None])[1]
        res_e = jax.lax.scan(f, self.e, self.e[0][None])[1]
        return Q(x=res_x[0], e=res_e[0])
