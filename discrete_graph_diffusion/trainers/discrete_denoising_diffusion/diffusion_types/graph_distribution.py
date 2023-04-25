from jax import numpy as np, Array
import jax
from jax.experimental.checkify import check
import ipdb
import jax_dataclasses as jdc
from mate.jax import SFloat, SInt, typed, SBool, Key
from jaxtyping import Float, Bool
from .geometric import to_dense
from jax import random
from .data_batch import DataBatch
from .q import Q

check = lambda x, y: None  # to be replaced with JAX's checkify.check function

XDistType = Float[Array, "b n en"]
EDistType = Float[Array, "b n n ee"]
YDistType = Float[Array, "b ey"]
MaskType = Bool[Array, "b n"]


@jdc.pytree_dataclass
class GraphDistribution(jdc.EnforcedAnnotationsMixin):
    x: XDistType
    e: EDistType
    y: YDistType
    mask: MaskType
    _created_internally: SBool  # trick to prevent users from creating this class directly

    @classmethod
    @typed
    def unmasked(
        cls,
        x: XDistType,
        e: EDistType,
        y: YDistType,
    ) -> "GraphDistribution":
        return cls(
            x=x,
            e=e,
            y=y,
            mask=np.ones(x.shape[:2], dtype=bool),
            _created_internally=True,
        )

    @classmethod
    @typed
    def masked(
        cls,
        x: XDistType,
        e: EDistType,
        y: YDistType,
        mask: MaskType,
    ) -> "GraphDistribution":
        x_mask = mask[..., None]  # bs, n, 1
        e_mask1 = x_mask[:, :, None]  # bs, n, 1, 1
        e_mask2 = x_mask[:, None]  # bs, 1, n, e_mask1
        x = x * x_mask
        e = e * e_mask1 * e_mask2
        check(np.allclose(e, np.transpose(e, (0, 2, 1, 3))), "whoops")
        return cls(x=x, e=e, y=y, mask=mask, _created_internally=True)

    def __str__(self):
        def arr_str(arr: Array):
            return f"array:({arr.dtype}, {arr.shape})"

        def self_arr(props: dict):
            return ", ".join(
                [
                    f"{key}: {arr_str(val)}"
                    for key, val in props.items()
                    if not key.startswith("_")
                ]
            )

        return f"GraphDistribution({self_arr(self.__dict__)})"

    @property
    def shape(self) -> dict[str, tuple[int, ...]]:
        return dict(
            x=self.x.shape, e=self.e.shape, y=self.y.shape, mask=self.mask.shape
        )

    @property
    def batch_size(self) -> int:
        return self.x.shape[0]

    def __repr__(self):
        return self.__str__()

    def __mul__(
        self, other: "GraphDistribution" | SFloat | SInt
    ) -> "GraphDistribution":
        if isinstance(other, (SFloat, SInt)):
            return GraphDistribution.masked(
                x=self.x * other,
                e=self.e * other,
                y=self.y * other,
                mask=self.mask,
            )
        else:
            return GraphDistribution.masked(
                x=self.x * other.x,
                e=self.e * other.e,
                y=self.y * other.y,
                mask=self.mask,
            )

    def __truediv__(
        self, other: "GraphDistribution" | SFloat | SInt
    ) -> "GraphDistribution":
        if isinstance(other, (SFloat, SInt)):
            return GraphDistribution.masked(
                x=self.x / other,
                e=self.e / other,
                y=self.y / other,
                mask=self.mask,
            )
        else:
            return GraphDistribution.masked(
                x=self.x / other.x,
                e=self.e / other.e,
                y=self.y / other.y,
                mask=self.mask,
            )

    def set(
        self,
        key: str,
        value: XDistType | EDistType | YDistType | MaskType,
    ) -> "GraphDistribution":
        """Sets the values of X, E, y."""
        new_vals = {
            k: v
            for k, v in (self.__dict__.copy() | {key: value}).items()
            if not k.startswith("__")
        }
        return GraphDistribution(**new_vals)

    @classmethod
    def from_sparse(cls, data_batch: DataBatch):
        x, e, node_mask = to_dense(
            data_batch.x, data_batch.edge_index, data_batch.edge_attr, data_batch.batch
        )
        return cls(x=x, e=e, y=data_batch.y, mask=node_mask, _created_internally=True)

    @classmethod
    def from_sparse_torch(cls, data_batch_torch):
        data_batch = DataBatch.from_torch(data_batch_torch)
        x, e, node_mask = to_dense(
            data_batch.x, data_batch.edge_index, data_batch.edge_attr, data_batch.batch
        )
        return cls(x=x, e=e, y=data_batch.y, mask=node_mask, _created_internally=True)

    @typed
    def astype(self, x: Array) -> "GraphDistribution":
        """Changes the device and dtype of X, E, y."""
        dtype = x.dtype
        return GraphDistribution(
            x=self.x.astype(dtype),
            e=self.e.astype(dtype),
            y=self.y.astype(dtype),
            mask=self.mask,
            _created_internally=True,
        )

    @typed
    def sample_one_hot(self, rng_key: Key) -> "GraphDistribution":
        """Sample features from multinomial distribution with given probabilities (probX, probE, proby)
        :param probX: bs, n, dx_out        node features
        :param probE: bs, n, n, de_out     edge features
        :param rng_key: random.PRNGKey     random key for JAX operations
        """
        bs, n, ne = self.x.shape
        _, _, _, ee = self.e.shape
        mask = self.mask
        prob_x = self.x
        prob_e = self.e
        # Noise X
        # The masked rows should define probability distributions as well
        # probX = probX.at[~node_mask].set(1 / probX.shape[-1])  # , probX)
        probX = np.where(
            (~mask)[:, :, None],
            1 / prob_x.shape[-1],
            prob_x,
        )

        # Flatten the probability tensor to sample with categorical distribution
        probX = probX.reshape(bs * n, -1)  # (bs * n, dx_out)

        # Sample X
        rng_key, subkey = random.split(rng_key)
        x_t = random.categorical(
            subkey, jax.scipy.special.logit(probX), axis=-1
        )  # (bs * n,)
        x_t = x_t.reshape(bs, n)  # (bs, n)

        # Noise E
        # The masked rows should define probability distributions as well
        inverse_edge_mask = ~(mask[:, None] * mask[:, :, None])
        diag_mask = np.eye(n)[None].astype(bool).repeat(bs, axis=0)
        prob_e = np.where(inverse_edge_mask[..., None], 1 / prob_e.shape[-1], prob_e)
        prob_e = np.where(diag_mask[..., None], 1 / prob_e.shape[-1], prob_e)

        probE = prob_e.reshape(bs * n * n, -1)  # (bs * n * n, de_out)

        # Sample E
        rng_key, subkey = random.split(rng_key)
        e_t = random.categorical(
            subkey, jax.scipy.special.logit(probE), axis=-1
        )  # (bs * n * n,)
        e_t = e_t.reshape(bs, n, n)  # (bs, n, n)
        e_t = np.triu(e_t, k=1)
        e_t = e_t + np.transpose(e_t, (0, 2, 1))

        # return Q(x=X_t, e=E_t, y=np.zeros((bs, 0), dtype=X_t.dtype))
        embedded_x = jax.nn.one_hot(x_t, num_classes=ne)
        embedded_e = jax.nn.one_hot(e_t, num_classes=ee)
        return GraphDistribution.masked(
            x=embedded_x, e=embedded_e, y=np.zeros((bs, 0)), mask=self.mask
        )

    # overrides the pipe operator, that takes another EmbeddedGraph as input. This concatenates the two graphs.
    # def __or__(self, other: "GraphDistribution") -> "GraphDistribution":
    #     x = np.concatenate((self.x, other.x), axis=2)
    #     e = np.concatenate((self.e, other.e), axis=3)
    #     y = np.hstack((self.y, other.y))
    #     return GraphDistribution(x=x, e=e, y=y, mask=self.mask)

    def __matmul__(self, q: Q) -> "GraphDistribution":
        x = self.x @ q.x
        e = self.e @ q.e[:, None]
        # TODO: why isn't y applied? It seems like it's like this in the original code.
        return GraphDistribution(
            x=x, e=e, y=self.y, mask=self.mask, _created_internally=True
        )
