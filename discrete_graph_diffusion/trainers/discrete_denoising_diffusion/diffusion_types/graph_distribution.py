from jax import Array
from jax import numpy as np
from jax.experimental.checkify import check
import ipdb
import jax_dataclasses as jdc
from mate.jax import typed, SBool
from jaxtyping import Float, Bool
from .geometric import to_dense
from .data_batch import DataBatch
from .q import Q

check = lambda x, y: None  # to be replaced with JAX's checkify.check function

XType = Float[Array, "b n en"]
EType = Float[Array, "b n n ee"]
YType = Float[Array, "b ey"]
MaskType = Bool[Array, "b n"]


@jdc.pytree_dataclass
class GraphDistribution(jdc.EnforcedAnnotationsMixin):
    x: XType
    e: EType
    y: YType
    mask: MaskType
    _created_internally: SBool  # trick to prevent users from creating this class directly

    @classmethod
    @typed
    def unmasked(
        cls,
        x: XType,
        e: EType,
        y: YType,
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
        x: XType,
        e: EType,
        y: YType,
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
        return dict(x=self.x.shape, e=self.e.shape, y=self.y.shape, mask=self.mask.shape)

    def __repr__(self):
        return self.__str__()

    def set(
        self,
        key: str,
        value: XType | EType | YType | MaskType,
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

    # overrides the pipe operator, that takes another EmbeddedGraph as input. This concatenates the two graphs.
    def __or__(self, other: "GraphDistribution") -> "GraphDistribution":
        x = np.concatenate((self.x, other.x), axis=2)
        e = np.concatenate((self.e, other.e), axis=3)
        y = np.hstack((self.y, other.y))
        return GraphDistribution(x=x, e=e, y=y, mask=self.mask)

    def __matmul__(self, q: Q) -> "GraphDistribution":
        x = self.x @ q.x
        e = self.e @ q.e[:, None]
        # TODO: why isn't y applied? It seems like it's like this in the original code.
        return GraphDistribution(
            x=x, e=e, y=self.y, mask=self.mask, _created_internally=True
        )
