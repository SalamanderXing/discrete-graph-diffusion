from jax import Array
from jax import numpy as np
import ipdb
import jax_dataclasses as jdc
from mate.jax import typed
from jaxtyping import Float, Bool
from .geometric import to_dense
from .data_batch import DataBatch
from .q import Q


@jdc.pytree_dataclass
class GraphDistribution(jdc.EnforcedAnnotationsMixin):
    x: Float[Array, "b n en"]
    e: Float[Array, "b n n ee"]
    y: Float[Array, "b ey"]
    mask: Bool[Array, "b"]

    collapse: bool = False

    @classmethod
    @typed
    def with_trivial_mask(
        cls,
        x: Float[Array, "b n en"],
        e: Float[Array, "b n n ee"],
        y: Float[Array, "b ey"],
    ):
        return cls(x=x, e=e, y=y, mask=np.ones(x.shape[:2], dtype=bool))

    def __post_init__(self):
        """
        assert (
            self.x.shape[0] == self.e.shape[0] == self.y.shape[0]
        ), f"{self.x.shape}, {self.e.shape}, {self.y.shape}"
        assert (
            self.x.shape[1] == self.e.shape[1] == self.e.shape[2]
        ), (f"{self.x.shape}, {self.e.shape}, {self.y.shape}", ipdb.set_trace())
        # assert self.x.shape[2] == self.y.shape[1], (f"{self.x.shape}, {self.y.shape}", ipdb.set_trace())
        """
        self.__mask(self.mask)

    def __str__(self):
        return f"Graph(X: {self.x.shape}, E: {self.e.shape}, y: {self.y.shape})"

    def __repr__(self):
        return self.__str__()

    def set_values(
        self,
        x: Array | None,
        e: Array | None,
        y: Array | None,
        mask: Array | None,
    ) -> "GraphDistribution":
        """Sets the values of X, E, y."""
        if x is None:
            x = self.x
        if e is None:
            e = self.e
        if y is None:
            y = self.y
        if mask is None:
            mask = self.mask
        return GraphDistribution(x=x, e=e, y=y, mask=mask)

    @classmethod
    def from_sparse(cls, data_batch: DataBatch):
        x, e, node_mask = to_dense(
            data_batch.x, data_batch.edge_index, data_batch.edge_attr, data_batch.batch
        )
        return cls(x=x, e=e, y=data_batch.y, mask=node_mask)

    @classmethod
    def from_sparse_torch(cls, data_batch_torch):
        data_batch = DataBatch.from_torch(data_batch_torch)
        x, e, node_mask = to_dense(
            data_batch.x, data_batch.edge_index, data_batch.edge_attr, data_batch.batch
        )
        return cls(x=x, e=e, y=data_batch.y, mask=node_mask)

    def type_as(self, x: Array) -> "GraphDistribution":
        """Changes the device and dtype of X, E, y."""
        dtype = x.dtype
        return GraphDistribution(
            x=self.x.astype(dtype),
            e=self.e.astype(dtype),
            y=self.y.astype(dtype),
            mask=self.mask,
        )

    def __mask(self, node_mask: Array):
        x_mask = np.expand_dims(node_mask, -1)  # bs, n, 1
        e_mask1 = np.expand_dims(x_mask, 2)  # bs, n, 1, 1
        e_mask2 = np.expand_dims(x_mask, 1)  # bs, 1, n, e_mask1
        if self.collapse:
            x = np.argmax(self.x, axis=-1)
            e = np.argmax(self.e, axis=-1)

            x[node_mask == 0] = -1
            e[(e_mask1 * e_mask2).squeeze(-1) == 0] = -1
        else:
            x = self.x * x_mask
            try:
                e = self.e * e_mask1 * e_mask2
            except:
                ipdb.set_trace()
            assert np.allclose(e, np.transpose(e, (0, 2, 1, 3)))
        object.__setattr__(self, "x", x)
        object.__setattr__(self, "e", e)

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
        return GraphDistribution(x=x, e=e, y=self.y, mask=self.mask)
