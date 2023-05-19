from jax import numpy as np, Array
import jax
from jax.experimental.checkify import check
import ipdb
import jax_dataclasses as jdc
from mate.jax import SFloat, SInt, typed, SBool, Key
from jaxtyping import Float, Bool, Int
from jax.scipy.special import logit

# from .geometric import to_dense
from jax import random

# from .data_batch import DataBatch
from .q import Q
from ....shared.graph import SimpleGraphDist

check = lambda x, y: None  # to be replaced with JAX's checkify.check function

XDistType = Float[Array, "b n en"]
EDistType = Float[Array, "b n n ee"]
MaskType = Bool[Array, "b n"]
EdgeCountType = Int[Array, "b"]


def sample(prob_x: XDistType, prob_e: EDistType, mask: MaskType, rng_key: Key):
    """Sample features from multinomial distribution with given probabilities (probX, probE, proby)
    :param probX: bs, n, dx_out        node features
    :param probE: bs, n, n, de_out     edge features
    :param rng_key: random.PRNGKey     random key for JAX operations
    """
    bs, n, ne = prob_x.shape
    _, _, _, ee = prob_e.shape
    epsilon = 1e-8
    # Noise X
    # The masked rows should define probability distributions as well
    # probX = probX.at[~node_mask].set(1 / probX.shape[-1])  # , probX)
    prob_x = np.where(
        (~mask)[:, :, None],
        1 / prob_x.shape[-1],
        prob_x,
    )
    prob_x = np.clip(prob_x, epsilon, 1 - epsilon)
    # Flatten the probability tensor to sample with categorical distribution
    prob_x = prob_x.reshape(bs * n, -1)  # (bs * n, dx_out)

    # Sample X
    rng_key, subkey = random.split(rng_key)
    x_t = random.categorical(subkey, logit(prob_x), axis=-1)  # (bs * n,)
    x_t = x_t.reshape(bs, n)  # (bs, n)

    # Noise E
    # The masked rows should define probability distributions as well
    inverse_edge_mask = ~(mask[:, None] * mask[:, :, None])
    diag_mask = np.eye(n)[None].astype(bool).repeat(bs, axis=0)
    prob_e = np.where(inverse_edge_mask[..., None], 1 / prob_e.shape[-1], prob_e)
    prob_e = np.where(diag_mask[..., None], 1 / prob_e.shape[-1], prob_e)

    prob_e = prob_e.reshape(bs * n * n, -1)  # (bs * n * n, de_out)
    prob_e = np.clip(prob_e, epsilon, 1 - epsilon)
    # Sample E
    rng_key, subkey = random.split(rng_key)
    e_t = random.categorical(subkey, logit(prob_e), axis=-1)  # (bs * n * n,)
    e_t = e_t.reshape(bs, n, n)  # (bs, n, n)
    e_t = np.triu(e_t, k=1)
    e_t = e_t + np.transpose(e_t, (0, 2, 1))

    # return Q(x=X_t, e=E_t, y=np.zeros((bs, 0), dtype=X_t.dtype))
    embedded_x = jax.nn.one_hot(x_t, num_classes=ne)
    embedded_e = jax.nn.one_hot(e_t, num_classes=ee)
    return GraphDistribution.masked(x=embedded_x, e=embedded_e, mask=mask)


def safe_div(a: Array, b: Array):
    mask = b == 0
    return np.where(mask, 0, a / np.where(mask, 1, b))


@jdc.pytree_dataclass
class GraphDistribution(jdc.EnforcedAnnotationsMixin):
    x: XDistType
    e: EDistType
    edges_counts: EdgeCountType
    nodes_counts: EdgeCountType
    # mask: MaskType
    _created_internally: SBool  # trick to prevent users from creating this class directly

    @classmethod
    @typed
    def from_simple(
        cls,
        simple: SimpleGraphDist,
    ) -> "GraphDistribution":
        return cls(
            x=simple.nodes,
            e=simple.edges,
            edges_counts=simple.edges_counts,
            nodes_counts=simple.nodes_counts,
            _created_internally=True,
        )

    @classmethod
    @typed
    def unmasked(
        cls,
        x: XDistType,
        e: EDistType,
        edges_counts: EdgeCountType,
        nodes_counts: EdgeCountType,
    ) -> "GraphDistribution":
        return cls(
            x=x,
            e=e,
            edges_counts=edges_counts,
            nodes_counts=nodes_counts,
            _created_internally=True,
        )

    @classmethod
    @typed
    def masked(
        cls,
        x: XDistType,
        e: EDistType,
        edges_counts: EdgeCountType,
        nodes_counts: EdgeCountType,
    ) -> "GraphDistribution":
        # x_mask = mask[..., None]  # bs, n, 1
        # e_mask1 = x_mask[:, :, None]  # bs, n, 1, 1
        # e_mask2 = x_mask[:, None]  # bs, 1, n, e_mask1
        # x = x * x_mask
        # e = e * e_mask1 * e_mask2
        check(np.allclose(e, np.transpose(e, (0, 2, 1, 3))), "whoops")
        return cls(
            x=x,
            e=e,
            edges_counts=edges_counts,
            nodes_counts=nodes_counts,
            _created_internally=True,
        )

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
        return dict(x=self.x.shape, e=self.e.shape)  # , mask=self.mask.shape)

    @property
    def batch_size(self) -> int:
        return self.x.shape[0]

    @property
    def n(self) -> int:
        return self.x.shape[1]

    def __repr__(self):
        return self.__str__()

    def __mul__(
        self, other: "GraphDistribution | SFloat | SInt"
    ) -> "GraphDistribution":
        if isinstance(other, (SFloat, SInt)):
            return GraphDistribution.masked(
                x=self.x * other,
                e=self.e * other,
                nodes_counts=self.nodes_counts,
                edges_counts=self.edges_counts,
            )
        elif isinstance(other, GraphDistribution):
            return GraphDistribution.masked(
                x=self.x * other.x,
                e=self.e * other.e,
                nodes_counts=self.nodes_counts,
                edges_counts=self.edges_counts,
            )

    # overrides the left multiplication
    def __rmul__(
        self, other: "GraphDistribution | SFloat | SInt"
    ) -> "GraphDistribution":
        return self.__mul__(other)

    def __truediv__(
        self, other: "GraphDistribution | SFloat | SInt"
    ) -> "GraphDistribution":
        if isinstance(other, (SFloat, SInt)):
            return GraphDistribution.masked(
                x=self.x / other,
                e=self.e / other,
                nodes_counts=self.nodes_counts,
                edges_counts=self.edges_counts,
            )
        else:
            return GraphDistribution.masked(
                x=safe_div(self.x, other.x),
                e=safe_div(self.e, other.e),
                nodes_counts=self.nodes_counts,
                edges_counts=self.edges_counts,
            )

    def set(
        self,
        key: str,
        value: XDistType | EDistType | MaskType,
    ) -> "GraphDistribution":
        """Sets the values of X, E, y."""
        new_vals = {
            k: v
            for k, v in (self.__dict__.copy() | {key: value}).items()
            if not k.startswith("__")
        }
        return GraphDistribution(**new_vals)

    def probs_at(self, one_hot: "GraphDistribution") -> SFloat:
        """Returns the probability of the given one-hot vector."""
        probs_at_vector = self * one_hot
        # sums over all the nodes and edges, but not over the batch
        return probs_at_vector.x.sum(axis=(1, 2)) + probs_at_vector.e.sum(
            axis=(1, 2, 3)
        )

    # @classmethod
    # def from_sparse(cls, data_batch: DataBatch):
    #     x, e, _ = to_dense(
    #         data_batch.x, data_batch.edge_index, data_batch.edge_attr, data_batch.batch
    #     )
    #     return cls(
    #         x=x,
    #         e=e,
    #         y=data_batch.y,
    #         # mask=node_mask,
    #         _created_internally=True,
    #     )

    # @classmethod
    # def from_sparse_torch(cls, data_batch_torch):
    #     data_batch = DataBatch.from_torch(data_batch_torch)
    #     x, e, node_mask = to_dense(
    #         data_batch.x, data_batch.edge_index, data_batch.edge_attr, data_batch.batch
    #     )
    #     return cls(
    #         x=x,
    #         e=e,
    #         # mask=node_mask,
    #         _created_internally=True,
    #     )

    @classmethod
    def sample_from_uniform(
        cls,
        batch_size: SInt,
        n: SInt,
        node_embedding_size: SInt,
        edge_embedding_size: SInt,
        key: Key,
    ) -> "GraphDistribution":
        x_prob = np.ones(node_embedding_size) / node_embedding_size
        e_prob = np.ones(edge_embedding_size) / edge_embedding_size
        x = np.repeat(x_prob[None, None, :], batch_size, axis=0)
        e = np.repeat(e_prob[None, None, None, :], batch_size, axis=0)
        mask = np.ones((batch_size, n), dtype=bool)
        uniform = cls(
            x=x, e=e, edges_counts=self.edges_counts, _created_internally=True
        )  # mask=mask,
        return uniform.sample_one_hot(key)

    @typed
    def sample_one_hot(self, rng_key: Key) -> "GraphDistribution":
        """Sample features from multinomial distribution with given probabilities (probX, probE, proby)
        :param probX: bs, n, dx_out        node features
        :param probE: bs, n, n, de_out     edge features
        :param rng_key: random.PRNGKey     random key for JAX operations
        """
        bs, n, ne = self.x.shape
        _, _, _, ee = self.e.shape
        epsilon = 1e-8
        # mask = self.mask
        prob_x = self.x
        prob_e = self.e

        # Noise X
        # The masked rows should define probability distributions as well
        # probX = probX.at[~node_mask].set(1 / probX.shape[-1])  # , probX)
        prob_x = np.clip(prob_x, epsilon, 1 - epsilon)
        # Flatten the probability tensor to sample with categorical distribution
        prob_x = prob_x.reshape(bs * n, ne)  # (bs * n, dx_out)

        # Sample X
        rng_key, subkey = random.split(rng_key)
        x_t = random.categorical(subkey, logit(prob_x), axis=-1)  # (bs * n,)
        x_t = x_t.reshape(bs, n)  # (bs, n)

        # Noise E
        # The masked rows should define probability distributions as well
        # inverse_edge_mask = ~(mask[:, None] * mask[:, :, None])
        # diag_mask = np.eye(n)[None].astype(bool).repeat(bs, axis=0)
        # prob_e = np.where(inverse_edge_mask[..., None], 1 / prob_e.shape[-1], prob_e)
        # prob_e = np.where(diag_mask[..., None], 1 / prob_e.shape[-1], prob_e)
        #
        # prob_e = prob_e.reshape(bs * n * n, -1)  # (bs * n * n, de_out)
        # prob_e = np.clip(prob_e, epsilon, 1 - epsilon)
        prob_e = prob_e.reshape(bs * n * n, ee)  # (bs * n * n, de_out)
        # # Sample E
        rng_key, subkey = random.split(rng_key)
        e_t = random.categorical(subkey, logit(prob_e), axis=-1)  # (bs * n * n,)
        e_t = e_t.reshape(bs, n, n)  # (bs, n, n)
        e_t = np.triu(e_t, k=1)
        e_t = e_t + np.transpose(e_t, (0, 2, 1))

        # return Q(x=X_t, e=E_t, y=np.zeros((bs, 0), dtype=X_t.dtype))
        embedded_x = jax.nn.one_hot(x_t, num_classes=ne)
        embedded_e = jax.nn.one_hot(e_t, num_classes=ee)
        return GraphDistribution.masked(
            x=embedded_x,
            e=embedded_e,
            nodes_counts=self.nodes_counts,
            edges_counts=self.edges_counts,
        )  # , mask=self.mask)

    # overrides the pipe operator, that takes another EmbeddedGraph as input. This concatenates the two graphs.
    # def __or__(self, other: "GraphDistribution") -> "GraphDistribution":
    #     x = np.concatenate((self.x, other.x), axis=2)
    #     e = np.concatenate((self.e, other.e), axis=3)
    #     y = np.hstack((self.y, other.y))
    #     return GraphDistribution(x=x, e=e, y=y, mask=self.mask)

    def __matmul__(self, q: Q) -> "GraphDistribution":
        x = self.x @ q.x
        e = self.e @ q.e[:, None]
        return GraphDistribution(
            x=x,
            e=e,
            edges_counts=self.edges_counts,
            nodes_counts=self.nodes_counts,
            _created_internally=True,
        )  # mask=self.mask,
