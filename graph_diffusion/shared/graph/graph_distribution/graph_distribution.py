from jax import numpy as np, Array
import jax
import networkx as nx
import optax
from jax.experimental.checkify import check
from rich import print
import ipdb
import jax_dataclasses as jdc
from mate.jax import SFloat, SInt, typed, SBool, Key
from jaxtyping import Float, Bool, Int
from typing import Sequence
from jax.scipy.special import logit
import einops as e
import wandb

# from .geometric import to_dense
from jax import random

# from .data_batch import DataBatch
from .q import Q

# from ..graph import SimpleGraphDist

check = lambda x, y: None  # to be replaced with JAX's checkify.check function

NodeDistribution = Float[Array, "b n en"]
EdgeDistribution = Float[Array, "b n n ee"]
MaskType = Bool[Array, "b n"]
EdgeCountType = Int[Array, "b"]


def safe_div(a: Array, b: Array):
    mask = b == 0
    return np.where(mask, 0, a / np.where(mask, 1, b))


@typed
def pseudo_assert(condition: SBool):
    """
    When debugging NaNs, the following function will raise an exception
    """
    return np.where(condition, 0, np.nan)


@jdc.pytree_dataclass
class GraphDistribution(jdc.EnforcedAnnotationsMixin):
    nodes: NodeDistribution
    edges: EdgeDistribution
    edges_counts: EdgeCountType
    nodes_counts: EdgeCountType
    # mask: MaskType
    _created_internally: SBool  # trick to prevent users from creating this class directly

    @classmethod
    @typed
    def from_simple(cls, simple) -> "GraphDistribution":  #: SimpleGraphDist,
        return cls(
            nodes=simple.nodes,
            edges=simple.edges,
            edges_counts=simple.edges_counts,
            nodes_counts=simple.nodes_counts,
            _created_internally=True,
        )

    def node_mask(self):
        n_range = np.arange(self.n)
        mask_x = n_range[None].repeat(self.batch_size, 0) < self.nodes_counts[:, None]
        return mask_x

    def pseudo_assert(self):
        is_nodes_dist = np.array(is_dist(self.nodes))
        pseudo_assert(is_nodes_dist)
        is_edges_dist = np.array(is_dist(self.edges))
        pseudo_assert(is_edges_dist)

    @classmethod
    @typed
    def create(
        cls,
        nodes: NodeDistribution,
        edges: EdgeDistribution,
        edges_counts: EdgeCountType,
        nodes_counts: EdgeCountType,
        _safe: SBool = True,
    ) -> "GraphDistribution":
        # check(np.allclose(e, np.transpose(e, (0, 2, 1, 3))), "whoops")

        is_nodes_dist = np.logical_or(is_dist(nodes), (~_safe))
        # if not is_nodes_dist:
        #     ipdb.set_trace()
        pseudo_assert(is_nodes_dist)
        is_edges_dist = np.logical_or(is_dist(edges), (~_safe))
        # if not is_edges_dist:
        #     ipdb.set_trace()
        pseudo_assert(is_edges_dist)
        return cls(
            nodes=nodes,
            edges=edges,
            edges_counts=edges_counts,
            nodes_counts=nodes_counts,
            _created_internally=True,
        )

    @staticmethod
    def to_symmetric(edges: EdgeDistribution) -> EdgeDistribution:
        upper = e.rearrange(
            np.triu(np.ones((edges.shape[1], edges.shape[2]))), "n1 n2 -> 1 n1 n2 1"
        )
        return np.where(upper, edges, edges.transpose((0, 2, 1, 3)))

    def __str__(self):
        def arr_str(arr: Array):
            return f"Array({arr.shape}, max={arr.max():.2f}, min={arr.min():.2f}, mean={arr.mean():.2f}, dtype={arr.dtype}, is_dist={is_dist(arr)})"

        def self_arr(props: dict):
            return ",\n ".join(
                [
                    f"{key}: {arr_str(val)}"
                    for key, val in props.items()
                    if not key.startswith("_")
                ]
            )

        return f"{self.__class__.__name__}(\n {self_arr(self.__dict__)}\n)"

    @property
    def shape(self) -> dict[str, tuple[int, ...]]:
        return dict(
            nodes=self.nodes.shape, edges=self.edges.shape
        )  # , mask=self.mask.shape)

    @property
    def batch_size(self) -> int:
        return self.nodes.shape[0]

    @property
    def n(self) -> int:
        return self.nodes.shape[1]

    def __repr__(self):
        return self.__str__()

    def __mul__(
        self, other: "GraphDistribution | SFloat | SInt", _safe: SBool = True
    ) -> "GraphDistribution":
        if isinstance(other, Array):
            return self.__class__.create(
                nodes=self.nodes * other[..., None, None],
                edges=self.edges * other[..., None, None, None],
                nodes_counts=self.nodes_counts,
                edges_counts=self.edges_counts,
                _safe=_safe,
            )
        elif isinstance(other, GraphDistribution):
            return self.__class__.create(
                nodes=self.nodes * other.nodes,
                edges=self.edges * other.edges,
                nodes_counts=self.nodes_counts,
                edges_counts=self.edges_counts,
                _safe=_safe,
            )

    def __rmul__(
        self, other: "GraphDistribution | SFloat | SInt"
    ) -> "GraphDistribution":
        return self.__mul__(other)

    def __truediv__(
        self, other: "GraphDistribution | SFloat | SInt"
    ) -> "GraphDistribution":
        if isinstance(other, (SFloat, SInt)):
            return self.__class__.create(
                nodes=self.nodes / other,
                edges=self.edges / other,
                nodes_counts=self.nodes_counts,
                edges_counts=self.edges_counts,
            )
        elif isinstance(other, GraphDistribution):
            pseudo_assert((other.nodes > 0).all())
            pseudo_assert((other.edges > 0).all())
            new_nodes = self.nodes / other.nodes
            new_edges = self.edges / other.edges
            try:
                pseudo_assert(np.allclose(new_nodes.sum(-1), 1, 0.15))
            except:
                misfit = np.where(
                    ~np.allclose(new_nodes.sum(-1, keepdims=True), 1, 0.15)
                )
                ipdb.set_trace()
            pseudo_assert(np.allclose(new_edges.sum(-1), 1, 0.15))
            new_nodes = new_nodes / new_nodes.sum(-1, keepdims=True)
            new_edges = new_edges / new_edges.sum(-1, keepdims=True)
            return self.__class__.create(
                nodes=new_nodes,
                edges=new_edges,
                nodes_counts=self.nodes_counts,
                edges_counts=self.edges_counts,
            )

    def set(
        self,
        key: str,
        value: NodeDistribution | EdgeDistribution | MaskType,
    ) -> "GraphDistribution":
        """Sets the values of X, E, y."""
        new_vals = {
            k: v
            for k, v in (self.__dict__.copy() | {key: value}).items()
            if not k.startswith("__")
        }
        return GraphDistribution(**new_vals)

    def masks(self) -> tuple[MaskType, MaskType]:
        n_range = np.arange(self.n)
        mask_x = n_range[None].repeat(self.batch_size, 0) < self.nodes_counts[:, None]
        e_ranges = mask_x[:, :, None].repeat(self.n, -1)
        mask_e = e_ranges & e_ranges.transpose(0, 2, 1)
        return mask_x, mask_e

    def logprobs_at(self, one_hot: "GraphDistribution") -> SFloat:
        """Returns the probability of the given one-hot vector."""
        probs_at_vector = self.__mul__(one_hot, _safe=False)
        mask_x, mask_e = self.masks()
        nodes_logprob = np.log(probs_at_vector.nodes.sum(-1)) * mask_x
        edges_logprob = np.log(probs_at_vector.edges.sum(-1)) * mask_e
        res = nodes_logprob.sum(-1) + edges_logprob.sum((-1, -2))
        return res

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
        nodes = np.repeat(x_prob[None, None, :], batch_size, axis=0)
        edges = np.repeat(e_prob[None, None, None, :], batch_size, axis=0)
        mask = np.ones((batch_size, n), dtype=bool)
        uniform = cls(
            nodes=nodes,
            edges=edges,
            edges_counts=self.edges_counts,
            _created_internally=True,
        )  # mask=mask,
        return uniform.sample_one_hot(key)

    @typed
    def sample_one_hot(self, rng_key: Key) -> "GraphDistribution":
        """Sample features from multinomial distribution with given probabilities (probX, probE, proby)
        :param probX: bs, n, dx_out        node features
        :param probE: bs, n, n, de_out     edge features
        :param rng_key: random.PRNGKey     random key for JAX operations
        """
        bs, n, ne = self.nodes.shape
        _, _, _, ee = self.edges.shape
        epsilon = 1e-8
        # mask = self.mask
        prob_x = self.nodes
        prob_e = self.edges

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
        # symmetrize e (while preseving the fact that they're one hot encoded)
        return self.__class__.create(
            nodes=embedded_x,
            edges=embedded_e,
            nodes_counts=self.nodes_counts,
            edges_counts=self.edges_counts,
        )  # , mask=self.mask)

    # overrides te addition operator
    def __add__(self, other) -> "GraphDistribution":
        if isinstance(other, GraphDistribution):
            return self.__class__.create(
                nodes=self.nodes + other.nodes,
                edges=self.edges + other.edges,
                nodes_counts=self.nodes_counts,
                edges_counts=self.edges_counts,
            )
        else:
            return self.__class__.create(
                nodes=self.nodes + other,
                edges=self.edges + other,
                nodes_counts=self.nodes_counts,
                edges_counts=self.edges_counts,
            )

    __radd__ = __add__

    def __sub__(self, other) -> "GraphDistribution":
        if isinstance(other, GraphDistribution):
            return self.__class__.create(
                nodes=self.nodes - other.nodes,
                edges=self.edges - other.edges,
                nodes_counts=self.nodes_counts,
                edges_counts=self.edges_counts,
            )
        elif isinstance(other, (int, float, Array)):
            return self.__class__.create(
                nodes=self.nodes - other,
                edges=self.edges - other,
                nodes_counts=self.nodes_counts,
                edges_counts=self.edges_counts,
            )

    __rsub__ = __sub__

    @typed
    def sum(self) -> Array:
        node_mask, edge_mask = self.masks()
        nodes = (self.nodes * node_mask[..., None]).mean(-1)
        edges = (self.edges * edge_mask[..., None]).mean(-1)
        return np.einsum("bn->b", nodes) + np.einsum("bjk->b", edges)

    def __matmul__(self, q: Q) -> "GraphDistribution":
        x = self.nodes @ q.nodes
        e = self.edges @ q.edges[:, None]
        # x = x / x.sum(-1, keepdims=True)
        # e = e / e.sum(-1, keepdims=True)
        return GraphDistribution.create(
            nodes=x,
            edges=e,
            edges_counts=self.edges_counts,
            nodes_counts=self.nodes_counts,
        )  # mask=self.mask,

    def __pow__(self, n: int) -> "GraphDistribution":
        return self.__class__.create(
            nodes=self.nodes**n,
            edges=self.edges**n,
            nodes_counts=self.nodes_counts,
            edges_counts=self.edges_counts,
        )

    def mask(self) -> "GraphDistribution":
        nodes = np.where(
            (self.nodes.sum(-1) > 0)[..., None], self.nodes, 1 / self.nodes.shape[-1]
        )
        nodes = nodes / nodes.sum(-1, keepdims=True)
        edges = np.where(
            (self.edges.sum(-1) > 0)[..., None], self.edges, 1 / self.edges.shape[-1]
        )
        edges = edges / edges.sum(-1, keepdims=True)
        return GraphDistribution.create(
            nodes=nodes,
            edges=edges,
            nodes_counts=self.nodes_counts,
            edges_counts=self.edges_counts,
        )

    def argmax(self) -> "GraphDistribution":
        id_nodes = np.eye(self.nodes.shape[-1])
        id_edges = np.eye(self.edges.shape[-1])
        nodes = id_nodes[self.nodes.argmax(-1)]
        edges = id_edges[self.edges.argmax(-1)]
        return self.__class__.create(
            nodes=nodes,
            edges=edges,
            nodes_counts=self.nodes_counts,
            edges_counts=self.edges_counts,
        )

    # overrides the square bracket indexing
    def __getitem__(self, key: Int[Array, "n"] | slice) -> "GraphDistribution":
        return self.__class__.create(
            nodes=self.nodes[key],
            edges=self.edges[key],
            nodes_counts=self.nodes_counts[key],
            edges_counts=self.edges_counts[key],
        )  # , mask=self.mask[key])

    def repeat(self, n: int) -> "GraphDistribution":
        return self.__class__.create(
            nodes=np.repeat(self.nodes, n, axis=0),
            edges=np.repeat(self.edges, n, axis=0),
            nodes_counts=np.repeat(self.nodes_counts, n, axis=0),
            edges_counts=np.repeat(self.edges_counts, n, axis=0),
        )

    def __len__(self):
        return self.batch_size

    @classmethod
    @typed
    def concatenate(cls, items: Sequence["GraphDistribution"]) -> "GraphDistribution":
        return cls.create(
            nodes=np.concatenate(tuple(item.nodes for item in items)),
            edges=np.concatenate(tuple(item.edges for item in items)),
            nodes_counts=np.concatenate(tuple(item.nodes_counts for item in items)),
            edges_counts=np.concatenate(tuple(item.edges_counts for item in items)),
        )

    @staticmethod
    @typed
    def plot(
        rows: Sequence["GraphDistribution"],
        location: str | None = None,
        share_position_among_graphs: bool = False,
        title: str | None = None,
    ):
        location = (
            f"{location}.png"
            if location is not None and not location.endswith(".png")
            else location
        )
        original_len = len(rows[0])
        skip = (len(rows[0]) // 15) if len(rows[0]) > 15 else 1
        rows = [
            GraphDistribution.concatenate((row[::skip], row[np.array([-1])]))
            for row in rows
        ]

        # assert 1 <= len(rows) <= 2, "can only plot 1 or 2 rows"
        import matplotlib.pyplot as plt
        import numpy
        from tqdm import tqdm

        fig, axs = plt.subplots(
            len(rows),
            len(rows[0]),
            figsize=(100, 10),
        )
        if len(axs.shape) == 1:
            axs = axs[None, :]

        cmap_edge = plt.cm.viridis(np.linspace(0, 1, rows[0].edges.shape[-1] - 1))
        cmap_node = plt.cm.viridis(np.linspace(0, 1, rows[0].nodes.shape[-1] - 1))
        cmap_edge = numpy.concatenate([cmap_edge, numpy.zeros((1, 4))], axis=0)
        cmap_node = numpy.concatenate([cmap_node, numpy.zeros((1, 4))], axis=0)
        node_size = 10.0
        positions = [None] * len(rows[0])

        # x s.t len(row) / x = 15
        # => x = len(row) / 15

        for i, (row, ax_row) in enumerate(zip(rows, axs)):
            xs = row.nodes.argmax(-1)
            es = row.edges.argmax(-1)
            n_nodes = row.nodes_counts[0]
            for j in range(len(row)):
                ax = ax_row[j]
                x = xs[j]
                e = es[j]

                nodes = x[:n_nodes]
                edges_features = e[:n_nodes, :n_nodes]

                indices = np.indices(edges_features.shape).reshape(2, -1).T
                mask = edges_features.flatten() != 0
                edges = indices[mask]

                # c_values_edges = np.array([cmap_edge[i] for i in edges[:, -1]])

                G = nx.Graph()
                for i in range(n_nodes):
                    G.add_node(i)
                for i in range(edges.shape[0]):
                    G.add_edge(edges[i, 0].tolist(), edges[i, 1].tolist())

                if positions[j] is None:
                    if j > 0 and share_position_among_graphs:
                        positions[j] = positions[0]
                    else:
                        positions[j] = nx.spring_layout(G)  # positions for all nodes

                color_nodes = numpy.array([cmap_node[i] for i in nodes])
                color_edges = numpy.array(
                    [cmap_edge[edges_features[i, j]] for (i, j) in G.edges]
                )
                nx.draw(
                    G,
                    positions[j],
                    node_size=node_size,
                    edge_color=color_edges,
                    node_color=color_nodes,
                    ax=ax,
                )
                ax.set_title(f"t={j*skip if not j == len(row)-1 else original_len-1}")
                ax.set_aspect("equal")
                ax.set_axis_off()

        plt.subplots_adjust(left=0, right=1, bottom=0, top=1)

        if title is not None:
            plt.suptitle(title, fontsize=16)

        if location is None:
            plt.show()
        else:
            plt.savefig(location)
            plt.clf()
            plt.close()
            # wandb.log({"prediction": wandb.Image(location)})
        plt.clf()
        plt.close()

    @classmethod
    @typed
    def noise(
        cls,
        key: Key,
        num_node_features: int,
        num_edge_features: int,
        num_nodes: int,
        batch_size: int,
    ) -> "GraphDistribution":
        nodes_counts = np.array(
            [num_nodes] * batch_size
        )  # jax.random.randint(key, shape=(batch_size,),   int)
        # nodes_counts = jax.random.randint(
        #     key, shape=(batch_size,), minval=8, maxval=num_nodes + 1
        # )
        edges_counts = np.array(
            [10] * batch_size
        )  # jax.random.randint(key, shape=(batch_size,), minval=1, maxval=10)
        edges = jax.nn.softmax(
            random.normal(key, (batch_size, num_nodes, num_nodes, num_edge_features))
        )
        edges = GraphDistribution.to_symmetric(edges)
        # sets the diagonal to 0

        tmp = e.repeat(
            np.eye(edges.shape[-1])[0],
            "f -> b n1 n2 f",
            n1=num_nodes,
            n2=num_nodes,
            b=batch_size,
        )
        edges = np.where(np.eye(num_nodes)[None, :, :, None], tmp, edges)
        return cls.create(
            nodes=jax.nn.softmax(
                random.normal(key, (batch_size, num_nodes, num_node_features))
            ),
            edges=edges,
            nodes_counts=nodes_counts,
            edges_counts=edges_counts,
        )


def is_dist(x):
    return (
        (np.min(x) >= 0).all() & (np.max(x) <= 1).all() & np.allclose(np.sum(x, -1), 1)
    )
