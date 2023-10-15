from mate.jax import typed, SInt
from mate.types import Interface
import networkx as nx
from jaxtyping import Array, Float, Bool, Int, Shaped
from numpy import ndarray
import einops as e
from flax.struct import dataclass
from mate.jax import typed, Key, SInt, SBool
from mate.types import Interface
import jax
from jax import numpy as jnp
from collections.abc import Sequence
import ipdb

# from .graph import Masks, Graph, pseudo_assert

EncodedNodes = Float[Array, "b n"]
EncodedEdges = Float[Array, "b n n"]
Nodes = Int[Array, "b n"]
Edges = Int[Array, "b n n"]
Masks = Int[Array, "b"]


#
@dataclass
class EncodedGraph:
    nodes: EncodedNodes
    edges: EncodedEdges
    edges_counts: Masks
    nodes_counts: Masks
    edge_vocab_size: SInt
    node_vocab_size: SInt
    _internal: bool

    @classmethod
    @typed
    def create(
        cls,
        nodes: EncodedNodes,
        edges: EncodedEdges,
        edges_counts: Masks,
        nodes_counts: Masks,
        edge_vocab_size: SInt,
        node_vocab_size: SInt,
        safe: SBool = True,
    ) -> "EncodedGraph":
        valid_node_counts = (nodes_counts >= 0).all()
        edges_counts_assert = (edges_counts >= 0).all()
        edges_symmetric = (edges == edges.transpose((0, 2, 1))).all()
        try:
            pseudo_assert(valid_node_counts and edges_counts_assert and edges_symmetric)
        except:
            ipdb.set_trace()

        return cls(
            nodes=nodes,
            edges=edges,
            edges_counts=edges_counts,
            nodes_counts=nodes_counts,
            edge_vocab_size=edge_vocab_size,
            node_vocab_size=node_vocab_size,
            _internal=True,
        )

    @staticmethod
    def to_symmetric(edges: Shaped[Array, "b n n"]) -> Shaped[Array, "b n n"]:
        upper = jnp.triu(jnp.ones(edges.shape[1:]))[None]
        return jnp.where(upper, edges, edges.transpose((0, 2, 1)))

    @typed
    def __add__(self, other) -> "EncodedGraph":
        if isinstance(other, EncodedGraph):
            return EncodedGraph.create(
                nodes=self.nodes + other.nodes,
                edges=self.edges + other.edges,
                nodes_counts=self.nodes_counts,
                edges_counts=self.edges_counts,
                edge_vocab_size=self.edge_vocab_size,
                node_vocab_size=self.node_vocab_size,
            )
        else:
            return EncodedGraph.create(
                nodes=self.nodes + other,
                edges=self.edges + other,
                nodes_counts=self.nodes_counts,
                edges_counts=self.edges_counts,
                edge_vocab_size=self.edge_vocab_size,
                node_vocab_size=self.node_vocab_size,
            )

    @classmethod
    @typed
    def concatenate(cls, targets: Sequence["EncodedGraph"]) -> "EncodedGraph":
        """
        concatenates along the first (batch) dimension
        """
        return cls.create(
            nodes=jnp.concatenate([t.nodes for t in targets], axis=0),
            edges=jnp.concatenate([t.edges for t in targets], axis=0),
            nodes_counts=jnp.concatenate([t.nodes_counts for t in targets], axis=0),
            edges_counts=jnp.concatenate([t.edges_counts for t in targets], axis=0),
            edge_vocab_size=targets[0].edge_vocab_size,
            node_vocab_size=targets[0].node_vocab_size,
        )

    @typed
    def __getitem__(self, key: Int[Array, "s"]) -> "EncodedGraph":
        """
        slices along the first (batch) dimension. Input must be a 1D JAX array because a graph must have a batch dimension.
        """
        return EncodedGraph.create(
            nodes=self.nodes[key],
            edges=self.edges[key],
            nodes_counts=self.nodes_counts[key],
            edges_counts=self.edges_counts[key],
            edge_vocab_size=self.edge_vocab_size,
            node_vocab_size=self.node_vocab_size,
        )

    @property
    def batch_size(self) -> Int:
        return self.nodes.shape[0]

    @typed
    def __mul__(self, other) -> "EncodedGraph":
        if isinstance(other, EncodedGraph):
            return EncodedGraph.create(
                nodes=self.nodes * other.nodes,
                edges=self.edges * other.edges,
                nodes_counts=self.nodes_counts,
                edges_counts=self.edges_counts,
                edge_vocab_size=self.edge_vocab_size,
                node_vocab_size=self.node_vocab_size,
            )
        else:
            try:
                return EncodedGraph.create(
                    nodes=self.nodes * e.rearrange(other, "b -> b 1"),
                    # nodes=self.nodes * other,
                    edges=self.edges * e.rearrange(other, "b -> b 1 1"),
                    # edges=self.edges * other,
                    nodes_counts=self.nodes_counts,
                    edges_counts=self.edges_counts,
                    edge_vocab_size=self.edge_vocab_size,
                    node_vocab_size=self.node_vocab_size,
                )
            except:
                ipdb.set_trace()

    @typed
    def __sub__(self, other, safe: bool = True) -> "EncodedGraph":
        if isinstance(other, EncodedGraph):
            return EncodedGraph.create(
                nodes=self.nodes - other.nodes,
                edges=self.edges - other.edges,
                nodes_counts=self.nodes_counts,
                edges_counts=self.edges_counts,
                edge_vocab_size=self.edge_vocab_size,
                node_vocab_size=self.node_vocab_size,
                safe=safe,
            )
        else:
            return EncodedGraph.create(
                nodes=self.nodes - other,
                edges=self.edges - other,
                nodes_counts=self.nodes_counts,
                edges_counts=self.edges_counts,
                edge_vocab_size=self.edge_vocab_size,
                node_vocab_size=self.node_vocab_size,
                safe=safe,
            )

    @typed
    def __truediv__(self, other) -> "EncodedGraph":
        if isinstance(other, EncodedGraph):
            return EncodedGraph.create(
                nodes=self.nodes / other.nodes,
                edges=self.edges / other.edges,
                nodes_counts=self.nodes_counts,
                edges_counts=self.edges_counts,
                edge_vocab_size=self.edge_vocab_size,
                node_vocab_size=self.node_vocab_size,
            )
        else:
            return EncodedGraph.create(
                nodes=self.nodes / other,
                edges=self.edges / other,
                nodes_counts=self.nodes_counts,
                edges_counts=self.edges_counts,
                edge_vocab_size=self.edge_vocab_size,
                node_vocab_size=self.node_vocab_size,
            )

    @typed
    def __pow__(self, other) -> "EncodedGraph":
        if isinstance(other, EncodedGraph):
            return EncodedGraph.create(
                nodes=self.nodes**other.nodes,
                edges=self.edges**other.edges,
                nodes_counts=self.nodes_counts,
                edges_counts=self.edges_counts,
                edge_vocab_size=self.edge_vocab_size,
                node_vocab_size=self.node_vocab_size,
            )
        else:
            return EncodedGraph.create(
                nodes=self.nodes**other,
                edges=self.edges**other,
                nodes_counts=self.nodes_counts,
                edges_counts=self.edges_counts,
                edge_vocab_size=self.edge_vocab_size,
                node_vocab_size=self.node_vocab_size,
            )

    @classmethod
    @typed
    def noise(
        cls, b: int, n: int, node_vocab_size: int, edge_vocab_size: int, key: Key
    ):
        edge_noise = jax.random.normal(key, (b, n, n))
        edge_noise = (edge_noise + edge_noise.transpose((0, 2, 1))) / 2
        nodes = jax.random.normal(key, (b, n))
        nodes_counts = jax.random.randint(key, (b,), 1, n)
        edges_counts = jax.random.randint(key, (b,), 1, n * (n - 1))
        return EncodedGraph.create(
            nodes=nodes,
            edges=edge_noise,
            nodes_counts=nodes_counts,
            edges_counts=edges_counts,
            edge_vocab_size=edge_vocab_size,
            node_vocab_size=node_vocab_size,
        )

    @typed
    def noise_like(self, key: Key) -> "EncodedGraph":
        edge_noise = jax.random.normal(key, self.edges.shape)
        edge_noise = (edge_noise + edge_noise.transpose((0, 2, 1))) / 2
        return EncodedGraph.create(
            nodes=jax.random.normal(key, self.nodes.shape),
            edges=edge_noise,
            nodes_counts=self.nodes_counts,
            edges_counts=self.edges_counts,
            edge_vocab_size=self.edge_vocab_size,
            node_vocab_size=self.node_vocab_size,
        )

    @typed
    def node_mask(self) -> Bool[Array, "b n"]:
        arange = e.repeat(
            jnp.arange(self.nodes.shape[1]), "a -> b a", b=self.nodes.shape[0]
        )
        return arange < e.rearrange(self.nodes_counts, "b -> b 1")

    @typed
    def edge_mask(self):
        mask = (
            jnp.broadcast_to(
                jnp.arange(self.edges.shape[1])[None, :, None],
                self.edges.shape,
            )
            < self.edges_counts[:, None, None]
        )
        return mask

    def sum(self) -> Float[Array, "b"]:
        nodes = self.nodes * self.nodes_mask
        edges = self.edges * self.edges_mask
        return jnp.einsum("bn->b", nodes) + jnp.einsum("bjk->b", edges)

    __radd__ = __add__
    __rmul__ = __mul__
    __rsub__ = __sub__
    __rtruediv__ = __truediv__

    # @typed
    # def __call__(self, g_0: Float[Array, "b"]):
    #     # For initialization purposes
    #     h = self.encode()
    #     return h.decode(g_0)
    #
    @typed
    def __encode_single(self, x, vocab_size: SInt):
        x = x.round()
        return 2 * ((x + 0.5) / vocab_size) - 1

    @typed
    def decode(self, g_0) -> "EncodedGraph":
        z = self
        zn = z.nodes
        ze = z.edges
        return EncodedGraph.create(
            nodes=self.__decode_node(zn, g_0),
            edges=self.__decode_edge(ze, g_0),
            nodes_counts=z.nodes_counts,
            edges_counts=z.edges_counts,
            edge_vocab_size=z.edge_vocab_size,
            node_vocab_size=z.node_vocab_size,
        )

    @typed
    def decode_no_probs(self) -> "Graph":
        """Decode graph without probabilities. Simply scale back to original domain and then rounds"""
        unscaled_nodes = jnp.clip(self.nodes, 0, 1) * self.node_vocab_size
        unscaled_edges = jnp.clip(self.edges, 0, 1) * self.edge_vocab_size
        # unscaled_nodes = ((jnp.clip(self.nodes, -1, 1) + 1) / 2) * self.node_vocab_size
        # unscaled_edges = ((jnp.clip(self.edges, -1, 1) + 1) / 2) * self.edge_vocab_size
        # reordering = jnp.array([2, 0, 1, 3, 4])
        # unscaled_nodes = reordering[unscaled_nodes.round().astype("int32")]
        # unscaled_edges = reordering[unscaled_edges.round().astype("int32")]
        id = jnp.eye(unscaled_edges.shape[1])[None]
        unscaled_edges = jnp.where(id, 0, unscaled_edges)
        return Graph.create(
            nodes=unscaled_nodes.round().astype("int32"),
            edges=unscaled_edges.round().astype("int32"),
            nodes_counts=self.nodes_counts,
            edges_counts=self.edges_counts,
            edge_vocab_size=self.edge_vocab_size,
            node_vocab_size=self.node_vocab_size,
        )

    # @typed
    # def __logprob_nodes(
    #     self, x: EncodedNodes, z: EncodedNodes, g_0
    # ) -> Float[Array, "b"]:
    #     # x = x.round().astype("int32")
    #     # unscale x
    #     # x_unscaled = ((x + 0.5) * self.node_vocab_size).round().astype("int32")
    #     x_onehot = jax.nn.one_hot(x, self.node_vocab_size)
    #     logprobs = self.__decode_node(z, g_0)
    #     logprob = jnp.sum(x_onehot * logprobs, axis=(1, 2))
    #     # jax.debug.breakpoint()
    #     return logprob
    #
    # @typed
    # def __logprob_edges(
    #     self, x: EncodedEdges, z: EncodedEdges, g_0
    # ) -> Float[Array, "b"]:
    #     # x = x.round().astype("int32")
    #     # x_unscaled = ((x + 0.5) * self.node_vocab_size).round().astype("int32")
    #     x_onehot = jax.nn.one_hot(x, self.edge_vocab_size)
    #     logprobs = self.__decode_edge(z, g_0)
    #     logprob = jnp.sum(x_onehot * logprobs, axis=(1, 2, 3))
    #     return logprob
    #
    # @typed
    # def logprob(self, z: "EncodedGraph", g_0) -> Float[Array, "b"]:
    #     x = self
    #     xn = x.nodes
    #     xg = x.edges
    #     zn = z.nodes
    #     zg = z.edges
    #     logprob_nodes = self.__logprob_nodes(xn, zn, g_0)
    #     logprob_edges = self.__logprob_edges(xg, zg, g_0)
    #     return logprob_nodes + logprob_edges


@typed
def pseudo_assert(condition: SBool):
    """
    When debugging NaNs, the following function will raise an exception
    """
    return jnp.where(condition, 0, jnp.nan)


@dataclass
class Graph(metaclass=Interface):
    nodes: Nodes
    edges: Edges
    edges_counts: Masks
    nodes_counts: Masks
    edge_vocab_size: SInt
    node_vocab_size: SInt
    _internal: bool

    @classmethod
    @typed
    def create(
        cls,
        nodes: Nodes,
        edges: Edges,
        edges_counts: Masks,
        nodes_counts: Masks,
        edge_vocab_size: SInt,
        node_vocab_size: SInt,
        safe: bool = True,
    ) -> "Graph":
        pseudo_assert(
            ((nodes_counts >= 0).all() and (nodes[-1] <= node_vocab_size).all())
            or (not safe)
        )
        pseudo_assert(
            ((edges_counts >= 0).all() and (edges[-1] <= edge_vocab_size).all())
            or (not safe)
        )
        # makes sure that the edges contain no self loops
        pseudo_assert((jnp.trace(jnp.abs(edges), axis1=1, axis2=2) == 0).all())

        # makes sure that the edges are symmetric
        pseudo_assert((edges == edges.transpose((0, 2, 1))).all())

        return cls(
            nodes=nodes,
            edges=edges,
            edges_counts=edges_counts,
            nodes_counts=nodes_counts,
            edge_vocab_size=edge_vocab_size,
            node_vocab_size=node_vocab_size,
            _internal=True,
        )

    @classmethod
    @typed
    def concatenate(cls, targets: Sequence["Graph"]) -> "Graph":
        """
        concatenates along the first (batch) dimension
        """
        return cls.create(
            nodes=jnp.concatenate([t.nodes for t in targets], axis=0),
            edges=jnp.concatenate([t.edges for t in targets], axis=0),
            nodes_counts=jnp.concatenate([t.nodes_counts for t in targets], axis=0),
            edges_counts=jnp.concatenate([t.edges_counts for t in targets], axis=0),
            edge_vocab_size=targets[0].edge_vocab_size,
            node_vocab_size=targets[0].node_vocab_size,
        )

    @typed
    def __getitem__(self, key: Int[Array, "s"]) -> "Graph":
        """
        slices along the first (batch) dimension. Input must be a 1D JAX array because a graph must have a batch dimension.
        """
        return Graph.create(
            nodes=self.nodes[key],
            edges=self.edges[key],
            nodes_counts=self.nodes_counts[key],
            edges_counts=self.edges_counts[key],
            edge_vocab_size=self.edge_vocab_size,
            node_vocab_size=self.node_vocab_size,
        )

    @typed
    def repeat(self, m: int) -> "Graph":
        return Graph.create(
            nodes=jnp.repeat(self.nodes, m, axis=0),
            edges=jnp.repeat(self.edges, m, axis=0),
            nodes_counts=jnp.repeat(self.nodes_counts, m, axis=0),
            edges_counts=jnp.repeat(self.edges_counts, m, axis=0),
            edge_vocab_size=self.edge_vocab_size,
            node_vocab_size=self.node_vocab_size,
        )

    @staticmethod
    @typed
    def plot(
        rows: Sequence["Graph"],
        location: str | None = None,
        share_position_among_graphs: bool = False,
    ):
        import numpy as np

        # assert 1 <= len(rows) <= 2, "can only plot 1 or 2 rows"
        import matplotlib.pyplot as plt
        import numpy
        from tqdm import tqdm

        _, axs = plt.subplots(
            3,
            len(rows[0]),
            figsize=(100, 10),
        )

        cmap_edge = plt.cm.viridis(np.linspace(0, 1, rows[0].edges.shape[-1] - 1))
        cmap_node = plt.cm.viridis(np.linspace(0, 1, rows[0].nodes.shape[-1] - 1))
        cmap_edge = np.concatenate([cmap_edge, np.zeros((1, 4))], axis=0)
        cmap_node = np.concatenate([cmap_node, np.zeros((1, 4))], axis=0)
        node_size = 10.0
        positions = [None] * len(rows[0])

        for i, (row, ax_row) in enumerate(zip(rows, axs)):
            xs = row.nodes
            es = row.edges
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
                ax.set_title(f"t={j}")
                ax.set_aspect("equal")

        plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
        if location is None:
            plt.show()
        else:
            plt.savefig(location)

    @property
    def shape(self) -> dict[str, tuple[int, ...]]:
        return {k: v.shape for k, v in self.__dict__.items() if not k.startswith("_")}

    @property
    def batch_size(self) -> Int:
        return self.nodes.shape[0]

    @typed
    def noise_like(self, key: Key) -> "Graph":
        edge_noise = jax.random.normal(key, self.edges.shape)
        edge_noise = (edge_noise + edge_noise.transpose((0, 2, 1))) / 2
        return Graph.create(
            nodes=jax.random.normal(key, self.nodes.shape),
            edges=edge_noise,
            nodes_counts=self.nodes_counts,
            edges_counts=self.edges_counts,
            edge_vocab_size=self.edge_vocab_size,
            node_vocab_size=self.node_vocab_size,
        )

    @typed
    def node_mask(self) -> Bool[Array, "b n"]:
        arange = e.repeat(
            jnp.arange(self.nodes.shape[1]), "a -> b a", b=self.nodes.shape[0]
        )
        return arange < e.rearrange(self.nodes_counts, "b -> b 1")

    def __len__(self):
        return self.nodes.shape[0]

    @typed
    def edge_mask(self):
        mask = (
            jnp.broadcast_to(
                jnp.arange(self.edges.shape[1])[None, :, None],
                self.edges.shape,
            )
            < self.edges_counts[:, None, None]
        )
        return mask

    def sum(self) -> Float[Array, "b"]:
        nodes = self.nodes * self.node_mask()
        edges = self.edges * self.edge_mask()
        return jnp.einsum("bn->b", nodes) + jnp.einsum("bjk->b", edges)

    @typed
    def __decode_edge(
        self, z: EncodedEdges, g_0: Float[Array, "k"]
    ) -> Float[Array, "b n n edge_vocab_size"]:
        # Logits are exact if there are no dependencies between dimensions of x
        x_vals_raw = jnp.arange(0, self.edge_vocab_size)[None, None, None]
        # x_vals = jnp.repeat(x_vals, 3, 1)
        x_vals = self.__encode_single(x_vals_raw, self.edge_vocab_size)
        inv_stdev = jnp.exp(-0.5 * g_0[..., None])
        logits = -0.5 * jnp.square((z[..., None] - x_vals) * inv_stdev)
        logprobs = jax.nn.log_softmax(logits)
        return logprobs

    @typed
    def __decode_node(
        self, z: EncodedNodes, g_0: Float[Array, "k"]
    ) -> Float[Array, "b n node_vocab_size"]:
        # z.shape = (b, n , 1)
        # x_vals_raw.shape (b, n, vocab_size)

        # Logits are exact if there are no dependencies between dimensions of x
        x_vals_raw = jnp.arange(0, self.node_vocab_size)[None, None]
        # x_vals = jnp.repeat(x_vals, 3, 1)
        x_vals = self.__encode_single(x_vals_raw, self.node_vocab_size)
        # .transpose([1, 0])[
        #     None, None, None, :, :
        # ]
        inv_stdev = jnp.exp(-0.5 * g_0[..., None])
        logits = -0.5 * jnp.square((z[..., None] - x_vals) * inv_stdev)
        logprobs = jax.nn.log_softmax(logits)
        # jax.debug.breakpoint()
        return logprobs

    # @typed
    # def __encode_single(self, x, vocab_size: SInt):
    #     reordering = jnp.array([1, 2, 0, 3, 4])
    #     x_reorder = reordering[x]
    #     return 2 * (x_reorder / vocab_size) - 1
    @typed
    def __encode_single(self, x: Array, vocab_size: SInt):
        return x / vocab_size

    @typed
    def encode(self) -> "EncodedGraph":
        nodes = self.nodes
        edges = self.edges
        # This transforms x from discrete values (0, 1, ...)
        # to the domain (-1,1).
        # Rounding here just a safeguard to ensure the input is discrete
        # (although typically, x is a discrete variable such as uint8)
        return EncodedGraph.create(
            nodes=self.__encode_single(nodes, self.node_vocab_size),
            edges=self.__encode_single(edges, self.edge_vocab_size),
            nodes_counts=self.nodes_counts,
            edges_counts=self.edges_counts,
            edge_vocab_size=self.edge_vocab_size,
            node_vocab_size=self.node_vocab_size,
        )

    @typed
    def __logprob_nodes(self, x: Nodes, z: EncodedNodes, g_0) -> Float[Array, "b"]:
        # x = x.round().astype("int32")
        # unscale x
        # x_unscaled = ((x + 0.5) * self.node_vocab_size).round().astype("int32")
        x_onehot = jax.nn.one_hot(x, self.node_vocab_size)
        logprobs = self.__decode_node(z, g_0)
        logprob = jnp.sum(x_onehot * logprobs, axis=(1, 2))
        # jax.debug.breakpoint()
        return logprob

    @typed
    def __logprob_edges(self, x: Edges, z: EncodedEdges, g_0) -> Float[Array, "b"]:
        # x = x.round().astype("int32")
        # x_unscaled = ((x + 0.5) * self.node_vocab_size).round().astype("int32")
        x_onehot = jax.nn.one_hot(x, self.edge_vocab_size)
        logprobs = self.__decode_edge(z, g_0)
        logprob = jnp.sum(x_onehot * logprobs, axis=(1, 2, 3))
        return logprob

    @typed
    def logprob(self, z: "EncodedGraph", g_0) -> Float[Array, "b"]:
        x = self
        xn = x.nodes
        xg = x.edges
        zn = z.nodes
        zg = z.edges
        logprob_nodes = self.__logprob_nodes(xn, zn, g_0)
        logprob_edges = self.__logprob_edges(xg, zg, g_0)
        return logprob_nodes + logprob_edges
