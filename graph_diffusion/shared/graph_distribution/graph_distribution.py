from jax import numpy as np, Array
import jax
import networkx as nx
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
from ..graph import SimpleGraphDist

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
    return GraphDistribution.create(nodes=embedded_x, edges=embedded_e, mask=mask)


def safe_div(a: Array, b: Array):
    mask = b == 0
    return np.where(mask, 0, a / np.where(mask, 1, b))


@jdc.pytree_dataclass
class GraphDistribution(jdc.EnforcedAnnotationsMixin):
    nodes: XDistType
    edges: EDistType
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

    @classmethod
    @typed
    def create(
        cls,
        nodes: XDistType,
        edges: EDistType,
        edges_counts: EdgeCountType,
        nodes_counts: EdgeCountType,
    ) -> "GraphDistribution":
        # check(np.allclose(e, np.transpose(e, (0, 2, 1, 3))), "whoops")
        return cls(
            nodes=nodes,
            edges=edges,
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
        self, other: "GraphDistribution | SFloat | SInt"
    ) -> "GraphDistribution":
        if isinstance(other, (SFloat, SInt)):
            new_e = self.edges * other
            tmp = np.ones((self.n, self.n), bool)
            tmp = np.triu(tmp, k=1)
            new_e = np.where(
                tmp[None, :, :, None], new_e, np.transpose(new_e, (0, 2, 1, 3))
            )

            return GraphDistribution.create(
                nodes=self.nodes * other,
                edges=new_e,
                nodes_counts=self.nodes_counts,
                edges_counts=self.edges_counts,
            )
        elif isinstance(other, GraphDistribution):
            new_e = self.edges * other.edges
            tmp = np.ones((self.n, self.n), bool)
            tmp = np.triu(tmp, k=1)
            new_e = np.where(
                tmp[None, :, :, None], new_e, np.transpose(new_e, (0, 2, 1, 3))
            )
            return GraphDistribution.create(
                nodes=self.nodes * other.nodes,
                edges=new_e,
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
            return GraphDistribution.create(
                nodes=self.nodes / other,
                edges=self.edges / other,
                nodes_counts=self.nodes_counts,
                edges_counts=self.edges_counts,
            )
        else:
            return GraphDistribution.create(
                nodes=safe_div(self.nodes, other.nodes),
                edges=safe_div(self.edges, other.edges),
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

    def masks(self) -> tuple[MaskType, MaskType]:
        n_range = np.arange(self.n)
        mask_x = n_range[None].repeat(self.batch_size, 0) < self.nodes_counts[:, None]
        e_ranges = mask_x[:, :, None].repeat(self.n, -1)
        mask_e = e_ranges & e_ranges.transpose(0, 2, 1)
        return mask_x, mask_e

    def logprobs_at(self, one_hot: "GraphDistribution") -> SFloat:
        """Returns the probability of the given one-hot vector."""
        probs_at_vector = self * one_hot
        mask_x, mask_e = self.masks()
        uba = np.log(probs_at_vector.nodes.sum(-1)) * mask_x
        alla = np.log(probs_at_vector.edges.sum(-1)) * mask_e
        res = uba.sum(-1) + alla.sum((-1, -2))
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
        # symmetrize e (while preseving the fact that they're one hot encoded)
        return GraphDistribution.create(
            nodes=embedded_x,
            edges=embedded_e,
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
        x = self.nodes @ q.x
        e = self.edges @ q.e[:, None]
        x = x / x.sum(-1, keepdims=True)
        e = e / e.sum(-1, keepdims=True)

        tmp = np.ones((self.n, self.n), bool)
        tmp = np.triu(tmp, k=1)
        new_e = np.where(tmp[None, :, :, None], e, np.transpose(e, (0, 2, 1, 3)))

        # assert is_dist(x), ipdb.set_trace()
        # assert is_dist(e), ipdb.set_trace()
        return GraphDistribution(
            nodes=x,
            edges=new_e,
            edges_counts=self.edges_counts,
            nodes_counts=self.nodes_counts,
            _created_internally=True,
        )  # mask=self.mask,

    def plot(self):
        import matplotlib.pyplot as plt
        import numpy
        from tqdm import tqdm

        _, axs = plt.subplots(1, self.batch_size, figsize=(100, 10))

        for batch_index in tqdm(range(self.batch_size)):
            x = self.nodes[batch_index].argmax(-1)
            e = self.edges[batch_index].argmax(-1)
            n_nodes = self.nodes_counts[batch_index]

            nodes = x[:n_nodes]
            edges_features = e[:n_nodes, :n_nodes]
            # c_values_nodes = np.array([cmap_node[0 if i == 0 else i + 10] for i in nodes])

            indices = np.indices(edges_features.shape).reshape(2, -1).T
            mask = edges_features.flatten() != 0
            edges = indices[mask]

            # c_values_edges = np.array([cmap_edge[i] for i in edges[:, -1]])

            G = nx.Graph()
            # for i in range(n_nodes):
            #     G.add_node(i)
            for i in range(edges.shape[0]):
                G.add_edge(edges[i, 0].tolist(), edges[i, 1].tolist())
            pos = nx.spring_layout(G)  # positions for all nodes

            cmap_edge = plt.cm.viridis(np.linspace(0, 1, self.edges.shape[-1]))
            cmap_node = plt.cm.viridis(np.linspace(0, 1, self.nodes.shape[-1]))

            color_nodes = numpy.array([cmap_node[i] for i in nodes])
            color_edges = numpy.array(
                [cmap_edge[edges_features[i, j]] for (i, j) in G.edges]
            )
            nx.draw(
                G,
                pos,
                edge_color=color_edges,
                node_color=color_nodes,
                ax=axs[batch_index],
            )
            axs[batch_index].set_title(f"t={batch_index}")
            axs[batch_index].set_aspect("equal")

        plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
        plt.show()

    # overrides the square bracket indexing
    def __getitem__(self, key: Int[Array, "n"]) -> "GraphDistribution":
        return GraphDistribution.create(
            nodes=self.nodes[key],
            edges=self.edges[key],
            nodes_counts=self.nodes_counts[key],
            edges_counts=self.edges_counts[key],
        )  # , mask=self.mask[key])

    def repeat(self, n: int) -> "GraphDistribution":
        return GraphDistribution.create(
            nodes=np.repeat(self.nodes, n, axis=0),
            edges=np.repeat(self.edges, n, axis=0),
            nodes_counts=np.repeat(self.nodes_counts, n, axis=0),
            edges_counts=np.repeat(self.edges_counts, n, axis=0),
        )

    def __len__(self):
        return self.batch_size


def is_dist(x):
    return np.min(x) >= 0 and np.max(x) <= 1 and np.allclose(np.sum(x, -1), 1)
