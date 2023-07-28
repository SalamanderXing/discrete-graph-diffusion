import torch
from jax import Array, numpy as np


def to_torch(x):
    return torch.tensor(x.tolist())


def tt(args):
    if isinstance(args, Array):
        return to_torch(args)
    elif isinstance(args, (tuple, list)):
        return tuple(tt(x) for x in args)
    elif isinstance(args, dict):
        return {key: tt(val) for key, val in args.items()}
    elif isinstance(args, gd.GraphDistribution):
        return PlaceHolder(
            X=tt(args.nodes), E=tt(args.edges), y=torch.zeros(args.edges.shape[0])
        )
    else:
        return args


def to_gd_one_hot(x, orig):
    nodes = np.array(x.X)
    edges = np.array(x.E)
    nodes = np.where(orig.nodes_mask, nodes, 0)
    edges = np.where(orig.edges_mask, edges, 0)
    nodes = jax.nn.one_hot(nodes, num_classes=orig.nodes.shape[-1])
    edges = jax.nn.one_hot(np.array(x.E), num_classes=orig.edges.shape[-1])
    nodes = np.where(orig.nodes_mask[..., None], nodes, 0)
    edges = np.where(orig.edges_mask[..., None], edges, 0)

    return gd.create_one_hot(
        nodes=nodes,
        edges=edges,
        nodes_mask=orig.nodes_mask,
        edges_mask=orig.edges_mask,
    )


def to_gd_dense(x, orig):
    nodes = np.array(x.X)
    nodes = np.where(np.isnan(nodes), 0, nodes)
    edges = np.array(x.E.reshape(orig.edges.shape))
    edges = np.where(np.isnan(edges), 0, edges)
    return gd.create_dense(
        nodes=nodes,
        edges=edges,
        nodes_mask=orig.nodes_mask,
        edges_mask=orig.edges_mask,
    )


def to_q(x):
    return gd.Q(
        nodes=np.array(x.X),
        edges=np.array(x.E),
    )


def q_to_torch(q):
    return PlaceHolder(
        X=to_torch(q.nodes),
        E=to_torch(q.edges),
        y=None,
    )


class DummyModel(torch.nn.Module):
    def __init__(self, p):
        super().__init__()
        self.p = p

    def forward(self, X, E, y, node_mask):
        y = y.squeeze(-1)
        res = self.p(
            gd.create_one_hot_minimal(
                nodes=np.array(X), edges=np.array(E), nodes_mask=np.array(node_mask)
            ),
            np.array(y),
        )
        res_torch = tt(res)
        return res_torch
