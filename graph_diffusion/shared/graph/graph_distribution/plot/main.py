from jaxtyping import jaxtyped
from beartype import beartype
from typing import Sequence
import matplotlib.pyplot as plt
import ipdb
import numpy as np
import networkx as nx
from ..graph_distribution import GraphDistribution
from ..functional import concatenate


def get_position(G, j, positions, position_option_option):
    if position_option_option is None:
        return nx.spring_layout(G)

    if positions[j] is None:
        if j > 0 and position_option_option in (
            "row",
            "all",
        ):
            positions[j] = positions[0]
        else:
            positions[j] = nx.spring_layout(G)  # positions for all nodes
    position = positions[j]
    return position


def plot(
    rows: Sequence[GraphDistribution] | GraphDistribution,
    location: str | None = None,
    shared_position_option: str | None = None,
    title: str | None = None,
):
    if isinstance(rows, GraphDistribution):
        rows = [rows]
    assert isinstance(rows, Sequence)
    assert shared_position_option in [None, "row", "col", "all"]
    location = (
        (
            f"{location}.png"
            if location is not None and not location.endswith(".png")
            else location
        )
        if location != "wandb"
        else "wandb"
    )
    original_len = len(rows[0])
    skip = (len(rows[0]) // 15) if len(rows[0]) > 15 else 1
    rows_skips = [(row[::skip], row[np.array([-1])]) for row in rows]
    rows = [concatenate(row) for row in rows_skips]
    lrows = len(rows)
    lcolumn = len(rows[0])
    _, axs = plt.subplots(
        lrows,
        lcolumn,
        figsize=(100, 10),
    )
    if len(axs.shape) == 1:
        axs = axs[None, :]

    if rows[0].nodes.shape[-1] > 2:
        cmap_edge = plt.cm.viridis(np.linspace(0, 1, rows[0].edges.shape[-1] - 1))
        cmap_node = plt.cm.viridis(np.linspace(0, 1, rows[0].nodes.shape[-1]))
        cmap_edge = np.concatenate([np.zeros((1, 4)), cmap_edge], axis=0)
    else:
        cmap_edge = plt.cm.viridis(np.linspace(0, 1, rows[0].edges.shape[-1] - 1))
        # cmap_node = plt.cm.viridis(np.linspace(0, 1, rows[0].nodes.shape[-1] - 1))
        cmap_edge = np.concatenate([np.zeros((1, 4)), cmap_edge], axis=0)
        # cmap_node = numpy.concatenate([numpy.zeros((1, 4)), cmap_node], axis=0)
        # node will be black
        cmap_node = np.array([[0, 0, 0, 1]])
    node_size = 10.0
    positions = [None] * len(rows[0])

    for i, (row, ax_row) in enumerate(zip(rows, axs)):
        node_values = row.nodes.argmax(-1)
        edge_values = row.edges.argmax(-1)
        current_node_counts = row.nodes_counts
        for j, ax in enumerate(ax_row):
            j *= skip
            n_nodes = current_node_counts[j]
            nodes = node_values[j, :n_nodes]
            edge_features = edge_values[j, :n_nodes, :n_nodes]
            indices = np.indices(edge_features.shape).reshape(2, -1).T
            mask = edge_features.flatten() != 0
            edges = indices[mask]
            G = nx.Graph()
            for i in range(n_nodes):
                G.add_node(i)
            for i in range(edges.shape[0]):
                G.add_edge(edges[i, 0].tolist(), edges[i, 1].tolist())
            color_nodes = (
                np.array([cmap_node[i - 1] for i in nodes])
                if rows[0].nodes.shape[-1] > 2
                else np.array([cmap_node[i] for i in nodes])
            )
            color_edges = np.array(
                [cmap_edge[edge_features[i, j]] for (i, j) in G.edges]
            )
            position = get_position(G, j, positions, shared_position_option)
            nx.draw(
                G,
                position,
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
    elif location == "wandb":
        wandb.log({"prediction": wandb.Image(plt)})
    else:
        plt.savefig(location)
        plt.clf()
        plt.close()
        # wandb.log({"prediction": wandb.Image(location)})
    plt.clf()
    plt.close()
