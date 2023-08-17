from rich.table import Table
from rich.style import Style
from flax.struct import dataclass
from jaxtyping import Float, Array
from jax import numpy as np
import jax

from ...shared.graph import graph_distribution as gd


@dataclass
class ValidationResultWrapper:
    """
    Just a wrapper around a dict of losses. Useful for wandb and console logging.
    """

    data: dict[str, dict[str, Float[Array, "batch_size"]]]

    @property
    def nll(self):
        return np.mean(np.array([val["nll"] for val in self.values()]))

    def flatten(self):
        return {
            f"{supkey}/{key}": val
            for supkey, subdict in self.items()
            for key, val in subdict.items()
        }

    def __getitem__(self, key):
        return self.data[key]

    def __lt__(self, other):
        return self.nll < other.nll

    def keys(self):
        return self.data.keys()

    def values(self):
        return self.data.values()

    def items(self):
        return self.data.items()

    def unshard(self):
        return jax.tree_map(
            lambda x: x.reshape(-1),
            self,
        )

    def to_dict(self):
        return self.data

    def mean(self):
        return jax.tree_map(
            lambda x: np.mean(x),
            self,
        )

    def __repr__(self):
        return jax.tree_map(lambda x: x.shape, self).__dict__.__repr__()

    def __str__(self):
        return jax.tree_map(lambda x: x.shape, self).__dict__.__str__()

    def __add__(self, other: "ValidationResultWrapper"):
        return jax.tree_map(
            lambda x, y: np.concatenate((x, y)),
            self,
            other,
        )

    def __len__(self):
        keys = tuple(self.keys())
        subkeys = tuple(self[keys[0]].keys())
        return len(self[keys[0]][subkeys[0]])

    def update(self, other: "ValidationResultWrapper"):
        self.data.update(other.data)

    def convert_to_shuffle_conding_metric(self, target: gd.GraphDistribution):
        tot_edges_mask = target.edges.argmax(-1) > 0
        edges_count = tot_edges_mask.sum((1, 2))
        return jax.tree_map(
            lambda x: x / (edges_count * np.log(2)),
            self,
        )

    def to_rich_table(self, title: str, epoch: int | None = None):
        data = self.data
        table = Table(
            title=f"{title}"
            + (f" [dim](epoch {epoch})[/dim]" if epoch is not None else ""),
            title_style=Style(color="green", bold=True),
            header_style=Style(color="magenta", bold=True),
            show_header=True,
            style=Style(color="green"),
            # row_styles=["", "dim"],
        )
        table.add_column(
            "Metrics",
            style=Style(color="yellow", bold=True),
        )
        table.add_column("NLL", style=Style(color="cyan"))
        keys = tuple(data.keys())
        all_values = []
        for i, key in enumerate(keys):
            if i == 0:
                for subkey in data[key].keys():
                    if subkey != "nll":
                        table.add_column(subkey, style=Style(color="cyan"))
            values = [data[key]["nll"]] + [
                v for k, v in data[key].items() if k != "nll"
            ]
            all_values.append(values)
            table.add_row(
                key,
                *[str(round(val.item(), 4)) for val in values],
            )
        table.add_row(
            "Mean",
            *[str(round(val.item(), 4)) for val in np.mean(np.array(all_values), 0)],
        )
        return table
