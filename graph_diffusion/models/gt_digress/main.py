from flax import linen as nn
from flax.struct import dataclass
from jax import numpy as np
from typing import no_type_check
from mate.jax import Key
import jax
from ...shared.graph import graph_distribution as gd
from einops import rearrange, repeat
import ipdb
from jaxtyping import jaxtyped
from beartype import beartype
from typing import no_type_check


class Xtoy(nn.Module):
    dy: int

    @nn.compact
    def __call__(self, X: gd.NodeDistribution):
        """Map node features to global features"""
        m = X.mean(axis=1)
        mi = X.min(axis=1)
        ma = X.max(axis=1)
        # std = X.std(axis=1) FIXME: this gives NaN during gradient computation. Why?
        z = np.concatenate(
            (
                m,
                mi,
                ma,  # std
            ),
            -1,
        )
        out = nn.Dense(self.dy)(z)
        return out


class Etoy(nn.Module):
    dy: int

    @nn.compact
    def __call__(self, E: gd.EdgeDistribution):
        """Map edge features to global features."""
        super().__init__()
        m = E.mean(axis=(1, 2))
        mi = E.min(axis=2).min(axis=1)
        ma = E.max(axis=2).max(axis=1)
        # std = np.std(E, axis=(1, 2))
        z = np.concatenate(
            (
                m,
                mi,
                ma,  # std
            ),
            axis=-1,
        )
        out = nn.Dense(self.dy)(z)
        return out


max_neg_value = -np.finfo(np.float32).max


def masked_softmax(x, mask, axis=-1):
    x_masked = np.where(
        mask[..., None], x, max_neg_value
    )  # x_masked[mask == 0] = -float("inf")
    return jax.nn.softmax(x_masked, axis=axis)


class XEyTransformerLayer(nn.Module):
    dx: int
    de: int
    dy: int
    n_head: int
    dim_ffX: int
    dim_ffE: int
    dim_ffy: int
    dropout: float = 0.1
    layer_norm_eps: float = 1e-5
    device = None
    dtype = None

    @nn.compact
    def __call__(self, X, E, y, node_mask, deterministic):
        act = nn.relu
        self_attn = NodeEdgeBlock(
            self.dx,
            self.de,
            self.dy,
            self.n_head,
        )
        linX1 = nn.Dense(
            self.dim_ffX,
        )
        linX2 = nn.Dense(
            self.dx,
        )
        normX1 = nn.LayerNorm(
            epsilon=self.layer_norm_eps,
        )
        normX2 = nn.LayerNorm(
            epsilon=self.layer_norm_eps,
        )
        dropoutX1 = nn.Dropout(self.dropout)
        dropoutX2 = nn.Dropout(self.dropout)
        dropoutX3 = nn.Dropout(self.dropout)

        linE1 = nn.Dense(
            self.dim_ffE,
        )
        linE2 = nn.Dense(
            self.de,
        )
        normE1 = nn.LayerNorm(
            epsilon=self.layer_norm_eps,
        )
        normE2 = nn.LayerNorm(
            epsilon=self.layer_norm_eps,
        )
        dropoutE1 = nn.Dropout(self.dropout)
        dropoutE2 = nn.Dropout(self.dropout)
        dropoutE3 = nn.Dropout(self.dropout)

        lin_y1 = nn.Dense(
            self.dim_ffy,
        )
        lin_y2 = nn.Dense(
            self.dy,
        )
        norm_y1 = nn.LayerNorm(
            epsilon=self.layer_norm_eps,
        )
        norm_y2 = nn.LayerNorm(
            epsilon=self.layer_norm_eps,
        )
        dropout_y1 = nn.Dropout(self.dropout)
        dropout_y2 = nn.Dropout(self.dropout)
        dropout_y3 = nn.Dropout(self.dropout)

        newX, newE, new_y = self_attn(
            X, E, y, node_mask=node_mask, deterministic=deterministic
        )

        newX_d = dropoutX1(newX, deterministic=deterministic)
        X = normX1(X + newX_d)

        newE_d = dropoutE1(newE, deterministic=deterministic)
        E = normE1(E + newE_d)

        new_y_d = dropout_y1(new_y, deterministic=deterministic)
        y = norm_y1(y + new_y_d)

        ff_outputX = linX2(dropoutX2(act(linX1(X)), deterministic=deterministic))
        ff_outputX = dropoutX3(ff_outputX, deterministic=deterministic)
        X = normX2(X + ff_outputX)

        ff_outputE = linE2(dropoutE2(act(linE1(E)), deterministic=deterministic))
        ff_outputE = dropoutE3(ff_outputE, deterministic=deterministic)
        E = normE2(E + ff_outputE)

        ff_output_y = lin_y2(dropout_y2(act(lin_y1(y)), deterministic=deterministic))
        ff_output_y = dropout_y3(ff_output_y, deterministic=deterministic)
        y = norm_y2(y + ff_output_y)

        return X, E, y


class NodeEdgeBlock(nn.Module):
    dx: int
    de: int
    dy: int
    n_head: int

    @nn.compact
    def __call__(self, X, E, y, node_mask, deterministic):
        n = X.shape[1]
        dx = self.dx
        de = self.de
        df = self.dx // self.n_head

        q, k, v = nn.Dense(dx), nn.Dense(dx), nn.Dense(dx)

        e_add, e_mul = nn.Dense(dx), nn.Dense(dx)

        y_e_mul, y_e_add = nn.Dense(dx), nn.Dense(dx)

        y_x_mul, y_x_add = nn.Dense(dx), nn.Dense(dx)

        x_out, e_out, y_out = (
            nn.Dense(dx),
            nn.Dense(de),
            nn.Sequential(
                (
                    nn.Dense(self.dy),
                    nn.relu,
                    nn.Dense(self.dy),
                )
            ),
        )

        x_mask = rearrange(node_mask, "bs n -> bs n 1")
        e_mask = rearrange(node_mask, "bs n -> bs n 1 1") * rearrange(
            node_mask, "bs n -> bs 1 n 1"
        )

        # 1. Map X to keys and queries
        # 2. Reshape to (bs, n, n_head, df) with dx = n_head * df
        Q = rearrange(
            q(X) * x_mask, "bs n (n_head df) -> bs 1 n n_head df", n_head=self.n_head
        )
        K = rearrange(
            k(X) * x_mask, "bs n (n_head df) -> bs n 1 n_head df", n_head=self.n_head
        )
        # Compute unnormalized attentions. Y is (bs, n, n, n_head, df)
        Y = Q * K / df
        E1 = rearrange(
            e_mul(E) * e_mask,
            "bs n1 n2 (n_head df) -> bs n1 n2 n_head df",
            n_head=self.n_head,
        )

        E2 = rearrange(
            e_add(E) * e_mask,
            "bs n1 n2 (n_head df) -> bs n1 n2 n_head df",
            n_head=self.n_head,
        )

        # Incorporate edge features to the self attention scores.
        Y = Y * (E1 + 1) + E2  # (bs, n, n, n_head, df)

        # Incorporate y to E
        newE = rearrange(Y, "bs n1 n2 n_head df -> bs n1 n2 (n_head df)")
        # .flatten(start_axis=3)  # bs, n, n, dx
        ye1 = rearrange(y_e_add(y), "bs de -> bs 1 1 de")
        # .unsqueeze(1).unsqueeze(1)  # bs, 1, 1, de
        ye2 = rearrange(y_e_mul(y), "bs de -> bs 1 1 de")  # .unsqueeze(1).unsqueeze(1)
        newE = ye1 + (ye2 + 1) * newE

        # Output E
        newE = e_out(newE) * e_mask  # bs, n, n, de

        # Compute attentions. attn is still (bs, n, n, n_head, df)
        softmax_mask = repeat(
            node_mask, "bs n1 -> bs n1 n2 n_head", n2=n, n_head=self.n_head
        )  # .expand(-1, n, -1, self.n_head)  # bs, 1, n, 1
        attn = masked_softmax(Y, softmax_mask, axis=2)  # bs, n, n, n_head

        V = rearrange(
            v(X) * x_mask,
            "bs n (n_head df) -> bs 1 n n_head df",
            n_head=self.n_head,
            df=df,
        )

        # Compute weighted values
        weighted_V = (attn * V).sum(axis=2)

        # Send output to input dim
        weighted_V = rearrange(weighted_V, "bs n n_head df -> bs n (n_head df)")

        # Incorporate y to X
        yx1 = rearrange(y_x_add(y), "bs ne -> bs 1 ne")
        yx2 = rearrange(y_x_mul(y), "bs ne -> bs 1 ne")
        newX = yx1 + (yx2 + 1) * weighted_V

        # Output X
        newX = x_out(newX) * x_mask
        # diffusion_utils.assert_correctly_masked(newX, x_mask)

        # Process y based on X axnd E
        y = nn.Dense(self.dy)(y)
        e_y = Etoy(self.dy)(E)  # FIXME: this is EToy in original
        x_y = Xtoy(self.dy)(X)  # FIXME: this is XToy in original
        new_y = y + x_y + e_y
        new_y = y_out(new_y)  # bs, dy

        return newX, newE, new_y


@dataclass
class Dims:
    x: int
    e: int
    y: int


@dataclass
class HiddenDims:
    x: int
    e: int
    y: int
    n_head: int
    dim_ffx: int
    dim_ffe: int
    dim_ffy: int


from mate.jax import SBool


@no_type_check
class GraphTransformer(nn.Module):
    n_layers: int
    hidden_mlp_dims: Dims = Dims(x=128, e=128, y=128)
    hidden_dims: HiddenDims = HiddenDims(
        x=256, e=64, y=64, n_head=8, dim_ffx=256, dim_ffe=128, dim_ffy=2048
    )

    @classmethod
    @jaxtyped
    @beartype
    def initialize(
        cls,
        key: Key,
        in_node_features: int,
        in_edge_features: int,
        number_of_nodes: int,
        num_layers: int,
    ):
        batch_size = 2
        dummy = gd.OneHotGraph.noise(
            key,
            in_node_features,
            in_edge_features,
            number_of_nodes,
        ).repeat(batch_size)
        key, subkey = jax.random.split(key)
        model = cls(num_layers)
        params = model.init(
            subkey,
            dummy,
            np.zeros((batch_size, 129)),
            deterministic=True,
        )
        return model, params

    @jaxtyped
    @beartype
    @nn.compact
    def __call__(self, g: gd.GraphDistribution, y, deterministic: SBool):
        act_fn_in = nn.relu
        act_fn_out = nn.relu
        X = g.nodes
        E = g.edges
        nodes_mask = g.nodes_mask[..., None]
        edges_mask = g.edges_mask[..., None]
        n_layers = self.n_layers
        input_dims = Dims(x=X.shape[-1], e=E.shape[-1], y=y.shape[-1])
        bs, n = X.shape[0], X.shape[1]
        diag_mask = rearrange(
            repeat(~np.eye(n, dtype=bool), "n1 n2 -> bs n1 n2", bs=bs),
            "bs n1 n2 -> bs n1 n2 1",
        )
        mlp_in_X = nn.Sequential(
            (
                nn.Dense(self.hidden_mlp_dims.x),
                act_fn_in,
                nn.Dense(self.hidden_dims.x),
                act_fn_in,
            )
        )

        mlp_in_E = nn.Sequential(
            (
                nn.Dense(self.hidden_mlp_dims.e),
                act_fn_in,
                nn.Dense(self.hidden_dims.e),
                act_fn_in,
            )
        )

        mlp_in_y = nn.Sequential(
            (
                nn.Dense(self.hidden_mlp_dims.y),
                act_fn_in,
                nn.Dense(self.hidden_dims.y),
                act_fn_in,
            )
        )
        tf_layers = [
            XEyTransformerLayer(
                dx=self.hidden_dims.x,
                de=self.hidden_dims.e,
                dy=self.hidden_dims.y,
                n_head=self.hidden_dims.n_head,
                dim_ffX=self.hidden_dims.dim_ffx,
                dim_ffE=self.hidden_dims.dim_ffe,
                dim_ffy=self.hidden_dims.dim_ffy,
            )
            for _ in range(n_layers)
        ]

        mlp_out_X = nn.Sequential(
            (
                nn.Dense(self.hidden_mlp_dims.x),
                act_fn_out,
                nn.Dense(input_dims.x),
            )
        )

        mlp_out_E = nn.Sequential(
            (
                nn.Dense(self.hidden_mlp_dims.e),
                act_fn_out,
                nn.Dense(input_dims.e),
            )
        )

        X_to_out = X[..., : input_dims.x]
        E_to_out = E[..., : input_dims.e]

        new_E = mlp_in_E(E)
        new_E = (new_E + rearrange(new_E, "bs n1 n2 ne -> bs n2 n1 ne")) / 2

        X = mlp_in_X(X) * nodes_mask
        E = new_E * edges_mask
        y = mlp_in_y(y)

        for layer in tf_layers:
            X, E, y = layer(X, E, y, g.nodes_mask, deterministic)

        X = mlp_out_X(X)
        E = mlp_out_E(E)

        X += X_to_out
        E = (E + E_to_out) * diag_mask
        E = (E + rearrange(E, "bs n1 n2 ne -> bs n2 n1 ne")) / 2
        X *= nodes_mask
        E *= edges_mask
        return gd.DenseGraphDistribution.create(
            nodes=X,
            edges=E,
            nodes_mask=g.nodes_mask,
            edges_mask=g.edges_mask,
            unsafe=True,
        )
