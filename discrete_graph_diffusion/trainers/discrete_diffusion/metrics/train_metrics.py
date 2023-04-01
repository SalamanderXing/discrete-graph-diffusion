from re import A
from jax import numpy as np
from jax import Array
from .cross_entropy import CrossEntropyMetric


class TrainLossDiscrete:
    def __init__(self, lambda_train):
        self.lambda_train = lambda_train
        self.node_loss = CrossEntropyMetric()
        self.edge_loss = CrossEntropyMetric()
        self.y_loss = CrossEntropyMetric()

    def __call__(
        self,
        masked_pred_x: Array,
        masked_pred_e: Array,
        pred_y: Array,
        true_x: Array,
        true_e: Array,
        true_y: Array,
        log: bool,
    ):
        """Compute train metrics
        masked_pred_X : tensor -- (bs, n, dx)
        masked_pred_E : tensor -- (bs, n, n, de)
        pred_y : tensor -- (bs, )
        true_X : tensor -- (bs, n, dx)
        true_E : tensor -- (bs, n, n, de)
        true_y : tensor -- (bs, )
        log : boolean."""
        true_x = true_x.reshape(-1, true_x.shape[-1])  # (bs*n, dx)
        true_e = true_e.reshape(-1, true_e.shape[-1])  # (bs*n*n, de)
        masked_pred_x = masked_pred_x.reshape(-1, masked_pred_x.shape[-1])  # (bs*n, dx)
        masked_pred_e = masked_pred_e.reshape(
            -1, masked_pred_e.shape[-1]
        )  # (bs*n*n, de)

        # Remove masked rows
        mask_x = (true_x != 0).any(axis=-1)
        mask_e = (true_e != 0).any(axis=-1)

        flat_true_x = true_x[mask_x]
        flat_true_e = true_e[mask_e]

        flat_pred_x = masked_pred_x[mask_x]
        flat_pred_e = masked_pred_e[mask_e]

        # Compute metrics
        loss_x = self.node_loss(flat_pred_x, flat_true_x) if true_x.size else 0
        loss_e = self.edge_loss(flat_pred_e, flat_true_e) if true_e.size else 0
        loss_y = self.y_loss(pred_y, true_y)

        if log:
            to_log = {
                "train_loss/batch_CE": (loss_x + loss_e + loss_y),
                "train_loss/X_CE": self.edge_loss,
                "train_loss/E_CE": loss_e,  # if true_e > 0 else -1,
                "train_loss/y_CE": self.node_loss.compute(),
            }
            # TODO: log it somewhere

        return loss_x + self.lambda_train[0] * loss_e + self.lambda_train[1] * loss_y
