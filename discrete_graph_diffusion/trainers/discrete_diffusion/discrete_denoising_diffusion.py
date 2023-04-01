from jax import numpy as np
import flax.linen as nn
from jax import Array
import optax

from .metrics import TrainLossDiscrete
from .metrics import Metric
from .metrics.sum_except_batch import SumExceptBatchMetric
from .metrics.sum_except_batch_kl import SumExceptBatchKL
from .metrics.nll import NLL
from .noise_schedule import PredefinedNoiseScheduleDiscrete
from .noise_transition import DiscreteUniformTransition, MarginalUniformTransition
from .config import GeneralConfig
from .utils.placeholder import PlaceHolder
from .utils.geometric import to_dense


def training_step(self, data, i):
    if data.edge_index.numel() == 0:
        print("Found a batch with no edges. Skipping.")
        return
    dense_data, node_mask = to_dense(
        data.x, data.edge_index, data.edge_attr, data.batch
    )
    dense_data = dense_data.mask(node_mask)
    X, E = dense_data.X, dense_data.E
    if i == 0:
        ipdb.set_trace()
    noisy_data = self.apply_noise(X, E, data.y, node_mask)
    extra_data = self.compute_extra_data(noisy_data)
    pred = self.forward(noisy_data, extra_data, node_mask)
    loss = self.train_loss(
        masked_pred_X=pred.X,
        masked_pred_E=pred.E,
        pred_y=pred.y,
        true_X=X,
        true_E=E,
        true_y=data.y,
        log=i % self.log_every_steps == 0,
    )

    self.train_metrics(
        masked_pred_X=pred.X,
        masked_pred_E=pred.E,
        true_X=X,
        true_E=E,
        log=i % self.log_every_steps == 0,
    )

    return {"loss": loss}


class DiscreteDenoisingDiffusion:
    def __init__(
        self,
        model: nn.Module,
        cfg: GeneralConfig,
        sampling_metrics: Metric,
    ):
        self.model = model
        self.cfg = cfg
        self.train_loss = TrainLossDiscrete(self.cfg.train.lambda_train)

        self.val_nll = NLL()
        self.val_X_kl = SumExceptBatchKL()
        self.val_E_kl = SumExceptBatchKL()
        self.val_X_logp = SumExceptBatchMetric()
        self.val_E_logp = SumExceptBatchMetric()

        self.test_nll = NLL()
        self.test_X_kl = SumExceptBatchKL()
        self.test_E_kl = SumExceptBatchKL()
        self.test_X_logp = SumExceptBatchMetric()
        self.test_E_logp = SumExceptBatchMetric()

        self.noise_schedule = PredefinedNoiseScheduleDiscrete(
            cfg.train.diffusion_noise_schedule, timesteps=cfg.train.diffusion_steps
        )

        if cfg.train.transition == "uniform":
            self.transition_model = DiscreteUniformTransition(
                x_classes=self.cfg.dataset.out_dims.X,
                e_classes=self.cfg.dataset.out_dims.E,
                y_classes=self.cfg.dataset.out_dims.y,
            )
            x_limit = np.ones(self.cfg.dataset.out_dims.X) / self.cfg.dataset.out_dims.X
            e_limit = np.ones(self.cfg.dataset.out_dims.E) / self.cfg.dataset.out_dims.E
            y_limit = np.ones(self.cfg.dataset.out_dims.y) / self.cfg.dataset.out_dims.y
            self.limit_dist = PlaceHolder(x=x_limit, e=e_limit, y=y_limit)

        elif cfg.train.transition == "marginal":
            node_types = self.cfg.dataset.node_types.astype(float)
            x_marginals = node_types / np.sum(node_types)

            edge_types = self.cfg.dataset.edge_types.astype(float)
            e_marginals = edge_types / np.sum(edge_types)
            print(
                f"Marginal distribution of the classes: {x_marginals} for nodes, {e_marginals} for edges"
            )
            self.transition_model = MarginalUniformTransition(
                x_marginals=x_marginals,
                e_marginals=e_marginals,
                y_classes=self.cfg.dataset.out_dims.y,
            )
            self.limit_dist = PlaceHolder(
                x=x_marginals,
                e=e_marginals,
                y=np.ones(self.cfg.dataset.out_dims.y) / self.cfg.dataset.out_dims.y,
            )

        # self.save_hyperparameters(ignore=[train_metrics, sampling_metrics]) # TODO: implement this maybe?
        self.start_epoch_time = None
        self.train_iterations = None
        self.val_iterations = None
        self.log_every_steps = cfg.train.log_every_steps
        self.number_chain_steps = cfg.train.number_chain_steps
        self.best_val_nll = 1e8
        self.val_counter = 0

    def init_optimizer(self):
        return optax.adam(self.cfg.train.learning_rate)

    def on_validation_epoch_start(self) -> None:
        self.val_nll.reset()
        self.val_X_kl.reset()
        self.val_E_kl.reset()
        self.val_X_logp.reset()
        self.val_E_logp.reset()
        self.sampling_metrics.reset()
