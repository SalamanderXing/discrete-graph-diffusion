import jax
import optax
import flax
from flax.training.common_utils import get_metrics, onehot
from flax.training import train_state
import jax.numpy as jnp


class ReduceLROnPlateau:
    def __init__(self, factor=0.1, patience=10, min_lr=0.00001):
        self.factor = factor
        self.patience = patience
        self.min_lr = min_lr
        self.best_loss = jnp.inf
        self.wait = 0

    def step(self, loss):
        if loss < self.best_loss:
            self.best_loss = loss
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.wait = 0
                return True
        return False


# Initialize the learning rate scheduler
lr_scheduler = ReduceLROnPlateau()


def train_step(state, batch, lr_scheduler):
    def loss_fn(params):
        logits = state.apply_fn({"params": params}, batch)
        loss = optax.softmax_cross_entropy(logits, onehot(batch["label"], 10)).mean()
        return loss, logits

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (_, logits), grads = grad_fn(state.params)
    loss = optax.softmax_cross_entropy(logits, onehot(batch["label"], 10)).mean()

    # Update the learning rate if the loss has plateaued
    if lr_scheduler.step(loss):
        new_lr = max(
            state.tx.opts.learning_rate * lr_scheduler.factor, lr_scheduler.min_lr
        )
        state = state.replace(tx=optax.adam(new_lr))

    state = state.apply_gradients(grads=grads)
    metrics = get_metrics(logits, batch["label"])
    return state, metrics
