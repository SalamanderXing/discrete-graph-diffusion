import jax.numpy as np
from jax import jit, lax
from jax import Array


class AverageMetric:
    def __init__(self):
        super().__init__()
        self._accumulator = 0.0
        self._count = 0.0

    def __call__(self, *args, **kwargs):
        return self.update(*args, **kwargs)

    def update(self, preds: Array):
        self._accumulator += preds.sum()
        self._count += preds.size

    def compute(self):
        return self._accumulator / self._count

    def reset(self):
        self._accumulator = 0.0
        self._count = 0.0

    """
    def add_state(
        self,
        name: str,
        default: Any,
        reduction_fn: Optional[str] = "sum",
        sync_fn: Optional[str] = "sum",
    ):
        self._state[name] = {
            "value": default,
            "reduction_fn": reduction_fn
            if reduction_fn is None
            else self.__reduction_fns[reduction_fn],
            "sync_fn": sync_fn if sync_fn is None else self.__reduction_fns[sync_fn],
        }

    def _sync_states(self):
        for name, state in self._state.items():
            state["value"] = state["sync_fn"](state["value"])

    @property
    def sync_fn(self):
        return self._sync_states

    @jit
    def __call__(self, preds, target):
        self.update(preds, target)
        self.sync_fn()
        return self.compute()

    def __getitem__(self, key):
        return self._state[key]["value"]

    def __setitem__(self, key, value):
        self._state[key]["value"] = value
    """
