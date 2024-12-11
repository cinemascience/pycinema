import jax
import jax.numpy as jnp
import optax
from flax.training.train_state import TrainState
from synema.models.cinema import CinemaScalarImage

from pycinema import Filter


class SynemaModelFactory(Filter):
    """Create a Synema model with empty states."""

    @staticmethod
    def create_empty_state(model):
        schedule_fn = optax.exponential_decay(init_value=1e-3,
                                              transition_begin=600,
                                              transition_steps=200,
                                              decay_rate=0.5)
        optimizer = optax.adam(learning_rate=schedule_fn)

        init_state = TrainState.create(apply_fn=model.apply,
                                       params=model.init(jax.random.PRNGKey(0),
                                                         jnp.empty((1024, 3))),
                                       tx=optimizer)
        return init_state

    # TODO: add capability to choose Scalar or RGB image model.
    def __init__(self):
        super().__init__(
            outputs={
                'model_state': {}
            }
        )
        model = CinemaScalarImage()
        state = self.create_empty_state(model)

        self.outputs.model_state.set({'model': model, 'state': state})

    def _update(self):
        return 1
