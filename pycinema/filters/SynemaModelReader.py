import logging as log
from os.path import exists

import jax
import jax.numpy as jnp
import optax
import orbax.checkpoint
from flax.training.train_state import TrainState
from synema.models.cinema import CinemaScalarImage

from pycinema import Filter


class SynemaModelReader(Filter):

    # TODO: move to SynemaModel?
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

    def __init__(self):
        super().__init__(
            inputs={
                'path': ''
            },
            outputs={
                'model_state': {},
            }
        )

    def _update(self):
        model = CinemaScalarImage()
        state = self.create_empty_state(model)

        checkpoint_path = self.inputs.path.get()
        if not checkpoint_path:
            self.outputs.model.set(state)
            return 0

        if not exists(checkpoint_path):
            log.error("No checkpoint file found: " + checkpoint_path)
            return 0

        target = {'state': state}
        checkpointer = orbax.checkpoint.PyTreeCheckpointer()
        raw_restored = checkpointer.restore(checkpoint_path, item=target)

        self.outputs.model_state.set({'model': model, 'state': raw_restored['state']})

        return 1
