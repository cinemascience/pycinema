import logging as log
from os.path import exists

import orbax.checkpoint

from pycinema import Filter


class SynemaModelReader(Filter):
    def __init__(self):
        super().__init__(
            inputs={
                'model_state': {},
                'path': ''
            },
            outputs={
                'model_state': {},
            }
        )

    def _update(self):
        model_state = self.inputs.model_state.get()
        if not model_state['model']:
            return 0

        checkpoint_path = self.inputs.path.get()
        if not checkpoint_path:
            self.outputs.model_state.set(model_state)
            return 0

        if not exists(checkpoint_path):
            log.error("No checkpoint file found: " + checkpoint_path)
            return 0

        target = {'state': self.inputs.model_state.get()['state']}
        checkpointer = orbax.checkpoint.PyTreeCheckpointer()
        raw_restored = checkpointer.restore(checkpoint_path, item=target)

        self.outputs.model_state.set({'model': self.inputs.model_state.get()['model'],
                                      'state': raw_restored['state']})

        return 1
