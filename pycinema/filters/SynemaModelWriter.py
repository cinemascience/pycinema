import os
import shutil

import orbax.checkpoint

from pycinema import Filter


class SynemaModelWriter(Filter):
    def __init__(self):
        super().__init__(
            inputs={
                'path': '',
                'model_state': []
            }
        )

    def _update(self):
        ckpt_dir = self.inputs.path.get()
        if os.path.exists(ckpt_dir):
            shutil.rmtree(ckpt_dir)

        ckpt = {'state': self.inputs.model_state.get()['state']}

        checkpointer = orbax.checkpoint.PyTreeCheckpointer()
        # save_args = orbax_utils.save_args_from_target(ckpt)
        checkpointer.save(ckpt_dir, ckpt)

        return 1
