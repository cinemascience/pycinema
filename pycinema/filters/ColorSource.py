from pycinema import Filter

import numpy

class ColorSource(Filter):

    def __init__(self):
        super().__init__(
          inputs={
            'rgba': (0,0,0,255)
          },
          outputs={
            'rgba': (0,0,0,255)
          }
        )

    def _update(self):
        self.outputs.rgba.set(
          self.inputs.rgba.get()
        )
        return 1
