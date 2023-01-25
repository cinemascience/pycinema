from .Core import *
import numpy

class ColorSource(Filter):

    def __init__(self):
        super().__init__()

        self.addInputPort('rgba', (0,0,0,255))
        self.addOutputPort('rgba', (0,0,0,255))

    def update(self):

        self.outputs.rgba.set(
          self.inputs.rgba.get()
        )

        return 1
