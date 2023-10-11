from pycinema import Filter

import PIL
import numpy

#
# PlotBarItem
#
# To be paired with a plot view
# Question: should this be a filter, or some new thing?
# Doesn't seem to fit the design of a view or filter
#
class PlotBarItem(Filter):

    def __init__(self):
        super().__init__(
          inputs={
            'y'     : 'none',
            'color' : 'default',
            'width' : 1.0 
          },
          outputs={
            'item' : 'none'
          }
        )

    def _update(self):
        out = [ self.inputs.y.get(), 
                self.inputs.color.get(), self.inputs.width.get()]
        self.outputs.item.set(out)

        return 1
