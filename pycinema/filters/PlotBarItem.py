from .PlotItem import * 

import numpy as np

#
# PlotBarItem
#
# To be paired with a plot view
# Question: should this be a filter, or some new thing?
# Doesn't seem to fit the design of a view or filter
#
class PlotBarItem(PlotItem):

    def __init__(self):
        super().__init__(
          inputs={
            'table'     : None,
            'x'         : 'none',
            'y'         : 'none',
            'brushcolor': 'default',
            'width'     : 1.0 
          },
          outputs={
            'plotitem' : {} 
          }
        )

    def _update(self):
        xID = self._getColumnIndex(self.inputs.x.get())
        xdata = self._getColumnFromTable(xID)
        yID = self._getColumnIndex(self.inputs.y.get())
        ydata = self._getColumnFromTable(yID)

        out = { 'x' : {
                        'label' : self.inputs.x.get(),
                        'data'  : xdata
                      },
                'y' : {
                        'label' : self.inputs.y.get(),
                        'data'  : ydata
                      },
                'brush' : {
                            'color' : self.inputs.brushcolor.get(), 
                        },
                'width' : self.inputs.width.get() 
              }
        self.outputs.plotitem.set({})
        self.outputs.plotitem.set(out)

        return 1
