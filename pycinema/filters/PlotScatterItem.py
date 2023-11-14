from .PlotItem import * 

import numpy as np

#
# PlotScatterItem
#
# To be paired with a plot view
# Question: should this be a filter, or some new thing?
# Doesn't seem to fit the design of a view or filter
#
class PlotScatterItem(PlotItem):

    def __init__(self):
        super().__init__(
          inputs={
            'table' : None,
            'x'     : 'none',
            'y'     : 'none',
            'pencolor'  : 'default',
            'penwidth'  : 1.0,
            'brushcolor': 'default',
            'symbol'    : 'x',
            'size'      : 1.0 
          },
          outputs={
            'plotitem' : {} 
          }
        )

    def _update(self):
        xdata = self._getColumnFromTable(self.inputs.x.get())
        ydata = self._getColumnFromTable(self.inputs.y.get())

        out = { 'x' : {
                        'label' : self.inputs.x.get(),
                        'data'  : xdata
                      },
                'y' : {
                        'label' : self.inputs.y.get(),
                        'data'  : ydata
                      },
                'pen' : {
                            'color' : self.inputs.pencolor.get(), 
                            'width' : self.inputs.penwidth.get(),
                        },
                'brush' : {
                            'color' : self.inputs.brushcolor.get() 
                          },
                'symbol': self.inputs.symbol.get(),
                'size'  : self.inputs.size.get() 
              }
        self.outputs.plotitem.set({})
        self.outputs.plotitem.set(out)

        return 1
