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
        xdata = self._getColumnFromTable(self.inputs.x.get())
        ydata = self._getColumnFromTable(self.inputs.y.get())

        cleanData = self._castAndCleanData([xdata, ydata])

        out = { 'x' : {
                        'label' : self.inputs.x.get(),
                        'data'  : cleanData[0]
                      },
                'y' : {
                        'label' : self.inputs.y.get(),
                        'data'  : cleanData[1]
                      },
                'brush' : {
                            'color' : self.inputs.brushcolor.get(), 
                        },
                'width' : self.inputs.width.get() 
              }
        self.outputs.plotitem.set({})
        self.outputs.plotitem.set(out)

        return 1
