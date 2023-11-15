from .PlotItem import *

#
# PlotLineItem
#
# To be paired with a plot view
# Question: should this be a filter, or some new thing?
# Doesn't seem to fit the design of a view or filter
#
class PlotLineItem(PlotItem):

    def __init__(self):
        super().__init__(
          inputs={
            'table'     : None,
            'x'         : 'none',
            'y'         : 'none',
            'penstyle'  : 'default',
            'pencolor'  : 'default',
            'penwidth'  : 1.0
          },
          outputs={
            'plotitem' : None
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
                'pen'  : {
                            'style' : self.inputs.penstyle.get(),
                            'color' : self.inputs.pencolor.get(),
                            'width' : self.inputs.penwidth.get()
                          }
              }
        self.outputs.plotitem.set({})
        self.outputs.plotitem.set(out)

        return 1
