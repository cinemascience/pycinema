from pycinema import Filter

from PySide6 import QtCore, QtWidgets, QtGui

from .FilterView import FilterView

import pprint
import numpy as np
import pyqtgraph as pg

class PlotLineView(Filter, FilterView):
    PenStyles = {
                    'dash'      : QtCore.Qt.DashLine,
                    'dashdot'   : QtCore.Qt.DashDotLine,
                    'dashdotdot': QtCore.Qt.DashDotDotLine,
                    'default'   : QtCore.Qt.SolidLine,
                    'dot'       : QtCore.Qt.DotLine,
                    'solid'     : QtCore.Qt.SolidLine
                }

    def __init__(self):
        FilterView.__init__(
          self,
          filter=self,
          delete_filter_on_close = True
        )

        Filter.__init__(
          self,
          inputs={
            'title'     : 'Plot Title',
            'background': 'white',
            'plotitem'  : 'none'
          }
        )
        
    def generateWidgets(self):
        self.plot = pg.PlotWidget() 
        self.content.layout().addWidget(self.plot)

    def _update(self):
        # clear on update
        self.plot.clear()

        # get plot items
        # in the future there will be more than one item, but for now only one
        item = self.inputs.plotitem.get()

        # pen
        pencolor = item['line']['color']
        if pencolor == 'default':
            pencolor = 'black'
        newpen = pg.mkPen(color = pencolor, style=self.PenStyles[item['line']['type']],width=item['line']['width'])

        # set up the plot
        self.plot.setBackground(self.inputs.background.get())
        self.plot.setTitle(self.inputs.title.get())
        self.plot.setLabel("left", item['y']['label']) 
        self.plot.setLabel("bottom", item['x']['label']) 
        self.plot.plot(item['x']['data'], item['y']['data'], pen = newpen)

        return 1
