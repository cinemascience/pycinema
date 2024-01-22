from pycinema import Filter

from PySide6 import QtCore, QtWidgets, QtGui

from .FilterView import FilterView

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
            'plotitem'  : []
          }
        )

    def generateWidgets(self):
        self.plot = pg.PlotWidget()
        self.content.layout().addWidget(self.plot)

    def _update(self):
        # clear on update
        self.plot.clear()

        # get plot items
        items = self.inputs.plotitem.get()
        if not isinstance(items, list):
          items = [items]

        self.plot.setBackground(self.inputs.background.get())
        self.plot.setTitle(self.inputs.title.get())

        # pen
        for item in items:
          self.plot.setLabel("left", item['y']['label'])
          self.plot.setLabel("bottom", item['x']['label'])
          pencolor = item['pen']['color']
          if pencolor == 'default':
              pencolor = 'black'
          itempen = pg.mkPen(color = pencolor, style=self.PenStyles[item['pen']['style']],width=item['pen']['width'])
          self.plot.plot(item['x']['data'], item['y']['data'], pen = itempen)

        return 1
