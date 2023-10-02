from pycinema import Filter

from PySide6 import QtCore, QtWidgets, QtGui

from .FilterView import FilterView

import pprint
import numpy as np
import pyqtgraph as pg

class PlotView(Filter, FilterView):
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
            'plotitems' : 'none',
            'table'     : []
          }
        )
        
    def getColumnIndex(self, colname):
        ID = 0

        colnames = self.inputs.table.get()[0]
        ID = colnames.index(colname)

        return ID 

    def generateWidgets(self):
        self.plot = pg.PlotWidget() 
        self.content.layout().addWidget(self.plot)

    def _update(self):
        # clear on update
        self.plot.clear()

        # get plot items
        items = self.inputs.plotitems.get()
        for i in items:
            axes = i[0]

            # get the ids of the value names
            xID = self.getColumnIndex(axes[0])
            yID = self.getColumnIndex(axes[1])

            # convert the table data and get the two arrays
            data = self.inputs.table.get()
            t = np.array(data)
            column = t[:, xID]
            xdata = column[1:].astype(float)
            row = t[:, yID]
            ydata = row[1:].astype(float)

            # pen
            newpen = pg.mkPen(color = i[2], style=self.PenStyles[i[1]],width=i[3])
            newpen.setStyle(self.PenStyles[i[1]])

            # set up the plot
            self.plot.setBackground(self.inputs.background.get())
            self.plot.setTitle(self.inputs.title.get())
            self.plot.setLabel("left", axes[0]) 
            self.plot.setLabel("bottom", axes[1]) 
            self.plot.plot(xdata, ydata, pen = newpen)

        return 1
