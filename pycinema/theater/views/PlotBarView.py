from pycinema import Filter

from PySide6 import QtCore, QtWidgets, QtGui

from .FilterView import FilterView

import pprint
import numpy as np
import pyqtgraph as pg

class PlotBarView(Filter, FilterView):
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
        # for now, there is only one item, but there will be a list in the future
        item = self.inputs.plotitems.get()

        # get the ids of the value names
        yID = self.getColumnIndex(item[0])

        # convert the table data and get the two arrays
        data = self.inputs.table.get()
        t = np.array(data)
        row = t[:, yID]
        ydata = row[1:].astype(float)
        # xdata is an array using the axes 0 value
        xdata = np.arange(len(ydata)) 

        # color
        barcolor = item[1]
        if barcolor == 'default':
            barcolor = 'black'

        # graph item
        bgItem = pg.BarGraphItem(x=xdata, height=ydata, width=item[2], brush=barcolor)

        # set up the plot
        self.plot.setBackground(self.inputs.background.get())
        self.plot.setTitle(self.inputs.title.get())
        self.plot.setLabel("left", item[0]) 
        self.plot.addItem(bgItem)

        return 1
