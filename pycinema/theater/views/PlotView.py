from pycinema import Filter

from PySide6 import QtCore, QtWidgets, QtGui

from .FilterView import FilterView

import pprint
import numpy as np
import pyqtgraph as pg

class PlotView(Filter, FilterView):

    def __init__(self):
        FilterView.__init__(
          self,
          filter=self,
          delete_filter_on_close = True
        )

        Filter.__init__(
          self,
          inputs={
            'title'      : 'Plot Title',
            'background' : 'white',
            'x_values'   : 'none',
            'y_values'   : 'none',
            'table'      : []
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

        # get the ids of the value names
        xID = self.getColumnIndex(self.inputs.x_values.get())
        yID = self.getColumnIndex(self.inputs.y_values.get())

        # convert the table data and get the two arrays
        data = self.inputs.table.get()
        t = np.array(data)
        column = t[:, xID]
        xdata = column[1:].astype(float)
        row = t[:, yID]
        ydata = row[1:].astype(float)

        # set up the plot
        self.plot.setBackground(self.inputs.background.get())
        self.plot.setTitle(self.inputs.title.get())
        self.plot.setLabel("left", self.inputs.y_values.get())
        self.plot.setLabel("bottom", self.inputs.x_values.get())
        self.plot.plot(xdata, ydata)

        return 1
