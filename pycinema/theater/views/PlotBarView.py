from pycinema import Filter

from PySide6 import QtCore, QtWidgets, QtGui

from .FilterView import FilterView

import numpy as np
import pyqtgraph as pg

class PlotBarView(Filter, FilterView):

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
        # for now, there is only one item, but there will be a list in the future
        item = self.inputs.plotitem.get()

        # brush
        brushcolor = item['brush']['color']
        if brushcolor == 'default':
            brushcolor = 'black'

        # graph item
        bgItem = pg.BarGraphItem(x=item['x']['data'], height=item['y']['data'], width=item['width'], brush=brushcolor)

        # set up the plot
        self.plot.setBackground(self.inputs.background.get())
        self.plot.setTitle(self.inputs.title.get())
        self.plot.setLabel("left", item['y']['label']) 
        self.plot.addItem(bgItem)

        return 1
