from pycinema import Filter
from .FilterView import FilterView

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
        # clear and set attributes 
        self.plot.clear()
        self.plot.setBackground(self.inputs.background.get())
        self.plot.setTitle(self.inputs.title.get())

        # get and plot items 
        items = self.inputs.plotitem.get()
        if not isinstance(items, list):
            items = [items]

        for item in items:
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
