from pycinema import Filter, isNumber, getColumnFromTable

import PIL
import numpy as np

#
# PlotItem
#
# To be paired with a plot view
# Question: should this be a filter, or some new thing?
# Doesn't seem to fit the design of a view or filter
#
class PlotItem(Filter):

    def __init__(self, inputs={}, outputs={}):
        super().__init__(inputs, outputs)

    def _getColumnFromTable(self, colname):
        return getColumnFromTable(self.inputs.table.get(), colname)
