from pycinema import Filter, isNumber

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

    def _getColumnIndex(self, colname):
        ID = 0

        colnames = self.inputs.table.get()[0]
        ID = colnames.index(colname)

        return ID 

    def _getColumnFromTable(self, colID):
        data = self.inputs.table.get()
        t = np.array(data)
        coldata = t[:, colID]
        column = coldata[1:]

        if isNumber(column[0]): 
            return column.astype(float)
        else:
            return column
