from pycinema import Filter, isNumber, getColumnFromTable, getLiteralValueOfList

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

    #
    # cast and clean data so that it can be passed on to plots
    #
    # - NaN values are passed on
    # - 'NaN' strings are converted to np.nan
    # - tuples with missing values are assumed to be invalid and
    #   are removed across all tuples
    #
    def _castAndCleanData(self, rawData):
        # empty results
        results = []

        # find indices of missing items
        missing = []
        for d in rawData:
            missing += [i for i,x in enumerate(d) if not x]

        missing = list(set(missing))

        # remove missing tuples across all data 
        for d in rawData:
            for i in sorted(missing, reverse=True):
                d.pop(i)

        # cast the remaining items
        for d in rawData:
            lval = getLiteralValueOfList(d) 

            if isinstance(lval, int):
                for i, v in enumerate(d):
                    if v.lower() == "nan":
                        d[i] = np.nan
                    else:
                        d[i] = int(v)
                # results.append([int(i) for i in d])
                results.append(d)

            elif isinstance(lval, float):
                for i, v in enumerate(d):
                    if v.lower() == "nan":
                        d[i] = np.nan
                    else:
                        d[i] = float(v)
                # results.append([float(i) for i in d])
                results.append(d)

            elif isinstance(lval, str):
                results.append(d)

        return results 
