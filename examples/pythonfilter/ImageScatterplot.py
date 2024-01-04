from pycinema import getColumnFromTable, imageFromMatplotlibFigure
import matplotlib.pyplot as plt
import numpy as np

# use offscreen backend
import matplotlib as mpl
mpl.use('Agg')

outputs = []

# a general loop for looping over column pairings and marks
for cols, mark in zip([["phi", "theta"]], ['x']):

    figure = plt.figure(figsize=(10,8), dpi=200)
    x = getColumnFromTable(inputs, cols[0])
    xvals = np.asarray(x, dtype='float')
    # print(xvals)

    y = getColumnFromTable(inputs, cols[1])
    yvals = np.asarray(y, dtype='float')
    # print(yvals)

    plt.scatter(xvals, yvals, marker=mark)
    plt.xlabel(cols[0])
    plt.ylabel(cols[1])
    plt.title(cols[1] + " vs. " + cols[0])

    outputs.append( imageFromMatplotlibFigure(figure) )

plt.close(figure)

