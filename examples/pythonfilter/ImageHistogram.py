from pycinema import getColumnFromTable, imageFromMatplotlibFigure
import matplotlib.pyplot as plt
import numpy as np
import PIL

# use offscreen backend
import matplotlib as mpl
mpl.use('Agg')

outputs = []

imDPI = 200

# a general loop for looping over column pairings and marks
for image in inputs: 
    rgbImage = PIL.Image.fromarray(image.channels['rgba'])
    # for this layout, make the plots the same size as the images
    width, height = rgbImage.size

    r, g, b, a = rgbImage.split()
    counts, bins = np.histogram(r)

    figure = plt.figure(figsize=(float(2*width/imDPI),float(2*height/imDPI)), dpi=imDPI, linewidth=1, edgecolor="black")
    figure.patch.set_facecolor('whitesmoke')
    plt.stairs(counts, bins, fill=True)
    plt.xlabel("bins")
    plt.ylabel("number of pixels")
    plt.title("Red Channel Histogram")

    outputs.append( imageFromMatplotlibFigure(figure) )

    plt.close(figure)

