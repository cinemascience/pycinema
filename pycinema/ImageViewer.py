from .Core import *

import matplotlib.pyplot as plt
import math

class ImageViewer(Filter):

    def __init__(self):
        super().__init__()

        self.addInputPort("images", [])
        self.addInputPort("container", None)

        self.nImages = -1

        plt.ioff()
        self.fig = plt.figure()
        self.fig.tight_layout()

    def update(self):

        container = self.inputs.container.get()
        if container == None:
            self.fig.show()
        else:
            with container:
                self.fig.show()

        images = self.inputs.images.get()
        nImages = len(images)

        if self.nImages != nImages:
            self.nImages = nImages
            self.fig.clear()
            self.plots = []
            dim = math.ceil(math.sqrt(self.nImages))
            for i,image in enumerate(images):
                axis = self.fig.add_subplot(dim, dim, i+1)
                axis.set_axis_off()

                if not 'rgba' in image.channels:
                    self.plots.append( [axis,None] )
                else:
                    im = axis.imshow(image.channels['rgba'])
                    self.plots.append( [axis,im] )

        for i,image in enumerate(images):
            if not 'rgba' in image.channels:
                continue
            if self.plots[i][1] == None:
                self.plots[i][1] = self.plots[i][0].imshow(image.channels['rgba'])
            else:
                self.plots[i][1].set_data(image.channels['rgba'])

        self.fig.subplots_adjust(
            left=0,
            bottom=0,
            right=1,
            top=1,
            wspace=0.05,
            hspace=0.05
        )

        self.fig.canvas.draw()

        return 1
