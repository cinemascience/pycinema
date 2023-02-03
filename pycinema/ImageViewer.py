from .Core import *

# import math
import IPython
from ipycanvas import Canvas
import ipywidgets

import numpy

class ImageViewer(Filter):

    def __init__(self):
        super().__init__()

        self.addInputPort("images", [])
        self.addInputPort("container", None)

        self.nImages = -1

        self.canvases = []

    def _update(self):

        container = self.inputs.container.get()
        if container==None:
            container = ipywidgets.HBox()
            IPython.display.display(container)

        images = self.inputs.images.get()
        nImages = len(images)

        if self.nImages != nImages:
            self.nImages = nImages

            canvases = []
            container.layout.flex_flow='row wrap'
            for i,image in enumerate(images):
                canvas = Canvas(width=image.shape[1], height=image.shape[0])
                canvas.layout.min_width=str(image.shape[1])+"px"
                canvas.layout.min_height=str(image.shape[0])+"px"
                canvas.layout.object_fit= 'contain'
                canvas.layout.margin= '0 0.5em 1em 0.5em'
                canvas.layout.border= '0.01em solid #ccc'
                canvases.append(canvas)
            container.children = canvases

        for i,image in enumerate(images):
            # container.children[i].clear_rect(0, 0, image.channels['rgba'].shape[1], image.channels['rgba'].shape[0])
            container.children[i].put_image_data(image.channels['rgba'], 0, 0)

        return 1
