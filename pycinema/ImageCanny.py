from .Core import *

import cv2

# ImageCanny filter
# Runs opencv canny edge detection filter, and creates an image of
# those edges, with transparancy everywhere but the edges
class ImageCanny(Filter):

    def __init__(self):
        super().__init__();
        self.addInputPort("thresholds", [100, 150]);
        self.addInputPort("images", []);
        self.addOutputPort("images", []);

    def _update(self):
        result = []
        # iterate over all the images in the input images
        for image in self.inputs.images.get():
            # convert the input data into a form that cv uses
            cvimage = cv2.cvtColor(image.channels['rgba'], cv2.COLOR_RGB2BGR)
            thresholds = self.inputs.thresholds.get()

            # run the canny algorithm, using this object's thresholds
            canny = cv2.Canny(cvimage, thresholds[0], thresholds[1])/255

            outImage = image.copy()
            outImage.channels['canny'] = canny
            result.append(outImage)

        self.outputs.images.set(result)

        return 1;
