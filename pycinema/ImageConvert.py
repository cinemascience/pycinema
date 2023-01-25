from .Core import *

import cv2

class ImageConvert(Filter):

    def __init__(self):
        super().__init__();
        self.addInputPort("conversion", cv2.COLOR_RGB2GRAY);
        self.addInputPort("images", []);
        self.addOutputPort("images", []);

    def update(self):
        result = []
        for image in self.inputs.images.get():

            if self.inputs.conversion.get() == cv2.COLOR_RGB2GRAY:
                cvi = cv2.cvtColor(image.channels['rgba'], cv2.COLOR_RGB2GRAY)
                cvfinal = cv2.cvtColor(cvi, cv2.COLOR_BGR2RGB)
                outImage = image.copy()
                outImage.channels['rgba'] = cvfinal
                result.append(outImage)
            elif self.inputs.conversion.get() == cv2.COLOR_RGB2BGR:
                cvi = cv2.cvtColor(image.channels['rgba'], cv2.COLOR_RGB2BGR)
                cvfinal = cv2.cvtColor(cvi, cv2.COLOR_BGR2RGB)
                outImage = image.copy()
                outImage.channels['rgba'] = cvfinal
                result.append(outImage)

        self.outputs.images.set(result)

        return 1;
