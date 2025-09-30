from pycinema import Filter

import cv2
import numpy as np

class VideoWriter(Filter):

    def __init__(self):
        self.cache = {}

        super().__init__(
          inputs={
            'images': [],
            'path': '',
            'fps': 30,
          },
          outputs={
          }
        )

    def _update(self):

        images = self.inputs.images.get()
        path = self.inputs.path.get()
        fps = self.inputs.fps.get()
        if not path or len(images)<1:
            return 1

        height, width, channels = images[0].channels['rgba'].shape

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4 format
        out = cv2.VideoWriter(path, fourcc, fps, (width, height))

        for image in images:
            frame_bgr = cv2.cvtColor(image.channels['rgba'], cv2.COLOR_RGBA2BGR)
            out.write(frame_bgr)

        out.release()

        return 1
