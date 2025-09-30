from pycinema import Filter, Image

import numpy as np
import libra
import cv2

from pycinema.filters import ImageInterpolator

class InterpolationCompression(Filter):

    def __init__(self):
        super().__init__(
          inputs={
            'images': [],
            'model_path': '',
            'error': 0.98
          },
          outputs={
            'images': []
          }
        )

    def _update(self):

      images = self.inputs.images.get()
      model_path = self.inputs.model_path.get()
      error = self.inputs.error.get()
      nImages = len(images)

      if nImages<1 or len(model_path)<1:
        self.outputs.images.set([])
        return 1

      outputs = []

      outputs.append(images[0])
      outputs[-1].meta['interpolated'] = False
      outputs[-1].meta['MS_SSIM'] = 1
      oIdx = 0

      model = ImageInterpolator.initialize_model(model_path)

      old_interpolations = []
      old_errors = []

      for i in range(1,nImages):
        a = outputs[-1]
        b = images[i]

        nInterpolations = i-oIdx-1
        if nInterpolations<1:
          continue

        interpolations = ImageInterpolator.inference_adaptive(
          model,
          a.channels['rgba'],
          b.channels['rgba'],
          nInterpolations
        )

        errors = []
        for j in range(0, len(interpolations)):
          errors.append(
            libra.compute_metric_(
              images[oIdx+1+j].channels['rgba'],
              interpolations[j],
              'MS-SSIM', 'LAB'
            )
          )

        print(i,errors)
        if any([e<=error for e in errors]) or i==nImages-1:
          print(i-1,'STOP')
          for j in range(0, len(old_interpolations)):
            outputs.append(Image(
              {'rgba':old_interpolations[j]},
              images[oIdx+1+j].meta
            ))
            outputs[-1].meta['interpolated'] = True
            outputs[-1].meta['MS_SSIM'] = old_errors[j]

          outputs.append(images[i-1])
          outputs[-1].meta['interpolated'] = False
          outputs[-1].meta['MS_SSIM'] = 1
          oIdx = i-1
          i-=1
          old_interpolations = []
          old_errors = []
        else:
          old_interpolations = interpolations
          old_errors = errors

      print(len(outputs))

      self.outputs.images.set(outputs)

      return 1;
