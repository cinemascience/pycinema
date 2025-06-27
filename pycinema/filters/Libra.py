from pycinema import Filter, Image

# Add the third-party repoSrcDir to sys.path
# third_party_dir = os.path.join(os.path.dirname(__file__), 'third-party', 'libra')
# sys.path.insert(0, third_party_dir)

import numpy as np
import libra
import cv2

class Libra(Filter):

    def __init__(self):
        super().__init__(
          inputs={
            'imagesA': [],
            'imagesB': [],
          },
          outputs={
            'images': []
          }
        )

    def _update(self):

      imagesA = self.inputs.imagesA.get()
      imagesB = self.inputs.imagesB.get()

      outputs = []

      metrics = [
        'MSE',
        'SSIM',
        'MS-SSIM',
        'PSNR',
        'VSI',
        'SR-SIM',
        'MS-GMSD',
        'LPIPS',
        'PieAPP',
        'DISTS',
        "MDSI",
        "DSS",
        "IW-SSIM",
        "VIFp",
        "GMSD",
        "HaarPSI",
        "PHASH",

        # "BRISQUE", # expensive
        # "NIQE", # expensive
        # "MUSIQ", # expensive
        # "NIMA", # expensive

        # 'FSIM', # broken
        # "CLIPIQA", # broken
      ]
          # metrics = libra.list_metrics()

      for i in range(0,len(imagesA)):
        a = imagesA[i].channels['rgba']
        b = imagesB[i].channels['rgba']

        heatmap, heatmapEq = libra.diff_images_(a, b, 0,'HSV')
        outputs.append(
          Image({
              'error': heatmap
            },
            {
              m: libra.compute_metric_(a, b, m, 'LAB') for m in metrics
            }
          )
        )

      self.outputs.images.set(outputs)

      return 1;
