import astra
import numpy as np

from .Core import *


class ImageGeneratorXinemax2D(Filter):

    def __init__(self):
        super().__init__()
        self.addInputPort("Model", "PathToModel")
        self.addOutputPort("Images", [])

    def update(self):
        super().update()

        # recreate the volume geometry. We know by fact lenna is a 128x128 image.
        vol_geom = astra.create_vol_geom(128, 128)

        # load the projected data
        sinogram = np.load(self.inputs.Model.get())['arr_0']
        # recreate projection geometry, again, parallel projection, 1 pixel, 128 detectors
        # and 30 scans
        proj_geom = astra.create_proj_geom('parallel', 1.0, 128, np.linspace(0, np.pi, 30, False))

        # create the internal data structure for the sinogram
        sinogram_id = astra.data2d.create('-sino', proj_geom, sinogram)

        # recreate the projector
        proj_id = astra.create_projector('linear', proj_geom, vol_geom)

        # Reconstruction, allocate internal memory for the result, i.e. 128x128 image
        rec_id = astra.data2d.create('-vol', vol_geom)

        # use the SIRT algorithm
        cfg = astra.astra_dict('SIRT')
        cfg['ReconstructionDataId'] = rec_id
        cfg['ProjectionDataId'] = sinogram_id
        cfg['ProjectorId'] = proj_id
        alg_id = astra.algorithm.create(cfg)

        # run the algorithm
        astra.algorithm.run(alg_id)

        # get the reconstructed image
        # TODO: how to turn the scalar field back to RGBA image with
        # colormap?
        rec = astra.data2d.get(rec_id)
        rgba = np.zeros((128, 128, 4))
        rgba[:, :, 0] = rec
        rgba[:, :, 1] = rec
        rgba[:, :, 2] = rec
        rgba[:, :, 3] = 1.0
        image = Image(
            {"rgba": rgba},
        )

        self.outputs.Images.set([image])
