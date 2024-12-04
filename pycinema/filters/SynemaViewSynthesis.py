from functools import partial

import jax
import numpy as np
from synema.renderers.ray_gen import Parallel
from synema.renderers.volume import Hierarchical
from synema.samplers.pixel import Dense

from pycinema import Filter
from pycinema.Core import Image


class SynemaViewSynthesis(Filter):

    def __init__(self):
        super().__init__(
            inputs={
                'model_state': [],
                'camera': [30, 30]
            },
            outputs={
                'images': []
            }
        )
        self.key = jax.random.PRNGKey(1997)
        self.width = 100
        self.height = 100
        self.pixel_coordinates = Dense(width=self.width, height=self.height)()
        self.ray_generator = Parallel(width=self.width, height=self.height, viewport_height=1.)
        self.renderer = Hierarchical()

    @partial(jax.jit, static_argnums=0)
    def generate_images(self, pose, key):
        ray_bundle = self.ray_generator(self.pixel_coordinates,
                                        pose,
                                        t_near=0.,
                                        t_far=1.)

        _, scalar_recon, _, depth_recon = self.renderer(self.model.bind(self.state.params),
                                                        self.model.bind(self.state.params),
                                                        ray_bundle,
                                                        key).values()
        return scalar_recon, depth_recon

    def _update(self):
        azimuthal, polar = self.inputs.camera.get()
        polar = np.radians(polar)
        azimuthal = np.radians(azimuthal)

        camera_up = np.array([0., 0., 1.])
        camera_w = np.array([np.sin(polar) * np.cos(azimuthal),
                             np.sin(polar) * np.sin(azimuthal),
                             np.cos(polar)])
        camera_u = np.cross(camera_up, camera_w)
        camera_u = camera_u / np.linalg.norm(camera_u)
        camera_v = np.cross(camera_w, camera_u)
        camera_v = camera_v / np.linalg.norm(camera_v)

        camera_pos_normalized = 0.5 * camera_w

        pose = np.zeros((4, 4))
        pose[:3, 0] = camera_u
        pose[:3, 1] = camera_v
        pose[:3, 2] = camera_w
        pose[:3, 3] = camera_pos_normalized
        pose[3, 3] = 1

        self.model = self.inputs.model_state.get()['model']
        self.state = self.inputs.model_state.get()['state']
        with jax.transfer_guard("log"), jax.log_compiles():
            scalar_recon, depth_recon = self.generate_images(pose, jax.random.fold_in(key=self.key, data=self.time))

            channels = {'scalar_recon': scalar_recon.reshape((self.width, self.height)),
                        'depth_recon': depth_recon.reshape((self.width, self.height, 1))}
            meta = {'resolution': np.array([self.width, self.height]),
                    'polar': polar, 'azimuthal': azimuthal,
                    'id': id}
            images = [Image(channels=channels, meta=meta)]
            self.outputs.images.set(images)

        return 1
