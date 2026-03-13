from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from skimage.util import img_as_ubyte
from synema.renderers.ray_gen import Parallel
from synema.renderers.volume import Hierarchical
from synema.samplers.pixel import Dense

from pycinema import Filter
from pycinema.Core import Image


class SynemaColorImageViewSynthesis(Filter):

    def __init__(self):
        super().__init__(
            inputs={
                'model_state': [],
                'camera': [0, 0]
            },
            outputs={
                'images': []
            }
        )
        self.key = jax.random.PRNGKey(1997)
        self.width = 100
        self.height = 100
        self.renderer = Hierarchical()

    @partial(jax.jit, static_argnums={0, 1})
    def generate_images(self, model, state, pose, key):
        pixel_coordinates = Dense(width=self.width, height=self.height)()
        ray_generator = Parallel(width=self.width, height=self.height, viewport_height=1.)

        ray_bundle = ray_generator(pixel_coordinates,
                                   pose,
                                   t_near=0.,
                                   t_far=1.)

        _, rgb_recon, alpha_recon, _ = self.renderer(model.bind(state.params),
                                                     model.bind(state.params),
                                                     ray_bundle,
                                                     key).values()
        return jnp.concatenate([rgb_recon, alpha_recon[:, None]], axis=-1)

    def _update(self):
        azimuthal, elevation = self.inputs.camera.get()
        polar = np.radians(90. - elevation)
        azimuthal = np.radians(azimuthal)

        # construct camera orientation matrix
        camera_u = np.array([-np.sin(azimuthal), np.cos(azimuthal), 0])
        camera_v = -np.array([np.cos(polar) * np.cos(azimuthal),
                              np.cos(polar) * np.sin(azimuthal),
                              -np.sin(polar)])
        camera_w = np.array([np.sin(polar) * np.cos(azimuthal),
                             np.sin(polar) * np.sin(azimuthal),
                             np.cos(polar)])

        # normalize the bbox to [-0.5, 0.5]^3 to prevent vanishing gradient.
        camera_pos_normalized = 0.5 * camera_w

        pose = np.zeros((4, 4))
        pose[:3, 0] = camera_u
        pose[:3, 1] = camera_v
        pose[:3, 2] = camera_w
        pose[:3, 3] = camera_pos_normalized
        pose[3, 3] = 1

        model = self.inputs.model_state.get()['model']
        state = self.inputs.model_state.get()['state']
        with jax.log_compiles():
            rgb_recon = self.generate_images(model, state,
                                             pose,
                                             jax.random.fold_in(key=self.key, data=abs(self.time)))
            rgb_recon = jnp.clip(rgb_recon, 0, 1)
            channels = {'rgba': img_as_ubyte(jax.device_get(rgb_recon).reshape((self.width, self.height, 4)))}
            meta = {'resolution': np.array([self.width, self.height]),
                    'elevation': elevation,
                    'azimuthal': np.degrees(azimuthal),
                    'id': id}
            self.outputs.images.set([Image(channels=channels, meta=meta)])

        return 1
