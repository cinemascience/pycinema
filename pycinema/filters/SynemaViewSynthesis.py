import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.training.train_state import TrainState
from synema.models.cinema import CinemaScalarImage
from synema.renderers.ray_gen import Parallel
from synema.renderers.rays import RayBundle
from synema.renderers.volume import DepthGuidedTrain, Hierarchical
from synema.samplers.pixel import Dense
from tqdm import tqdm

from pycinema import Filter
from pycinema.Core import Image


class SynemaViewSynthesis(Filter):

    @staticmethod
    def create_train_steps(key, model, optimizer):
        init_state = TrainState.create(apply_fn=model.apply,
                                       params=model.init(key, jnp.empty((1024, 3))),
                                       tx=optimizer)
        train_renderer = DepthGuidedTrain()

        def loss_fn(params, ray_bundle: RayBundle, targets, key: jax.random.PRNGKey):
            scalar, alpha, depth = train_renderer(field_fn=model.bind(params),
                                                  ray_bundle=ray_bundle,
                                                  rng_key=key,
                                                  depth_gt=targets['depth']).values()
            return (jnp.mean(optax.l2_loss(scalar, targets['scalar'])) +
                    1.e-3 * jnp.mean(jnp.abs(depth - jnp.nan_to_num(targets['depth']))))

        @jax.jit
        def train_step(state, ray_bundle: RayBundle, targets, key: jax.random.PRNGKey):
            loss_val, grads = jax.value_and_grad(loss_fn)(state.params, ray_bundle, targets, key)
            new_state = state.apply_gradients(grads=grads)
            return new_state, loss_val

        return train_step, init_state

    def __init__(self):
        super().__init__(
            inputs={
                'images': []
            },
            outputs={
                'images': []
            }
        )

        self.model = CinemaScalarImage()
        self.schedule_fn = optax.exponential_decay(init_value=1e-3,
                                                   transition_begin=600,
                                                   transition_steps=200,
                                                   decay_rate=0.5)
        self.optimizer = optax.adam(learning_rate=self.schedule_fn)
        key = jax.random.PRNGKey(12345)
        self.train_step, self.model_state = self.create_train_steps(key, self.model, self.optimizer)

    def training_loop(self, poses, depths, scalars):
        height, width = scalars.shape[1], scalars.shape[2]
        pixel_coordinates = Dense(width=width, height=height)()
        ray_generator = Parallel(width=width, height=height, viewport_height=1.)

        key = jax.random.PRNGKey(1377)
        pbar = tqdm(range(5000))
        for i in pbar:
            key, subkey = jax.random.split(key)
            image_idx = jax.random.randint(subkey, shape=(1,),
                                           minval=0, maxval=scalars.shape[0])[0]
            pose = poses[image_idx]
            depth = depths[image_idx]
            scalar = scalars[image_idx]
            targets = {'depth': depth.reshape((-1, 1)), 'scalar': scalar.reshape((-1, 1))}

            ray_bundle = ray_generator(pixel_coordinates,
                                       pose,
                                       t_near=0.,
                                       t_far=1.)
            key, subkey = jax.random.split(key)
            self.model_state, loss = self.train_step(self.model_state,
                                                     ray_bundle,
                                                     targets,
                                                     subkey)
            pbar.set_description("Loss: {:.4f}".format(loss))

    def generate_images(self):
        width, height = 256, 256

        polar_angles = np.linspace(15, 180, 3, endpoint=False)
        azimuthal_angles = np.linspace(0, 360, 3, endpoint=False)
        camera_angles = np.meshgrid(polar_angles, azimuthal_angles)

        pixel_coordinates = Dense(width=width, height=height)()
        ray_generator = Parallel(width=width, height=height, viewport_height=1.)

        key = jax.random.PRNGKey(1997)
        renderer = Hierarchical()
        images = []
        id = 0
        camera_up = np.array([0., 0., 1.])
        for polar, azimuthal in np.stack(camera_angles, axis=-1).reshape(-1, 2):
            polar = np.radians(polar)
            azimuthal = np.radians(azimuthal)
            camera_w = np.array([np.sin(polar) * np.cos(azimuthal),
                                 np.sin(polar) * np.sin(azimuthal),
                                 np.cos(polar)])
            camera_u = np.cross(camera_up, camera_w)
            camera_u = camera_u / np.linalg.norm(camera_u)
            camera_v = np.cross(camera_w, camera_u)
            camera_v = camera_v / np.linalg.norm(camera_v)
            # camera_u = np.array([-np.sin(azimuthal),
            #                      -np.cos(azimuthal),
            #                      0])
            # camera_v = np.array([np.cos(polar) * np.cos(azimuthal),
            #                      np.cos(polar) * np.sin(azimuthal),
            #                      np.sin(polar)])
            camera_pos_normalized = 0.5 * camera_w

            pose = np.zeros((4, 4))
            pose[:3, 0] = camera_u
            pose[:3, 1] = camera_v
            pose[:3, 2] = camera_w
            pose[:3, 3] = camera_pos_normalized
            pose[3, 3] = 1

            ray_bundle = ray_generator(pixel_coordinates,
                                       pose,
                                       t_near=0.,
                                       t_far=1.)

            key, subkey = jax.random.split(key)
            _, scalar_recon, _, depth_recon = renderer(self.model.bind(self.model_state.params),
                                                       self.model.bind(self.model_state.params),
                                                       ray_bundle,
                                                       subkey).values()
            channels = {'depth_recon': depth_recon.reshape((256, 256, 1)),
                        'scalar_recon': scalar_recon.reshape((256, 256))}
            meta = {'resolution': np.array([width, height]),
                    'polar': polar, 'azimuthal': azimuthal,
                    'id': id}
            images.append(Image(channels=channels, meta=meta))
            id += 1
        self.outputs.images.set(images)
        return 1

    def _update(self):
        input_images = self.inputs.images.get()

        poses = []
        depths = []
        scalars = []
        for image in input_images:
            meta = image.meta
            camera_dir = meta['CameraDir']
            camera_up = meta['CameraUp']

            # construct camera orientation matrix
            camera_w = -camera_dir / np.linalg.norm(camera_dir)
            camera_u = np.cross(camera_up, camera_w)
            camera_u = camera_u / np.linalg.norm(camera_u)
            camera_v = np.cross(camera_w, camera_u)
            camera_v = camera_v / np.linalg.norm(camera_v)

            # normalize the bbox to [-0.5, 0.5]^3 to prevent vanishing gradient.
            camera_pos_normalized = 0.5 * camera_w

            pose = np.zeros((4, 4))
            pose[:3, 0] = camera_u
            pose[:3, 1] = camera_v
            pose[:3, 2] = camera_w
            pose[:3, 3] = camera_pos_normalized
            pose[3, 3] = 1
            poses.append(pose)

            channels = image.channels
            depths.append(channels['Depth'])
            scalars.append(channels['Elevation'])

        poses = np.stack(poses, axis=0)
        depths = np.stack(depths, axis=0)
        depths = jnp.where(depths == 1., jnp.nan, depths)
        scalars = np.stack(scalars, axis=0)
        scalars = jnp.nan_to_num(scalars)

        self.training_loop(poses, depths, scalars)
        self.generate_images()
