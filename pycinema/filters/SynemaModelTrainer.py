import jax
import jax.numpy as jnp
import numpy as np
import optax
from synema.renderers.ray_gen import Parallel
from synema.renderers.rays import RayBundle
from synema.renderers.volume import DepthGuidedTrain
from synema.samplers.pixel import Dense
from tqdm import tqdm

from pycinema import Filter


class SynemaModelTrainer(Filter):
    @staticmethod
    def create_train_steps(model):
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

        return train_step

    def __init__(self):
        super().__init__(
            inputs={
                'model_state': [],
                'images': [],
                'epochs': 0
            },
            outputs={
                'model_state': [],
            }
        )

    def training_loop(self, poses, depths, scalars):
        height, width = scalars.shape[1], scalars.shape[2]
        pixel_coordinates = Dense(width=width, height=height)()
        ray_generator = Parallel(width=width, height=height, viewport_height=1.)

        key = jax.random.PRNGKey(1377)
        pbar = tqdm(range(self.inputs.epochs.get()))
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
            self.state, loss = self.train_step(self.state,
                                               ray_bundle,
                                               targets,
                                               subkey)
            pbar.set_description("Loss: {:.4f}".format(loss))

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

        self.model = self.inputs.model_state.get()['model']
        self.state = self.inputs.model_state.get()['state']
        self.train_step = self.create_train_steps(self.model)

        self.training_loop(poses, depths, scalars)

        self.outputs.model_state.set({'model': self.model, 'state': self.state})
