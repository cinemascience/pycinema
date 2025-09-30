from pycinema import Filter, Image

import bisect
import os
from tqdm import tqdm
import torch
import numpy as np
import cv2

def pad_batch(batch, align):
    height, width = batch.shape[1:3]
    height_to_pad = (align - height % align) if height % align != 0 else 0
    width_to_pad = (align - width % align) if width % align != 0 else 0

    crop_region = [height_to_pad >> 1, width_to_pad >> 1, height + (height_to_pad >> 1), width + (width_to_pad >> 1)]
    batch = np.pad(batch, ((0, 0), (height_to_pad >> 1, height_to_pad - (height_to_pad >> 1)),
                           (width_to_pad >> 1, width_to_pad - (width_to_pad >> 1)), (0, 0)), mode='constant')
    return batch, crop_region

def to_padded_rgb(data, align=64):
    rgb_image = data[..., :3]
    rgb_image = rgb_image.astype(np.float32) / 255.0
    image_batch, crop_region = pad_batch(np.expand_dims(rgb_image, axis=0), align)
    return image_batch, crop_region

def interpolate_meta(meta1,meta2,l):
    meta = {}
    for key in meta1:
      a = meta1[key]
      b = meta2[key]
      if isinstance(a, str) and a.replace('.','',1).isdigit():
        a = float(a)
        b = float(b)
      if isinstance(a, str) or isinstance(a, set):
        continue

      meta[key] = (1 - l) * a + l * b

    return meta

class ImageInterpolator(Filter):

    def __init__(self):
        self.cache = {}

        super().__init__(
          inputs={
            'images': [],
            'model_path': '',
            'nFrames': 1,
            'adaptive': True
          },
          outputs={
            'images': []
          }
        )

    def initialize_model(model_path):
      model = torch.jit.load(model_path, map_location='cpu')
      model.eval()
      model.float()
      if torch.cuda.is_available():
        model = model.cuda()
      return model

    def inference_linear(model, channel1, channel2, nFrames, half=False):
      img_batch_1, crop_region_1 = to_padded_rgb(channel1)
      img_batch_2, crop_region_2 = to_padded_rgb(channel2)

      gpu = torch.cuda.is_available()

      frame1 = torch.from_numpy(img_batch_1).permute(0, 3, 1, 2).cuda()
      frame2 = torch.from_numpy(img_batch_2).permute(0, 3, 1, 2).cuda()

      results = []
      for i in tqdm(range(1, nFrames + 1), desc="Generating interpolated frames"):
          interpolation_parameter = i / (nFrames + 1)

          dt = torch.tensor([[interpolation_parameter]], dtype=torch.float32, device='cuda')
          with torch.no_grad():
              prediction = model(frame1, frame2, dt)

          results.append(prediction.clamp(0, 1).cpu().float())

      y1, x1, y2, x2 = crop_region_1
      return [
        np.ascontiguousarray(np.concatenate(
          [((tensor[0] * 255).byte().permute(1, 2, 0).numpy()[y1:y2, x1:x2]), np.ones((y2 - y1, x2 - x1, 1), dtype=np.uint8) * 255], axis=-1
        ))
        for tensor in results[1:-1]
      ]

    def inference_adaptive(model, channel1, channel2, nFrames, half=False):
      img_batch_1, crop_region_1 = to_padded_rgb(channel1)
      img_batch_2, crop_region_2 = to_padded_rgb(channel2)

      img_batch_1 = torch.from_numpy(img_batch_1).permute(0, 3, 1, 2)
      img_batch_2 = torch.from_numpy(img_batch_2).permute(0, 3, 1, 2)

      gpu = torch.cuda.is_available()

      results = [
          img_batch_1,
          img_batch_2
      ]

      idxes = [0, nFrames + 1]
      remains = list(range(1, nFrames + 1))

      splits = torch.linspace(0, 1, nFrames + 2)

      for _ in tqdm(range(len(remains)), 'Generating in-between frames'):
          starts = splits[idxes[:-1]]
          ends = splits[idxes[1:]]
          distances = ((splits[None, remains] - starts[:, None]) / (ends[:, None] - starts[:, None]) - .5).abs()
          matrix = torch.argmin(distances).item()
          start_i, step = np.unravel_index(matrix, distances.shape)
          end_i = start_i + 1

          x0 = results[start_i]
          x1 = results[end_i]

          if gpu:
              if half:
                  x0 = x0.half()
                  x1 = x1.half()
              x0 = x0.cuda()
              x1 = x1.cuda()

          dt = x0.new_full((1, 1), (splits[remains[step]] - splits[idxes[start_i]])) / (splits[idxes[end_i]] - splits[idxes[start_i]])

          with torch.no_grad():
              prediction = model(x0, x1, dt)
          insert_position = bisect.bisect_left(idxes, remains[step])
          idxes.insert(insert_position, remains[step])
          results.insert(insert_position, prediction.clamp(0, 1).cpu().float())
          del remains[step]

      y1, x1, y2, x2 = crop_region_1
      return [
        np.ascontiguousarray(np.concatenate(
          [((tensor[0] * 255).byte().permute(1, 2, 0).numpy()[y1:y2, x1:x2]), np.ones((y2 - y1, x2 - x1, 1), dtype=np.uint8) * 255], axis=-1
        ))
        for tensor in results[1:-1]
      ]

    def _update(self):

        images = self.inputs.images.get()
        model_path = self.inputs.model_path.get()
        nFrames = self.inputs.nFrames.get()
        adaptive = self.inputs.adaptive.get()

        if len(images)<1 or nFrames<1 or len(model_path)<1:
          self.outputs.images.set([])
          return 1

        # model
        model = ImageInterpolator.initialize_model(model_path)

        # generate images
        output_images = []
        for i in range(0,len(images)-1):
          rgba1 = images[i].channels['rgba']
          rgba2 = images[i+1].channels['rgba']
          if adaptive:
            rgbaI = ImageInterpolator.inference_adaptive( model, rgba1, rgba2, nFrames )
          else:
            rgbaI = ImageInterpolator.inference_linear( model, rgba1, rgba2, nFrames )

          output_images.append(Image(
            {'rgba': rgba1},
            interpolate_meta(images[i].meta,images[i+1].meta,0)
          ))
          for l in range(0,len(rgbaI)):
            output_images.append(Image(
              {'rgba':rgbaI[l]},
              interpolate_meta(images[i].meta,images[i+1].meta,(l+1)/(nFrames+1))
            ))
        output_images.append(Image(
          {'rgba':images[-1].channels['rgba']},
          interpolate_meta(images[-1].meta,images[-1].meta,1)
        ))

        self.outputs.images.set(output_images)

        return 1
