from .Core import *

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

################################################################
class BasicModel(nn.Module):
  def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
               padding=1, activation=F.relu, upsample=True):
    super(BasicModel, self).__init__()

    self.activation = activation
    self.upsample = upsample
    self.conv_res = None
    if self.upsample or in_channels != out_channels:
      self.conv_res = nn.Conv2d(in_channels, out_channels,
                                1, 1, 0, bias=False)

    self.bn0 = nn.BatchNorm2d(in_channels)
    self.conv0 = nn.Conv2d(in_channels, out_channels, kernel_size,
                           stride, padding, bias=False)

    self.bn1 = nn.BatchNorm2d(out_channels)
    self.conv1 = nn.Conv2d(out_channels, out_channels, kernel_size,
                           stride, padding, bias=False)

  def forward(self, x):
    residual = x
    if self.upsample:
      residual = F.interpolate(residual, scale_factor=2)
    if self.conv_res is not None:
      residual = self.conv_res(residual)

    out = self.bn0(x)
    out = self.activation(out)
    if self.upsample:
      out = F.interpolate(out, scale_factor=2)
    out = self.conv0(out)

    out = self.bn1(out)
    out = self.activation(out)
    out = self.conv1(out)

    return out + residual

class Model(nn.Module):
    
    def __init__(self, vp=2, vpo=256, ch=16):
        super(Model, self).__init__()

        self.vp = vp
        self.vpo = vpo
        self.ch = ch

        #  subnet for parameters
        self.vparams_subnet = nn.Sequential(
            nn.Linear(vp, 2*vpo)
            ,nn.ReLU()
            ,nn.Linear(2*vpo, 2*vpo)
            ,nn.ReLU()
            ,nn.Linear(2*vpo,  4 * 4 * 32 * ch, bias=False)
        )

        # subnet for image 
        self.img_subnet = nn.Sequential(
            BasicModel(ch * 32, ch * 16, kernel_size=3, stride=1, padding=1), # in = 4x4, out = 8x8 
            BasicModel(ch * 16, ch * 8, kernel_size=3, stride=1, padding=1), # in = 8x8, out = 16x16
            BasicModel(ch * 8, ch * 4, kernel_size=3, stride=1, padding=1), # in = 16x16, out = 32x32
            BasicModel(ch * 4, ch * 2, kernel_size=3, stride=1, padding=1), # in = 32x32, out = 64x64
            BasicModel(ch * 2, ch * 1, kernel_size=3, stride=1, padding=1), # in = 64x64, out = 128x128
            nn.BatchNorm2d(ch),
            nn.ReLU(),
            nn.Conv2d(ch, 3, kernel_size=3, stride=1, padding=1),
            nn.Tanh()
        )
        
    # Forward function
    def forward(self,vp):
        vp = self.vparams_subnet(vp)
        x = vp.view(vp.size(0), self.ch * 32, 4, 4)
        x = self.img_subnet(x)
        return x

class ImageGeneratorCNN(Filter):

  def __init__(self):
    super().__init__();
    #self.addInputPort("Query", "SELECT * FROM input");
    self.addInputPort("Params",[[0.0,0.0]]);
    self.addInputPort("Model", "PathToModel");
    self.addInputPort("Channel",8);
    self.addInputPort("VP",2);
    self.addInputPort("VPO",256);
    self.addInputPort("Device","cpu");
    self.addOutputPort("Images", []);
    #self.addOutputPort("Pop", []);

  def update(self):
    super().update()

    # Load the trained model
    model = Model(  vp=self.inputs.VP.get(), 
                    vpo=self.inputs.VPO.get(), 
                    ch=self.inputs.Channel.get());
  
    checkpoint = torch.load(self.inputs.Model.get(), map_location=self.inputs.Device.get());
    model.load_state_dict(checkpoint["model_state_dict"]);
    
    param = torch.from_numpy(np.asarray(self.inputs.Params.get(), dtype='float32'))

    model.eval()
    out_image = model(param)
    out_image = ((out_image+ 1.) * .5)

    # Discard first axis from output
    nparray = out_image.detach().numpy()[0,:,:,:]
    # Change the range to 0 - 255 with uint8 type
    nparray = (nparray*255.0).astype(np.uint8)
    # Swap 0 - 2 axes to go from pytorch to numpy array form
    nparray = np.swapaxes(nparray,0,2)

    #self.outputs.Pop.set(out_image.detach())

    params = self.inputs.Params.get()
    if len(params[0]) == 3:
        generatedImage = Image(
            {
                'rgba': nparray # get numpy array from pytorch
            },
            {
                'Phi':      params[0][0],
                'Theta':    params[0][1],
                'Time':     params[0][2],
                'Source':   'Estimated'
            }
        )
    else:
        generatedImage = Image(
            {
                'rgba': nparray # get numpy array from pytorch
            },
            {
                'Phi':      params[0][0],
                'Theta':    params[0][1],
                'Source':   'Estimated'
            }
        )

    self.outputs.Images.set([generatedImage]);

    return 1;
