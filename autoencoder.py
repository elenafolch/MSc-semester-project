import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from PIL import Image #pillow library used for handling images
from torchvision import transforms
import numpy as np

from astropy.io import fits

class ConvBlock(nn.Module):
   def __init__(self, in_channels, out_channels): #input shape (in_channels, 60, 60)
      super().__init__()
      self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=2) #output shape (out_channels, 30, 30)
      self.groupnorm = nn.GroupNorm(8, out_channels)
   def forward(self, x):
    x = self.conv(x)
    x = self.groupnorm(x)
    x = F.relu(x)

    return x

class ConvTBlock(nn.Module):
   def __init__(self, in_channels, out_channels):
      super().__init__()
      self.convT = nn.ConvTranspose2d(in_channels, out_channels, 3, stride=2, padding=1, output_padding=1)
   def forward(self, x):
      x = self.convT(x)
      x = F.relu(x)

      return x
    

class autoenc(nn.Module):
    def __init__(self, in_channels):  #(1, 60, 60)
        super().__init__()
        #encoder
        self.conv1 = ConvBlock(in_channels, 32) #(32, 30, 30)
        self.conv2 = ConvBlock(32, 64) #(64, 15, 15)
        self.conv3 = ConvBlock(64, 128) #(128, 8, 8)

        #bottleneck
        self.bottleneck = ConvBlock(128, 256) #(256, 4, 4)

        #decoder
        self.convT1 = ConvTBlock(256, 128) #(128, 8, 8)
        self.convT2 = ConvTBlock(128, 64) #(64, 16, 16)
        self.convT3 = ConvTBlock(64, 32) #(32, 32, 32)

        self.output = nn.ConvTranspose2d(32, 1, kernel_size=3, stride=2, padding=3, output_padding=1) #(1, 60, 60) 
       

    def forward(self, x):
       e1 = self.conv1(x)
       e2 = self.conv2(e1)
       e3 = self.conv3(e2)

       latent = self.bottleneck(e3)

       d3 = self.convT1(latent)
       d2 = self.convT2(d3)
       d1 = self.convT3(d2)

       output = F.relu(self.output(d1))
       output = (output - output.min()) / (output.max() - output.min() + 1e-9) #rescaling to [0, 1]

       layers = [e1, e2, e3, d3, d2, d1]

       output_hdu = fits.PrimaryHDU(data=output.detach().numpy().astype(float))

       return latent, output, output_hdu, layers


class FizeauDataset(Dataset):
   def __init__(self, files): #files are .fits
      super().__init__()
      self.files = files
   def __len__(self):
      return len(self.files)

   def rescale(self, tens, lower, upper):
      mx = tens.max()
      mn = tens.min()
      return (tens - mn) / (mx - mn + 1e-9) * (upper - lower) + lower

   def __getitem__(self, index):
      img, hdr = fits.getdata(self.files[index], header=True)

      opd = torch.tensor(float(hdr["OPD"]), dtype=torch.float32)

      img = np.astype(img, np.float32)
      img = torch.from_numpy(img)
      img = torch.unsqueeze(img, dim=0)
      img = self.rescale(img, 0, 1)

      return img , opd
    

    
    
