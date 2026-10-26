import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from PIL import Image #pillow library used for handling images
from torchvision import transforms
import numpy as np




class ConvBlock(nn.Module):
   def __init__(self, in_channels, out_channels): #input shape (in_channels, 128, 128)
      super().__init__()
      self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1) #output shape (out_channels, 128, 128)
      self.groupnorm = nn.GroupNorm(8, out_channels)

   def forward(self, x):
    x = self.conv(x)
    x = self.groupnorm(x)
    x = F.relu(x)

    return x

class denoiser(nn.Module):
    def __init__(self):
      super().__init__()
        #input has shape (3, 128, 128)=(channels, height, width) , 3 channels because RGB


      #encoder layers:  
      self.conv1 = ConvBlock(3, 32) 
      self.conv2 = ConvBlock(32, 64) 
      self.conv3 = ConvBlock(64, 128)

      self.pool = nn.MaxPool2d(2) #halves height and width of image

      #latent representation
      self.latent = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)

      #decoder layers:
      self.convT1 = nn.ConvTranspose2d(256, 128, stride=2, kernel_size=2)
      self.convT2 = nn.ConvTranspose2d(256, 64, stride=2, kernel_size=2)
      self.convT3 = nn.ConvTranspose2d(128, 32, stride=2, kernel_size=2)

      #output layer
      self.output = nn.Conv2d(64, 3, kernel_size=3, padding=1)

    def forward(self, x):
       e1 = self.conv1(x) #output shape (32, 128, 128)
       e2 = self.conv2(self.pool(e1)) #(64, 64, 64)
       e3 = self.conv3(self.pool(e2)) #(128, 32, 32)

       latent = self.latent(self.pool(e3)) #(256, 16, 16)

       d1 = self.convT1(latent) #(128, 32, 32)
       d1 = F.relu(d1)
       d2 = self.convT2(torch.cat([d1, e3], dim=1)) #(64, 64, 64)
       d2 = F.relu(d2)
       d3 = self.convT3(torch.cat([d2, e2], dim=1)) #(32, 128, 128) 
       d3 = F.relu(d3)

      
       output = self.output(torch.cat([d3, e1], dim=1)) #(3, 128, 128)
       output = F.sigmoid(output)
       
       return latent, output

   




      
class denoiser_dataset(Dataset):  #subclass torch's Dataset module
    def __init__(self, files):
      self.files = files

    def __len__(self):
       return len(self.files)
    
    def __getitem__(self, i):
       img = Image.open(self.files[i])
       img = img.convert("RGB") #convert to RGB st every image has the same shape

       y = transforms.Resize((128, 128))(img)
       y = transforms.ToTensor()(y) #values ranging from 0 to 1

       noise = torch.rand_like(y) * 0.2 #torch.rand gives random values between 0 and 1 -> muliply by desired std
       x = y + noise
       x.clamp(0, 1) #values ranging from 0 to 1

       return x, y #input data and target
       
   

      
    