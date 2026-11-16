import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from PIL import Image #pillow library used for handling images
from torchvision import transforms
import numpy as np

from astropy.io import fits
import json


class ConfigureModel:
    def __init__(self, n_encoder_layers):
        self.n_encoder_layers = n_encoder_layers



class configAutoenc(nn.Module):
    def __init__(self, in_channels, in_height, configuration: ConfigureModel):  #(1, 60, 60)
            super().__init__()
            self.in_channels = in_channels
            self.in_height = in_height
            self.configuration = configuration
            self.n_encoder_layers = self.configuration.n_encoder_layers

            height = in_height
            input_maps = in_channels
            feature_maps = 32
            kernel_size = 3
            padding = 1
            stride = 2
            dilation = 1 #default value

            for i in range(self.n_encoder_layers):
                  encoder_layer = nn.Sequential(nn.Conv2d(input_maps, feature_maps, kernel_size=kernel_size, padding=padding, stride=stride), 
                                                  nn.GroupNorm(num_groups=8, num_channels=feature_maps), 
                                                  nn.ReLU())
                  setattr(self, f"e{i}", encoder_layer)
                  
                  input_maps = feature_maps
                  feature_maps = 2*feature_maps
                  height = int((height + 2*padding - dilation * (kernel_size - 1) - 1)/stride + 1)

            self.bottleneck = nn.Sequential(nn.Conv2d(input_maps, feature_maps, kernel_size=kernel_size, padding=padding, stride=stride), 
                                                  nn.GroupNorm(8, feature_maps), 
                                                  nn.ReLU())
            
            decoded_feature_maps = input_maps
            height = int((height + 2*padding - dilation * (kernel_size - 1) - 1)/stride + 1)
            self.bottleneck_dim = height

            kernel_size = 3
            stride=2
            padding=1
            output_padding=1

            for i in range(self.n_encoder_layers):
                  decoder_layer = nn.Sequential(nn.ConvTranspose2d(in_channels=feature_maps, out_channels=decoded_feature_maps, kernel_size=kernel_size, stride=stride, padding=padding, output_padding=output_padding), 
                                     nn.GroupNorm(8, decoded_feature_maps), 
                                     nn.ReLU())
                  setattr(self, f"d{i}", decoder_layer)
                  
                  feature_maps = decoded_feature_maps
                  decoded_feature_maps = decoded_feature_maps//2

            self.output = nn.Sequential(nn.ConvTranspose2d(feature_maps, self.in_channels, kernel_size=3, stride=2, padding=3, output_padding=1))


    def forward(self, x):
            latent_reps = []
            for i in range(self.n_encoder_layers):
                  new_x = getattr(self, f"e{i}")(x)
                  latent_reps.append(new_x)
                  x = new_x

            bottleneck_rep = self.bottleneck(x)
            x = bottleneck_rep

            for i in range(self.n_encoder_layers):
                              new_x = getattr(self, f"d{i}")(x)
                              latent_reps.append(new_x)
                              x = new_x

            output = self.output(x)
            output = (output - output.min()) / (output.max() - output.min() + 1e-9)
            
            return output, bottleneck_rep, latent_reps, self.bottleneck_dim
            
def random_indices(low, high, size, tuples):
       """
       returns an array containing random tuples of indices without repetition
       """
       index_tuples = np.zeros((tuples, size))

       for i in range(tuples):
              indices = np.array([np.random.randint(low=low, high=high, size=size)])

              for k in range(i):
                     if np.array_equal(index_tuples[k, :], indices):
                        break
                     else:
                        index_tuples[i, :] = indices

       return index_tuples
