import torch
import torch.nn as nn
import torch.nn.functional as F

class onedim_model(nn.Module):
    def __init__(self, in_channels, num_classes):
      super().__init__()

      self.fc1 = nn.Linear(in_features=100, out_features=100) 
      self.fc2 = nn.Linear(100, 100)
    
    def forward(self, x):
      
      x = self.fc1(x)
      x = F.relu(x)
      x = self.fc2(x)

      return x