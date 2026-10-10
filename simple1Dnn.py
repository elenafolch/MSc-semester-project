import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

class onedim_model(nn.Module):
    def __init__(self, in_channels, num_classes):
      super().__init__()

      self.fc1 = nn.Linear(in_features=1, out_features=100) 
      self.fc2 = nn.Linear(100, 100)
      self.fc3 = nn.Linear(100, 100)
      self.fc4 = nn.Linear(100, 1)
    
    def forward(self, x):
      
      x = self.fc1(x)
      x = F.relu(x)
      x = self.fc2(x)
      x = F.relu(x)
      x = self.fc3(x)
      x = F.relu(x)
      x = self.fc4(x)

      return x

class xy_dataset(Dataset):
    def __init__(self, x, y, split):
        # NN layers expect shape [N, 1] and float32
        self.x = torch.as_tensor(x, dtype=torch.float32).reshape(-1, 1)
        self.y = torch.as_tensor(y, dtype=torch.float32).reshape(-1, 1)
        self.split = split #"train" or "test"

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]