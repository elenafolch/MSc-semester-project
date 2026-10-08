import torch
import torch.nn as nn
import torch.nn.functional as F


#defining neural network

class network(nn.Module):
    #paremeters belonging to layers in init are being tracked, moved to GPU and to optimizer for backpropagation
    def __init__(self, in_channels, num_classes):
      super().__init__()

      # First 2D convolutional layer, taking in 1 input channel (image),
      # uses 32 filters (kernels of size 3x3) and outputs 32 convolutional feature maps
      self.conv1 = nn.Conv2d(in_channels=1, out_channels=20, kernel_size=3, stride=1)
      # Second 2D convolutional layer, taking in the 32 input layers (feature maps),
      # using 2 filters here?
      # outputting 64 convolutional feature maps, with a square kernel size of 3
      self.conv2 = nn.Conv2d(20, 40, 3, 1)

      # Designed to ensure that adjacent pixels are either all 0s or all active
      # with an input probability
      #do this to prevent overfitting
      self.dropout1 = nn.Dropout2d(0.25)
      self.dropout2 = nn.Dropout(0.5) #nn.Dropout works better for tensor of shape (#images, #features)

      # First fully connected layer (4096 input neurons fully connected to 128 output neurons)
      self.fc1 = nn.Linear(in_features=2560, out_features=80) #input dimension must equal output dimension of pooling
      # Second fully connected layer that outputs our 10 labels
      self.fc2 = nn.Linear(80, 10)

    #defining how data passes through network
    # x represents our data
    def forward(self, x):
      # Pass data through conv1
      x = self.conv1(x)
      # Use the rectified-linear activation function over x
      x = F.relu(x)

      x = self.conv2(x)
      x = F.relu(x)

      # Run max pooling over x
      x = F.max_pool2d(input = x, kernel_size = 2, stride=3) #stride = kernel_size unless stride is specified
      # Pass data through dropout1
      x = self.dropout1(x)
      # Flatten x with start_dim=1
      x = torch.flatten(x, start_dim = 1)
      # Pass data through ``fc1``
      x = self.fc1(x)
      x = F.relu(x)
      x = self.dropout2(x)
      x = self.fc2(x)

      # Apply softmax to x
      output = F.log_softmax(x, dim=1)
      return output


    