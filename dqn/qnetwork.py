"""
    DQN network....
"""

import gym
import torch
from torch import nn

def conv2d_size_out(size, kernel_size = 1, stride = 1):
    """
        Helper function used to compute the size of the output of a conv2d layer.
        This is useful because the last layer of our model is a linear layer whose
        size depends on the previous conv layers.
    """
    return (size - (kernel_size - 1) - 1) // stride  + 1

class QNetwork(nn.Module):

    def __init__(self, observation_space, action_space, layers=[(16, 8, 1), (32, 4, 1)]):
        super().__init__()
        # Check that observation space is box
        assert isinstance(observation_space, gym.spaces.Box), "Only works for box environments."
        # Check action space is discrete
        assert isinstance(action_space, gym.spaces.Discrete), "DQN works only for discrete action."

        # Get input and output shapes
        self.input_shape = observation_space.shape
        self.output_shape = action_space.n

        # Declare the convolutional layers: (kernels, kernel_size, stride)
        self.conv_layers = []
        size0, size1, channels = self.input_shape
        # Check images are squared
        assert size0 == size1, "A squared observation is required."
        # Loop over the requested layers
        for layer in layers:
            kernels, kernel_size, stride = layer
            # Declare a conv layer followed by a relu
            self.conv_layers.append(nn.Sequential(nn.Conv2d(channels, kernels, kernel_size, stride), nn.ReLU()))
            size0 = conv2d_size_out(size0, kernel_size, stride)
            channels = kernels

        # Transform list of module to single sequential module
        self.conv_layers = nn.Sequential(*self.conv_layers)

        # Declare the last linear layer, input size is the last conv layer flattened
        self.head = nn.Sequential(nn.Linear(size0 * size0 * channels, 256),
                                  nn.ReLU(),
                                  nn.Linear(256, self.output_shape))

    def forward(self, batch):
        # Pass the batch through the conv layers
        conv_out = self.conv_layers(batch)
        # Flatten the output
        conv_out = conv_out.view(batch.size(0), -1)
        # Pass through the last linear layer and return
        return self.head(conv_out)
