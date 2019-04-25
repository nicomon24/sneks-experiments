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

    def __init__(self, observation_space, action_space, arch='nature'):
        super().__init__()
        # Check that observation space is box
        assert isinstance(observation_space, gym.spaces.Box), "Only works for box environments."
        # Check action space is discrete
        assert isinstance(action_space, gym.spaces.Discrete), "DQN works only for discrete action."

        # Get input and output shapes
        self.input_shape = observation_space.shape
        self.output_shape = action_space.n

        # Get the arch specification
        self.arch = arch
        arch_conv_layers, arch_fc_layers = self.layers_from_arch(self.arch)

        # Declare the convolutional layers: (kernels, kernel_size, stride)
        self.conv_layers = []
        # NCHW
        channels, size0, size1 = self.input_shape
        # NHWC
        # size0, size1, channels = self.input_shape
        # Check images are squared
        assert size0 == size1, "A squared observation is required."
        # Loop over the requested layers
        for layer in arch_conv_layers:
            kernels, kernel_size, stride = layer
            # Declare a conv layer followed by a relu
            self.conv_layers.append(nn.Sequential(nn.Conv2d(channels, kernels, kernel_size, stride), nn.ReLU()))
            size0 = conv2d_size_out(size0, kernel_size, stride)
            channels = kernels

        # Transform list of module to single sequential module
        self.conv_layers = nn.Sequential(*self.conv_layers)

        flattened_size = fc_size = size0 * size0 * channels
        fc_layers = []
        for fc_layer in arch_fc_layers:
            fc_layers.append(nn.Sequential(nn.Linear(fc_size, fc_layer), nn.ReLU()))
            fc_size = fc_layer
        fc_layers.append(nn.Linear(fc_size, self.output_shape))

        # Declare the last linear layer, input size is the last conv layer flattened
        self.head = nn.Sequential(*fc_layers)

    def layers_from_arch(self, arch):
        if arch == 'nature':
            return [(32, 8, 4), (64, 4, 2), (64, 3, 1)], [512]
        elif arch == 'smally':
            return [(64, 4, 1)], [256]
        elif arch == 'smally512':
            return [(128, 3, 1)], [512]
        elif arch == 'minimal':
            return [(32, 1, 1)], [512]
        else:
            raise Exception('Unrecognized architecture.')

    def forward(self, batch):
        # Pass the batch through the conv layers
        conv_out = self.conv_layers(batch.float())
        # Flatten the output
        conv_out = conv_out.view(batch.size(0), -1)
        # Pass through the last linear layer and return
        return self.head(conv_out)
