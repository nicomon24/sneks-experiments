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

    def __init__(self, observation_space, action_space, arch='nature', dueling=False):
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
        self.dueling = dueling
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
        self.fc_layers = []
        for fc_layer in arch_fc_layers:
            self.fc_layers.append(nn.Sequential(nn.Linear(fc_size, fc_layer), nn.ReLU()))
            fc_size = fc_layer
        self.fc_layers = nn.Sequential(*self.fc_layers)

        # Last layer
        self.head = nn.Linear(fc_size, self.output_shape)
        if self.dueling:
            self.value_head = nn.Linear(fc_size, 1)

    def layers_from_arch(self, arch):
        if arch == 'nature':
            return [(32, 8, 4), (64, 4, 2), (64, 3, 1)], [512]
        elif arch == 'smally':
            return [(16, 3, 1), (32, 3, 1)], [256]
        else:
            raise Exception('Unrecognized architecture.')

    def forward(self, batch):
        # Pass the batch through the conv layers
        conv_out = self.conv_layers(batch.float())
        # Flatten the output
        conv_out = conv_out.view(batch.size(0), -1)
        # Pass through the linear layers
        lin_out = self.fc_layers(conv_out)

        if self.dueling:
            v = self.value_head(lin_out)
            adv = self.head(lin_out)
            return v + adv - adv.mean()
        else:
            return self.head(lin_out)

    def get_extended_state(self):
        return {
            'arch': self.arch,
            'dueling': self.dueling,
            'state_dict': self.state_dict()
        }
