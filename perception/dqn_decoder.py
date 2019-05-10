'''
    Define a decoder for a DQN network
'''

import torch
from torch import nn

class DqnDecoder(nn.Module):

    def __init__(self, dqn):
        super().__init__()

        self.fc_layers_inverse = []
        # Inverse linear layers
        for layer in dqn.fc_layers:
            # Get the first module of the layer, since it is [fc, activation]
            fc_layer = layer[0]
            inverse_layer = nn.Linear(fc_layer.out_features, fc_layer.in_features)
            self.fc_layers_inverse.append(nn.Sequential(inverse_layer, nn.ReLU()))
        self.fc_layers_inverse = nn.Sequential(*self.fc_layers_inverse)
        # Inverse conv layers
        self.conv_layers_inverse = []
        for i, layer in enumerate(dqn.conv_layers[::-1]):
            # Get the first module of the layer, since it is [fc, activation]
            conv_layer = layer[0]
            inverse_layer = nn.ConvTranspose2d(conv_layer.out_channels, conv_layer.in_channels, conv_layer.kernel_size)
            self.conv_layers_inverse.append(nn.Sequential(inverse_layer, nn.ReLU() if i < len(dqn.conv_layers)-1 else nn.Sigmoid()))
        self.conv_layers_inverse = nn.Sequential(*self.conv_layers_inverse)

    def forward(self, observation, dqn):
        # Use the dqn model to get the hidden state
        dqn_conv_out = dqn.conv_layers(observation.float())
        dqn_conv_out_flat = dqn_conv_out.view(observation.size(0), -1)
        h = dqn.fc_layers(dqn_conv_out_flat)
        # Apply the inverse layers
        inverse_lin_out = self.fc_layers_inverse(h)
        inverse_lin_out_2d = inverse_lin_out.view(dqn_conv_out.shape)
        # Apply inverse conv layers
        reconstruction = self.conv_layers_inverse(inverse_lin_out_2d)
        return reconstruction
