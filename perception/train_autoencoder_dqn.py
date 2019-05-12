#!/usr/bin/env python3
'''
    We create an auto-encoder starting from a trained network. We then train it
    on observations from the environment gathered from playing the policy.
'''

import gym
import ptan
import argparse
import time
import numpy as np
import sneks

import torch
from torch import nn, optim
import torch.nn.functional as F
import torchvision.utils as vutils
from dqn.qnetwork import QNetwork
from tensorboardX import SummaryWriter
from common.atari_wrappers import ScaledFloatFrame
from common.pytorch_utils import ImageToPyTorch
from perception.dqn_decoder import DqnDecoder
from torch.optim.lr_scheduler import StepLR

EPSILON = 0.0

def make_env(env_name, rnd_seed):
    env = gym.make(env_name)
    env = ScaledFloatFrame(env)
    env = ImageToPyTorch(env)
    env.seed(rnd_seed)
    return env

def generate_batch(env, dqn, device, batch_size=16):
    obs = env.reset()
    buffer = []
    while True:
        torch_obs = torch.from_numpy(np.expand_dims(obs, 0)).to(device)
        buffer.append(torch_obs)
        if len(buffer) == batch_size:
            yield torch.cat(buffer, dim=0)
            buffer = []
        action = dqn(torch_obs).to(device).argmax(dim=1)[0]
        obs, reward, done, _ = env.step(action)
        if done:
            obs = env.reset()

def train(env_name, iterations, seed=42, model=None, render=True, lr=1e-3, batch_size=16, loss_base=2):
    # Create the environment
    env = make_env(env_name, seed)
    # Get PyTorch device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Declare writer
    writer = SummaryWriter('runs/reconstruction_logs/')
    # Create Qnetwork
    state = torch.load(model, map_location="cuda" if torch.cuda.is_available() else "cpu")
    net = QNetwork(env.observation_space, env.action_space, arch=state['arch'], dueling=state.get('dueling', False)).to(device)
    net.load_state_dict(state['state_dict'])

    # Create the decoder and the optimizer
    dqn_decoder = DqnDecoder(net).to(device)
    optimizer = optim.Adam(dqn_decoder.parameters(), lr=lr)
    scheduler = StepLR(optimizer, step_size=1000, gamma=0.5)

    # Training loop
    data_generator = generate_batch(env, net, device, batch_size)
    for i in range(iterations):
        scheduler.step()
        writer.add_scalar('reconstruction/lr', scheduler.get_lr()[0], i)
        optimizer.zero_grad()
        # Get batch and AE reconstruction
        batch = next(data_generator)
        reconstruction = dqn_decoder(batch, net)
        # Compute loss and backpropagate
        loss = ((batch - reconstruction) ** loss_base).sum() / batch.size(0)
        loss.backward()
        optimizer.step()
        writer.add_scalar('reconstruction/loss', loss, i)

        # Visualize
        if i % 50 == 0:
            original = vutils.make_grid(batch[:4], normalize=True, scale_each=True)
            reco = vutils.make_grid(reconstruction[:4], normalize=True, scale_each=True)
            writer.add_image('original', original, i)
            writer.add_image('reconstruction', reco, i)

    # Save the decoder
    output_name = model.split('/')[-1].split('.')[0] + '_decoder' + '.pth'
    torch.save(dqn_decoder.state_dict(), output_name)

if __name__ == '__main__':
    # Check also for scientific notation
    def int_scientific(x):
        return int(float(x))
    # Declare and parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--env_name', help='Environment to train on.', type=str, default='snek-rgb-16-v1')
    parser.add_argument('--iterations', help='Number of training iterations', type=int_scientific, default=10)
    parser.add_argument('--model', help='Path of the model to load.', type=str, default=None)
    parser.add_argument('--seed', help='Random seed.', type=int, default=42)
    parser.add_argument('--batch_size', help='Batch size', type=int, default=16)
    parser.add_argument('--lr', help='Learning rate.', type=float, default=1e-3)
    parser.add_argument('--loss_base', help='Loss cardinality.', type=int, default=2)
    args = parser.parse_args()
    # Call the train function with arguments
    train(**vars(args))
