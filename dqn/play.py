'''
    Play episodes using the trained policy.
'''
#!/usr/bin/env python3
import gym
import ptan
import argparse
import time
import numpy as np
import sneks

import torch
from torch import nn, optim
import torch.nn.functional as F
from qnetwork import QNetwork
from tensorboardX import SummaryWriter
from common.atari_wrappers import EpisodicLifeEnv, NoopResetEnv, MaxAndSkipEnv, FireResetEnv, WarpFrame, FrameStack, ClipRewardEnv, ScaledFloatFrame
from common.pytorch_utils import ImageToPyTorch

EPSILON = 0.0

def make_env(env_name, rnd_seed):
    env = gym.make(env_name)
    #env = EpisodicLifeEnv(env)
    #env = NoopResetEnv(env)
    #env = MaxAndSkipEnv(env)
    #env = FireResetEnv(env)
    #env = WarpFrame(env)
    #env = FrameStack(env, 4)
    #env = ClipRewardEnv(env)
    env = ScaledFloatFrame(env)
    env = ImageToPyTorch(env)
    env.seed(rnd_seed)
    return env

def play(env_name, seed=42, model=None):
    # Create the environment
    env = make_env(env_name, seed)
    # Get PyTorch device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Create Qnetwork
    state = torch.load(model, map_location="cuda" if torch.cuda.is_available() else "cpu")
    net = QNetwork(env.observation_space, env.action_space, arch=state['arch']).to(device)
    net.load_state_dict(state['state_dict'])

    obs, ep_return, ep_len = env.reset(), 0, 0
    while True:

        action = net(torch.from_numpy(np.expand_dims(obs, 0)).to(device)).argmax(dim=1)[0]
        obs, reward, done , _ = env.step(action)
        env.render()
        ep_return += reward
        ep_len += 1

        if done:
            print("Episode:", ep_return, '\t\t', ep_len)
            obs, ep_return, ep_len = env.reset(), 0, 0

        time.sleep(0.05)

if __name__ == '__main__':
    # Check also for scientific notation
    def int_scientific(x):
        return int(float(x))
    # Declare and parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--env_name', help='Environment to train on.', type=str, default='snek-rgb-16-v1')
    parser.add_argument('--model', help='Path of the model to load.', type=str, default=None)
    parser.add_argument('--seed', help='Random seed.', type=int, default=42)
    args = parser.parse_args()
    # Call the train function with arguments
    play(**vars(args))
