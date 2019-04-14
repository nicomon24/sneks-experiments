"""
    Train a DQN policy using the vanilla DQN algorithm.
    ....
"""

# Imports
import torch
from torch import nn, optim
import torch.nn.functional as F
import gym
import numpy as np
import sneks
from sneks.wrappers import NormalizeInt8
import copy

# Local Imports
from qnetwork import QNetwork
from experience_replay import ExperienceReplay
from common.vec_env.shmem_vec_env import ShmemVecEnv
from common.pytorch_utils import NCHW_from_NHWC, NHWC_from_NCHW

EPSILON_START = 1.0
EPSILON_STOP = 0.05

def train(env_name, seed=42, num_envs=1, timesteps=1, epsilon_decay_last_step=1000,
            er_capacity=1e4, batch_size=16, lr=1e-3, gamma=1.0):
    # Parallel environment thunk function
    def make_env_generator(rnd_seed):
        def make_env():
            env = gym.make(env_name)
            env = NormalizeInt8(env)
            env.seed(rnd_seed)
            return env
        return make_env
    # Create the parallel environment
    vec_env = ShmemVecEnv([make_env_generator(seed + i) for i in range(num_envs)])
    # Init Q networks
    policy_network = QNetwork(vec_env.observation_space, vec_env.action_space)
    target_network = QNetwork(vec_env.observation_space, vec_env.action_space)
    # Copy the policy network
    target_network = copy.deepcopy(policy_network)
    # Init the experience replay
    memory = ExperienceReplay(capacity=er_capacity)
    # Define the optimizer
    optimizer = optim.RMSprop(policy_network.parameters(), lr=lr)

    # Initialize the environments
    obs = vec_env.reset()
    ep_rew = np.zeros(num_envs)
    ep_len = np.zeros(num_envs)
    episode_returns = []
    episode_lens = []
    losses = []

    # Loop for the selected number of timesteps
    for timestep in range(0, timesteps, num_envs):
        # Epsilon starts from EPSILON_START and linearly decreases till epsilon_decay_last_step to EPSILON_STOP
        epsilon = EPSILON_STOP + max(0, (EPSILON_START - EPSILON_STOP)*(epsilon_decay_last_step-timestep)/epsilon_decay_last_step)
        # Get the selected action
        greedy_action = policy_network(torch.from_numpy(NCHW_from_NHWC(obs))).argmax(dim=1).detach().numpy()
        epsilon_prob = np.random.rand(num_envs)
        actions = np.array([greedy_action[i] if epsilon_prob[i] > epsilon else vec_env.action_space.sample() for i in range(num_envs)])

        # Sample num_envs steps
        previous_obs = obs
        vec_env.step_async(actions)
        obs, rew, done, _ = vec_env.step_wait()
        ep_rew += rew
        ep_len += 1
        for i in range(num_envs):
            if done[i]:
                episode_returns.append(ep_rew[i])
                episode_lens.append(ep_len[i])
                ep_rew[i] = 0.0
                ep_len[i] = 0
                episode_returns = episode_returns[-100:]
                episode_lens = episode_lens[-100:]

        # Add new transitions to experience replay
        memory.push(state=previous_obs, action=actions, reward=rew, next_state=obs, done=done)

        # Sample batch of experience
        batch_state, batch_action, batch_reward, batch_next_state, batch_done = memory.sample(batch_size)
        # Compute the next_Q prediction using the target network
        next_Q, _ = target_network(torch.from_numpy(NCHW_from_NHWC(batch_next_state))).max(dim=1)
        # Get the estimate for the current Q updated, set to r if the state is terminal
        mask = torch.Tensor(np.invert(batch_done).astype('float'))
        Q_estimate = torch.from_numpy(batch_reward).float() + gamma * next_Q * mask
        # Compute the loss w.r.t. the prediction for the selected actions
        Q_preds = policy_network(torch.from_numpy(NCHW_from_NHWC(batch_state))).gather(1, torch.from_numpy(batch_action).unsqueeze(1))
        loss = F.mse_loss(Q_preds, Q_estimate) # We can use MSE instead of Huber because we can directly clip gradients
        losses.append(loss.item())
        # Optimizer step
        optimizer.zero_grad()
        loss.backward()
        for param in policy_network.parameters():
            param.grad.data.clamp_(-1, 1)
        optimizer.step()

        # Copy the policy network to the target network
        if (timestep // num_envs) % 100 == 0:
            target_network = copy.deepcopy(policy_network)
            losses = losses[-100:]

        if (timestep // num_envs) % 100 == 0:
            print('-------------------------')
            print('LOSS:', np.mean(losses[-100:]))
            print('EPISODE:', np.mean(episode_lens), '\t', np.mean(episode_returns))
            print('EPSILON:', epsilon)
            print('TIMESTEPS:', timestep)

if __name__ == '__main__':
    # Declare and parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--env_name', help='Environment to train on.', type=str, default='snek-rgb-16-v1')
    parser.add_argument('--num_envs', help='Number of parallel environments.', type=int, default=1)
    parser.add_argument('--timesteps', help='Number of parallel environments.', type=int, default=1)
    parser.add_argument('--seed', help='Random seed.', type=int, default=42)
    parser.add_argument('--er_capacity', help='Experience replay capacity.', type=int, default=1e4)
    parser.add_argument('--epsilon_decay_last_step', help='Step at which the epsilon plateau is reached.', type=int, default=1e4)
    parser.add_argument('--batch_size', help='Experience batch size.', type=int, default=16)
    parser.add_argument('--lr', help='Optimizer learning rate.', type=float, default=1e-3)
    parser.add_argument('--gamma', help='Discount factor for the MDP.', type=float, default=1.0)
    args = parser.parse_args()
    # Call the train function with arguments
    train(**vars(args))
