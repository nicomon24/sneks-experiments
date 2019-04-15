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
from tensorboardX import SummaryWriter
import time

# Local Imports
from qnetwork import QNetwork
from experience_replay import ExperienceReplay
from common.pytorch_utils import NCHW_from_NHWC, NHWC_from_NCHW
from common.atari_wrappers import EpisodicLifeEnv, NoopResetEnv, MaxAndSkipEnv, FireResetEnv, WarpFrame, FrameStack, ClipRewardEnv, ScaledFloatFrame

EPSILON_START = 1.0
EPSILON_STOP = 0.02
PLAY_STEPS = 1

class MultidimWrapper(gym.Wrapper):

    def reset(self):
        obs = self.env.reset()
        return np.expand_dims(np.array(obs), axis=0)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        return np.expand_dims(np.array(obs), axis=0), np.expand_dims(np.array(reward), axis=0), np.expand_dims(np.array(done), axis=0), info


def make_env(env_name, rnd_seed):
    env = gym.make(env_name)
    env = EpisodicLifeEnv(env)
    env = NoopResetEnv(env)
    env = MaxAndSkipEnv(env)
    env = FireResetEnv(env)
    env = WarpFrame(env)
    env = FrameStack(env, 4)
    env = ClipRewardEnv(env)
    env = ScaledFloatFrame(env)
    env.seed(rnd_seed)
    return env

def obs_to_pytorch(obs):
    if len(obs.shape) == 3:
        # To 4D
        obs = np.expand_dims(obs, axis=0)
    # To NCHW
    obs = NCHW_from_NHWC(obs)
    # To pytorch
    return torch.from_numpy(obs)

def train(env_name, seed=42, timesteps=1, epsilon_decay_last_step=1000,
            er_capacity=1e4, batch_size=16, lr=1e-3, gamma=1.0,  update_target=16,
            logdir='logs', init_timesteps=100):

    env = make_env(env_name, seed)

    # Get PyTorch device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Init Q networks
    policy_network = QNetwork(env.observation_space, env.action_space).to(device)
    target_network = QNetwork(env.observation_space, env.action_space).to(device)
    # Copy the policy network
    target_network = copy.deepcopy(policy_network)
    target_network.eval()
    # Init the experience replay
    memory = ExperienceReplay(capacity=er_capacity)
    # Define the optimizer
    optimizer = optim.Adam(policy_network.parameters(), lr=lr)
    # Declare the summary writer for tensorboard
    writer = SummaryWriter(logdir)

    # Initialize the environments
    obs = env.reset()
    # Process the observation to pytorch format
    obs = obs_to_pytorch(obs).to(device)
    # Init
    ep_rew, ep_len = 0, 0
    completed_episodes = 0
    start_time = time.time()

    # Loop for the selected number of timesteps
    for timestep in range(0, timesteps):

        # Epsilon starts from EPSILON_START and linearly decreases till epsilon_decay_last_step to EPSILON_STOP
        epsilon = EPSILON_STOP + max(0, (EPSILON_START - EPSILON_STOP)*(epsilon_decay_last_step-timestep)/epsilon_decay_last_step)
        # Get the selected action
        greedy_action = policy_network(obs).argmax(dim=1).cpu().detach().numpy()
        epsilon_prob = np.random.rand()
        action = np.array([greedy_action[0] if epsilon_prob > epsilon else env.action_space.sample()])[0]

        # Sample num_envs steps
        previous_obs = obs
        obs, rew, done, _ = env.step(action)
        ep_rew += rew
        ep_len += 1
        env.render()

        if done:
            # Compute speed
            speed = (ep_len) / (time.time() - start_time)
            print(speed)
            start_time = time.time()
            # Write performance
            writer.add_scalar('performance/return', ep_rew, completed_episodes)
            writer.add_scalar('performance/length', ep_len, completed_episodes)
            writer.add_scalar('performance/speed', speed, completed_episodes)
            completed_episodes += 1
            # Reset
            ep_rew, ep_len = 0, 0
            obs = env.reset()

        # Process memories to pytorch format
        obs = obs_to_pytorch(obs).to(device)
        rew = torch.from_numpy(np.expand_dims(np.array([rew]), 0)).float().to(device)
        action = torch.from_numpy(np.expand_dims(np.array([action]), 0)).to(device)
        mask = torch.Tensor(np.invert(np.expand_dims(np.array([done]), 0)).astype('float')).to(device)

        # Add new transitions to experience replay
        memory.push(state=previous_obs, action=action, reward=rew, next_state=obs, done=mask)

        if timestep > init_timesteps and timestep % PLAY_STEPS == 0:
            # Sample batch of experience
            batch_state, batch_action, batch_reward, batch_next_state, batch_mask = memory.sample(batch_size * PLAY_STEPS)
            # Compute the next_Q prediction using the target network
            next_Q, _ = target_network(batch_next_state).max(dim=1)
            # Get the estimate for the current Q updated, set to r if the state is terminal
            Q_estimate = batch_reward + gamma * next_Q * batch_mask
            # Compute the loss w.r.t. the prediction for the selected actions
            Q_preds = policy_network(batch_state).gather(1, batch_action)
            loss = F.mse_loss(Q_preds, Q_estimate) # We can use MSE instead of Huber because we can directly clip gradients
            '''
            # Optimizer step
            optimizer.zero_grad()
            loss.backward()
            for param in policy_network.parameters():
                param.grad.data.clamp_(-1, 1)
            optimizer.step()

            # Copy the policy network to the target network
            if timestep % update_target == 0:
                target_network = copy.deepcopy(policy_network)
                target_network.eval()

            writer.add_scalar('internals/loss', loss.item(), timestep)
            '''
        writer.add_scalar('internals/epsilon', epsilon, timestep)
        writer.add_scalar('internals/episodes', completed_episodes, timestep)
        writer.add_scalar('internals/timesteps', timestep, timestep)

    # Ending things
    writer.close()
    env.close()

    # Save network
    torch.save(policy_network.state_dict(), 'qn.pth')

if __name__ == '__main__':
    # Check also for scientific notation
    def int_scientific(x):
        return int(float(x))
    # Declare and parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--env_name', help='Environment to train on.', type=str, default='snek-rgb-16-v1')
    parser.add_argument('--logdir', help='Directory to save the logs to.', type=str, default='logs/')
    #parser.add_argument('--num_envs', help='Number of parallel environments.', type=int, default=1)
    parser.add_argument('--timesteps', help='Number of parallel environments.', type=int_scientific, default=1)
    parser.add_argument('--init_timesteps', help='Number of initial steps to fill the buffer.', type=int_scientific, default=1e4)
    parser.add_argument('--seed', help='Random seed.', type=int, default=42)
    parser.add_argument('--er_capacity', help='Experience replay capacity.', type=int_scientific, default=1e4)
    parser.add_argument('--epsilon_decay_last_step', help='Step at which the epsilon plateau is reached.', type=int_scientific, default=1e4)
    parser.add_argument('--batch_size', help='Experience batch size.', type=int, default=16)
    parser.add_argument('--update_target', help='Number of iterations for each target update.', type=int, default=16)
    parser.add_argument('--lr', help='Optimizer learning rate.', type=float, default=1e-3)
    parser.add_argument('--gamma', help='Discount factor for the MDP.', type=float, default=1.0)
    args = parser.parse_args()
    # Call the train function with arguments
    train(**vars(args))
