#!/usr/bin/env python3
import gym
import ptan
import argparse
import time

import torch
import torch.optim as optim

from qnetwork import QNetwork

from tensorboardX import SummaryWriter

EPSILON_START = 1.0
EPSILON_STOP = 0.02
PLAY_STEPS = 1

def make_env(env_name, rnd_seed):
    env = gym.make(env_name)
    env = ptan.common.wrappers.wrap_dqn(env)
    return env

def train(env_name, seed=42, timesteps=1, epsilon_decay_last_step=1000,
            er_capacity=1e4, batch_size=16, lr=1e-3, gamma=1.0,  update_target=16,
            logdir='logs', init_timesteps=100):

    # Create the environment
    env = make_env(env_name, seed)
    # Get PyTorch device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Create tensorboard writer
    writer = SummaryWriter(logdir)

    # Create the Q network
    net = QNetwork(env.observation_space, env.action_space).to(device)
    tgt_net = ptan.agent.TargetNet(net)
    selector = ptan.actions.EpsilonGreedyActionSelector(epsilon=EPSILON_START)
    agent = ptan.agent.DQNAgent(net, selector, device=device)

    exp_source = ptan.experience.ExperienceSourceFirstLast(env, agent, gamma=gamma, steps_count=1)
    buffer = ptan.experience.ExperienceReplayBuffer(exp_source, buffer_size=er_capacity)
    optimizer = optim.Adam(net.parameters(), lr=lr)

    ep_start_step, ep_start_time = 0, time.time()

    for timestep in range(timesteps):

        # Epsilon starts from EPSILON_START and linearly decreases till epsilon_decay_last_step to EPSILON_STOP
        epsilon = EPSILON_STOP + max(0, (EPSILON_START - EPSILON_STOP)*(epsilon_decay_last_step-timestep)/epsilon_decay_last_step)
        selector.epsilon = epsilon
        # Add one step to the buffer
        buffer.populate(1)
        new_rewards = exp_source.pop_total_rewards()
        if new_rewards:
            ep_len = timestep - ep_start_step
            speed = (ep_len) / (time.time() - ep_start_time)
            ep_start_step, ep_start_time = timestep, time.time()
            # Write performance
            writer.add_scalar('performance/return', new_rewards[0], timestep)
            writer.add_scalar('performance/length', ep_len, timestep)
            writer.add_scalar('performance/speed', speed, timestep)

        if len(buffer) < init_timesteps:
            continue

        optimizer.zero_grad()
        batch = buffer.sample(batch_size)

        from dis_lib import common
        loss_v = common.calc_loss_dqn(batch, net, tgt_net.target_model, gamma=gamma, cuda=False)
        writer.add_scalar('loss', loss_v, timestep)
        loss_v.backward()
        optimizer.step()

        if timestep % update_target == 0:
            tgt_net.sync()

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
