#!/usr/bin/env python3
# ======== IMPORTS ========
# General import
import argparse, time
import gym, sneks
import ptan
import numpy as np
# PyTorch imports
import torch
from torch import nn, optim
import torch.nn.functional as F
import torch.multiprocessing as mp
from torch.optim.lr_scheduler import StepLR
from tensorboardX import SummaryWriter
# Local imports
from common.atari_wrappers import EpisodicLifeEnv, NoopResetEnv, MaxAndSkipEnv, FireResetEnv, WarpFrame, FrameStack, ClipRewardEnv, ScaledFloatFrame
from common.pytorch_utils import ImageToPyTorch
from common.logger import Logger
from qnetwork import QNetwork

# ======== CONSTANTS ========
EPSILON_START = 1.0
EPSILON_STOP = 0.05
PLAY_STEPS = 2
LR_STEPS = 1000

def make_env(env_name, rnd_seed):
    """
        Create the environment suitable for PyTorch, specifically:
        - Create environment from env id
        - Translate the observation from range [0,255] to [0,1]
        - Change format from NHWC to NCHW
        - Set the random seed
    """
    env = gym.make(env_name)
    env = ScaledFloatFrame(env)
    env = ImageToPyTorch(env)
    env.seed(rnd_seed)
    return env

def unpack_batch(batch):
    """
        Takes an experience batch from the experience replay and unpack it
        to get states, actions, rewards, dones and next_states.
    """
    states, actions, rewards, dones, last_states = [], [], [], [], []
    for exp in batch:
        state = np.array(exp.state, copy=False)
        states.append(state)
        actions.append(exp.action)
        rewards.append(exp.reward)
        dones.append(exp.last_state is None)
        if exp.last_state is None:
            last_states.append(state)       # the result will be masked anyway
        else:
            last_states.append(np.array(exp.last_state, copy=False))
    return np.array(states, copy=False), np.array(actions), np.array(rewards, dtype=np.float32), \
           np.array(dones, dtype=np.uint8), np.array(last_states, copy=False)

def play_func(env_name, net, exp_queue, logger, seed=42, timesteps=1, epsilon_decay_last_step=1000, gamma=1.0):
    """
        Function called to play episodes in parallel.
    """
    # Create the environment
    env = make_env(env_name, seed)
    # Get PyTorch device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Create agent
    selector = ptan.actions.EpsilonGreedyActionSelector(epsilon=EPSILON_START)
    agent = ptan.agent.DQNAgent(net, selector, device=device)
    # Create experience source, i.e. the wrapped environment.
    exp_source = ptan.experience.ExperienceSourceFirstLast(env, agent, gamma=gamma, steps_count=1)
    exp_source_iter = iter(exp_source)

    # Start the playing loop
    timestep, ep_start_step, ep_start_time = 0, 0, time.time()
    while timestep < timesteps:
        # Epsilon starts from EPSILON_START and linearly decreases till epsilon_decay_last_step to EPSILON_STOP
        epsilon = EPSILON_STOP + max(0, (EPSILON_START - EPSILON_STOP)*(epsilon_decay_last_step-timestep)/epsilon_decay_last_step)
        logger.log_kv('internals/epsilon', epsilon, timestep)
        selector.epsilon = epsilon
        # Do one step
        timestep += 1
        exp = next(exp_source_iter)
        new_rewards = exp_source.pop_total_rewards()
        ep_info = None
        # Check if the episode has ended
        if new_rewards:
            ep_len = timestep - ep_start_step
            speed = (ep_len) / (time.time() - ep_start_time)
            ep_start_step, ep_start_time = timestep, time.time()
            ep_info = (new_rewards[0], ep_len, speed)
        exp_queue.put((exp, ep_info))
    # End
    exp_queue.put((None, None))

def train(env_name, seed=42, timesteps=1, epsilon_decay_last_step=1000,
            er_capacity=1e4, batch_size=16, lr=1e-3, gamma=1.0,  update_target=16,
            exp_name='test', init_timesteps=100, save_every_steps=1e4, arch='nature',
            dueling=False):
    """
        Main training function. Calls the subprocesses to get experience and
        train the network.
    """
    # Multiprocessing method
    mp.set_start_method('spawn')

    # Get PyTorch device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Create logger
    logger = Logger(exp_name, loggers=['tensorboard'])

    # Create the Q network
    _env = make_env(env_name, seed)
    net = QNetwork(_env.observation_space, _env.action_space, arch=arch, dueling=dueling).to(device)
    # Create the target network as a copy of the Q network
    tgt_net = ptan.agent.TargetNet(net)
    # Create buffer and optimizer
    buffer = ptan.experience.ExperienceReplayBuffer(experience_source=None, buffer_size=er_capacity)
    optimizer = optim.Adam(net.parameters(), lr=lr)
    lr_scheduler = StepLR(optimizer, step_size=LR_STEPS, gamma=0.99)

    # Multiprocessing queue
    exp_queue = mp.Queue(maxsize=PLAY_STEPS * 2)
    play_proc = mp.Process(target=play_func, args=(env_name, net, exp_queue, logger, seed, timesteps, epsilon_decay_last_step, gamma))
    play_proc.start()

    # Main training loop
    timestep = 0
    while play_proc.is_alive() and timestep < timesteps:
        timestep += PLAY_STEPS

        # Query the environments and log results if the episode has ended
        for _ in range(PLAY_STEPS):
            exp, ep_info = exp_queue.get()
            if exp is None:
                play_proc.join()
                break
            buffer._add(exp)
            if ep_info is not None:
                logger.log_kv('performance/return', ep_info[0], timestep)
                logger.log_kv('performance/length', ep_info[1], timestep)
                logger.log_kv('performance/speed', ep_info[2], timestep)

        # Check if we are in the starting phase
        if len(buffer) < init_timesteps:
            continue

        scheduler.step()
        logger.log_kv('internals/lr', scheduler.get_lr(), timestep)
        # Get a batch from experience replay
        optimizer.zero_grad()
        batch = buffer.sample(batch_size * PLAY_STEPS)
        # Unpack the batch
        states, actions, rewards, dones, next_states = unpack_batch(batch)
        states_v = torch.tensor(states).to(device)
        next_states_v = torch.tensor(next_states).to(device)
        actions_v = torch.tensor(actions).to(device)
        rewards_v = torch.tensor(rewards).to(device)
        done_mask = torch.ByteTensor(dones).to(device)
        # Optimize defining the loss function
        state_action_values = net(states_v).gather(1, actions_v.unsqueeze(-1)).squeeze(-1)
        next_state_values = tgt_net.target_model(next_states_v).max(1)[0]
        next_state_values[done_mask] = 0.0
        expected_state_action_values = next_state_values.detach() * gamma + rewards_v
        loss = F.mse_loss(state_action_values, expected_state_action_values)
        logger.log_kv('internals/loss', loss.item(), timestep)
        loss.backward()
        # Clip the gradients to avoid to abrupt changes (this is equivalent to Huber Loss)
        for param in net.parameters():
            param.grad.data.clamp_(-1, 1)
        optimizer.step()

        # Check if the target network need to be synched
        if timestep % update_target == 0:
            tgt_net.sync()

        # Check if we need to save a checkpoint
        if timestep % save_every_steps == 0:
            torch.save(net.get_extended_state(), exp_name + '.pth')

if __name__ == '__main__':
    # Check also for scientific notation
    def int_scientific(x):
        return int(float(x))
    # Declare and parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--env_name', help='Environment to train on.', type=str, default='snek-rgb-16-v1')
    parser.add_argument('--arch', help='Net architecture.', type=str, default='nature')
    parser.add_argument('--exp_name', help='Experiment name, used in log dir', type=str, default='test')
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
    parser.add_argument("--dueling", default=False, action="store_true", help="Dueling network.")
    args = parser.parse_args()
    # Call the train function with arguments
    train(**vars(args))
