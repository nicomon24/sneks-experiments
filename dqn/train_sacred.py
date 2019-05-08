#!/usr/bin/env python3
# ======== IMPORTS ========
# General import
import argparse, time, os
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
from common.atari_wrappers import ScaledFloatFrame
from common.pytorch_utils import ImageToPyTorch
from common.logger import Logger
from qnetwork import QNetwork
# Sacred imports
from sacred import Experiment

# ======== SACRED CONFIG ========
if os.environ.get('EXPERIMENT_NAME') is not None:
    ex = Experiment(os.environ.get('EXPERIMENT_NAME'))
else:
    ex = Experiment('DQN')

@ex.config
def base_config():
    experiment_name = os.environ.get('EXPERIMENT_NAME', 'test')    # Experiment name
    env_name = 'snek-rgb-16-v1'         # Environment to play
    arch = 'nature'                     # Architecture for the network
    timesteps = int(1e6)                # Training timesteps
    init_timesteps = int(1e4)           # Timestep to bootstrap
    seed = 42                           # Random seed
    er_capacity = int(1e5)              # Capacity of †he experience replay
    epsilon_start = 1.0                 # Starting epsilon
    epsilon_stop = 0.05                 # Final epsilon after decay
    epsilon_decay_stop = int(1e5)       # Timestep at which we end decay
    batch_size = 16                     # Batch size
    target_sync = int(1e3)              # How frequently we sync target net
    lr = 1e-3                           # Starting learning rate
    gamma = 0.99                        # Discount factor
    dueling = False                     # Use dueling network
    play_steps = 2                      # How many play steps per iteration
    lr_steps = int(1e4)                 # How frequently we decay LR
    lr_gamma = 0.99                     # Decay factor for LR
    save_steps = int(5e4)               # How frequently we save a checkpoint

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

def play_func(env_name, net, exp_queue, seed=42, timesteps=1, epsilon_schedule=(1.0, 0.05, 1000), gamma=1.0):
    """
        Function called to play episodes in parallel.
    """
    # Parse epsilon config
    epsilon_start, epsilon_stop, epsilon_decay_stop = epsilon_schedule
    # Create the environment
    env = make_env(env_name, seed)
    # Get PyTorch device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Create agent
    selector = ptan.actions.EpsilonGreedyActionSelector(epsilon=epsilon_start)
    agent = ptan.agent.DQNAgent(net, selector, device=device)
    # Create experience source, i.e. the wrapped environment.
    exp_source = ptan.experience.ExperienceSourceFirstLast(env, agent, gamma=gamma, steps_count=1)
    exp_source_iter = iter(exp_source)

    # Start the playing loop
    timestep, ep_start_step, ep_start_time = 0, 0, time.time()
    while timestep < timesteps:
        # Epsilon starts from EPSILON_START and linearly decreases till epsilon_decay_last_step to EPSILON_STOP
        epsilon = epsilon_stop + max(0, (epsilon_start - epsilon_stop)*(epsilon_decay_stop-timestep)/epsilon_decay_stop)
        selector.epsilon = epsilon
        # Do one step
        timestep += 1
        exp = next(exp_source_iter)
        new_rewards = exp_source.pop_total_rewards()
        info = { 'epsilon': (epsilon, timestep) }
        # Check if the episode has ended
        if new_rewards:
            ep_len = timestep - ep_start_step
            speed = (ep_len) / (time.time() - ep_start_time)
            ep_start_step, ep_start_time = timestep, time.time()
            info['ep_reward'] = new_rewards[0]
            info['ep_length'] = ep_len
            info['speed'] = speed
        exp_queue.put((exp, info))
    # End
    exp_queue.put((None, {}))

@ex.capture
def train(env_name, arch, timesteps=1, init_timesteps=0, seed=42, er_capacity=1,
            epsilon_start=1.0, epsilon_stop=0.05, epsilon_decay_stop=1, batch_size=16,
            target_sync=16, lr=1e-3, gamma=1.0, dueling=False, play_steps=1, lr_steps=1e4,
            lr_gamma=0.99, save_steps=5e4, logger=None, experiment_name='test'):
    """
        Main training function. Calls the subprocesses to get experience and
        train the network.
    """
    # Casting params which are expressable in scientific notation
    def int_scientific(x):
        return int(float(x))
    timesteps, er_capacity, epsilon_decay_stop = map(int_scientific, [timesteps, er_capacity, epsilon_decay_stop])

    # Multiprocessing method
    mp.set_start_method('spawn')

    # Get PyTorch device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create the Q network
    _env = make_env(env_name, seed)
    net = QNetwork(_env.observation_space, _env.action_space, arch=arch, dueling=dueling).to(device)
    # Create the target network as a copy of the Q network
    tgt_net = ptan.agent.TargetNet(net)
    # Create buffer and optimizer
    buffer = ptan.experience.ExperienceReplayBuffer(experience_source=None, buffer_size=er_capacity)
    optimizer = optim.Adam(net.parameters(), lr=lr)
    scheduler = StepLR(optimizer, step_size=lr_steps, gamma=0.99)

    # Multiprocessing queue
    epsilon_schedule = (epsilon_start, epsilon_stop, epsilon_decay_stop)
    exp_queue = mp.Queue(maxsize=play_steps * 2)
    play_proc = mp.Process(target=play_func, args=(env_name, net, exp_queue, seed, timesteps, epsilon_schedule, gamma))
    play_proc.start()

    # Main training loop
    timestep = 0
    while play_proc.is_alive() and timestep < timesteps:
        timestep += play_steps
        # Query the environments and log results if the episode has ended
        for _ in range(play_steps):
            exp, info = exp_queue.get()
            if exp is None:
                play_proc.join()
                break
            buffer._add(exp)
            logger.log_kv('internals/epsilon', info['epsilon'][0], info['epsilon'][1])
            if 'ep_reward' in info.keys():
                logger.log_kv('performance/return', info['ep_reward'], timestep)
                logger.log_kv('performance/length', info['ep_length'], timestep)
                logger.log_kv('performance/speed', info['speed'], timestep)

        # Check if we are in the starting phase
        if len(buffer) < init_timesteps:
            continue

        scheduler.step()
        logger.log_kv('internals/lr', scheduler.get_lr()[0], timestep)
        # Get a batch from experience replay
        optimizer.zero_grad()
        batch = buffer.sample(batch_size * play_steps)
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
        if timestep % save_steps == 0:
            torch.save(net.get_extended_state(), experiment_name + '.pth')

@ex.automain
def main(experiment_name, _run):
    # Create logger
    logger = Logger(experiment_name, loggers=['tensorboard', 'sacred'], sacred_run=_run)
    # Call train
    train(logger=logger, experiment_name=experiment_name)
