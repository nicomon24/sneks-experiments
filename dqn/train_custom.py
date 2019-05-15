#!/usr/bin/env python3
# ======== IMPORTS ========
# General import
import argparse, time, copy
import gym, sneks
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
from experience import ExperienceReplay

# ======== CONSTANTS ========
EPSILON_START = 1.0
EPSILON_STOP = 0.05
LR_STEPS = 1e4

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

def play_func(id, env_name, obs_queue, transition_queue, action_queue, seed=42):
    """
        Function called to play episodes in parallel. Returns transitions on the
        sampler_queue as:
        (id, previous_observation, action, reward, observation, done)
        If previous observation is None, the environment has been reset and
        we only need an action without storing it to experience replay
    """
    # Create the environment
    print("Worker %d: creating environment with seed %d" % (id, seed))
    env = make_env(env_name, seed)
    obs = env.reset()
    while True:
        prev_obs = obs
        # Ask and wait for the action
        obs_queue.put((id, obs))
        action = action_queue.get()
        # Check for the termination signal
        if action is None:
            return
        # Perform the action
        obs, r, done, _ = env.step(action)
        # Transition to queue
        transition_queue.put((id, prev_obs, action, r, done, obs))
        # Reset environment if done
        if done:
            obs = env.reset()

def train(env_name, seed=42, timesteps=1, epsilon_decay_last_step=1000,
            er_capacity=1e4, batch_size=16, lr=1e-3, gamma=1.0,  update_target=16,
            exp_name='test', init_timesteps=100, save_every_steps=1e4, arch='nature',
            dueling=False, play_steps=2):
    """
        Main training function. Calls the subprocesses to get experience and
        train the network.
    """
    # Multiprocessing method
    mp.set_start_method('spawn')

    # Get PyTorch device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Set random seed for PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # Create logger
    logger = Logger(exp_name, loggers=['tensorboard'])

    # Create the Q network
    _env = make_env(env_name, seed)
    net = QNetwork(_env.observation_space, _env.action_space, arch=arch, dueling=dueling).to(device)
    # Create the target network as a copy of the Q network
    target_net = copy.deepcopy(net)
    # Create buffer and optimizer
    buffer = ExperienceReplay(capacity=int(er_capacity))
    optimizer = optim.Adam(net.parameters(), lr=lr)
    scheduler = StepLR(optimizer, step_size=LR_STEPS, gamma=0.99)

    # Multiprocessing queue
    obs_queue = mp.Queue(maxsize=play_steps)
    transition_queue = mp.Queue(maxsize=play_steps)
    workers, action_queues = [], []
    for i in range(play_steps):
        action_queue = mp.Queue(maxsize=1)
        _seed = seed + i * 1000
        play_proc = mp.Process(target=play_func, args=(i, env_name, obs_queue, transition_queue, action_queue, _seed))
        play_proc.start()
        workers.append(play_proc)
        action_queues.append(action_queue)

    # Vars to keep track of performances and time
    timestep = 0
    current_reward, current_len = np.zeros(play_steps), np.zeros(play_steps, dtype=np.int64)
    current_time = [time.time() for _ in range(play_steps)]
    # Training loop
    while timestep < timesteps:
        # Compute the current epsilon
        epsilon = EPSILON_STOP + max(0, (EPSILON_START - EPSILON_STOP)*(epsilon_decay_last_step-timestep)/epsilon_decay_last_step)
        logger.log_kv('internals/epsilon', epsilon, timestep)
        # Gather observation N_STEPS
        ids, obs_batch = zip(*[obs_queue.get() for _ in range(play_steps)])
        # Pre-process observation_batch for PyTorch
        obs_batch = torch.from_numpy(np.array(obs_batch)).to(device)
        # Select greedy action from policy, apply epsilon-greedy selection
        greedy_actions = net(obs_batch).argmax(dim=1).cpu().detach().numpy()
        probs = torch.rand(greedy_actions.shape)
        actions = np.where(probs < epsilon, _env.action_space.sample(), greedy_actions)
        # Send actions
        for id, action in zip(ids, actions):
            action_queues[id].put(action)
        # Add transitions to experience replay
        transitions = [transition_queue.get() for _ in range(play_steps)]
        buffer.pushTransitions(transitions)
        # Check if we need to update rewards, time and lengths
        _, _, _, reward, done, _ = zip(*transitions)
        current_reward += reward
        current_len += 1
        for i, done in enumerate(done):
            if done:
                # Log quantities
                logger.log_kv('performance/return', current_reward[i], timestep)
                logger.log_kv('performance/length', current_len[i], timestep)
                logger.log_kv('performance/speed', current_len[i] / (time.time() - current_time[i]), timestep)
                # Reset counters
                current_reward[i] = 0.0
                current_len[i] = 0
                current_time = time.time()

        # Update number of steps
        timestep += play_steps

        # Check if we are in the warm-up phase, otherwise go on with policy update
        if timestep < init_timesteps:
            continue
        # Learning rate upddate and log
        scheduler.step()
        logger.log_kv('internals/lr', scheduler.get_lr()[0], timestep)
        # Clear grads
        optimizer.zero_grad()
        # Get a batch from experience replay
        batch = buffer.sampleTransitions(batch_size)
        def batch_preprocess(batch_item):
            return torch.tensor(batch_item, dtype=(torch.long if isinstance(batch_item[0], np.int64) else None)).to(device)
        ids, states_batch, actions_batch, rewards_batch, done_batch, next_states_batch = map(batch_preprocess, zip(*batch))
        # Compute the loss function
        state_action_values = net(states_batch).gather(1, actions_batch.unsqueeze(-1)).squeeze(-1)
        next_state_values = target_net(next_states_batch).max(1)[0]
        next_state_values[done_batch] = 0.0
        expected_state_action_values = next_state_values.detach() * gamma + rewards_batch
        loss = F.mse_loss(state_action_values, expected_state_action_values)
        logger.log_kv('internals/loss', loss.item(), timestep)
        loss.backward()
        # Clip the gradients to avoid to abrupt changes (this is equivalent to Huber Loss)
        for param in net.parameters():
            param.grad.data.clamp_(-1, 1)
        optimizer.step()

        if timestep % update_target == 0:
            target_net.load_state_dict(net.state_dict())

        # Check if we need to save a checkpoint
        if timestep % save_every_steps == 0:
            torch.save(net.get_extended_state(), exp_name + '.pth')

    # Ending
    for i, worker in enumerate(workers):
        action_queues[i].put(None)
        worker.join()

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
    parser.add_argument('--play_steps', help='Number of parallel environments.', type=int, default=2)
    parser.add_argument('--update_target', help='Number of iterations for each target update.', type=int, default=16)
    parser.add_argument('--lr', help='Optimizer learning rate.', type=float, default=1e-3)
    parser.add_argument('--gamma', help='Discount factor for the MDP.', type=float, default=1.0)
    parser.add_argument("--dueling", default=False, action="store_true", help="Dueling network.")
    args = parser.parse_args()
    # Call the train function with arguments
    train(**vars(args))
