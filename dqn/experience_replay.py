"""
    Here we define a class to implement the Experience Replay described in the paper.
    Experience Replay is able to store a fixed amount of transitions of the form
    (state, action, reward, next_state). It is also capable of sampling a batch
    of these transitions to train the Q model.
"""

import numpy as np
import torch

class ExperienceReplay(object):

    def __init__(self, capacity=1e4):
        # Store the maximum capacity of transitions
        self.capacity = capacity
        # Create lazy-loading memory structure
        self.memory = {
            'state': [],
            'action': [],
            'reward': [],
            'next_state': [],
            'done': []
        }

    def push(self, **kwargs):
        """
            Add a transition to the experience replay buffer.
        """
        # Update the memory dict (all the keys)
        for key in self.memory.keys():
            self.memory[key].append(kwargs[key])
            # Remove the head of the memory if the capacity is reached
            self.memory[key] = self.memory[key][-self.capacity:]

    def sample(self, batch_size):
        """
            Sample batch_size transitions from the experience replay buffer.
        """
        indexes = np.arange(len(self))
        selected = np.random.choice(indexes, batch_size)
        result = []
        for key in ['state', 'action', 'reward', 'next_state', 'done']:
            selection = torch.cat([self.memory[key][i] for i in selected], 0)
            result.append(selection)
        return tuple(result)

    def __len__(self):
        """
            Return the current number of transitions in the experience replay buffer.
        """
        return len(self.memory['state'])
