"""
    Here we define a class to implement the Experience Replay described in the paper.
    Experience Replay is able to store a fixed amount of transitions of the form
    (state, action, reward, next_state). It is also capable of sampling a batch
    of these transitions to train the Q model.
"""

import numpy as np

class ExperienceReplay(object):

    def __init__(self, capacity=1e4):
        # Store the maximum capacity of transitions
        self.capacity = capacity
        # Create lazy-loading memory structure
        self.memory = {
            'state': None,
            'action': None,
            'reward': None,
            'next_state': None,
            'done': None
        }

    def push(self, **kwargs):
        """
            Add a transition to the experience replay buffer.
        """
        # Update the memory dict (all the keys)
        for key in self.memory.keys():
            self.memory[key] = kwargs[key] if self.memory[key] is None else np.concatenate((self.memory[key], kwargs[key]), axis=0)
            # Remove the head of the memory if the capacity is reached
            self.memory[key] = self.memory[key][-self.capacity:]

    def sample(self, batch_size):
        """
            Sample batch_size transitions from the experience replay buffer.
        """
        indexes = np.arange(len(self))
        selected = np.random.choice(indexes, batch_size)
        return (self.memory['state'][selected],
               self.memory['action'][selected],
               self.memory['reward'][selected],
               self.memory['next_state'][selected],
               self.memory['done'][selected])

    def __len__(self):
        """
            Return the current number of transitions in the experience replay buffer.
        """
        return 0 if self.memory['state'] is None else self.memory['state'].shape[0]
