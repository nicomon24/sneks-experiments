'''
    Implement the DQN's experience replay. Tasks:
    - add a transition to the memory
    - batch a sample of transitions
'''
from collections import deque
import numpy as np

class ExperienceReplay:

    def __init__(self, capacity=1):
        # Init the buffer
        self.capacity = capacity
        self.buffer = deque(maxlen=self.capacity)

    def pushTransitions(self, transitions):
        self.buffer.extend(transitions)

    def sampleTransitions(self, size=1):
        keys = np.random.choice(len(self.buffer), size=size, replace=True)
        return [self.buffer[key] for key in keys] 

    def __len__(self):
        return len(self.buffer)
