"""
    Hard-coded deterministic policy, using as a benchmark. Simply avoids bumping
    into walls if possible.

    ACTIONS:
    0: UP
    1: RIGHT
    2: DOWN
    3: LEFT
"""

import numpy as np

DIRECTIONS = [np.array([-1,0]), np.array([0,1]), np.array([1,0]), np.array([0,-1])]

class HighlanderPolicy():
    """
        Random policy which avoids to bump into walls (does not check auto-bumping)
    """

    def __init__(self, env, head_color=(0, 77, 0)):
        self.size = env.SIZE[0]
        self.previous_head = None
        self.head_color = head_color
        self.walls = 1 if env.add_walls else 0

    def search_rgb(self, observation, color):
        return np.where((observation[:,:,0] == color[0]) & (observation[:,:,1] == color[1]) & (observation[:,:,2] == color[2]))

    def act(self, observation):
        # Get snek position and current direction
        head_y, head_x = map(lambda x: x[0], self.search_rgb(observation, self.head_color))
        actions = [0, 1, 2, 3]
        # Check left border
        if head_x <= self.walls:
            actions.remove(3)
            # Check also if we are moving horizontally, if so also don't go right
            if head_y == self.previous_head[0]:
                actions.remove(1)
        # Check upper border
        if head_y <= self.walls:
            actions.remove(0)
            # Check also if we are moving vertically
            if head_x == self.previous_head[1]:
                actions.remove(2)
        # Check right border
        if head_x >= self.size - 1 - self.walls:
            actions.remove(1)
            # Check also if moving horizontally
            if head_y == self.previous_head[0]:
                actions.remove(3)
        # Check bottom border
        if head_y >= self.size - 1 - self.walls:
            actions.remove(2)
            # Check also moving vertically
            if head_x == self.previous_head[1]:
                actions.remove(0)

        if len(actions) > 0:
            a = np.random.choice(actions)
        else:
            a = 0
        self.previous_head = (head_y, head_x)
        return a
