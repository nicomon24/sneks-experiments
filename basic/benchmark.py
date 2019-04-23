"""
    Hard-coded deterministic policy, using as a benchmark.

    ACTIONS:
    0: UP
    1: RIGHT
    2: DOWN
    3: LEFT
"""

import numpy as np

DIRECTIONS = [np.array([-1,0]), np.array([0,1]), np.array([1,0]), np.array([0,-1])]

class BasicPolicy():

    def __init__(self, size=16):
        self.current_direction = 0
        self.current_counter = 0
        self.size = size

    def next_position(self, current_y, current_x, action):
        ady, adx = DIRECTIONS[action]
        return current_y + ady, current_x + adx

    def act(self, observation):
        # Get food position
        food_y, food_x = map(lambda x: x[0], np.where(observation[:,:,0] == 255))
        # Get snek position and current direction
        head_y, head_x = map(lambda x: x[0], np.where(observation[:,:,1] == 77))
        # Get distance
        dy, dx = food_y - head_y, food_x - head_x
        # Get possible direction (don't eat yourself, don't go outside)
        possible_directions = []
        for action in [0, 1, 2, 3]:
            next_y, next_x = self.next_position(head_y, head_x, action)
            if 0 <= next_x < self.size and 0 <= next_y < self.size:
                if observation[next_y, next_x, 1] != 204:
                    possible_directions.append(action)

        # Check the new action
        ady, adx = DIRECTIONS[self.current_direction]
        is_correct_direction = (dy * ady) + (dx * adx) > 0
        action = 0 # Fuckit action
        if self.current_direction in possible_directions and is_correct_direction:
                action = self.current_direction
                self.current_counter += 1
        else:
            for i in range(1, 4):
                _action = (self.current_direction + i) % 4
                if _action in possible_directions:
                    action = _action
                    break
            self.current_counter = 0

        self.current_direction = action
        return action

if __name__== '__main__':
    import gym
    import sneks
    import time

    pi = BasicPolicy()

    env = gym.make('snek-rgb-16-v1')
    obs = env.reset()
    done = False
    while not done:
        obs, r, done, _ = env.step(pi.act(obs))
        env.render()
        time.sleep(0.05)
    env.close()
