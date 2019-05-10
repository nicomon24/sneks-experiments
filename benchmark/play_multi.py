'''
    Play in multi-agent mode. Declare all of the players.
'''

from benchmark.base_policy import HighlanderPolicy
import gym
import sneks
import time

ENV = 'sneks2-rgb-16-v1'

# Create env
env = gym.make(ENV)
# Create agent
agents = [HighlanderPolicy(env, head_color=(0, 77, 0)), HighlanderPolicy(env, head_color=(0, 0, 77))]

# Play loop
obs = env.reset()
while True:
    actions = [agent.act(obs) for agent in agents]
    obs, r, done, _ = env.step(actions)
    print(r)
    env.render()
    if all(done):
        env.reset()
    time.sleep(0.25)
env.close()
