'''
    Play in single-agent mode. Declare which policy need to play.
'''

from benchmark.base_policy import HighlanderPolicy
import gym
import sneks
import time

ENV = 'snek-rgb-16-v1'

# Create env
env = gym.make(ENV)
# Create agent
agent = HighlanderPolicy(env)

# Play loop
obs = env.reset()
while True:
    action = agent.act(obs)
    obs, r, done, _ = env.step(action)
    env.render()
    if done:
        env.reset()
    time.sleep(0.05)
env.close()
