<p align="center"><img src="http://i0.kym-cdn.com/photos/images/original/001/185/731/ed3.png" height="164"></p>

# Sneks solutions
This repository contains a set of different solutions and studies for the many [sneks environments](https://github.com/nicomon24/Sneks). This has multiple goals:
- Easy to read and explained implementation of various algorithm, from the more classical ones to state-of-the-art methods. The algorithms will be implemented using PyTorch.
- Studies to understand state representation issues in complex visual policies in reinforcement-learning
- Studies to explore the instability of multi-agent systems, as suggested by [this OpenAI post](https://openai.com/blog/requests-for-research-2/).

## Draft of Roadmap
1. Implementation of vanilla DQN and tests on different environments.
2. Implementation of PPO and tests on different environments.
3. Basic multi-agent competitive setting: learning by self-play

## Ideas
- Perception studies: is the trained policy able to understand a particular scene? Image reconstruction?
- Recurrent policies and partial observability?
- RL fast optimization + evolutionary slow optimization?
