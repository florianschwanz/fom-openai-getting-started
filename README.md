# FOM OpenAI Getting Started

## Preparation

Install gym

`
pip install 'gym[all]'
`

Re-install numpy

`
pip install --upgrade --force-reinstall numpy
`

## OpenAI docs

* http://gym.openai.com/docs/
`
00-getting-started-CartPole-v0.py
`

* simple random agent playing CartPole

## Medium Introduction: Reinforcement Learning with OpenAI Gym 

* https://towardsdatascience.com/reinforcement-learning-with-openai-d445c2c687d2

`
01-getting-started-Acrobot-v1.py
`

* simple random agent playing Acrobot

`
02-getting-started-MountainCarContinuous-v0.py
`

* simple agent with log output playing MountainCarContinuous

`
03-getting-started-FrozenLake8x8-v0.py
`

* agent playing FrozenLake8x8 using Q-Learning for a fixed number of steps per episode

`
04-getting-started-FrozenLake8x8-v0-until-done.py
`

* agent playing FrozenLake8x8 using Q-Learning until done per episode

## REINFORCEMENT LEARNING (DQN) TUTORIAL

* https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html

`
05-getting-started-CartPole-v0-dqn.py
`

* agent playing CartPole using Deep Q-Learning
