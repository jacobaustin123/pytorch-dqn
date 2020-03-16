import gym
import torch
import torchvision
import time

from environment import Breakout

env = Breakout(skip=4)

for _ in range(3):
    env.reset()
    state, _, done, _ = env.step(env.action_space.sample())

    while not done:
        env.render()
        action = env.action_space.sample()
        print(action)
        action = input()
        print(env.ale.lives())
        next_state, reward, done, info = env.step(int(action))