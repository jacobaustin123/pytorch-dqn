import gym
import torch
import torchvision
import time

from utils import save_gif
from environment import Environment
import numpy as np

env = Environment(game='centipede')

frames = []

for _ in range(3):
    env.reset()
    state, _, done, _ = env.step(env.action_space.sample())

    while not done:
        env.render()
        action = env.action_space.sample()
        # action = input()
        # print(env.ale.lives())
        next_state, reward, done, info = env.step(int(action))
        frames.append(next_state[0,0])

save_gif(frames, name="example.gif")