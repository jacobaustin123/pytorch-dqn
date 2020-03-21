import gym
import torch
import torchvision
import time
import moviepy

from environment import Environment
import moviepy.editor as mpy
import numpy as np

env = Environment(game='pong')

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

num = 0
def make_frame(t):
    global num
    if num < len(frames):
        num += 1
        return (np.array(frames[num - 1]) * 255).astype(np.uint8)
    else: 
        return (np.array(frames[-1]) * 255).astype(np.uint8)

fps = 20
clip = mpy.VideoClip(make_frame, duration=len(frames) // fps)
clip.write_gif("example.gif",fps=fps)
