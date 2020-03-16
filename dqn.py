import gym
import torch
import numpy as np
import torchvision
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import argparse
from gym.wrappers import Monitor
import datetime
import os

from model import Model
from plotting import VisdomLinePlotter
from environment import Breakout
from utils import Memory, EpsilonScheduler, make_log_dir

parser = argparse.ArgumentParser(
    description='An implementation of the 2015 DeepMind DQN Paper'
)

parser.add_argument('--weights', type=str, help='weights file for pretrained weights')

args = parser.parse_args()

MEM_SIZE = int(6e5) # this is either 250k or 1 million in the paper (size of replay memory)
EPISODES = int(1e5)
BATCH_SIZE = 32 # 32
GAMMA = 0.99
TARGET_UPDATE = 10
EPS_START = 1.0
EPS_END = 0.1
EPS_STEPS = 1e6
STORAGE_DEVICE = ['cpu', 'cuda:0', 'cuda:1']
DEVICE = 'cuda:0'
LR = 0.00001 # 0.01
START_LEARNING_AT = 100 * BATCH_SIZE # in paper, 50k frames or ~ 500 episodes
TEST_EVERY = 100
PLOT_EVERY = 10
SAVE_EVERY = 1000
EXPERIMENT_DIR = "experiments"

root_dir, weight_dir, video_dir = make_log_dir(EXPERIMENT_DIR)
plot_title = "DQN ({})".format(datetime.datetime.now().strftime("%d/%m/%y %H:%M"))

scheduler = EpsilonScheduler(init_value=EPS_START, lower_bound=EPS_END, max_steps=EPS_STEPS)
env = Breakout()
mem = Memory(MEM_SIZE, storage_device=STORAGE_DEVICE, target_device=DEVICE)

q_func = Model(4).to(DEVICE)
if args.weights:
    q_func.load_state_dict(torch.load(args.weights))

target_q_func = Model(4).to(DEVICE)
target_q_func.load_state_dict(q_func.state_dict())
# optimizer = optim.RMSprop(q_func.parameters(), lr=LR, alpha=0.95, eps=0.01) # betas=(0.5, 0.999), alpha=0.95, eps=0.01
optimizer = optim.Adam(q_func.parameters(), lr=LR) # betas=(0.5, 0.999)

loss_func = nn.SmoothL1Loss()
plotter = VisdomLinePlotter()
env = Monitor(env, directory=video_dir, video_callable=lambda count: count % 500 == 0, force=True)

def test():
    env.reset()
    state, _, done, _ = env.step(env.action_space.sample())
    total_reward = 0
    frame = 0

    while not done:
        # env.render()
        q_values = q_func(state.to(DEVICE))
        if np.random.random() > 0.05: # small epsilon-greedy
            action = torch.argmax(q_values, dim=1).item()
        else:
            action = env.action_space.sample()
        
        next_state, reward, done, info = env.step(action)
        breakpoint()
        total_reward += reward
        state = next_state
        frame += 1
        # print(f"[TESTING {frame}] Action: {action}, Q-Values: {np.array(q_values.cpu().detach())}, Reward: {reward}, Cumm. Reward: {total_reward}, Terminal: {done}")

    plotter.plot("DQN", "Total Test Reward", plot_title, scheduler.step_count(), total_reward, xlabel='frames')
    print(f"[TESTING] Total Reward: {total_reward}")

    return total_reward

test()

avg_reward = 0
for episode in range(EPISODES):
    avg_loss = 0
    total_reward = 0
    frame = 0

    env.reset()
    state, _, done, _ = env.step(env.action_space.sample())

    while not done:
        q_values = q_func(state.to(DEVICE))
        if np.random.random() > scheduler.epsilon():
            action = torch.argmax(q_values, dim=1)
        else:
            action = env.action_space.sample()

        lives = env.ale.lives()
        next_state, reward, done, info = env.step(action)
        
        if env.ale.lives() != lives: # hack used in the paper
            mem.store(state, next_state, action, reward, True)
        else:
            mem.store(state, next_state, action, reward, done)

        state = next_state
        total_reward += reward
        frame += 1

        if mem.size() < START_LEARNING_AT:
            continue
        
        states, next_states, actions, rewards, terminals = mem.sample(BATCH_SIZE)

        mask = (1 - terminals).float()
        y = rewards + mask * GAMMA * torch.max(target_q_func(next_states), dim=1).values.view(-1, 1).detach()
        x = q_func(states)[range(BATCH_SIZE), actions.squeeze()]
        loss = loss_func(x, y.squeeze())
        optimizer.zero_grad()
        loss.backward()

        for param in q_func.parameters(): # gradient clipping
            param.grad.data.clamp_(-1, 1)

        optimizer.step()
        avg_loss += loss.item()
        
    avg_loss /= frame
    avg_reward = 0.9 * avg_reward + 0.1 * total_reward
    scheduler.step(frame)
    print(f"[EPISODE {episode}] Loss: {avg_loss}, Total Reward: {total_reward}, Frames: {frame}, Epsilon: {scheduler.epsilon()}, Total Frames: {scheduler.step_count()}, Memory Size: {mem.size()}")

    if episode % PLOT_EVERY == 0:
        plotter.plot("DQN", "Total Reward", plot_title, scheduler.step_count(), avg_reward, xlabel='frames')
        plotter.plot("DQN", "Epsilon", plot_title, scheduler.step_count(), scheduler.epsilon(), xlabel='frames')
        plotter.plot("DQN", "Episode Length (Frames)", plot_title, scheduler.step_count(), frame, xlabel='frames')
        plotter.plot("DQN", "Average Loss", plot_title, scheduler.step_count(), avg_loss, xlabel='frames')

    if episode % TARGET_UPDATE == 0: # reset target to source
        target_q_func.load_state_dict(q_func.state_dict())

    if episode % TEST_EVERY == 0:
        test_reward = test()

    if episode % SAVE_EVERY == 0:
        test_reward = test()
        path = f"episode-{episode}.pt"
        weight_path = os.path.join(weight_dir, path)
        info_path = os.path.join(root_dir, "info.txt")

        torch.save(q_func.state_dict(), weight_path)

        with open(info_path, "a+") as f:
            f.write(",".join([str(x) for x in [path, scheduler.step_count(), scheduler.epsilon(), episode, test_reward]]) + "\n")