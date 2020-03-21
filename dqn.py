import gym
import torch
import numpy as np
import torchvision
import matplotlib as mpl
mpl.use('Agg')
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

MEM_SIZE = int(1e6) # this is either 250k or 1 million in the paper (size of replay memory)
EPISODES = int(1e5) # total training episodes
BATCH_SIZE = 64 # minibatch update size
GAMMA = 0.99 # discount factor
EPS_START = 1.0 # starting epsilon-greedy
EPS_END = 0.1 # ending epsilon-greedy (minimum)
EPS_STEPS = 1e6 # period of decrease
STORAGE_DEVICES = ['cuda:1'] # list of devices to use for episode storage (need about 10GB for 1 million memories)
DEVICE = 'cuda:1' # list of devices for computation
UPDATE_FREQ = 4 # perform minibatch update once every UPDATE_FREQ
TARGET_UPDATE_EVERY = 10000 # (minibatch updates)
INIT_MEMORY_SIZE = 50000 # initial size of memory before minibatch updates begin

TEST_EVERY = 1000 # (episodes)
PLOT_EVERY = 10 # (episodes)
SAVE_EVERY = 1000 # (episodes)
EXPERIMENT_DIR = "experiments"
NUM_TEST = 20

root_dir, weight_dir, video_dir = make_log_dir(EXPERIMENT_DIR)
plot_title = "DQN ({})".format(datetime.datetime.now().strftime("%d/%m/%y %H:%M"))

scheduler = EpsilonScheduler(init_value=EPS_START, lower_bound=EPS_END, max_steps=EPS_STEPS)
env = Breakout()
mem = Memory(MEM_SIZE, storage_devices=STORAGE_DEVICES, target_device=DEVICE)

q_func = Model(env.action_space.n).to(DEVICE)
if args.weights:
    q_func.load_state_dict(torch.load(args.weights))

target_q_func = Model(env.action_space.n).to(DEVICE)
target_q_func.load_state_dict(q_func.state_dict())
target_q_func.eval()

# optimizer = optim.RMSprop(q_func.parameters(), lr=1e-3, alpha=0.95, momentum=0.95, eps=1e-2)
optimizer = optim.Adam(q_func.parameters(), lr=0.00001)

loss_func = nn.SmoothL1Loss()
plotter = VisdomLinePlotter()
env = Monitor(env, directory=video_dir, video_callable=lambda count: count % 500 == 0, force=True)

def test():
    print("[TESTING]")
    total_reward = 0

    for _ in range(NUM_TEST):
        env.reset(eval=True) # performs random actions to start
        state, _, done, _ = env.step(env.action_space.sample())
        frame = 0

        while not done:
            # env.render()
            q_values = q_func(state.to(DEVICE))
            if np.random.random() > 0.05: # small epsilon-greedy
                action = torch.argmax(q_values, dim=1).item()
            else:
                action = env.action_space.sample()

            lives = env.ale.lives()
            next_state, reward, done, info = env.step(action)
            if env.ale.lives() != lives: # lost life
                pass
                # plt.imshow(next_state[0,0])
                # plt.savefig(f"frame-{frame}.png")
                # print("LOST LIFE")

            total_reward += reward
            state = next_state
            frame += 1
            # print(f"[TESTING {frame}] Action: {action}, Q-Values: {np.array(q_values.cpu().detach())}, Reward: {reward}, Total Reward: {total_reward}, Terminal: {done}")
            # plt.imshow(state[0,0])
            # plt.savefig("frame-{}.png".format(frame))

    total_reward /= NUM_TEST
    plotter.plot("DQN", "Total Test Reward", plot_title, scheduler.step_count(), total_reward, xlabel='frames')
    print(f"[TESTING] Total Reward: {total_reward}")

    return total_reward

avg_reward = 0
avg_q = 0
num_parameter_updates = 0
for episode in range(EPISODES):
    avg_loss = 0
    total_reward = 0
    frame = 0

    env.reset()
    state, _, done, _ = env.step(env.action_space.sample())

    while not done:
        q_values = q_func(state.to(DEVICE))
        if np.random.random() > scheduler.epsilon(): # epsilon-random policy
            action = torch.argmax(q_values, dim=1)
        else:
            action = env.action_space.sample()

        avg_q = 0.9 * avg_q + 0.1 * q_values.mean().item() # record average q value

        lives = env.ale.lives() # get lives before action
        next_state, reward, done, info = env.step(action)
                
        if env.ale.lives() != lives: # hack used in the paper to make loss of life a terminal state.
            mem.store(state[0,0], action, reward, True) # we store the earliest frame in the window
        else:
            mem.store(state[0,0], action, reward, done)

        state = next_state
        total_reward += reward
        frame += 1
        scheduler.step(1)

        if mem.size() < INIT_MEMORY_SIZE:
            continue

        if scheduler.step_count() % UPDATE_FREQ == 0:
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
            num_parameter_updates += 1
        
        if num_parameter_updates % TARGET_UPDATE_EVERY == 0: # reset target to source
            target_q_func.load_state_dict(q_func.state_dict())

    avg_loss /= frame
    avg_reward = 0.9 * avg_reward + 0.1 * total_reward
    print(f"[EPISODE {episode}] Loss: {avg_loss}, Total Reward: {total_reward}, Frames: {frame}, Epsilon: {scheduler.epsilon()}, Total Frames: {scheduler.step_count()}, Memory Size: {mem.size()}, Average Q: {avg_q}")

    if episode % PLOT_EVERY == 0:
        plotter.plot("DQN", "Total Reward", plot_title, scheduler.step_count(), avg_reward, xlabel='frames')
        plotter.plot("DQN", "Epsilon", plot_title, scheduler.step_count(), scheduler.epsilon(), xlabel='frames')
        plotter.plot("DQN", "Episode Length (Frames)", plot_title, scheduler.step_count(), frame, xlabel='frames')
        plotter.plot("DQN", "Average Loss", plot_title, scheduler.step_count(), avg_loss, xlabel='frames')
        plotter.plot("DQN", "Average Q", plot_title, scheduler.step_count(), avg_q, xlabel='frames')

    if episode % TEST_EVERY == 0:
        test_reward = test()

    if episode % SAVE_EVERY == 0:
        path = f"episode-{episode}.pt"
        weight_path = os.path.join(weight_dir, path)
        info_path = os.path.join(root_dir, "info.txt")

        torch.save(q_func.state_dict(), weight_path)

        with open(info_path, "a+") as f:
            f.write(",".join([str(x) for x in [path, scheduler.step_count(), scheduler.epsilon(), episode, test_reward, str(optimizer).replace("\n", "").replace("  ", " ")]]) + "\n")
