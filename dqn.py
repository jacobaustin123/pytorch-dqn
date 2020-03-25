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
from environment import Environment
from utils import Memory, EpsilonScheduler, make_log_dir, save_gif

parser = argparse.ArgumentParser(description='An implementation of the 2015 DeepMind DQN Paper')

parser.add_argument('--weights', type=str, help='weights file for pretrained weights')
parser.add_argument('--nosave', default=False, action='store_true', help='do not save a record of the run')

args = parser.parse_args()

MEM_SIZE = int(1e6) # this is either 250k or 1 million in the paper (size of replay memory)
EPISODES = int(1e5) # total training episodes
BATCH_SIZE = 32 # minibatch update size
GAMMA = 0.99 # discount factor
STORAGE_DEVICES = ['cuda:1'] # list of devices to use for episode storage (need about 10GB for 1 million memories)
DEVICE = 'cuda:1' # list of devices for computation
UPDATE_FREQ = 4 # perform minibatch update once every UPDATE_FREQ
TARGET_UPDATE_EVERY = 10000 # in units of minibatch updates
INIT_MEMORY_SIZE = 200000 # initial size of memory before minibatch updates begin

TEST_EVERY = 1000 # (episodes)
PLOT_EVERY = 10 # (episodes)
SAVE_EVERY = 1000 # (episodes)
EXPERIMENT_DIR = "experiments"
NUM_TEST = 20
GAME = 'breakout'

scheduler = EpsilonScheduler(schedule=[(0, 1), (INIT_MEMORY_SIZE, 1), (2e6, 0.1), (30e6, 0.01)])

if not args.nosave:
    root_dir, weight_dir, video_dir = make_log_dir(EXPERIMENT_DIR, GAME)

    with open(os.path.join(EXPERIMENT_DIR, "current.txt"), "w") as f:
        f.write(os.path.abspath(video_dir))

plot_title = "{} DQN ({})".format(GAME, datetime.datetime.now().strftime("%d/%m/%y %H:%M"))

env = Environment(game=GAME)
mem = Memory(MEM_SIZE, storage_devices=STORAGE_DEVICES, target_device=DEVICE)

q_func = Model(env.action_space.n).to(DEVICE)
if args.weights:
    q_func.load_state_dict(torch.load(args.weights))

target_q_func = Model(env.action_space.n).to(DEVICE)
target_q_func.load_state_dict(q_func.state_dict())

# optimizer = optim.RMSprop(q_func.parameters(), lr=1e-3, alpha=0.95, momentum=0.95, eps=1e-2)
optimizer = optim.Adam(q_func.parameters(), lr=0.0000625, eps=1.5e-4) # 0.00001 for breakout, 0.00025 is faster for pong

loss_func = nn.SmoothL1Loss()
plotter = VisdomLinePlotter(disable=args.nosave)

if not args.nosave:
    env = Monitor(env, directory=video_dir, video_callable=lambda count: count % 500 == 0, force=True)

def test(save=False):
    print("[TESTING]")
    total_reward = 0
    unclipped_reward = 0

    for i in range(NUM_TEST):
        if i == 0 and save:
            frames = []

        env.reset(eval=True) # performs random actions to start
        state, _, done, _ = env.step(env.action_space.sample())
        frame = 0

        while not done:
            if i == 0 and save:
                frames.append(state[0,0])
            
            # env.render()
            q_values = q_func(state.to(DEVICE))
            if np.random.random() > 0.01: # small epsilon-greedy, sometimes 0.05
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

            unclipped_reward += info['unclipped_reward']
            total_reward += reward
            state = next_state
            frame += 1
            # print(f"[TESTING {frame}] Action: {action}, Q-Values: {np.array(q_values.cpu().detach())}, Reward: {reward}, Total Reward: {total_reward}, Terminal: {done}")
            # plt.imshow(state[0,0])
            # plt.savefig("frame-{}.png".format(frame))

        if i == 0 and save:
            frames.append(state[0,0])
            save_gif(frames, "{}.gif".format(os.path.join(video_dir, str(scheduler.step_count()))))

    total_reward /= NUM_TEST
    unclipped_reward /= NUM_TEST
    plotter.plot("DQN", "Total Test Reward", plot_title, scheduler.step_count(), total_reward, xlabel='frames')
    plotter.plot("DQN", "Total Unclipped Test Reward", plot_title, scheduler.step_count(), unclipped_reward, xlabel='frames')
    print(f"[TESTING] Total Reward: {total_reward}, Unclipped Reward: {unclipped_reward}")

    return total_reward

avg_reward = 0
avg_unclipped_reward = 0
avg_q = 0
num_parameter_updates = 0
for episode in range(EPISODES):
    avg_loss = 0
    total_reward = 0
    unclipped_reward = 0
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
        
        # hack to make learning faster (count loss of life as end of episode for memory purposes)
        mem.store(state[0,0], action, reward, done or (env.ale.lives() != lives))

        state = next_state
        total_reward += reward
        unclipped_reward += info['unclipped_reward']
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
    avg_unclipped_reward = 0.9 * avg_unclipped_reward + 0.1 * unclipped_reward

    print(f"[EPISODE {episode}] Loss: {avg_loss}, Total Reward: {total_reward}, Total Unclipped Reward: {unclipped_reward}, Frames: {frame}, Epsilon: {scheduler.epsilon()}, Total Frames: {scheduler.step_count()}, Memory Size: {mem.size()}, Average Q: {avg_q}")

    if episode % PLOT_EVERY == 0:
        plotter.plot("DQN", "Total Reward", plot_title, scheduler.step_count(), avg_reward, xlabel='frames')
        plotter.plot("DQN", "Total Unclipped Reward", plot_title, scheduler.step_count(), unclipped_reward, xlabel='frames')
        plotter.plot("DQN", "Epsilon", plot_title, scheduler.step_count(), scheduler.epsilon(), xlabel='frames')
        plotter.plot("DQN", "Episode Length (Frames)", plot_title, scheduler.step_count(), frame, xlabel='frames')
        plotter.plot("DQN", "Average Loss", plot_title, scheduler.step_count(), avg_loss, xlabel='frames')
        plotter.plot("DQN", "Average Q", plot_title, scheduler.step_count(), avg_q, xlabel='frames')

    if episode % TEST_EVERY == 0:
        test_reward = test(save=not args.nosave)

    if episode % SAVE_EVERY == 0 and not args.nosave:
        path = f"episode-{episode}.pt"
        weight_path = os.path.join(weight_dir, path)
        info_path = os.path.join(root_dir, "info.txt")

        torch.save(q_func.state_dict(), weight_path)

        with open(info_path, "a+") as f:
            f.write(",".join([str(x) for x in [path, scheduler.step_count(), scheduler.epsilon(), episode, test_reward]]) + "\n")
