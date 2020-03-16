import torch
import numpy as np
import datetime
import os

class Memory:
    def __init__(self, max_size, storage_device='cuda:0', target_device='cuda:0'):
        """
        Memory stores a replay buffer on device and samples from it for experience
        replay onto target_device. 

        Parameters:
            max_size: max size of replay buffer (usually memory constrained)
            storage_device: (string or list) storage device(s) for replay buffer (cuda or cpu, usually)
            target_device: device to server replay memories onto (usually cuda)
        """

        self.max_size = max_size
        self.full = False
        self.curr = 0
        self.target_device = target_device
        
        if isinstance(storage_device, str):
            self.n_devices = 1
            self.storage_device = [storage_device]
        else:
            self.n_devices = len(storage_device)
            self.storage_device = storage_device

        self.size_per_device = self.max_size // self.n_devices

        self.states = []
        self.next_states = []
        self.rewards = []
        self.actions = []
        self.terminals = []

        for device in self.storage_device:
            self.states.append(torch.zeros((self.size_per_device, 4, 84, 84), dtype=torch.uint8, device=device))
            self.next_states.append(torch.zeros((self.size_per_device, 4, 84, 84), dtype=torch.uint8, device=device))
            self.rewards.append(torch.zeros((self.size_per_device, 1), device=device))
            self.actions.append(torch.zeros((self.size_per_device, 1), dtype=torch.uint8, device=device))
            self.terminals.append(torch.zeros((self.size_per_device, 1), dtype=torch.uint8, device=device))
            
    def store(self, state, next_state, action, reward, terminal):
        device_id = self.current_device_id()
        device = self.current_device()
        idx = self.curr % self.size_per_device
        
        self.states[device_id][idx] = (state * 255).to(torch.uint8).to(device)
        self.next_states[device_id][idx] = (next_state * 255).to(torch.uint8).to(device)
        self.actions[device_id][idx] = action
        self.rewards[device_id][idx] = reward
        self.terminals[device_id][idx] = terminal

        self.increment()

    def size(self):
        if self.full:
            return self.max_size
        else:
            return self.curr

    def increment(self):
        if self.curr + 1 == self.max_size:
            self.full = True

        self.curr = (self.curr + 1) % self.max_size

    def index_from_device(self, idx, device_id):
        if len(idx) == 0 or self.size() - self.size_per_device * device_id < idx.max():
            raise ValueError("Not enough elements in cache to sample {} elements".format(len(idx)))

        return (self.states[device_id][idx].to(self.target_device).to(torch.float32) / 255.), \
                (self.next_states[device_id][idx].to(self.target_device).to(torch.float32) / 255.), \
                self.actions[device_id][idx].to(self.target_device).to(torch.long), \
                self.rewards[device_id][idx].to(self.target_device), \
                self.terminals[device_id][idx].to(self.target_device).to(torch.int16)
    
    def current_device_id(self):
        return self.curr // self.size_per_device

    def current_device(self):
        return self.storage_devices[self.curr // self.size_per_device]

    def sample(self, N):
        if self.size() < N:
            raise ValueError("Not enough elements in cache to sample {} elements".format(N))

        idx = np.random.choice(self.size(), N)
        device = (idx // self.size_per_device) % self.n_devices
        idx = idx % self.size_per_device

        states = []
        next_states = []
        actions = []
        rewards = []
        terminals = []
    
        for device_id in range(self.n_devices):
            if len(idx[device == device_id]) == 0:
                continue

            s, ns, a, r, t = self.index_from_device(idx[device == device_id], device_id)
            states.append(s)
            next_states.append(ns)
            actions.append(a)
            rewards.append(r)
            terminals.append(t)

        return torch.cat(states), \
               torch.cat(next_states), \
               torch.cat(actions), \
               torch.cat(rewards), \
               torch.cat(terminals)

class EpsilonScheduler:
    def __init__(self, init_value=1.0, lower_bound=0.1, max_steps=1e6):
        self.init_value = init_value
        self.lower_bound = lower_bound
        self.max_steps = max_steps
        self.steps = 0

    def step(self, n):
        self.steps += n

    def step_count(self):
        return self.steps

    def epsilon(self):
        progress = min(self.steps, self.max_steps) / self.max_steps
        return progress * self.lower_bound + (1 - progress) * self.init_value

def make_log_dir(experiment_dir):
    if not os.path.exists(experiment_dir):
        os.mkdir(experiment_dir)

    root = os.path.join(experiment_dir, datetime.datetime.now().strftime("dqn_experiment_%d_%m_%y_%H_%M"))
    weight_dir = os.path.join(root, "weights")
    video_dir = os.path.join(root, "videos")

    if not os.path.exists(root):
        os.mkdir(root)

    if not os.path.exists(weight_dir):
        os.mkdir(weight_dir)

    if not os.path.exists(video_dir):
        os.mkdir(video_dir)

    print(f"[INFO] saving experiment to {root}...")

    return root, weight_dir, video_dir
