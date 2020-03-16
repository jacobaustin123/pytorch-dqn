import gym
import torch
import torchvision

# mean = torch.Tensor([0.1])
# std = torch.Tensor([0.2])

class Breakout:
    def __init__(self, skip=4):
        self.env = gym.make('BreakoutNoFrameskip-v4')
        self.skip = skip
        
        self.ale = self.env.ale
        self.spec = self.env.spec
        self.action_space = self.env.action_space
        self.metadata = self.env.metadata
        self.observation_space = self.env.observation_space
        self.reward_range = self.env.reward_range

        self.transforms = torchvision.transforms.Compose([
            torchvision.transforms.ToPILImage(),
            torchvision.transforms.Grayscale(),
            lambda x : torchvision.transforms.functional.crop(x, 32, 8, 168, 140),
            torchvision.transforms.Resize((84, 84)),
            torchvision.transforms.ToTensor(),
            # torchvision.transforms.Normalize(mean = mean, std = std)
        ])

    def reset(self):
        self.env.reset()
        self.step(1)

    def render(self, mode=None):
        return self.env.render(mode)

    def clip(self, reward):
        return min(reward, 1)

    def step(self, action):
        total_reward = 0
        obs = []

        for i in range(self.skip):
            observation, reward, done, info = self.env.step(action)
            obs.append(self.transforms(observation))  # [32:200:2, 8:-8:2]
            total_reward += self.clip(reward)

        return torch.stack(obs).transpose(0, 1), total_reward, done, info

    def close(self):
        return self.env.close()