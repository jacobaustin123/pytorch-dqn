import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

class Model(nn.Module):
    def __init__(self, actions):
        super(Model, self).__init__()

        self.conv1 = nn.Conv2d(4, 32, 8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, 3, stride=1)
        self.fc1 = nn.Linear(7 * 7 * 64, 512)
        self.fc2 = nn.Linear(512, actions)

        torch.nn.init.kaiming_uniform_(self.conv1.weight, nonlinearity='relu')
        torch.nn.init.kaiming_uniform_(self.conv2.weight, nonlinearity='relu')
        torch.nn.init.kaiming_uniform_(self.conv3.weight, nonlinearity='relu')
        torch.nn.init.kaiming_uniform_(self.fc1.weight, nonlinearity='relu')
        torch.nn.init.kaiming_uniform_(self.fc2.weight, nonlinearity='relu')

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.fc1(x.view(x.shape[0], -1)))
        return self.fc2(x)

if __name__ == "__main__":
    img = torch.randn((1, 4, 84, 84))
    m = Model(4)
    m(img)