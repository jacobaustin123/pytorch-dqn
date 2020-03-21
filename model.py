import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

class Model(nn.Module):
    def __init__(self, actions):
        super(Model, self).__init__()

        self.conv1 = nn.Conv2d(4, 32, 8, stride=4)
        # self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        # self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, 3, stride=1)
        # self.bn3 = nn.BatchNorm2d(64)
        self.fc1 = nn.Linear(7 * 7 * 64, 1024)
        self.fc2 = nn.Linear(1024, actions)

        torch.nn.init.kaiming_normal_(self.conv1.weight, nonlinearity='leaky_relu')
        torch.nn.init.kaiming_normal_(self.conv2.weight, nonlinearity='leaky_relu')
        torch.nn.init.kaiming_normal_(self.conv3.weight, nonlinearity='leaky_relu')
        torch.nn.init.kaiming_normal_(self.fc1.weight, nonlinearity='leaky_relu')
        torch.nn.init.kaiming_normal_(self.fc2.weight, nonlinearity='leaky_relu')

    def forward(self, x):
        x = F.leaky_relu(self.conv1(x), 0.01)
        x = F.leaky_relu(self.conv2(x), 0.01)
        x = F.leaky_relu(self.conv3(x), 0.01)
        x = F.leaky_relu(self.fc1(x.view(x.shape[0], -1)), 0.01)
        return self.fc2(x)

if __name__ == "__main__":
    img = torch.randn((1, 4, 84, 84))
    m = Model(4)
    m(img)