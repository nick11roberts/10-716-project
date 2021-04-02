import torch
import torch.nn as nn
import torch.nn.functional as F

class S_CONV(nn.Module):
    def __init__(self, alpha=200):
        super(S_CONV, self).__init__()
        in_size = 32 * 32 * 3
        self.conv1 = nn.Conv2d(3, alpha, 9, stride=2)
        self.bn1 = nn.BatchNorm2d(alpha)
        self.fc2 = nn.Linear(144 * alpha, 24 * alpha)
        self.bn2 = nn.BatchNorm1d(24 * alpha)
        self.fc3 = nn.Linear(24 * alpha, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        
        x = torch.flatten(x, 1)
        x = self.fc2(x)
        x = self.bn2(x)
        x = F.relu(x)

        x = self.fc3(x)
        output = F.log_softmax(x, dim=1)
        return output


class S_FC(nn.Module):
    def __init__(self, alpha=200):
        super(S_FC, self).__init__()
        in_size = 32 * 32 * 3
        self.fc1 = nn.Linear(in_size, 144 * alpha)
        self.bn1 = nn.BatchNorm1d(144 * alpha)
        self.fc2 = nn.Linear(144 * alpha, 24 * alpha)
        self.bn2 = nn.BatchNorm1d(24 * alpha)
        self.fc3 = nn.Linear(24 * alpha, 10)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.bn1(x)
        x = F.relu(x)
        
        x = self.fc2(x)
        x = self.bn2(x)
        x = F.relu(x)

        x = self.fc3(x)
        output = F.log_softmax(x, dim=1)
        return output
