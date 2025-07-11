import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNPolicy(nn.Module):
    def __init__(self, input_shape=(32, 21), num_actions=9):
        super().__init__()
        c, h, w = 1, *input_shape

        self.conv1 = nn.Conv2d(c, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)

        # Compute the output size after convolutions dynamically
        with torch.no_grad():
            dummy = torch.zeros(1, 1, h, w)
            dummy = F.relu(self.conv1(dummy))
            dummy = F.relu(self.conv2(dummy))
            dummy = F.relu(self.conv3(dummy))
            self.flat_dim = dummy.view(1, -1).size(1)

        # Use the computed flat_dim instead of hardcoded value
        self.fc1 = nn.Linear(self.flat_dim, 256)
        self.fc2 = nn.Linear(256, num_actions)

    def forward(self, x):
        x = x.float()
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        q_values = self.fc2(x)
        return q_values