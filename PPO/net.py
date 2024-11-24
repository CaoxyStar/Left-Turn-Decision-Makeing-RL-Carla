import torch
import torch.nn as nn

class Policy_Network(nn.Module):
    def __init__(self):
        """Initializes the policy network.

        Args:
            obs_space_dims: Dimension of the observation space
            hidden_dims: Dimension of the hidden layer
            action_space_dims: Dimension of the action space
        """
        super(Policy_Network, self).__init__()

        self.extract_net = nn.Sequential(
            nn.Conv2d(1, 3, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(3, 3, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(3, 1, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Flatten(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 16), nn.ReLU(),
            nn.Linear(16, 3)
        )
        self.prob_net = nn.Softmax(dim=1)

    def forward(self, x: torch.Tensor):
        feat = self.extract_net(x)
        action_porb = self.prob_net(feat)
        return action_porb


class Value_Network(nn.Module):
    def __init__(self):
        """Initializes the value network.

        Args:
            obs_space_dims: Dimension of the observation space
            hidden_dims: Dimension of the hidden layer
        """
        super(Value_Network, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 3, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(3, 6, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(6, 3, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Flatten(),
            nn.Linear(3 * 8 * 8, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 1)
        )
    
    def forward(self, x: torch.Tensor):
        value = self.model(x)
        return value