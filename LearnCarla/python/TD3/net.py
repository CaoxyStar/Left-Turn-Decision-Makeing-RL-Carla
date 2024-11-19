import torch
import torch.nn as nn


class PolicyNetwork(nn.Module):
    def __init__(self, obs_space_dims: int, hidden_dims: int, action_space_dims: int, action_bound):
        """Initializes the policy network.

        Args:
            obs_space_dims: Dimension of the observation space
            hidden_dims: Dimension of the hidden layer 
            action_space_dims: Dimension of the action space
            action_bound: Bound of the action
        """
        super(PolicyNetwork, self).__init__()
        self.action_factor = torch.tensor(action_bound[0])
        self.action_bias = torch.tensor(action_bound[1])
        self.net = nn.Sequential(
            nn.Linear(obs_space_dims, hidden_dims), nn.ReLU(),
            nn.Linear(hidden_dims, hidden_dims), nn.ReLU(),
            nn.Linear(hidden_dims, action_space_dims)
        )

    def forward(self, x: torch.Tensor):
        x = self.net(x)
        x = torch.tanh(x)
        x = x * self.action_factor + self.action_bias
        return x


class ValueNetwork(nn.Module):
    def __init__(self, obs_space_dims: int, hidden_dims: int, action_space_dims: int):
        """Initializes the value network.

        Args:
            obs_space_dims: Dimension of the observation space
            hidden_dims: Dimension of the hidden layer
            action_space_dims: Dimension of the action space
        """
        super(ValueNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_space_dims + action_space_dims, hidden_dims), nn.ReLU(),
            nn.Linear(hidden_dims, hidden_dims), nn.ReLU(),
            nn.Linear(hidden_dims, hidden_dims), nn.ReLU(),
            nn.Linear(hidden_dims, 1)
        )

    def forward(self, x: torch.Tensor, a: torch.Tensor):
        cat = torch.cat([x, a], dim=1)
        x = self.net(cat)
        return x