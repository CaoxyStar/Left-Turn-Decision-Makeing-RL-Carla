import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import collections
import random


class PolicyNetwork(nn.Module):
    def __init__(self, obs_space_dims: int, hidden_dims: int, action_space_dims: int):
        """Initializes the policy network.

        Args:
            obs_space_dims: Dimension of the observation space
            hidden_dims: Dimension of the hidden layer 
            action_space_dims: Dimension of the action space
        """
        super(PolicyNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_space_dims, hidden_dims), nn.ReLU(),
            nn.Linear(hidden_dims, hidden_dims), nn.ReLU(),
            nn.Linear(hidden_dims, action_space_dims)
        )

    def forward(self, x: torch.Tensor):
        x = self.net(x.float())
        x = torch.tanh(x)
        x = (x + torch.tensor((1.0, 0))) / torch.tensor((2.0, 1.0))
        # print(x)
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
            nn.Linear(hidden_dims, hidden_dims * 2), nn.ReLU(),
            nn.Linear(hidden_dims * 2, hidden_dims), nn.ReLU(),
            nn.Linear(hidden_dims, 1)
        )

    def forward(self, x: torch.Tensor, a: torch.Tensor):
        cat = torch.cat([x, a], dim=1)
        x = self.net(cat)
        return x


class ReplayBuffer:
    def __init__(self, capacity: int):
        """Initializes the replay buffer.

        Args:
            capacity: Capacity of the replay buffer
        """
        self.buffer = collections.deque(maxlen=capacity)

    def __len__(self):
        return len(self.buffer)

    def append(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        transitions = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*transitions)
        return state, np.array(action), np.array(reward), next_state, np.array(done)
    
    def convert_state(self, state):
        ego_vehicle_state = torch.FloatTensor(state['ego_state'])
        next_point = torch.FloatTensor(state['next_point'])
        state = torch.cat((ego_vehicle_state, next_point)).unsqueeze(0)
        return state


class DDPG_Agent:
    def __init__(self, obs_space_dims, hidden_dims, action_space_dims, actor_lr, critic_lr, sigma, gamma, tau, device, delay=2):
        """Initializes the DDPG algorithm.

        Args:
            obs_space_dims: Dimension of the observation space
            hidden_dims: Dimension of the hidden layer
            action_space_dims: Dimension of the action space
            actor_lr: Learning rate of the actor network
            critic_lr: Learning rate of the critic network
            sigma: Standard deviation of the noise
            gamma: Discount factor
            tau: Soft update parameter
            device: Device to run the algorithm
            delay: Delay of the actor network
        """
        self.actor = PolicyNetwork(obs_space_dims, hidden_dims, action_space_dims)
        self.actor_target = PolicyNetwork(obs_space_dims, hidden_dims, action_space_dims)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)

        self.critic_1 = ValueNetwork(obs_space_dims, hidden_dims, action_space_dims)
        self.critic_target_1 = ValueNetwork(obs_space_dims, hidden_dims, action_space_dims)
        self.critic_target_1.load_state_dict(self.critic_1.state_dict())
        self.critic_optimizer_1 = torch.optim.Adam(self.critic_1.parameters(), lr=critic_lr)

        self.critic_2 = ValueNetwork(obs_space_dims, hidden_dims, action_space_dims)
        self.critic_target_2 = ValueNetwork(obs_space_dims, hidden_dims, action_space_dims)
        self.critic_target_2.load_state_dict(self.critic_2.state_dict())
        self.critic_optimizer_2 = torch.optim.Adam(self.critic_2.parameters(), lr=critic_lr)

        self.obs_space_dims = obs_space_dims
        self.action_space_dims = action_space_dims
        self.sigma = sigma
        self.gamma = gamma
        self.tau = tau
        self.device = device

        self.delay = delay
        self.update_times = 0
    
    def sample_action(self, state, training=True):
        action = self.actor(state).detach()
        if training:
            action = action + self.sigma * np.random.randn(action.shape[0], action.shape[1])
        return action[0].tolist()
    
    def soft_update(self, net, target_net):
        for param_target, param in zip(target_net.parameters(), net.parameters()):
            param_target.data.copy_(param.data * self.tau + param_target.data * (1 - self.tau))
    
    def train(self, states, actions, rewards, next_states, dones):
        states = torch.cat(states, dim=0)
        actions = torch.FloatTensor(actions).view(-1, self.action_space_dims)
        rewards = torch.FloatTensor(rewards).view(-1, 1)
        next_states = torch.cat(next_states, dim=0)
        dones = torch.FloatTensor(dones).view(-1, 1)

        next_actions = self.actor_target(next_states)
        Q_1 = self.critic_target_1(next_states, next_actions)
        Q_2 = self.critic_target_2(next_states, next_actions)
        target_Q = rewards + (1 - dones) * self.gamma * torch.min(Q_1, Q_2)
        
        current_Q_1 = self.critic_1(states, actions)
        critic_loss_1 = torch.mean(F.mse_loss(current_Q_1, target_Q.detach()))
        self.critic_optimizer_1.zero_grad()
        critic_loss_1.backward()
        self.critic_optimizer_1.step()

        current_Q_2 = self.critic_2(states, actions)
        critic_loss_2 = torch.mean(F.mse_loss(current_Q_2, target_Q.detach()))
        self.critic_optimizer_2.zero_grad()
        critic_loss_2.backward()
        self.critic_optimizer_2.step()

        if self.update_times % self.delay != 0:
            return

        actor_loss = -self.critic_1(states, self.actor(states)).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.soft_update(self.critic_1, self.critic_target_1)
        self.soft_update(self.critic_2, self.critic_target_2)
        self.soft_update(self.actor, self.actor_target)
