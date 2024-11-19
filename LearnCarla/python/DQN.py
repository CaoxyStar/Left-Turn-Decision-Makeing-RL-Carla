import gymnasium as gym
import random
import torch
import torch.nn as nn
import numpy as np
import collections


class QNetwork(nn.Module):
    def __init__(self, obs_space_dims: int, action_space_dims: int):
        """Initializes the Q-network.

        Args:
            obs_space_dims: Dimension of the observation space
            action_space_dims: Dimension of the action space
        """
        super(QNetwork, self).__init__()

        # Create the shared network
        self.shared_net = nn.Sequential(
            nn.Linear(obs_space_dims, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU()
        )
        self.throttle_head = nn.Linear(16, action_space_dims[0])
        self.steering_head = nn.Linear(16, action_space_dims[1])
        self.brake_head = nn.Linear(16, action_space_dims[2])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns the Q-values of the Q-network.

        Args:
            x: Observation

        Returns:
            Q-values
        """
        hidden_state = self.shared_net(x.float())
        throttle_value = self.throttle_head(hidden_state)
        steering_value = self.steering_head(hidden_state)
        brake_value = self.brake_head(hidden_state)

        return throttle_value, steering_value, brake_value


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
        ego_vehicle_state[:3] = next_point - ego_vehicle_state[:3]
        state = ego_vehicle_state.unsqueeze(0)
        return state


class QLearningAgent:
    def __init__(self, obs_space_dims, action_space_dims):
        """Initializes the Q-learning agent.

        Args:
            obs_space_dims: Dimension of the observation space
            action_space_dims: Dimension of the action space
        """
        self.obs_space_dims = obs_space_dims
        self.action_space_dims = action_space_dims

        self.q_net = QNetwork(obs_space_dims, action_space_dims)
        self.target_q_net = QNetwork(obs_space_dims, action_space_dims)
        self.target_q_net.load_state_dict(self.q_net.state_dict())
        # self.q_net.to('cuda')
        # self.target_q_net.to('cuda')
        self.q_net.train()
        self.target_q_net.eval()

        lr = 5e-5
        self.gamma = 0.99
        self.epsilon = 0.2

        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()
    
    def update_target_network(self):
        self.target_q_net.load_state_dict(self.q_net.state_dict())

    def sample_action(self, state, training=True) -> int:
        if training and random.random() < self.epsilon:
            throttle = random.randint(0, self.action_space_dims[0] - 1)
            steering = random.randint(0, self.action_space_dims[1] - 1)
            brake = random.randint(0, self.action_space_dims[2] - 1)
            return throttle, steering, brake
        else:
            throttle_values, steering_values, brake_values = self.q_net(state)
            throttle = torch.argmax(throttle_values).item()
            steering = torch.argmax(steering_values).item()
            brake = torch.argmax(brake_values).item()
            return throttle, steering, brake
    
    def train(self, batch):
        states, actions, rewards, next_states, dones = batch
        states = torch.cat(states, dim=0)
        actions = torch.tensor(actions, dtype=torch.int64).view(-1, 3)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.cat(next_states, dim=0)
        dones = torch.FloatTensor(dones)

        throttle_values, steering_values, brake_values = self.q_net(states)
        value_t = throttle_values.gather(1, actions[:, 0].unsqueeze(-1)).squeeze(-1)
        value_s = steering_values.gather(1, actions[:, 1].unsqueeze(-1)).squeeze(-1)
        value_b = brake_values.gather(1, actions[:, 2].unsqueeze(-1)).squeeze(-1)
        q_values = value_t + value_s + value_b

        next_throttle_values, next_steering_values, next_brake_values = self.target_q_net(next_states)
        next_value_t = next_throttle_values.max(1)[0]
        next_value_s = next_steering_values.max(1)[0]
        next_value_b = next_brake_values.max(1)[0]
        next_q_values = next_value_t + next_value_s + next_value_b

        target_q_values = rewards + self.gamma * next_q_values * (1 - dones.float())
        loss = self.loss_fn(q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()