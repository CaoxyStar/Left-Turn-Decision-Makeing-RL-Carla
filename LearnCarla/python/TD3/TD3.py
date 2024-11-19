import torch
import numpy as np
import collections
import random

from .net import PolicyNetwork, ValueNetwork

class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buffer = collections.deque(maxlen=capacity)

    def __len__(self):
        return len(self.buffer)

    def append(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        transitions = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*transitions)
        return state, action, reward, next_state, done
    
    def convert_state(self, state):
        ego_vehicle_state = state['ego_state']
        next_point = state['next_point']
        ego_vehicle_state[:3] = next_point - ego_vehicle_state[:3]
        return ego_vehicle_state


class TD3_Agent:
    '''TD3 agent

    Args:
        obs_space_dims (int): observation space dimensions
        hidden_dims (int): hidden dimensions
        action_space_dims (int): action space dimensions
        action_bound (float): action bound
        actor_lr (float): actor learning rate
        critic_lr (float): critic learning rate
        gamma (float): discount factor
        tau (float): soft update rate
        sigma (float): noise factor
        delay (int): delay factor
    '''
    def __init__(self, 
                 obs_space_dims, 
                 hidden_dims, 
                 action_space_dims, 
                 action_bound, 
                 actor_lr = 1e-4, 
                 critic_lr = 1e-3,  
                 gamma = 0.99, 
                 tau = 0.005,
                 sigma = 0.1,
                 delay = 2):
        # Hyperparameters
        self.action_bound = action_bound
        self.action_space_dims = action_space_dims
        self.gamma = gamma
        self.tau = tau
        self.sigma = sigma
        self.delay = delay

        # Actor
        self.actor = PolicyNetwork(obs_space_dims, hidden_dims, action_space_dims, action_bound)
        self.actor_target = PolicyNetwork(obs_space_dims, hidden_dims, action_space_dims, action_bound)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)

        # Critics ( Improvement 1 based DDPG: Use two critics )
        self.critic_1 = ValueNetwork(obs_space_dims, hidden_dims, action_space_dims)
        self.critic_target_1 = ValueNetwork(obs_space_dims, hidden_dims, action_space_dims)
        self.critic_target_1.load_state_dict(self.critic_1.state_dict())
        self.critic_optimizer_1 = torch.optim.Adam(self.critic_1.parameters(), lr=critic_lr)

        self.critic_2 = ValueNetwork(obs_space_dims, hidden_dims, action_space_dims)
        self.critic_target_2 = ValueNetwork(obs_space_dims, hidden_dims, action_space_dims)
        self.critic_target_2.load_state_dict(self.critic_2.state_dict())
        self.critic_optimizer_2 = torch.optim.Adam(self.critic_2.parameters(), lr=critic_lr)

        # Initialize update counter for delay
        self.update_cnt = 0
    
    def get_action(self, state, training=True):
        state = torch.FloatTensor(state).unsqueeze(0)
        action = self.actor(state).detach()
        # Improvement 2 based DDPG: Add noise
        if training:
            action = action + self.sigma * np.random.randn(action.shape[0], action.shape[1])
        return action[0].tolist()
    
    def soft_update(self, net, target_net):
        for param_target, param in zip(target_net.parameters(), net.parameters()):
            param_target.data.copy_(param.data * self.tau + param_target.data * (1 - self.tau))

    def train(self, batch):
        # Inputs
        states, actions, rewards, next_states, dones = batch
        states = torch.FloatTensor(np.array(states))
        actions = torch.FloatTensor(np.array(actions))
        rewards = torch.FloatTensor(np.array(rewards)).view(-1, 1)
        next_states = torch.FloatTensor(np.array(next_states))
        dones = torch.FloatTensor(np.array(dones)).view(-1, 1)

        # Target Values
        next_actions = self.actor_target(next_states)
        Q_1 = self.critic_target_1(next_states, next_actions)
        Q_2 = self.critic_target_2(next_states, next_actions)
        target_Q = rewards + (1 - dones) * self.gamma * torch.min(Q_1, Q_2)
        
        # Update Critic
        current_Q_1 = self.critic_1(states, actions)
        critic_loss_1 = torch.mean(torch.nn.functional.mse_loss(current_Q_1, target_Q.detach()))
        self.critic_optimizer_1.zero_grad()
        critic_loss_1.backward()
        self.critic_optimizer_1.step()

        current_Q_2 = self.critic_2(states, actions)
        critic_loss_2 = torch.mean(torch.nn.functional.mse_loss(current_Q_2, target_Q.detach()))
        self.critic_optimizer_2.zero_grad()
        critic_loss_2.backward()
        self.critic_optimizer_2.step()

        # Update Counter ( Improvement 3 based DDPG: Add Delay )
        self.update_cnt += 1
        if self.update_cnt >= 1e4:
            self.update_cnt = 0
        if self.update_cnt % self.delay != 0:
            return

        # Update Actor
        actor_loss = -self.critic_1(states, self.actor(states)).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Update Target Networks
        self.soft_update(self.critic_1, self.critic_target_1)
        self.soft_update(self.critic_2, self.critic_target_2)
        self.soft_update(self.actor, self.actor_target)