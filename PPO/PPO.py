import torch
import torch.nn as nn
import numpy as np

from PPO.net import Policy_Network, Value_Network


class PPO_Agent:
    def __init__(self, 
                 actor_lr: float = 1e-4,
                 value_lr: float = 1e-3, 
                 eps: float = 0.2):
        """Initializes the PPO algorithm.

        Args:
            obs_space_dims: Dimension of the observation space
            hidden_dims: Dimension of the hidden layer
            action_space_dims: Dimension of the action space
            actor_lr: Learning rate for the actor network, default is 1e-4
            value_lr: Learning rate for the value network, default is 1e-3
            eps: Clipping parameter, default is 0.2
        """
        # Policy and Value networks
        self.actor_net = Policy_Network().to('cuda')
        self.value_net = Value_Network().to('cuda')

        # Optimizers
        self.actor_optimizer = torch.optim.Adam(self.actor_net.parameters(), lr=actor_lr)
        self.value_optimizer = torch.optim.Adam(self.value_net.parameters(), lr=value_lr)

        self.eps = eps

    def get_action(self, state: np.ndarray, training=True) -> float:
        actor_prob = self.actor_net(state.to('cuda'))

        # Output action with maximum probability directly at testing stage
        if not training:
            return torch.argmax(actor_prob).item()
        
        # Sample an action from the probability distribution at training stage
        action = np.random.choice(np.array([0, 1, 2], dtype=np.int64), p=actor_prob.detach().squeeze(0).cpu().numpy())
        return action, actor_prob
    
    def convert_state(self, state):
        ego_vehicle_state = torch.FloatTensor(state['ego_state'])
        next_point = torch.FloatTensor(state['next_point'])
        ego_vehicle_state[:3] = next_point - ego_vehicle_state[:3]
        state_follow = ego_vehicle_state.unsqueeze(0)
        img = state['image']
        state_brake = torch.FloatTensor(img).unsqueeze(0).unsqueeze(0)
        return state_follow, state_brake

    def train(self, states, actions, probs, returns, training_steps=3):
        # Inputs
        states = torch.cat(states, dim=0).to('cuda')
        actions = torch.stack([torch.tensor(item) for item in actions]).unsqueeze(1).to('cuda')
        probs = torch.cat(probs, dim=0).to('cuda')
        returns = torch.stack([torch.tensor(item) for item in returns]).unsqueeze(1).to('cuda')
        old_prob = probs.gather(1, actions).detach()

        # Training
        for _ in range(training_steps):
            # Compute new probability
            new_probs = self.actor_net(states)
            new_prob = new_probs.gather(1, actions)

            # Compute the value and advantage
            V = self.value_net(states)
            A_k = returns - V.detach()

            # Compute the actor loss
            ratio = new_prob / old_prob
            surr1 = ratio * A_k
            surr2 = torch.clamp(ratio, 1 - self.eps, 1 + self.eps) * A_k
            actor_loss = (-torch.min(surr1, surr2)).mean()

            # Update the actor network
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Compute the value loss and update the value network
            value_loss = nn.MSELoss()(V, returns)
            self.value_optimizer.zero_grad()
            value_loss.backward()
            self.value_optimizer.step()