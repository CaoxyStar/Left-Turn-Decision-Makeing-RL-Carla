from carla_env import CarlaEnv
from DDPG import DDPG_Agent, ReplayBuffer

import torch
import numpy as np
import collections

# parameters
state_space_dims = 8
action_space_dims = 2
hidden_dims = 16

actor_lr = 1e-4
critic_lr = 1e-3

sigma = 0.1
gamma = 0.99
tau = 0.005
device = 'cuda'

buffer_maxlen = int(1e4)

# initialize env and agent
env = CarlaEnv()
agent = DDPG_Agent(state_space_dims, hidden_dims, action_space_dims, actor_lr, critic_lr, sigma, gamma, tau, device)
replay_buffer = ReplayBuffer(buffer_maxlen)

# train
batch_size = 128
episode_num = int(1e5)

rewards_queue = collections.deque(maxlen=50)

for episode in range(episode_num):
    state = env.reset()
    state = replay_buffer.convert_state(state)
    episode_reward = 0

    done = False
    while not done:
        action = agent.sample_action(state)
        next_state, reward, done = env.step(action)
        next_state = replay_buffer.convert_state(next_state)
        replay_buffer.append(state, action, reward, next_state, done)
        state = next_state
        episode_reward += reward
    
    rewards_queue.append(episode_reward)

    if len(replay_buffer) > batch_size:
        for i in range(3):
            states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)
            agent.train(states, actions, rewards, next_states, dones)
    
    if (episode + 1) % 50 == 0:
        print(f"Episode: {episode + 1}, Avg reward: {np.mean(rewards_queue)}")
    
    if (episode + 1) % 100 == 0:
        torch.save(agent.actor_target.state_dict(), f"CarLearning_TD3_actor_{episode+1}.pth")
