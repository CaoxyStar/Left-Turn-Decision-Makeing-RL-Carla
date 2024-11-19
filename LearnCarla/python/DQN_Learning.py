from carla_env import CarlaEnv
from DQN import QLearningAgent, ReplayBuffer

import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import collections

# action mapping
def action_mapping(action):
    throttle, steering, brake = 0, 0, 0

    # throttle mapping
    if action[0] == 0:
        throttle = 0.3
    elif action[0] == 1:
        throttle = 0.5
    elif action[0] == 2:
        throttle = 0.7
    else:
        print("throttle error!")
    
    # steering mapping
    if action[1] == 0:
        steering = -0.4
    elif action[1] == 1:
        steering = -0.2
    elif action[1] == 2:
        steering = 0
    elif action[1] == 3:
        steering = 0.2
    elif action[1] == 4:
        steering = 0.4
    else:
        print("steering error!")

    # brake mapping
    if action[2] == 0:
        brake = 0
    elif action[2] == 1:
        brake = 0.0
    elif action[2] == 2:
        brake = 0.0
    else:
        print("brake error!")

    return throttle, steering, brake

# parameters
state_space_dims = 5
action_space_dims = (3, 5, 3)
buffer_maxlen = int(1e4)

# initialize env and agent
env = CarlaEnv()
agent = QLearningAgent(state_space_dims, action_space_dims)

# agent.q_net.load_state_dict(torch.load("weight/DQN_car_700_turn.pth", weights_only=True))
# agent.target_q_net.load_state_dict(agent.q_net.state_dict())
print('************************')
print('Load parameters successfully!')
print('************************')

replay_buffer = ReplayBuffer(buffer_maxlen)
writer = SummaryWriter("runs/DQN_car")

# train
episode_num = int(1e5)
batch_size = 128


for episode in range(episode_num):
    state = env.reset()
    state = replay_buffer.convert_state(state)
    episode_reward = 0
    w_reward = 0
    v_reward = 0
    p_reward = 0
    a_reward = 0

    t = 0

    done = False
    while not done:
        action = agent.sample_action(state, training=True)
        map_action = action_mapping(action)
        next_state, reward_list, done = env.step(map_action)
        next_state = replay_buffer.convert_state(next_state)
        reward = sum(reward_list)
        replay_buffer.append(state, action, reward, next_state, done)
        state = next_state

        if len(replay_buffer) > batch_size:
            batch = replay_buffer.sample(batch_size)
            agent.train(batch)

        episode_reward += reward
        w_reward += reward_list[1]
        v_reward += reward_list[2]
        p_reward += reward_list[3]
        a_reward += reward_list[4]
        t += 1
    
    w_reward = w_reward / t
    v_reward = v_reward / t
    p_reward = p_reward / t
    a_reward = a_reward / t

    
    if (episode + 1) % 50 == 0:
        agent.update_target_network()
        
    writer.add_scalar("modified_rewards/sum_reward", episode_reward, episode + 1)
    writer.add_scalar("modified_rewards/waypoint_reward", w_reward, episode + 1)
    writer.add_scalar("modified_rewards/velocity_reward", v_reward, episode + 1)
    writer.add_scalar("modified_rewards/position_reward", p_reward, episode + 1)
    writer.add_scalar("modified_rewards/angle_reward", a_reward, episode + 1)
    
    if (episode + 1) % 10 == 0:
        torch.save(agent.q_net.state_dict(), f"DQN_car_{episode+1}.pth")

writer.close()
env.close()