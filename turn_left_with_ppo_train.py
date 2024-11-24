import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import collections
import argparse

from carla_env import CarlaEnv
from DQN.DQN import QLearningAgent
from PPO.PPO import PPO_Agent


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
        brake = 0.4
    elif action[2] == 2:
        brake = 0.8
    else:
        print("brake error!")

    return throttle, steering, brake


# parser
parser = argparse.ArgumentParser()
parser.add_argument('--scene', type=str, default='normal')
args = parser.parse_args()

# parameters
state_space_dims = 5
action_space_dims = (3, 5, 3)

# initialize env and agent
env = CarlaEnv()

agent = QLearningAgent(state_space_dims, action_space_dims)
agent.q_net.load_state_dict(torch.load("weight/DQN_car_3250.pth", weights_only=True))
print('Load the parameters of path following agent successfully!')

brake_agent = PPO_Agent()


# train
writer = SummaryWriter("runs/PPO_Turn_Left")

episode_num = int(1e3)
queue = collections.deque(maxlen=25)

eps_high = 0.2
eps_low = 0.01

for episode in range(episode_num):
    state = env.reset(scene=args.scene)
    state_follow, state_brake = brake_agent.convert_state(state)
    
    episode_reward = 0
    done = False
    states = []
    actions = []
    probs = []
    rewards = []

    while not done:
        action_follow = agent.sample_action(state_follow, training=False)
        action_brake, brake_prob = brake_agent.get_action(state_brake, training=True)
        action = (action_follow[0], action_follow[1], action_brake)
        action = action_mapping(action)

        next_state, reward, terminated, truncated = env.step(action)
        next_state_follow, next_state_brake = brake_agent.convert_state(next_state)

        states.append(state_brake)
        actions.append(action_brake)
        probs.append(brake_prob)
        rewards.append(reward)
        
        state_follow = next_state_follow
        state_brake = next_state_brake
        done = terminated or truncated
        episode_reward += reward
    
    # Calculate cumulative rewards
    g = 0
    gamma = 0.99
    returns = []

    for reward in reversed(rewards):
        g = reward + gamma * g
        returns.insert(0, g)

    # Updata
    brake_agent.train(states, actions, probs, returns)

    # Update epsilon
    brake_agent.eps = eps_high - (eps_high - eps_low) / episode_num * episode
    
    queue.append(episode_reward)
    writer.add_scalar("ppo_rewards/step_reward", episode_reward, episode + 1)
    writer.add_scalar("ppo_rewards/avg_reward", np.mean(queue), episode + 1)
    
    if (episode + 1) % 10 == 0:
        torch.save(brake_agent.actor_net.state_dict(), f"PPO_Actor_{episode+1}.pth")
        torch.save(brake_agent.value_net.state_dict(), f"PPO_Value_{episode+1}.pth")

writer.close()
env.close()