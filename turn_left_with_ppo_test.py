import torch
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
buffer_maxlen = int(1e4)

# initialize env and agent
env = CarlaEnv()
agent = QLearningAgent(state_space_dims, action_space_dims)
agent.q_net.load_state_dict(torch.load("weight/DQN_Path_Following_3250e.pth", weights_only=True))

brake_agent = PPO_Agent()
if args.scene == 'normal':
    brake_agent.actor_net.load_state_dict(torch.load('weight/PPO_Turn_Left_Actor_780e.pth', weights_only=True))
elif args.scene == 'difficult':
    pass
else:
    print('Please choose normal or difficult.')
    exit()


# test
success = 0

for i in range(10):
    state = env.reset(scene=args.scene)
    state_follow, state_brake = brake_agent.convert_state(state)

    done = False
    while not done:
        action_follow = agent.sample_action(state_follow, training=False)
        action_brake = brake_agent.get_action(state_brake, training=False)
        action = (action_follow[0], action_follow[1], action_brake)
        action = action_mapping(action)
        next_state, reward, terminated, truncated = env.step(action)
        next_state_follow, next_state_brake = brake_agent.convert_state(next_state)
        state_follow = next_state_follow
        state_brake = next_state_brake
        done = terminated or truncated
    
    if terminated:
        success += 1

print(f"Success Rate: {success / 10}")

env.close()