from carla_env import CarlaEnv
from DQN import QLearningAgent, ReplayBuffer

import torch

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

agent.q_net.load_state_dict(torch.load("weight/DQN_car_3250.pth", weights_only=True))
buffer = ReplayBuffer(buffer_maxlen)


# test
state = env.reset()
state = buffer.convert_state(state)

done = False
while not done:
    action = agent.sample_action(state, training=False)
    map_action = action_mapping(action)
    print(map_action)
    next_state, reward, done = env.step(map_action)
    next_state = buffer.convert_state(next_state)
    state = next_state
env.close()