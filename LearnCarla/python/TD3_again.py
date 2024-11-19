import torch
import gymnasium as gym
import numpy as np
import argparse
from torch.utils.tensorboard import SummaryWriter
import collections

from TD3.TD3 import TD3_Agent, ReplayBuffer
from carla_env import CarlaEnv


def train():
    # Environment
    env = CarlaEnv()

    # Hyperparameters
    obs_space_dims = 5
    action_space_dims = 3
    action_bound = [(0.5, 1, 0.0), (0.5, 0, 0.0)]
    hidden_size = 16
    actor_lr = 1e-4
    critic_lr = 1e-3

    # Agent
    agent = TD3_Agent(obs_space_dims, hidden_size, action_space_dims, action_bound, actor_lr, critic_lr)

    # Training
    buffer_length = int(1e4)
    batch_size = 64
    total_num_episodes = int(1e4)

    replay_buffer = ReplayBuffer(buffer_length)
    writer = SummaryWriter("runs/TD3_car")

    rewards_queue = collections.deque(maxlen=25)
    r_w = collections.deque(maxlen=25)
    r_v = collections.deque(maxlen=25)
    r_p = collections.deque(maxlen=25)
    r_a = collections.deque(maxlen=25)

    for episode in range(total_num_episodes):
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
            action = agent.get_action(state, training=True)
            print(action)
            next_state, reward_list, done = env.step(action)
            next_state = replay_buffer.convert_state(next_state)
            reward = sum(reward_list)
            replay_buffer.append(state, action, reward, next_state, done)
            state = next_state

            episode_reward += reward
            w_reward += reward_list[1]
            v_reward += reward_list[2]
            p_reward += reward_list[3]
            a_reward += reward_list[4]
            t += 1

            if len(replay_buffer) > batch_size:
                batch = replay_buffer.sample(batch_size)
                agent.train(batch)

        rewards_queue.append(episode_reward)
        r_w.append(w_reward / t)
        r_v.append(v_reward / t)
        r_p.append(p_reward / t)
        r_a.append(a_reward / t)

        writer.add_scalar("modified_rewards/sum_reward", np.mean(rewards_queue), episode + 1)
        writer.add_scalar("modified_rewards/waypoint_reward", np.mean(r_w), episode + 1)
        writer.add_scalar("modified_rewards/velocity_reward", np.mean(r_v), episode + 1)
        writer.add_scalar("modified_rewards/position_reward", np.mean(r_p), episode + 1)
        writer.add_scalar("modified_rewards/angle_reward", np.mean(r_a), episode + 1)
        
        if (episode + 1) % 10 == 0:
            torch.save(agent.actor.state_dict(), f"TD3_car_{episode+1}_episode.pth")
    
    env.close()
    writer.close()
    

def test():
    env = gym.make("InvertedPendulum-v4", render_mode="human")

    obs_space_dims = env.observation_space.shape[0]
    action_space_dims = env.action_space.shape[0]
    action_bound = env.action_space.high[0]
    hidden_size = 32

    agent = TD3_Agent(obs_space_dims, hidden_size, action_space_dims, action_bound)
    agent.actor.load_state_dict(torch.load("weights/TD3_Inverted_Pendulum.pth", weights_only=True))

    state, info = env.reset(seed=seed)
    for _ in range(1000):
        action = agent.get_action(state, training=False)
        state, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            state, info = env.reset(seed=seed)
    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--test", action="store_true")
    args = parser.parse_args()

    if args.train:
        train()
    elif args.test:
        test()
    else:
        print("Please specify --train or --test")