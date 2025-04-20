# Left Turn Decision Making in Carla through Reinforcement Learning

## Intruduction

This repository provides **a solution for the problem of left-turn decision-making at intersections without traffic signals**. The ego vehicle needs to turn left while avoiding other cars. Additionally, we also **introduce an interactive environment within the CARLA simulator for reinforcement learning, named carla-env**.

The solution leverages deep reinforcement learning techniques, including PPO and DQN algorithms. We also adopt a curriculum learning approach to break the task into two phases: **the basic phase focuses on path-following using DQN, while the more advanced phase addresses left-turn decision-making with PPO**. In 100 random tests, we achieved a success rate of **81%**.

The observation space consists of two components: the planned future path and the semantic input in BEV (Bird's Eye View) space. The action space is discrete, with the following values: throttle {0.3, 0.5, 0.7}, steering {-0.4, -0.2, 0.0, 0.2, 0.4} and brake {0, 0.4, 0.8}.


## Network

<img src=demo/network.png title="Demo_1" width="600"/>


## Demo

<img src=demo/demo_1.gif title="Demo_1" width="250"/> <img src=demo/demo_2.gif title="Demo_2" width="250"/> <img src=demo/demo_3.gif title="Demo_3" width="250"/>


## Usage
Path Following Train:

`python path_following_with_dqn_train.py`

Path Following Test:

`python path_following_with_dqn_test.py`

Left-Turn Decision-Making Train:

`python turn_left_with_ppo_train.py --scene normal`

Left-Turn Decision-Making Test:

`python turn_left_with_ppo_test.py --scene normal`




