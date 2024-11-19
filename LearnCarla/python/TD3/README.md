# TD3

This is the implementation of TD3, which is a improved version based on DDPG.

There are three main improvements:

- Use two target cirtic networks to fix the overestimate problem
- Add noise to actions for exploration at training stage
- Delay the update speed of actor