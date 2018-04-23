import gym
import time
env = gym.make('SpaceInvaders-v0')
env.reset()

for i in range(1000):
    env.step(env.action_space.sample())
    env.render()
    time.sleep(0.05)
