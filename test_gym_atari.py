import gym
import time
env = gym.make('BreakoutDeterministic-v4')
env.reset()

for i in range(10000):
    observe, reward, done, info = env.step(env.action_space.sample())
    env.render()
    time.sleep(0.05)
    if done is True:
        print("The episode is done!")
        observe = env.reset()
