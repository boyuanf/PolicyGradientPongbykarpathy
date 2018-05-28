import gym
import time
env = gym.make('BreakoutDeterministic-v4')
env.reset()

start_life = 5

for i in range(10000):
    observe, reward, done, info = env.step(env.action_space.sample())
    print(reward)
    env.render()
    time.sleep(0.05)
    if start_life > info['ale.lives']:
        dead = True
        start_life = info['ale.lives']
    if done is True:
        print("The episode is done!")
        observe = env.reset()
        start_life = 5
