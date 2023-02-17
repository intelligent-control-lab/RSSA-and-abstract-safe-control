import gym
import time
import cv2
import numpy as np

env = gym.make('Walker2d-v2')
env.reset()

for i in range(1000):
    # rgb = env.render(mode='rgb_array')
    rgb = env.render(mode='human')
    env.step(np.random.rand())
    # cv2.imwrite(f'./src/pybullet-dynamics/walker_env/imgs/test/{i}.png', rgb)
    time.sleep(0.1)