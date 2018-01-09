import os
import time

import gym

from network import FullConnectionNetwork

# Load Policy Network
ENV_NAME = "CartPole-v0"

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SAVED_PATH = os.path.join(CURRENT_DIR, 'pre_trained', ENV_NAME)

policy_network = FullConnectionNetwork.load(SAVED_PATH)

# Play the game
env = gym.make(ENV_NAME)
env.seed(2)
observation = env.reset()
total_reward = 0
while True:
    action = policy_network.make_decision(observation)
    observation, reward, done, _ = env.step(action)
    env.render()
    time.sleep(0.01)
    total_reward += reward
    if done:
        break

print("Total reward: %d" % total_reward)
