"""
Solve Continuous observation discrete action problem in gym
"""
# TODO add continuous action support
# TODO implement neural network baseline

import argparse
import os
import time

import gym
import torch
from torch.autograd import Variable

from loss import PolicyGradientLoss
from network import FullConnectionNetwork

parser = argparse.ArgumentParser()
parser.add_argument('--env_name', type=str, dest="env_name")
parser.add_argument('--render', action='store_true', dest='render', help="Whether render the gym environment or not.")
parser.add_argument('--gamma', type=float, default=1.0, help="Discount factor")
parser.add_argument('--n_iter', type=int, default=1000, dest='n_iter', help="Number of iteration")
parser.add_argument('--batch_size', type=int, default=1000, dest="batch_size", help="min_timesteps_per_batch")
parser.add_argument('--episode_max_len', type=float, default=-1., dest="episode_max_len",
                    help="max path length in one episode")
parser.add_argument('--learning_rate', '-lr', type=float, default=5e-3, help="Learning rate for AdamOptimizer")
parser.add_argument('--reward_to_go', action='store_true')
parser.add_argument('--normalize_advantages', action='store_true')
parser.add_argument('--nn_baseline', '-bl', action='store_true')
parser.add_argument('--n_layers', type=int, default=1, dest='n_layers', help="Hidden Layer num for Policy Network")
parser.add_argument('--hidden_size', type=int, default=32, dest='hidden_size',
                    help="Specify how many units in each hidden layer")
args = parser.parse_args()

ENV_NAME = args.env_name
MAX_PATH_LENGTH = args.episode_max_len if args.episode_max_len > 0 else None
N_ITER = args.n_iter
N_LAYERS = args.n_layers
HIDDEN_SIZE = args.hidden_size
RENDER = args.render
BATCH_SIZE = args.batch_size
GAMMA = args.gamma

env = gym.make(ENV_NAME)

# Is this evn observation continuous, or discrete
OBSERVATION_DISCRETE = isinstance(env.observation_space, gym.spaces.Discrete)
print("discrete env observation: ", OBSERVATION_DISCRETE)

# Is this env action continuous, or discrete?
ACTION_DISCRETE = isinstance(env.action_space, gym.spaces.Discrete)
print("discrete env action: ", ACTION_DISCRETE)

# Maximum length for episodes
MAX_PATH_LENGTH = MAX_PATH_LENGTH or env.spec.max_episode_steps
print("max path length: ", MAX_PATH_LENGTH)

# Observation and action sizes
OBSERVATION_DIM = env.observation_space.shape[0]
ACTION_DIM = env.action_space.n if ACTION_DISCRETE else env.action_space.shape[0]

policy_network = FullConnectionNetwork(OBSERVATION_DIM, ACTION_DIM, [HIDDEN_SIZE for i in range(N_LAYERS)])
loss = PolicyGradientLoss(gamma=GAMMA)
optimizer = torch.optim.Adam(policy_network.parameters())

total_time_steps = 0

for itr in range(N_ITER):
    print("********** Iteration %i ************" % itr)
    # Collect paths until we have enough timesteps
    time_steps_this_batch = 0
    paths = []

    # Sample training data(paths) for update network parameters
    while True:
        observation = env.reset()
        observations, actions, rewards = [], [], []
        animate_this_episode = (len(paths) == 0 and (itr % 10 == 0) and RENDER)
        steps = 0
        while True:
            if animate_this_episode:
                env.render()
                time.sleep(0.05)
            observations.append(observation)
            action = policy_network.make_decision(observation)
            actions.append(action)
            observation, reward, done, _ = env.step(action)
            rewards.append(reward)
            steps += 1
            if done or steps > MAX_PATH_LENGTH:
                break
        path = {"observation": observations,
                "reward": rewards,
                "action": actions}
        paths.append(path)
        time_steps_this_batch += len(path["reward"])
        if time_steps_this_batch > BATCH_SIZE:
            break
    total_time_steps += time_steps_this_batch

    batch_observations = Variable(torch.cat([torch.FloatTensor(path['observation']) for path in paths]),
                                  requires_grad=True)
    batch_actions = [torch.LongTensor(path['action']) for path in paths]
    batch_rewards = [torch.FloatTensor(path['reward']) for path in paths]

    batch_outputs = policy_network(batch_observations)
    loss.reset()
    loss.eval_batch(outputs=batch_outputs, actions=batch_actions, rewards=batch_rewards)
    loss.backward()
    optimizer.step()

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SAVED_PATH = os.path.join(CURRENT_DIR, 'pre_trained', ENV_NAME)
policy_network.save(SAVED_PATH)
