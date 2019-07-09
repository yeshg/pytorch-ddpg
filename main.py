import argparse
import math
from collections import namedtuple
from itertools import count
import time


import numpy as np
#from gym import wrappers

import torch

from rl_algos.utils import OUNoise, AdaptiveParamNoiseSpec, NormalizedActions, Logger
from rl_algos.replay_buffer import ReplayBuffer
from rl_algos.algos import DDPG, TD3

import gym



#parser = argparse.ArgumentParser(description='PyTorch DDPG example')
parser = argparse.ArgumentParser()

# General args
parser.add_argument("--algo_name", default="TD3")
parser.add_argument('--env-name', default="Humanoid-v2",
                    help='name of the environment to run (default: Cassie-v0)')
parser.add_argument('--seed', type=int, default=0,
                    help='random seed (default: 0)')
parser.add_argument('--num_episodes', type=int, default=1000, metavar='N',
                    help='number of episodes before stoppping training (default: 1000)')

# Noise / early exploration args / hyperparameters
parser.add_argument('--exploration_end', type=int, default=100, metavar='N',
                    help='number of episodes with randomly sampled actions (default: 100)')
parser.add_argument('--ou_noise', type=bool, default=True)
parser.add_argument('--param_noise', type=bool, default=False)
parser.add_argument('--noise_scale', type=float, default=0.3, metavar='G',
                    help='initial noise scale (default: 0.3)')
parser.add_argument('--final_noise_scale', type=float, default=0.3, metavar='G',
                    help='final noise scale (default: 0.3)')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor for reward (default: 0.99)')
parser.add_argument('--tau', type=float, default=0.001, metavar='G',
                    help='discount factor for model (default: 0.001)')
parser.add_argument('--batch_size', type=int, default=128, metavar='N',
                    help='batch size (default: 128)')
parser.add_argument('--num_steps', type=int, default=1000, metavar='N',
                    help='max episode length (default: 1000)')
parser.add_argument('--hidden_size', type=int, default=256, metavar='N',
                    help='number of episodes (default: 256)')
parser.add_argument('--updates_per_step', type=int, default=5, metavar='N',
                    help='model updates per simulator step (default: 5)')
parser.add_argument('--replay_size', type=int, default=1000000, metavar='N',
                    help='size of replay buffer (default: 1000000)')
parser.add_argument("--logdir", type=str, default="/tmp/rl/experiments/",
                    help="Where to log diagnostics to")
parser.add_argument("--name", type=str, default="model")

# Unsorted args for TD3
parser.add_argument("--act_noise", default=0.2, type=float)		# Noise added to target policy during critic update
parser.add_argument("--noise_clip", default=0.5, type=float)		# Range to clip target policy noise
parser.add_argument("--policy_freq", default=2, type=int)	

args = parser.parse_args()


"""
Print out settings of this run
"""
file_name = "{}_{}_{}".format(args.algo_name, args.env_name, str(args.seed))
print("---------------------------------------")
print("Settings: {}".format(file_name))
print("---------------------------------------")


"""
Create logger
"""
logger = Logger(args, viz=True)


"""
Create environment
"""
if(args.env_name not in ["Cassie-v0", "Cassie-mimic-v0"]):
    env = NormalizedActions(gym.make(args.env_name))
else:
    # set up cassie environment
    import gym_cassie
    env = gym.make(args.env_name)


"""
Set seeds
"""
env.seed(args.seed)
torch.manual_seed(args.seed)
np.random.seed(args.seed)


"""
Initialize Replay Buffer
"""
memory = ReplayBuffer(args.replay_size)

"""
Action noise and parameter noise for exploration
"""
ounoise = OUNoise(env.action_space.shape[0]) if args.ou_noise else None
param_noise = AdaptiveParamNoiseSpec(initial_stddev=0.05, desired_action_stddev=args.noise_scale, adaptation_coefficient=1.05) if args.param_noise else None


"""
Create agent and start train
"""
if args.algo_name == "DDPG":
    agent = DDPG(args.gamma, args.tau, args.hidden_size, env.observation_space.shape[0], env.action_space,float(env.action_space.high[0]))
    agent.train(env, memory, args.num_episodes, ounoise, param_noise, args, logger=logger)

elif args.algo_name == "TD3":
    agent = TD3(args.gamma, args.tau, args.hidden_size, env.observation_space.shape[0], env.action_space, float(env.action_space.high[0]))
    agent.train(env, memory, args.num_episodes, ounoise, param_noise, args.act_noise, args.noise_clip, args.policy_freq, args, logger=logger)
# elif args.algo_name == "D4PG": #TBD
# elif args.algo_name == "D4PG_TD3": #TBD


"""
Start training loop
"""
    
env.close()