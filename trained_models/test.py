# TODO: organize this file
from rl_algos.utils import NormalizedActions
from rl_algos.algos import DDPG, TD3
import numpy as np
import argparse
import pickle
import torch

import time

import gym

#from cassie import CassieEnv
#from rl.envs import Normalize, Vectorize
#from rl.policies import GaussianMLP, EnsemblePolicy

import matplotlib
from matplotlib import pyplot as plt
plt.style.use('ggplot')

np.set_printoptions(precision=2, suppress=True)


# TODO: add .dt to all environments. OpenAI should do the same...
def visualize(env, agent, vlen, viz_target, dt=0.033, speedup=1):
    done = False
    R = []
    episode_reward = 0
    state = torch.Tensor([env.reset()])
    t = 0

    with torch.no_grad():

        while True:
            t += 1
            start = time.time()
            action = agent.select_action(np.array(state))
            #print(action)
            #print("policy time: ", time.time() - start)

            #start = time.time()
            next_state, reward, done, _ = env.step(action)
            #print("env time: ", time.time() - start)

            episode_reward += reward

            next_state = torch.Tensor([next_state])

            state = next_state
            if done:
                print(episode_reward)
                state = env.reset()
                R += [episode_reward]
                episode_reward = 0

            state = torch.Tensor(state)

            env.render()

            time.sleep(dt / speedup)

        if not done:
            R += [episode_reward]

        

        print("avg reward:", sum(R)/len(R))
        print("avg timesteps:", vlen / len(R))


parser = argparse.ArgumentParser(
    description="Run a model, including visualization and plotting.")
parser.add_argument("-p", "--model_path", type=str, default="./trained_models/td3",
                    help="File path for model to test")
parser.add_argument("-x", "--no-visualize", dest="visualize", default=True, action='store_false',
                    help="Don't render the policy.")
parser.add_argument("--viz-target", default=False, action='store_true',
                    help="Length of trajectory to visualize")
# parser.add_argument("-g", "--graph", dest="plot", default=False, action='store_true',
#                     help="Graph the output of the policy.")
# parser.add_argument("--glen", type=int, default=150,
#                     help="Length of trajectory to graph.")
parser.add_argument("--vlen", type=int, default=75,
                    help="Length of trajectory to visualize")
parser.add_argument("--noise", default=False, action="store_true",
                    help="Visualize policy with exploration.")
parser.add_argument('--env-name', default="Humanoid-v2",
                    help='name of the environment to run')

parser.add_argument('--algo_name', default="TD3",
                    help='name of the algo model to load')
args = parser.parse_args()

if(args.env_name not in ["Cassie-v0", "Cassie-mimic-v0"]):
    env = gym.make(args.env_name)
    env = NormalizedActions(env)
else:
    # set up cassie environment
    import gym_cassie
    env = NormalizedActions(gym.make(args.env_name))

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
max_action = float(env.action_space.high[0])


if args.algo_name == "DDPG":
    agent = DDPG(state_dim, action_dim, max_action)
    agent.load("./trained_models/DDPG")

elif args.algo_name == "TD3":
    agent = TD3(state_dim, action_dim, max_action)
    agent.load("./trained_models/TD3")
# elif args.algo_name == "D4PG": #TBD
# elif args.algo_name == "D4PG_TD3": #TBD


# work on making it such that you don't need to specify all of this stuff that won't be used (cause we are only testing)
# agent = DDPG(gamma=0.99, tau=0.001, hidden_size=256,
#                num_inputs=env.observation_space.shape[0], action_space=env.action_space, max_action=float(env.action_space.high[0]))


if(not args.visualize):
    visualize(env, agent, args.vlen, args.viz_target)
