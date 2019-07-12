import numpy as np
import torch

import argparse
import os

from rl_algos.replay_buffer import ReplayBuffer
from rl_algos.algos import TD3, DDPG
from rl_algos.utils import VisdomLinePlotter, NormalizedActions, AdaptiveParamNoiseSpec, distance_metric

import gym

def make_cassie_env(*args, **kwargs):
    def _thunk():
        return CassieEnv(*args, **kwargs)
    return _thunk

def gym_factory(path, **kwargs):
    from functools import partial

    """
    This is (mostly) equivalent to gym.make(), but it returns an *uninstantiated* 
    environment constructor.

    Since environments containing cpointers (e.g. Mujoco envs) can't be serialized, 
    this allows us to pass their constructors to Ray remote functions instead 
    (since the gym registry isn't shared across ray subprocesses we can't simply 
    pass gym.make() either)

    Note: env.unwrapped.spec is never set, if that matters for some reason.
    """
    spec = gym.envs.registry.spec(path)
    _kwargs = spec._kwargs.copy()
    _kwargs.update(kwargs)
    
    if callable(spec._entry_point):
        cls = spec._entry_point(**_kwargs)
    else:
        cls = gym.envs.registration.load(spec._entry_point)

    return partial(cls, **_kwargs)

# Runs policy for X episodes and returns average reward. Optionally render policy
def evaluate_policy(env, policy, eval_episodes=1):
    avg_reward = 0.
    for _ in range(eval_episodes):
        obs = env.reset()
        done = False
        while not done:
            action = policy.select_action(np.array(obs), param_noise=None)
            obs, reward, done, _ = env.step(action)
            avg_reward += reward

    avg_reward /= eval_episodes

    print("---------------------------------------")
    print("Evaluation over %d episodes: %f" % (eval_episodes, avg_reward))
    print("---------------------------------------")
    return avg_reward

if __name__ == "__main__":

    # General
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy_name", default="TD3")					            # Policy name
    
    parser.add_argument("--env_name", default="Cassie-mimic-walking-v0")            # OpenAI gym environment name
    parser.add_argument("--seed", default=0, type=int)                              # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--start_timesteps", default=1e4, type=int)                 # How many time steps purely random policy is run for
    parser.add_argument("--eval_freq", default=5e3, type=float)                     # How often (time steps) we evaluate
    parser.add_argument("--max_timesteps", default=1e7, type=float)                 # Max time steps to run environment for
    parser.add_argument("--save_models", default=True, action="store_true")         # Whether or not models are saved
    
    parser.add_argument("--act_noise", default=0.3, type=float)                     # Std of Gaussian exploration noise (used to be 0.1)
    parser.add_argument('--param_noise', type=bool, default=True)                   # param noise
    parser.add_argument('--noise_scale', type=float, default=0.3, metavar='G',
                        help='initial param noise scale (default: 0.3)')

    parser.add_argument("--batch_size", default=100, type=int)                      # Batch size for both actor and critic
    parser.add_argument("--discount", default=0.99, type=float)                     # Discount factor
    parser.add_argument("--tau", default=0.005, type=float)                         # Target network update rate

    # TD3 Specific
    # Noise added to target policy during critic update
    parser.add_argument("--policy_noise", default=0.2, type=float)
    # Range to clip target policy noise
    parser.add_argument("--noise_clip", default=0.5, type=float)
    # Frequency of delayed policy updates
    parser.add_argument("--policy_freq", default=2, type=int)

    # For visdom logger
    # Name of experiment
    parser.add_argument("--name", default="test")
    # Where to log diagnostics to
    parser.add_argument("--logdir", type=str, default="/tmp/rl/experiments/")

    # For rendering agent
    parser.add_argument('--render', action='store_true', default=False, help='render the environment')

    args = parser.parse_args()

    file_name = "%s_%s_%s" % (args.policy_name, args.env_name, str(args.seed))
    print("---------------------------------------")
    print("Settings: %s" % (file_name))
    print("---------------------------------------")
    
    # create visdom logger
    global plotter
    plotter = VisdomLinePlotter(env_name=file_name)

    if not os.path.exists("./results"):
        os.makedirs("./results")

    print("./trained_models/" + args.policy_name + "/")
    print(args.save_models)

    if args.save_models and not os.path.exists("./trained_models/" + args.policy_name + "/"):
        print("foo")
        os.makedirs("./trained_models/" + args.policy_name + "/")

    if(args.env_name not in ["Cassie-v0", "Cassie-mimic-v0", "Cassie-mimic-walking-v0"]):
        env = gym.make(args.env_name)
        env = NormalizedActions(env)
    else:
        # set up cassie environment
        import gym_cassie
        from gym_cassie import CassieEnv
        env_fn = make_cassie_env()
        env = env_fn()
        #env = gym.make(args.env_name)

    # should also work
    #env = gym.make(args.env_name)
    max_episode_steps = 10000
    #env = NormalizedActions(env)

    # Set seeds
    env.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    print("state_dim: {}".format(state_dim))
    print("action_dim: {}".format(action_dim))
    print("max_action dim: {}".format(max_action))

    # Initialize policy
    if args.policy_name == "TD3":
        policy = TD3(state_dim, action_dim, max_action, plotter)
    elif args.policy_name == "DDPG":
        policy = DDPG(state_dim, action_dim, max_action, plotter)

    replay_buffer = ReplayBuffer()

    # Initialize param noise (or set to None)
    param_noise = AdaptiveParamNoiseSpec(initial_stddev=0.05, desired_action_stddev=args.noise_scale, adaptation_coefficient=1.05) if args.param_noise else None

    # Evaluate untrained policy
    evaluations = [evaluate_policy(env, policy)]
    plotter.plot('return', 'eval', 'Agent Return', 0, evaluations[-1])

    total_timesteps = 0
    timesteps_since_eval = 0
    episode_num = 0
    done = True

    while total_timesteps < args.max_timesteps:

        if done:

            if total_timesteps != 0:
                # Plot stuff
                plotter.plot('return', 'train', 'Agent Return', total_timesteps, episode_reward)
                print("Total T: {} Episode Num: {} Episode T: {} Reward: {}".format(
                    total_timesteps, episode_num, episode_timesteps, episode_reward))
                if args.policy_name == "TD3":
                    policy.train(replay_buffer, episode_timesteps, args.batch_size, args.discount,
                                 args.tau, args.policy_noise, args.noise_clip, args.policy_freq)
                elif args.policy_name == "DDPG":
                    policy.train(replay_buffer, episode_timesteps,
                                 args.batch_size, args.discount, args.tau)

                # Update param_noise based on distance metric
                if args.param_noise and replay_buffer.ptr > 0:
                    # get tuple of states and actions from last training pass
                    states, perturbed_actions = replay_buffer.get_transitions_from_range(replay_buffer.ptr - (episode_timesteps - 1), replay_buffer.ptr)
                    unperturbed_actions = np.array([policy.select_action(state, param_noise=None) for state in states])
                    
                    dist = distance_metric(perturbed_actions, unperturbed_actions)
                    param_noise.adapt(dist)

            # Evaluate episode
            if timesteps_since_eval >= args.eval_freq:
                timesteps_since_eval %= args.eval_freq
                evaluations.append(evaluate_policy(env, policy))
                plotter.plot('return', 'eval', 'Agent Return', total_timesteps, evaluations[-1])

                if args.save_models:
                    policy.save()
                np.save("./results/%s" % (file_name), evaluations)


            # Reset environment
            obs = env.reset()
            done = False
            episode_reward = 0
            episode_timesteps = 0
            episode_num += 1

        # Param noise
        if args.param_noise:
            policy.perturb_actor_parameters(param_noise)

        # Select action randomly or according to policy
        if total_timesteps < args.start_timesteps:
            #action = env.action_space.sample()
            action = torch.randn(action_dim)
        else:
            action = policy.select_action(np.array(obs), param_noise)
            if args.act_noise != 0:
                action = (action + np.random.normal(0, args.act_noise,
                                                    size=env.action_space.shape[0])).clip(env.action_space.low, env.action_space.high)

        # Perform action
        new_obs, reward, done, _ = env.step(action)
        done_bool = 0 if episode_timesteps + 1 == max_episode_steps else float(done)
        episode_reward += reward

        # Store data in replay buffer
        replay_buffer.add((obs, new_obs, action, reward, done_bool))

        obs = new_obs

        episode_timesteps += 1
        total_timesteps += 1
        timesteps_since_eval += 1

    # Final evaluation
    evaluations.append(evaluate_policy(env, policy))
    if args.save_models:
        policy.save()
    np.save("./results/%s" % (file_name), evaluations)
