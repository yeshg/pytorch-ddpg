import torch
import torch.nn as nn
import torch.nn.functional as F 
import torch.autograd
from torch.autograd import Variable

from torch.optim import Adam

import numpy as np

import os
import time as time

"""
Import replay buffer, model, utils
"""
from rl_algos.replay_buffer import ReplayBuffer, Transition
from rl_algos.model import TD3Actor as Actor, TD3Critic as Critic
from rl_algos.utils import soft_update, hard_update


class DDPG(object):
    def __init__(self, gamma, tau, hidden_size, num_inputs, action_space):

        self.num_inputs = num_inputs
        self.action_space = action_space

        """
        Initialize actor and critic networks. Also initialize target networks
        """
        self.actor = Actor(hidden_size, self.num_inputs, self.action_space)
        self.actor_target = Actor(hidden_size, self.num_inputs, self.action_space)
        self.actor_perturbed = Actor(hidden_size, self.num_inputs, self.action_space)
        self.actor_optim = Adam(self.actor.parameters(), lr=1e-4)

        self.critic = Critic(hidden_size, self.num_inputs, self.action_space)
        self.critic_target = Critic(hidden_size, self.num_inputs, self.action_space)
        self.critic_optim = Adam(self.critic.parameters(), lr=1e-3)

        self.gamma = gamma
        self.tau = tau

        """
        Copy initial params of the actor and critic networks to their respective target networks
        """
        hard_update(self.actor_target, self.actor)  # Make sure target is with the same weight
        hard_update(self.critic_target, self.critic)


    def select_action(self, act_noise, noise_clip, mode, state):
        """
        Select action with non-target actor network and add actor noise for exploration
        """
        if(mode == "train"):
            mu = self.actor((Variable(state)))
            mu = mu.data
            noise = torch.FloatTensor(mu).data.normal_(0, act_noise)
            noise = noise.clamp(-noise_clip, noise_clip)
            mu = (mu + noise).clamp(-self.max_action, self.max_action)
        else:
            self.actor.eval() # https://stackoverflow.com/questions/48146926/whats-the-meaning-of-function-eval-in-torch-nn-module
            mu = self.actor((Variable(state)))
            mu = mu.data

        self.actor.train() # switch back to training mode
        

        return mu


    def update_parameters(self, batch):
        state_batch = Variable(torch.cat(batch.state))
        action_batch = Variable(torch.cat(batch.action))
        reward_batch = Variable(torch.cat(batch.reward))
        mask_batch = Variable(torch.cat(batch.mask))
        next_state_batch = Variable(torch.cat(batch.next_state))
        
        """
        In DDPG, next-state Q values are calculated with the target value network and target policy network
        Once this is calculated, use Bellman equation to calculated updated Q value
        """
        next_action_batch = self.actor_target(next_state_batch)
        next_state_action_values = self.critic_target(next_state_batch, next_action_batch)

        reward_batch = reward_batch.unsqueeze(1)
        mask_batch = mask_batch.unsqueeze(1)
        # This final result is the target Q value
        expected_state_action_batch = reward_batch + (self.gamma * mask_batch * next_state_action_values)

        """
        Minimize MSE loss between updated Q value and original Q value
        """
        state_action_batch = self.critic((state_batch), (action_batch))

        value_loss = F.mse_loss(state_action_batch, expected_state_action_batch)
        self.critic_optim.zero_grad()
        value_loss.backward()
        self.critic_optim.step()

        """
        Maxmize expected return for policy (actor) function
        """
        policy_loss = -self.critic((state_batch),self.actor((state_batch)))

        self.actor_optim.zero_grad()

        policy_loss = policy_loss.mean()
        policy_loss.backward()
        self.actor_optim.step()

        """
        Update the frozen target networks
        """
        soft_update(self.actor_target, self.actor, self.tau)
        soft_update(self.critic_target, self.critic, self.tau)

        return value_loss.item(), policy_loss.item()
    

    # TODO: eventually save replay buffer as well so training can be stopped and resumed
    def save(self):
        """Save all networks, including non-target"""

        #save_path = os.path.join("./trained_models", "ddpg")

        # try:
        #     os.makedirs(save_path)
        # except OSError:
        #     pass

        print("Saving model")

        if not os.path.exists('trained_models/td3/'):
            os.makedirs('trained_models/td3/')

        filetype = ".pt" # pytorch model
        torch.save(self.actor_target.state_dict(), os.path.join("./trained_models/td3", "target_actor_model" + filetype))
        torch.save(self.critic_target.state_dict(), os.path.join("./trained_models/td3", "target_critic_model" + filetype))
        torch.save(self.actor.state_dict(), os.path.join("./trained_models/td3", "actor_model" + filetype))
        torch.save(self.critic.state_dict(), os.path.join("./trained_models/td3", "critic_model" + filetype))

    def load_model(self, model_path):
        target_actor_path = os.path.join(model_path, "target_actor_model.pt")
        target_critic_path = os.path.join(model_path, "target_critic_model.pt")
        actor_path = os.path.join(model_path, "actor_model.pt")
        critic_path = os.path.join(model_path, "critic_model.pt")
        print('Loading models from {}, {}, {}, and {}'.format(target_actor_path, target_critic_path, actor_path, critic_path))
        if actor_path is not None:
            self.actor.load_state_dict(torch.load(actor_path))
            self.actor.eval()
        if critic_path is not None: 
            self.critic.load_state_dict(torch.load(critic_path))
            self.critic.eval()

    def train(self, env, memory, n_itr, act_noise=0.2, noise_clip=0.5, policy_freq=2, args, logger=None):

        rewards = []
        total_numsteps = 0
        updates = 0

        start_time = time.time()
        
        
        for itr in range(n_itr):    # n_itr == args.num_episodes
            print("********** Iteration {} ************".format(itr))

            # Observe initial state
            state = torch.Tensor([env.reset()])

            # if args.ou_noise:
            #     """
            #     As args.exploration_end is reached, gradually switch from args.noise_scale to args.final_noise_scale.
            #     Purpose of this is to improve exploration at start of training.
            #     """
            #     ounoise.scale = (args.noise_scale - args.final_noise_scale) * max(0, args.exploration_end - itr) / args.exploration_end + args.final_noise_scale
            #     ounoise.reset()

            episode_reward = 0
            episode_start = time.time()
            trainEpLen = 0
            while True:
                trainEpLen += 1
                """
                select action according to current policy and exploration noise
                """
                (self, act_noise, noise_clip, mode, state)
                action = self.select_action(mode="train", state, args.act_noise, param_noise)

                """
                execute action and observe reward and new state
                """
                next_state, reward, done, _ = env.step(action.numpy()[0])
                total_numsteps += 1
                episode_reward += reward

                action = torch.Tensor(action)
                mask = torch.Tensor([not done])
                next_state = torch.Tensor([next_state])
                reward = torch.Tensor([reward])

                """
                store transition tuple in replay buffer
                """
                memory.push(state, action, mask, next_state, reward)

                state = next_state

                if len(memory) > args.batch_size:
                    for _ in range(args.updates_per_step):
                        """
                        Sample random minibatch of (args.batch_size) transitions
                        """
                        transitions = memory.sample(args.batch_size)
                        batch = Transition(*zip(*transitions))

                        """
                        Calculate updated Q value (Bellman equation) and update parameters of all networks
                        """
                        value_loss, policy_loss = self.update_parameters(batch)

                        updates += 1
                if done:
                    break

            print("time elapsed: {:.2f} s".format(time.time() - start_time))
            print("episode time elapsed: {:.2f} s".format(time.time() - episode_start))

            # Update param_noise based on distance metric
            if args.param_noise:
                episode_transitions = memory.memory[memory.position-t:memory.position]
                states = torch.cat([transition[0] for transition in episode_transitions], 0)
                unperturbed_actions = self.select_action(states, None, None)
                perturbed_actions = torch.cat([transition[1] for transition in episode_transitions], 0)

                ddpg_dist = ddpg_distance_metric(perturbed_actions.numpy(), unperturbed_actions.numpy())
                param_noise.adapt(ddpg_dist)

            """
            Logging with visdom
            """
            if logger is not None:
                """
                Evaluate non-target actor
                """
                state = torch.Tensor([env.reset()])
                episode_reward = 0
                evaluate_start = time.time()
                testEpLen = 0
                while True:
                    testEpLen += 1
                    action = self.select_action(state)

                    next_state, reward, done, _ = env.step(action.numpy()[0])
                    episode_reward += reward

                    next_state = torch.Tensor([next_state])

                    state = next_state
                    if done:
                        print("evaluate time elapsed: {:.2f} s".format(time.time() - evaluate_start))
                        break

                rewards.append(episode_reward)
                logger.record("Return", rewards[-1])
                logger.record("Train EpLen", trainEpLen)
                logger.record("Test EpLen", testEpLen)
                logger.record("Time elapsed", time.time()-start_time)
                logger.dump()

                print("Iteration: {}, total numsteps: {}, reward: {}, average reward: {}".format(itr, total_numsteps, rewards[-1], np.mean(rewards[-10:])))
                self.save()