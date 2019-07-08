import torch
import torch.nn as nn
import torch.nn.functional as F

from rl.distributions import DiagonalGaussian
from .base import FFPolicy

import time

# NOTE: the fact that this has the same name as a parameter caused a NASTY bug
# apparently "if <function_name>" evaluates to True in python...
def normc_fn(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0, 1)
        m.weight.data *= 1 / torch.sqrt(m.weight.data.pow(2).sum(1, keepdim=True))
        if m.bias is not None:
            m.bias.data.fill_(0)

class GaussianMLPActor(FFPolicy):
    def __init__(self, 
                 num_inputs, 
                 action_dim, 
                 init_std=1, 
                 learn_std=True, 
                 nonlinearity="tanh", 
                 bounded=False,
                 normc_init=True,
                 obs_std=None,
                 obs_mean=None):
        super(GaussianMLPActor, self).__init__()

        actor_dims = (256, 256)

        # create actor network
        self.actor_layers = nn.ModuleList()
        self.actor_layers += [nn.Linear(num_inputs, actor_dims[0])]
        for l in range(len(actor_dims) - 1):
            in_dim = actor_dims[l]
            out_dim = actor_dims[l + 1]
            self.actor_layers += [nn.Linear(in_dim, out_dim)]
        
        self.mean = nn.Linear(actor_dims[-1], action_dim)

        self.dist = DiagonalGaussian(action_dim, init_std, learn_std)

        if nonlinearity == "relu":
            self.nonlinearity = F.relu
        elif nonlinearity == "tanh":
            self.nonlinearity = torch.tanh
        else:
            raise NotImplementedError

        # weight initialization scheme used in PPO paper experiments
        self.normc_init = normc_init

        self.bounded = bounded

        self.init_parameters()
        self.train()

    def init_parameters(self):
        if self.normc_init:
            print("Doing norm column initialization.")
            self.apply(normc_fn)

            if self.dist.__class__.__name__ == "DiagGaussian":
                self.mean.weight.data.mul_(0.01)

    def forward(self, inputs):
        if self.training == False:
            inputs = (inputs - self.obs_mean) / self.obs_std
        
        x = inputs
        for l in self.actor_layers:
            x = self.nonlinearity(l(x))
        x = self.mean(x)

        if self.bounded:
            mean = torch.tanh(x) 
        else:
            mean = x

        return mean

class GaussianMLPCritic(FFPolicy):
    def __init__(self, 
                 num_inputs, 
                 action_dim, 
                 init_std=1, 
                 learn_std=True, 
                 nonlinearity="tanh", 
                 bounded=False,
                 normc_init=True,
                 obs_std=None,
                 obs_mean=None):
        super(GaussianMLPCritic, self).__init__()

        critic_dims = (256, 256)

        # create critic network

        # Q1
        self.critic1_layers = nn.ModuleList()
        self.critic1_layers += [nn.Linear(num_inputs + action_dim, critic_dims[0])]
        for l in range(len(critic_dims) - 1):
            in_dim = critic_dims[l]
            out_dim = critic_dims[l + 1]
            self.critic1_layers += [nn.Linear(in_dim, out_dim)]
        
        self.vf1 = nn.Linear(critic_dims[-1], 1)

        # Q2
        self.critic2_layers = nn.ModuleList()
        self.critic2_layers += [nn.Linear(num_inputs + action_dim, critic_dims[0])]
        for l in range(len(critic_dims) - 1):
            in_dim = critic_dims[l]
            out_dim = critic_dims[l + 1]
            self.critic2_layers += [nn.Linear(in_dim, out_dim)]
        
        self.vf2 = nn.Linear(critic_dims[-1], 1)

        self.dist = DiagonalGaussian(action_dim, init_std, learn_std)

        if nonlinearity == "relu":
            self.nonlinearity = F.relu
        elif nonlinearity == "tanh":
            self.nonlinearity = torch.tanh
        else:
            raise NotImplementedError

        self.obs_std = obs_std
        self.obs_mean = obs_mean

        # weight initialization scheme used in PPO paper experiments
        self.normc_init = normc_init

        self.bounded = bounded

        self.init_parameters()
        self.train()

    def init_parameters(self):
        if self.normc_init:
            print("Doing norm column initialization.")
            self.apply(normc_fn)

            if self.dist.__class__.__name__ == "DiagGaussian":
                self.mean.weight.data.mul_(0.01)

    def forward(self, states, actions):
        inputs = torch.cat([states, actions], 1)

        if self.training == False:
            inputs = (inputs - self.obs_mean) / self.obs_std
        
        # Q1
        x = inputs
        for l in self.critic1_layers:
            x = self.nonlinearity(l(x))
        v1 = self.vf1(x)

        # Q2
        x = inputs
        for l in self.critic2_layers:
            x = self.nonlinearity(l(x))
        v2 = self.vf2(x)

        return v1, v2

    def Q1(self, states, actions):
        x = torch.cat([states, actions], 1)

        for l in self.critic1_layers:
            x = self.nonlinearity(l(x))
        v1 = self.vf1(x)

        return v1