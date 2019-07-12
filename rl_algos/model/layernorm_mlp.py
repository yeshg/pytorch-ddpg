# Straight from pedro's gaussian mlp code

import torch
import torch.nn as nn
import torch.nn.functional as F

# By default all the modules are initialized to train mode (self.training = True)


class LN_MLP_Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action, hidden_size1, hidden_size2, nonlinearity="tanh", bounded=False, layernorm=True):
        super(LN_MLP_Actor, self).__init__()

        actor_dims = (256, 256)
        if nonlinearity == "relu":
            self.nonlinearity = F.relu
        elif nonlinearity == "tanh":
            self.nonlinearity = torch.tanh
        else:
            raise NotImplementedError

        self.actor_layers = nn.ModuleList()
        self.actor_layers += [nn.Linear(state_dim, actor_dims[0])]
        self.actor_layers += [nn.LayerNorm(actor_dims[0])]
        for l in range(len(actor_dims) - 1):
            in_dim = actor_dims[l]
            out_dim = actor_dims[l + 1]
            self.actor_layers += [nn.Linear(in_dim, out_dim)]
            self.actor_layers += [nn.LayerNorm(out_dim)]

        self.mean = nn.Linear(actor_dims[-1], action_dim)

        self.bounded = bounded

    def forward(self, inputs):
        x = inputs
        for l in self.actor_layers:
            x = self.nonlinearity(l(x))
            
        x = self.mean(x)

        if self.bounded:
            mean = torch.tanh(x) 
        else:
            mean = x

        return x

class LN_MLP_DDPGCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size1, hidden_size2, nonlinearity="tanh"):
        super(LN_MLP_DDPGCritic, self).__init__()

        critic_dims = (256, 256)
        if nonlinearity == "relu":
            self.nonlinearity = F.relu
        elif nonlinearity == "tanh":
            self.nonlinearity = torch.tanh
        else:
            raise NotImplementedError

        self.critic_layers = nn.ModuleList()
        self.critic_layers += [nn.Linear(state_dim + action_dim, critic_dims[0])]
        self.critic_layers += [nn.LayerNorm(critic_dims[0])]
        for l in range(len(critic_dims) - 1):
            in_dim = critic_dims[l]
            out_dim = critic_dims[l + 1]
            self.critic_layers += [nn.Linear(in_dim, out_dim)]
            self.critic_layers += [nn.LayerNorm(out_dim)]

        self.vf = nn.Linear(critic_dims[-1], 1)

    def forward(self, inputs, actions):
        xu = torch.cat([inputs, actions], 1)

        x = xu
        for l in self.critic_layers:
            x = self.nonlinearity(l(x))
        value = self.vf(x)

        return x1

# critic uses 2 action-value functions (and uses smaller one to form targets)


class LN_MLP_TD3Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size1, hidden_size2, nonlinearity="tanh"):
        super(LN_MLP_TD3Critic, self).__init__()

        critic_dims = (256, 256)
        if nonlinearity == "relu":
            self.nonlinearity = F.relu
        elif nonlinearity == "tanh":
            self.nonlinearity = torch.tanh
        else:
            raise NotImplementedError

        # Q1 architecture
        self.critic1_layers = nn.ModuleList()
        self.critic1_layers += [nn.Linear(state_dim + action_dim, critic_dims[0])]
        self.critic1_layers += [nn.LayerNorm(critic_dims[0])]
        for l in range(len(critic_dims) - 1):
            in_dim = critic_dims[l]
            out_dim = critic_dims[l + 1]
            self.critic1_layers += [nn.Linear(in_dim, out_dim)]
            self.critic1_layers += [nn.LayerNorm(out_dim)]

        self.v1 = nn.Linear(critic_dims[-1], 1)

        # Q2 architecture
        self.critic2_layers = nn.ModuleList()
        self.critic2_layers += [nn.Linear(state_dim + action_dim, critic_dims[0])]
        self.critic2_layers += [nn.LayerNorm(critic_dims[0])]
        for l in range(len(critic_dims) - 1):
            in_dim = critic_dims[l]
            out_dim = critic_dims[l + 1]
            self.critic2_layers += [nn.Linear(in_dim, out_dim)]
            self.critic2_layers += [nn.LayerNorm(out_dim)]

        self.v2 = nn.Linear(critic_dims[-1], 1)

    def forward(self, inputs, actions):
        xu = torch.cat([inputs, actions], 1)

        x1 = xu
        for l in self.critic1_layers:
            x1 = self.nonlinearity(l(x1))
        value = self.v1(x1)

        x2 = xu
        for l in self.critic2_layers:
            x2 = self.nonlinearity(l(x2))
        value = self.v2(x2)

        return x1, x2

    def Q1(self, inputs, actions):
        xu = torch.cat([inputs, actions], 1)

        x1 = xu
        for l in self.critic1_layers:
            x1 = self.nonlinearity(l(x1))
        value = self.v1(x1)

        return x1