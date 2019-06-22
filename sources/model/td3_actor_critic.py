import torch
import torch.nn as nn
import torch.nn.functional as F 

# By default all the modules are initialized to train mode (self.training = True)

class Actor(nn.Module):
    def __init__(self, hidden_size, num_inputs, action_space):
        super(Actor, self).__init__()
        self.action_space = action_space
        num_outputs = action_space.shape[0]

        self.linear1 = nn.Linear(num_inputs, hidden_size)
        self.ln1 = nn.LayerNorm(hidden_size)

        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.ln2 = nn.LayerNorm(hidden_size)

        self.mu = nn.Linear(hidden_size, num_outputs)
        self.mu.weight.data.mul_(0.1)
        self.mu.bias.data.mul_(0.1)

    def forward(self, inputs):
        x = inputs
        x = self.linear1(x)
        x = self.ln1(x)
        x = F.relu(x)
        x = self.linear2(x)
        x = self.ln2(x)
        x = F.relu(x)
        mu = torch.tanh(self.mu(x))
        return mu

"""
TD3 uses two Q-functions
"""
class Critic(nn.Module):
    def __init__(self, hidden_size, num_inputs, action_space):
        super(Critic, self).__init__()
        self.action_space = action_space
        num_outputs = action_space.shape[0]

        # Q1 architecture
        self.linear1 = nn.Linear(num_inputs, hidden_size)
        self.ln1 = nn.LayerNorm(hidden_size)

        self.linear2 = nn.Linear(hidden_size+num_outputs, hidden_size)
        self.ln2 = nn.LayerNorm(hidden_size)

        self.V1 = nn.Linear(hidden_size, 1)
        self.V1.weight.data.mul_(0.1)
        self.V1.bias.data.mul_(0.1)

        # Q2 architecture
        self.linear3 = nn.Linear(num_inputs, hidden_size)
        self.ln3 = nn.LayerNorm(hidden_size)

        self.linear4 = nn.Linear(hidden_size+num_outputs, hidden_size)
        self.ln4 = nn.LayerNorm(hidden_size)

        self.V2 = nn.Linear(hidden_size, 1)
        self.V2.weight.data.mul_(0.1)
        self.V2.bias.data.mul_(0.1)

    def forward(self, inputs, actions):
        x1 = x2 = inputs

        x1 = self.linear1(x1)
        x1 = self.ln1(x1)
        x1 = F.relu(x1)

        x1 = torch.cat((x1, actions), 1)
        x1 = self.linear2(x1)
        x1 = self.ln2(x1)
        x1 = F.relu(x1)
        V1 = self.V(x1)

        x2 = self.linear1(x2)
        x2 = self.ln1(x2)
        x2 = F.relu(x2)

        x2 = torch.cat((x2, actions), 1)
        x2 = self.linear2(x2)
        x2 = self.ln2(x2)
        x2 = F.relu(x2)
        V2 = self.V(x2)
        return V1, V2

    def Q1_forward(self, inputs, actions):
        x1 = inputs

        x1 = self.linear1(x1)
        x1 = self.ln1(x1)
        x1 = F.relu(x1)

        x1 = torch.cat((x1, actions), 1)
        x1 = self.linear2(x1)
        x1 = self.ln2(x1)
        x1 = F.relu(x1)
        V1 = self.V(x1)
        return V1