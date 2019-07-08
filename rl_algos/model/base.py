import torch.nn as nn


class FFPolicy(nn.Module):
    def __init__(self):
        super(FFPolicy, self).__init__()
        self.env = None # Gym environment name string

    def forward(self, x):
        raise NotImplementedError

    def act(self, inputs, deterministic=False):
        x = self(inputs)
        action = self.dist.sample(x, deterministic=deterministic)
        return action.detach()

    def evaluate(self, inputs):
        value = self(inputs)
        return value, self.dist.evaluate(x)
