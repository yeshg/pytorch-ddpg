# pytorch-ddpg

This is an implementation of DDPG in Pytorch with action and parameter noise for exploration.

Real-time monitoring of both target and non-target actor networks is done with visdom.

Code structure based on https://github.com/p-morais/deep-rl

## TODO

- Add procedure to improve exploration at start of training: for fixed number of steps (some hyperparameter) have agent take actions sampled from uniform random distribution over valid actions.
- Clean up unneeded code.
- Streamline into https://github.com/yeshg/deep-rl