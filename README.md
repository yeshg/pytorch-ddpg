# rl algos

Collection of rl algorithms (so far just DDPG and TD3 are implemented.)

Real-time monitoring of training done with visdom.


#### DDPG
Deep Learning extension of deterministic policy gradients (DPG), an off-policy RL algorithm. My implementation uses gradually decreasing action and parameter noise to improve exploration at the start of training and then throughout the remainder of the steps.

(not parallelized)

#### TD3
In progress - DDPG with tweaks to counter the tendency of DDPG to overestimate Q-function later during learning.

(not parallelized)

#### D4PG
In progress - DDPG with Ape-X framework (using ray for this) and PER

(parallelized)

#### D4PG + TD3?

This is an implementation of DDPG in Pytorch with action and parameter noise for exploration.

(parallelized)


## TODO

- Add procedure to improve exploration at start of training: for fixed number of steps (some hyperparameter) have agent take actions sampled from uniform random distribution over valid actions.
- Clean up unneeded code.
- Integrate into https://github.com/yeshg/deep-rl


## Acknowledgements


Code structure, visdom logging from on https://github.com/p-morais/deep-rl

