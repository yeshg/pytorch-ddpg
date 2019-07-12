# rl algos

Collection of (off-policy) rl algorithms. Fully compatible with OpenAI gym.

Real-time monitoring of training done with visdom.


#### DDPG (not distributed)
Deep Learning extension of deterministic policy gradients (DPG), an off-policy RL algorithm. My implementation uses action and parameter noise to improve exploration at the start of training and then throughout the remainder of the steps.

#### TD3 (not distributed)
In progress - DDPG with tweaks to counter the tendency of DDPG to overestimate Q-function later during learning. Also uses action and parameter noise to improve exploration

#### D4PG (In progress, distributed)
In progress - DDPG with Ape-X framework (using ray for this) and PER

#### D4PG + TD3 (In progress, distributed)

This is an implementation of DDPG in Pytorch with action and parameter noise for exploration.

## TODO
- Implement prioritized experience replay
- Implement Ape-X distributed training framework with Ray
- Fix visdom logging
- Clean up unneeded code.
- Integrate into https://github.com/yeshg/deep-rl


## Acknowledgements


Code structure, visdom logging from on https://github.com/p-morais/deep-rl

Basic implementations of DDPG and TD3 from official TD3 release repo: https://github.com/sfujim/TD3