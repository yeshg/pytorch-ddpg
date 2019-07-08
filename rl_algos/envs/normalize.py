# Modified from https://github.com/openai/baselines/blob/master/baselines/common/vec_env/vec_normalize.py
# Thanks to the authors + OpenAI for the code

import time
import numpy as np
import functools
import torch

from .wrapper import WrapEnv
def get_normalization_params(iter, policy, env_fn, noise_std):
    print("Gathering input normalization data using {0} steps, noise = {1}...".format(iter, noise_std))

    env = WrapEnv(env_fn)

    states = np.zeros((iter, env.observation_space.shape[0]))

    state = env.reset()
    for t in range(iter):
        start_time = time.time()
        #env.render()
        states[t, :] = state

        state = torch.Tensor(state)

        action = policy.act(state, deterministic=True)

        # add gaussian noise to deterministic action
        action = action + torch.randn(action.size()) * noise_std

        state, _, done, _ = env.step(action.data.numpy())

        if done:
            state = env.reset()

    print("Done gathering input normalization data. Time taken = {} seconds".format(time.time()))

    return np.mean(states, axis=0), np.sqrt(np.var(states, axis=0) + 1e-8)