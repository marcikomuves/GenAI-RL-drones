# rollout.py
import numpy as np
import torch


class RolloutBuffer:

    def __init__(self, max_steps, n_envs, obs_shape, act_shape, device, store_logprob_value=False):
        self.max_steps = max_steps
        self.n_envs = n_envs
        self.device = device
        self.store_logprob_value = store_logprob_value
        self.obs_shape = obs_shape
        self.act_shape = act_shape
        self.reset()

    def __len__(self):
        return self.size


    def _t(self, x):
        if torch.is_tensor(x):
            x = x.detach()
        return torch.as_tensor(x, dtype=torch.float32, device=self.device)


    def append(self, obs, action, reward, terminated, truncated, next_obs, logprob=None, value=None):
        self.obs[self.ptr]        = self._t(obs)
        self.action[self.ptr]     = self._t(action)
        self.reward[self.ptr]     = self._t(reward)
        self.terminated[self.ptr] = self._t(terminated)
        self.truncated[self.ptr]  = self._t(truncated)
        self.next_obs[self.ptr]   = self._t(next_obs)
        if self.logprob is not None:
            self.logprob[self.ptr] = self._t(logprob).flatten()
            self.value[self.ptr]   = self._t(value).flatten()

        self.ptr = (self.ptr + 1) % self.max_steps
        self.size = min(self.size + 1, self.max_steps)


    def ordered(self):
        names = ["obs", "action", "reward", "terminated", "truncated", "next_obs"]
        if self.logprob is not None:
            names += ["logprob", "value"]
        out = {}
        for n in names:
            x = getattr(self, n)
            out[n] = x[:self.size] if self.size < self.max_steps else torch.roll(x, -self.ptr, dims=0)
        return out


    def sample(self, batch_size):
        names = ["obs", "action", "reward", "terminated", "truncated", "next_obs"]
        if self.logprob is not None:
            names += ["logprob", "value"]
        idx = torch.randint(0, self.size * self.n_envs, (batch_size,), device=self.device)
        t = idx // self.n_envs
        e = idx % self.n_envs
        out = {}
        for n in names:
            x = getattr(self, n)
            out[n] = x[t, e]
        return out

    def reset(self):
        self.ptr = 0     # next write position
        self.size = 0    # number of valid steps stored

        f = dict(dtype=torch.float32, device=self.device)
        # torch.empty: allocated but uninitialised (only [:size] is ever read)
        self.obs = torch.empty((self.max_steps, self.n_envs) + tuple(self.obs_shape), **f)
        self.action = torch.empty((self.max_steps, self.n_envs) + tuple(self.act_shape), **f)
        self.reward = torch.empty((self.max_steps, self.n_envs), **f)
        self.terminated = torch.empty((self.max_steps, self.n_envs), **f)
        self.truncated = torch.empty((self.max_steps, self.n_envs), **f)
        self.next_obs = torch.empty((self.max_steps, self.n_envs) + tuple(self.obs_shape), **f)
        self.logprob = torch.empty((self.max_steps, self.n_envs), **f) if self.store_logprob_value else None
        self.value = torch.empty((self.max_steps, self.n_envs), **f) if self.store_logprob_value else None