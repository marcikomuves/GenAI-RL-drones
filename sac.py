import torch
import numpy as np

LOG_STD_MAX = 2
LOG_STD_MIN = -5

class Actor(torch.nn.Module):
    def __init__(self, act_space, obs_space):
        super().__init__()
        self.fc1 = torch.nn.Linear(np.prod(obs_space.shape), 256)
        self.fc2 = torch.nn.Linear(256, 256)
        self.fc_mean = torch.nn.Linear(256, np.prod(act_space.shape))
        self.fc_logstd = torch.nn.Linear(256, np.prod(act_space.shape))
        self.register_buffer(
            "action_scale",
            torch.tensor((act_space.high - act_space.low) / 2.0, dtype=torch.float32,),
        )
        self.register_buffer(
            "action_bias",
            torch.tensor((act_space.high + act_space.low) / 2.0, dtype=torch.float32,),
        )

    def forward(self, x):
        x = torch.nn.functional.relu(self.fc1(x))
        x = torch.nn.functional.relu(self.fc2(x))
        mean = self.fc_mean(x)
        log_std = self.fc_logstd(x)
        log_std = torch.tanh(log_std)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)
        return mean, log_std

    def get_action(self, x):
        mean, log_std = self(x)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample() # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias # Action Bound
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean


class SoftQNetwork(torch.nn.Module):
    def __init__(self, act_space, obs_space, ln = False):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(np.prod(obs_space.shape) + np.prod(act_space.shape), 256),
            torch.nn.LayerNorm(256) if ln else torch.nn.Identity(),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 256),
            torch.nn.LayerNorm(256) if ln else torch.nn.Identity(),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 1)
        )

    def forward(self, x):
        return self.net(x)

    def get_value(self, x, act):
        return self(torch.cat([x, act], 1))