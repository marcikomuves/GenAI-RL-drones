import os
import numpy as np
from tqdm import tqdm
from color import bcolors
import torch
import itertools
from torch.utils.tensorboard import SummaryWriter
from gym_pybullet_drones.utils.enums import ActionType, DroneModel, Physics
import gymnasium as gym
from targetaviary import TargetAviary
from targetstategenerator import StaticTargetStateGenerator
from dronestategenerator import DroneStateGenerator
from rolloutbuffer import RolloutBuffer
from synther import SimpleDiffusionGenerator
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--run-name", required=True)
parser.add_argument("--n", type=int, default=2)
parser.add_argument("--m", type=int, default=2)
parser.add_argument("--utd", type=int, default=1)
parser.add_argument("--synthetic", action="store_true")
parser.add_argument("--layernorm", action="store_true")
parser.add_argument("--seed", type=int, default=0)
args = parser.parse_args()

# experiment-specific params

# soft Q-network ensemble params
N = args.n
M = args.m
UTD = args.utd
ENABLE_SYNTHETIC = args.synthetic
ENABLE_LAYERNORM = args.layernorm
RUN_NAME = args.run_name

# FIX PARAMS

# environment
DRONE_MODEL = DroneModel.CF2X
ACT = ActionType.RPM
PHYSICS = Physics.PYB # Physics.PYB_GND_DRAG_DW ?
SIMULATION_FREQ_HZ = 240
CONTROL_FREQ_HZ = 48
INIT_XYZ = np.array([0, 0, 1.0])

# SAC training
NUM_TRAINING_ENVS = 1   # number of parallel training environments
NUM_EVAL_ENVS = 6   # number of parallel evaluation environments
NUM_STEPS_TOTAL = 100_000   # number of training episodes
WARMUP_LENGTH = 5000    # warmup period length (action sampling without training)
REAL_BUFFER_SIZE = NUM_STEPS_TOTAL // NUM_TRAINING_ENVS
SAC_BATCH_SIZE = 256    # number of transitions to sample for an update
LOG_STD_MAX = 2
LOG_STD_MIN = -5
POLICY_LR = 3e-4    # learning rate of the policy network optimizer
Q_LR = 3e-4 # learning rate of the Q network network optimizer
GAMMA = 0.99    # discount factor
TAU = 5e-3  # target update coefficient
AUTOTUNE = True # flag to toggle automatic tuning of the entropy regularization coefficient
ALPHA = 0.2 # entropy regularization coefficient
POLICY_FREQ = 1 # delayed policy update frequency in global steps
TARGET_FREQ = 1 # delayed target q network update frequency in global steps
LOG_FREQ = 100  # log to tensorboard frequency in global steps

# synthetic sampling
SYNTHETIC_RATIO = 0.5 if ENABLE_SYNTHETIC else 0 # ratio of synthetic samples and the full SAC_BATCH_SIZE 
SYNTHETIC_DENOISING_STEPS = 128 # for noise schedule
SYNTHETIC_SAMPLES = 200_000 # number of synthetic transitions to sample after each synther retrain 
SYNTHETIC_SAMPLE_BATCH_SIZE = 100_000  # break SYNTHETIC_SAMPLES into smaller chunks to prevent OOM

# SynthER training
SYNTHER_STARTS = 10_000
SYNTHER_FREQ = 10_000
SYNTHER_TRAIN_STEPS = 100_000
SYNTHER_LR = 3e-4
SYNTHER_TRAIN_BATCH_SIZE = 256
SYNTHER_EMA_FREQ = 10
SYNTHER_EMA_AFTER = 100
SYNTHER_EMA_BETA = 0.995
SYNTHER_EMA_POWER = 2.0/3.0


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
    def __init__(self, act_space, obs_space):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(np.prod(obs_space.shape) + np.prod(act_space.shape), 256),
            torch.nn.LayerNorm(256) if ENABLE_LAYERNORM else torch.nn.Identity(),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 256),
            torch.nn.LayerNorm(256) if ENABLE_LAYERNORM else torch.nn.Identity(),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 1)
        )

    def forward(self, x):
        return self.net(x)

    def get_value(self, x, act):
        return self(torch.cat([x, act], 1))


def make_env():
    return TargetAviary(
        target_state_generator=StaticTargetStateGenerator(xyz=INIT_XYZ, randomize=False),
        init_state_generator=DroneStateGenerator(init_xyz=INIT_XYZ, randomize=True),
        max_t=10,
        act=ACT,
        physics=PHYSICS,
        drone_model=DRONE_MODEL,
        pyb_freq=SIMULATION_FREQ_HZ,
        ctrl_freq=CONTROL_FREQ_HZ,
    )


if __name__ == "__main__":

    print(f"[INFO] {bcolors.OKGREEN} Initializing training vectorized environments...{bcolors.ENDC}")
    train_envs = gym.vector.SyncVectorEnv(
        [make_env for _ in range(NUM_TRAINING_ENVS)],
        autoreset_mode=gym.vector.AutoresetMode.SAME_STEP,
    )

    print(f"[INFO] {bcolors.OKGREEN} Initializing evalutaion vectorized environments...{bcolors.ENDC}")
    eval_envs = gym.vector.SyncVectorEnv(
        [make_env for _ in range(NUM_EVAL_ENVS)],
        autoreset_mode=gym.vector.AutoresetMode.SAME_STEP,
    )

    print(f"[INFO] {bcolors.OKGREEN} Initializing device...{bcolors.ENDC}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] {device}")

    print(f"[INFO] {bcolors.OKGREEN} Initializing agent...{bcolors.ENDC}")

    qfs = []
    qf_targets = []
    for i in range(N):
        qf = SoftQNetwork(train_envs.single_action_space, train_envs.single_observation_space).to(device)
        qf_target = SoftQNetwork(train_envs.single_action_space, train_envs.single_observation_space).to(device)
        qf_target.load_state_dict(qf.state_dict())
        qfs.append(qf); qf_targets.append(qf_target)
    q_optimizer = torch.optim.Adam(itertools.chain.from_iterable(q.parameters() for q in qfs), lr=Q_LR)
    
    actor = Actor(train_envs.single_action_space, train_envs.single_observation_space).to(device)
    actor_optimizer = torch.optim.Adam(list(actor.parameters()), lr=POLICY_LR)
    if AUTOTUNE:
        target_entropy = -torch.prod(torch.Tensor(train_envs.single_action_space.shape).to(device)).item()
        log_alpha = torch.zeros(1, requires_grad=True, device=device)
        alpha = log_alpha.exp().item()
        a_optimizer = torch.optim.Adam([log_alpha], lr=Q_LR)
    else:
        alpha = ALPHA

    subdir  = "layernorm" if ENABLE_LAYERNORM else "no_layernorm"
    model_dir = f"models/sac/{subdir}/{RUN_NAME}/"
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
    model_path = os.path.join(model_dir, f"{RUN_NAME}.cleanrl_model")
    runs_dir = f"runs/sac/{subdir}/{RUN_NAME}/"
    if not os.path.exists(runs_dir):
        os.mkdir(runs_dir)
    writer = SummaryWriter(runs_dir)

    print(f"[INFO] {bcolors.OKGREEN} Starting training...{bcolors.ENDC}")

    real_buffer = RolloutBuffer(
        max_steps = REAL_BUFFER_SIZE,
        n_envs = NUM_TRAINING_ENVS,
        obs_shape = train_envs.single_observation_space.shape,
        act_shape = train_envs.single_action_space.shape,
        device = device,
    )

    if ENABLE_SYNTHETIC:
        synthetic_buffer = None
        generator = SimpleDiffusionGenerator(
            obs_space = train_envs.single_observation_space,
            act_space = train_envs.single_action_space,
            device=device,
            terminal_threshold=0.5,
        )

    obs, _ = train_envs.reset()
    pbar = tqdm(range(NUM_STEPS_TOTAL), desc="[SAC] Global step", mininterval=1, leave=False)
    for global_step in pbar:
        actor.train()
        # 1) Rollout episode (1 step at a time)
        if global_step >= WARMUP_LENGTH:
            with torch.no_grad():
                actions, _, _ = actor.get_action(torch.as_tensor(obs, dtype=torch.float32, device=device))
                actions = actions.cpu().numpy()
        else:
            actions = np.array([train_envs.single_action_space.sample() for _ in range(NUM_TRAINING_ENVS)])
        next_obs, rewards, terminations, truncations, infos = train_envs.step(actions)
        true_next_obs = next_obs.copy()
        for idx, done in enumerate(np.logical_or(terminations, truncations)):
            if done:
                true_next_obs[idx] = infos["final_obs"][idx]
        real_buffer.append(obs, actions, rewards, terminations, truncations, true_next_obs)
        obs = next_obs

        # 2) Updates
        if global_step >= WARMUP_LENGTH:
            # 2.1) SynthER update
            if ENABLE_SYNTHETIC and global_step >= SYNTHER_STARTS and global_step % SYNTHER_FREQ == 0:
                
                if synthetic_buffer is None:
                    synthetic_buffer = RolloutBuffer(
                        max_steps = SYNTHETIC_SAMPLES,
                        n_envs = 1,
                        obs_shape = train_envs.single_observation_space.shape,
                        act_shape = train_envs.single_action_space.shape,
                        device = device,
                    )
                else:
                    synthetic_buffer.reset()

                generator.retrain(
                    real_buffer=real_buffer,
                    train_steps=SYNTHER_TRAIN_STEPS,
                    batch_size=SYNTHER_TRAIN_BATCH_SIZE,
                    lr=SYNTHER_LR,
                    ema_freq=SYNTHER_EMA_FREQ,
                    ema_after=SYNTHER_EMA_AFTER,
                    beta=SYNTHER_EMA_BETA,
                    power=SYNTHER_EMA_POWER
                )

                synthetic_sample = zip(
                    *generator.sample(
                        num_samples=SYNTHETIC_SAMPLES,
                        num_sample_steps=SYNTHETIC_DENOISING_STEPS,
                        sample_batch_size=SYNTHETIC_SAMPLE_BATCH_SIZE,
                    )
                )

                for obs_synth, act_synth, reward_synth, next_obs_synth, terminated_synth in synthetic_sample:
                    synthetic_buffer.append(obs_synth, act_synth, reward_synth, terminated_synth, False, next_obs_synth)

            # 2.2) soft Q network updates
            for epoch in range(UTD):
                
                if ENABLE_SYNTHETIC and synthetic_buffer is not None:
                    real = real_buffer.sample(int(SAC_BATCH_SIZE * (1.0 - SYNTHETIC_RATIO)))
                    synthetic = synthetic_buffer.sample(int(SAC_BATCH_SIZE * SYNTHETIC_RATIO))
                    data = {k: torch.cat([real[k], synthetic[k]], dim=0) for k in real}
                else:
                    data = real_buffer.sample(SAC_BATCH_SIZE)

                with torch.no_grad():
                    next_state_actions, next_state_log_pi, _ = actor.get_action(data["next_obs"])
                    idx = idx = np.random.choice(N, M, replace=False)             # M = 2 in REDQ, fresh draw every update
                    q_next = torch.stack([qf_targets[i].get_value(data["next_obs"], next_state_actions) for i in idx]).min(0).values
                    y = data["reward"] + (1 - data["terminated"]) * GAMMA * (q_next - alpha * next_state_log_pi).view(-1)
                q_vals = [qf.get_value(data["obs"], data["action"]).view(-1) for qf in qfs]
                qf_losses = [torch.nn.functional.mse_loss(q, y) for q in q_vals]
                qf_loss = torch.stack(qf_losses).sum()
                q_optimizer.zero_grad()
                qf_loss.backward()
                q_optimizer.step()

            # 2.3) policy network update
            if global_step % POLICY_FREQ == 0:

                if ENABLE_SYNTHETIC and synthetic_buffer is not None:
                    real = real_buffer.sample(int(SAC_BATCH_SIZE * (1.0 - SYNTHETIC_RATIO)))
                    synthetic = synthetic_buffer.sample(int(SAC_BATCH_SIZE * SYNTHETIC_RATIO))
                    data = {k: torch.cat([real[k], synthetic[k]], dim=0) for k in real}
                else:
                    data = real_buffer.sample(SAC_BATCH_SIZE)

                for _ in range(POLICY_FREQ): # Compensate for the delay by doing POLICY_FREQ instead of 1
                    pi, log_pi, _ = actor.get_action(data["obs"])
                    q_mean = torch.stack([qf.get_value(data["obs"], pi) for qf in qfs]).mean(0)
                    actor_loss = (alpha * log_pi - q_mean).mean()
                    actor_optimizer.zero_grad()
                    actor_loss.backward()
                    actor_optimizer.step()

                    # 2.2.1) Update the actor's alpha if it is not a constant
                    if AUTOTUNE:
                        with torch.no_grad():
                            _, log_pi, _ = actor.get_action(data["obs"])
                        alpha_loss = (-log_alpha.exp() * (log_pi + target_entropy)).mean()
                        a_optimizer.zero_grad()
                        alpha_loss.backward()
                        a_optimizer.step()
                        alpha = log_alpha.exp().item()

            # 2.4) Update the target soft Q networks
            if global_step % TARGET_FREQ == 0:
                with torch.no_grad():
                    for qf, qf_t in zip(qfs, qf_targets):
                        for p, tp in zip(qf.parameters(), qf_t.parameters()):
                            tp.data.mul_(1 - TAU).add_(TAU * p.data)

            # 3) Log stats
            if global_step % LOG_FREQ == 0:

                # 3.1) General training stats:
                stats = {
                    "losses/qf_values": sum(qs.mean() for qs in q_vals) / N,
                    "losses/qf_loss": qf_loss.item() / N,
                    "losses/actor_loss": actor_loss.item(),
                    "losses/alpha": alpha,
                }

                # 3.2) Eval episode stats
                if global_step % (LOG_FREQ * 10) == 0:
                    actor.eval()
                    eval_obs, _ = eval_envs.reset(seed=list(range(NUM_EVAL_ENVS)))   # one seed per env

                    alive = np.ones(NUM_EVAL_ENVS, dtype=bool)
                    ep_rewards = np.zeros(NUM_EVAL_ENVS, dtype=np.float64)
                    ep_lengths = np.zeros(NUM_EVAL_ENVS, dtype=np.int64)

                    while alive.any():
                        with torch.no_grad():
                            obs_t = torch.as_tensor(eval_obs, dtype=torch.float32, device=device)
                            _, _, action_mean = actor.get_action(x=obs_t)      # already batched, no unsqueeze
                        eval_obs, reward, terminated, truncated, _ = eval_envs.step(action_mean.cpu().numpy())

                        ep_rewards += reward * alive          # only count envs that were still alive
                        ep_lengths += alive                   # the step that ends the episode still counts
                        alive &= ~(terminated | truncated)    # update AFTER accumulating

                    actor.train()
                    stats["eval/eval_ep_reward"] = ep_rewards.mean()
                    stats["eval/eval_ep_length"] = ep_lengths.mean()

                    pbar.set_postfix(eval_ep_reward = f"{ep_rewards.mean():.4f}",)

                    torch.save(actor.state_dict(), model_path)

                for key, value in stats.items():
                    writer.add_scalar(key, value, global_step/100)

                    