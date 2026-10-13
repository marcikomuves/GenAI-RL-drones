import os
import numpy as np
from tqdm import tqdm
from color import bcolors
import torch
import itertools
from torch.utils.tensorboard import SummaryWriter
from gym_pybullet_drones.utils.enums import ActionType, DroneModel, Physics
import gymnasium as gym
from trajectoryaviary import TrajectoryAviary
from trajectorygenerator import StaticTrajectory
from dronestategenerator import DroneStateGenerator
from rolloutbuffer import RolloutBuffer
from synther import SimpleDiffusionGenerator

# environment params
DRONE_MODEL = DroneModel.CF2X
ACT = ActionType.RPM
PHYSICS = Physics.PYB # Physics.PYB_GND_DRAG_DW
SIMULATION_FREQ_HZ = 240
CONTROL_FREQ_HZ = 48
INIT_XYZ = np.array([0, 0, 1.0])

# training params
RUN_NAME = "latest_real_synth_20utd"
NUM_TRAINING_ENVS = 1               # number of parallel game environments
NUM_STEPS_TOTAL = 100_000           # number of training episodes
WARMUP_LENGTH = 5000                # warmup period length (action sampling without training)
REAL_BUFFER_SIZE = NUM_STEPS_TOTAL // NUM_TRAINING_ENVS
BATCH_SIZE = 256                    # number of transitions to sample for an update
LOG_STD_MAX = 2
LOG_STD_MIN = -5
POLICY_LR = 3e-4                    # learning rate of the policy network optimizer
Q_LR = 3e-4                         # learning rate of the Q network network optimizer
GAMMA = 0.99                        # discount factor
TAU = 5e-3
AUTOTUNE = True                     # flag to toggle automatic tuning of the entropy regularization coefficient
ALPHA = 0.2                         # entropy regularization coefficient
UTD = 20                             # number of critic updates for each environment step
POLICY_FREQ = 1                     # delayed policy update frequency
TARGET_FREQ = 1                     # delayed target q network update frequency
LOG_FREQ = 100

N = 2
M = 2

ENABLE_SYNTHETIC = True
SYNTHETIC_STARTS = 10_000
SYNTHETIC_FREQ = 10_000
NUM_SYNTHETIC_SAMPLES = 200_000
SYNTHETIC_RATIO = 0.5 if ENABLE_SYNTHETIC else 0


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
            torch.nn.LayerNorm(256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 256),
            torch.nn.LayerNorm(256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 1)
        )

    def forward(self, x):
        return self.net(x)

    def get_value(self, x, act):
        return self(torch.cat([x, act], 1))


def make_env():
    return TrajectoryAviary(
        trajectory_generator=StaticTrajectory(xyz=INIT_XYZ, randomize=False),
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

    print(f"[INFO] {bcolors.OKGREEN} Initializing evalutaion environment...{bcolors.ENDC}")
    eval_env = make_env()

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

    model_dir = f"models/sac/{RUN_NAME}/"
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
    model_path = os.path.join(model_dir, f"{RUN_NAME}.cleanrl_model")
    runs_dir = f"runs/sac/{RUN_NAME}/"
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
            normalizer_type="standard",
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
            if ENABLE_SYNTHETIC and global_step >= SYNTHETIC_STARTS and global_step % SYNTHETIC_FREQ == 0:

                if synthetic_buffer is None:

                    synthetic_buffer = RolloutBuffer(
                        max_steps = NUM_SYNTHETIC_SAMPLES,
                        n_envs = 1,
                        obs_shape = train_envs.single_observation_space.shape,
                        act_shape = train_envs.single_action_space.shape,
                        device = device,
                    )

                else:

                    synthetic_buffer.reset()

                generator.retrain(
                    real_buffer=real_buffer,
                    train_steps=100_000,
                    batch_size=256,
                    lr=3e-4,
                    ema_freq=10,
                    ema_after=100,
                    beta=0.995,
                    power=2.0/3.0
                )

                synthetic_sample = zip(
                    *generator.sample(
                        num_samples=NUM_SYNTHETIC_SAMPLES,
                        num_sample_steps=128,
                        sample_batch_size=100_000,
                    )
                )

                for obs_synth, act_synth, reward_synth, next_obs_synth, terminated_synth in synthetic_sample:
                    synthetic_buffer.append(obs_synth, act_synth, reward_synth, terminated_synth, False, next_obs_synth)

            # 2.2) soft Q network updates
            for epoch in range(UTD):
                
                if ENABLE_SYNTHETIC and synthetic_buffer is not None:
                    real = real_buffer.sample(int(BATCH_SIZE * (1.0 - SYNTHETIC_RATIO)))
                    synthetic = synthetic_buffer.sample(int(BATCH_SIZE * SYNTHETIC_RATIO))
                    data = {k: torch.cat([real[k], synthetic[k]], dim=0) for k in real}
                else:
                    data = real_buffer.sample(BATCH_SIZE)

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
                    real = real_buffer.sample(int(BATCH_SIZE * (1.0 - SYNTHETIC_RATIO)))
                    synthetic = synthetic_buffer.sample(int(BATCH_SIZE * SYNTHETIC_RATIO))
                    data = {k: torch.cat([real[k], synthetic[k]], dim=0) for k in real}
                else:
                    data = real_buffer.sample(BATCH_SIZE)

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
                    eval_obs, _ = eval_env.reset(seed=0)
                    eval_done, eval_ep_rewards, eval_ep_length = False, 0.0, 0
                    while not eval_done:
                        with torch.no_grad():
                            _, _, action_mean = actor.get_action(x=torch.as_tensor(eval_obs, dtype=torch.float32, device=device).unsqueeze(0))
                        eval_obs, reward, terminated, truncated, _ = eval_env.step(action_mean[0].cpu().numpy())
                        eval_ep_rewards += reward
                        eval_ep_length += 1
                        eval_done = terminated or truncated
                    stats["eval/eval_ep_reward"] = eval_ep_rewards
                    stats["eval/eval_ep_length"] = eval_ep_length

                    pbar.set_postfix(eval_ep_reward = f"{eval_ep_rewards:.4f}",)

                    torch.save(actor.state_dict(), model_path)

                for key, value in stats.items():
                    writer.add_scalar(key, value, global_step/100)

                    