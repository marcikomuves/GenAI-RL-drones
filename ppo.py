# train_ppo.py
import numpy as np
from tqdm import tqdm
import os
import itertools
import torch
from torch.utils.tensorboard import SummaryWriter
from gym_pybullet_drones.utils.enums import ActionType, DroneModel, Physics
import gymnasium as gym
from targetaviary import TargetAviary
from targetstategenerator import StaticTargetStateGenerator
from dronestategenerator import DroneStateGenerator
from rolloutbuffer import RolloutBuffer

# environment params
DRONE_MODEL = DroneModel.CF2X             
ACT = ActionType.RPM
PHYSICS = Physics.PYB
SIMULATION_FREQ_HZ = 240
CONTROL_FREQ_HZ = 48
INIT_XYZ = np.array([0, 0, 1.0])

# training params
NEW_POLICY_NAME = "latest.cleanrl_model"
NUM_TRAINING_ENVS=16
MAX_STEPS_EP = 512
NUM_TRAINING_EPS = 200
BATCH_SIZE = NUM_TRAINING_ENVS * MAX_STEPS_EP
MINIBATCH_SIZE = 256 
NUM_MINIBATCHES = BATCH_SIZE//MINIBATCH_SIZE # = 16 * 512 / 256 = 32
UPDATE_EPOCHS = 4
LR = 2.5e-4
GAMMA = 0.99
LAMBDA = 0.95
EPSILON = 0.2
ENT_COEF = 0.01
VALUE_COEF = 0.5
MAX_GRAD_NORM = 0.5
KL_THRESHOLD = 0.02
CLIP_VLOSS = True
NORMALIZE_ADVANTAGES = True
ANNEAL_LR = True
KL_EARLY_STOPPING = True


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Actor(torch.nn.Module):
    def __init__(self, act_space, obs_space):
        super().__init__()
        self.mean = torch.nn.Sequential(
            layer_init(torch.nn.Linear(np.prod(obs_space.shape), 64)),
            torch.nn.Tanh(),
            layer_init(torch.nn.Linear(64, 64)),
            torch.nn.Tanh(),
            layer_init(torch.nn.Linear(64, np.prod(act_space.shape)), std=0.01))
        self.logstd = torch.nn.Parameter(torch.zeros(np.prod(act_space.shape)))

    def _distribution(self, x):
        action_mean = self.mean(x)
        action_logstd = self.logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        return torch.distributions.Normal(action_mean, action_std), action_mean

    def get_action(self, x, action=None):
        probs, action_mean = self._distribution(x)
        if action is None:
            action = probs.sample()
        log_prob = probs.log_prob(action).sum(-1)
        return action, log_prob, action_mean


class Critic(torch.nn.Module):
    def __init__(self, obs_space):
        super().__init__()
        self.net = torch.nn.Sequential(
            layer_init(torch.nn.Linear(np.prod(obs_space.shape), 64)),
            torch.nn.Tanh(),
            layer_init(torch.nn.Linear(64, 64)),
            torch.nn.Tanh(),
            layer_init(torch.nn.Linear(64, 1), std=1.0))

    def get_value(self, x):
        return self.net(x).flatten()



def make_env():
    return TargetAviary(
        trajectory_generator=StaticTargetStateGenerator(xyz=INIT_XYZ, randomize=False),
        init_state_generator=DroneStateGenerator(init_xyz=INIT_XYZ, randomize=False),
        max_t=10, 
        act=ACT, 
        physics=PHYSICS, 
        drone_model=DRONE_MODEL,
        pyb_freq=SIMULATION_FREQ_HZ, 
        ctrl_freq=CONTROL_FREQ_HZ,
    )

if __name__ == "__main__":

    print("[INFO] Initializing training environments...")
    train_envs = gym.vector.SyncVectorEnv(
        [make_env for _ in range(NUM_TRAINING_ENVS)], 
        autoreset_mode=gym.vector.AutoresetMode.SAME_STEP,
    )
    print("[INFO] Initializing evalutaion environment...")
    eval_env = make_env()

    print("[INFO] Initializing agent...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")
    #agent = PPOAgent(
    #    device=device,
    #    obs_dim=train_envs.single_observation_space.shape[0],
    #    act_dim=train_envs.single_action_space.shape[0]
    #)
    actor = Actor(act_space=train_envs.single_action_space, obs_space=train_envs.single_observation_space).to(device)
    critic = Critic(obs_space=train_envs.single_observation_space).to(device)
    optimizer = torch.optim.Adam(itertools.chain(actor.parameters(), critic.parameters()), lr=LR, eps=1e-5)

    writer = SummaryWriter("runs/ppo/")

    print("[INFO] Starting training...")
    buffer = RolloutBuffer(
        max_steps=MAX_STEPS_EP,
        n_envs=NUM_TRAINING_ENVS,
        obs_shape=train_envs.single_observation_space.shape,
        act_shape=train_envs.single_action_space.shape,
        device=device,
        store_logprob_value=True
    )

    obs, _ = train_envs.reset(seed=0)
    for i in tqdm(range(1, NUM_TRAINING_EPS + 1)):
        actor.train()

        # 1) LR tuning
        if ANNEAL_LR:
            frac = 1.0 - (i - 1.0) / NUM_TRAINING_EPS
            lrnow = frac * LR
            optimizer.param_groups[0]["lr"] = lrnow

        # 2) Training episode (rollout)
        for step in range(MAX_STEPS_EP):
            obs = torch.as_tensor(obs, dtype=torch.float32, device=device)    
            with torch.no_grad():
                action, logprob, _ = actor.get_action(obs)
                value = critic.get_value(obs)
            next_obs, reward, terminated, truncated, info = train_envs.step(action.cpu().numpy())
            true_next_obs = next_obs.copy()
            for idx, done in enumerate(np.logical_or(terminated, truncated)):
                if done:
                    true_next_obs[idx] = info["final_obs"][idx]
            buffer.append(obs, action, reward, terminated, truncated, true_next_obs, logprob, value)
            obs = torch.tensor(next_obs, dtype=torch.float32, device=device)

        # 3) compute GAE
        data = buffer.ordered()
        advantages = torch.zeros_like(data["reward"])
        with torch.no_grad():
            next_obs_flat = data["next_obs"].reshape(-1, data["next_obs"].shape[-1])
            next_values = critic.get_value(next_obs_flat).reshape(MAX_STEPS_EP, NUM_TRAINING_ENVS)
        lastgaelam = torch.zeros(NUM_TRAINING_ENVS, device=device)
        for t in reversed(range(MAX_STEPS_EP)):
            next_nonterminal = 1.0 - data["terminated"][t]
            next_nondone = 1.0 - torch.maximum(data["terminated"][t], data["truncated"][t])
            delta = data["reward"][t] + GAMMA * next_values[t] * next_nonterminal - data["value"][t]
            lastgaelam = delta + GAMMA * LAMBDA * next_nondone * lastgaelam
            advantages[t] = lastgaelam
        returns = advantages + data["value"]

        # 4) Prepare the batch
        b_obs        = data["obs"].reshape((-1,) + data["obs"].shape[2:])
        b_actions    = data["action"].reshape((-1,) + data["action"].shape[2:])
        b_logprobs   = data["logprob"].reshape(-1)
        b_values     = data["value"].reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_returns    = returns.reshape(-1)

        # 5) PPO update
        clipfracs = []
        for _ in range(UPDATE_EPOCHS):
            idxs = np.random.permutation(BATCH_SIZE)
            minibatch_kls = []
            for start in range(0, BATCH_SIZE, MINIBATCH_SIZE):
                mb = idxs[start:start + MINIBATCH_SIZE]
                # _, newlogprob, entropy = actor.get_action(b_obs[mb], b_actions[mb])
                probs, _ = actor._distribution(b_obs[mb])
                newlogprob = probs.log_prob(b_actions[mb]).sum(-1)
                entropy = probs.entropy().sum(-1)
                newvalue = critic.get_value(b_obs[mb])
                logratio = newlogprob - b_logprobs[mb]
                ratio = logratio.exp()
                with torch.no_grad():
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                clipfracs.append(((ratio - 1.0).abs() > EPSILON).float().mean().item())
                minibatch_kls.append(approx_kl.item())
                mb_adv = b_advantages[mb]
                if NORMALIZE_ADVANTAGES:
                    mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)
                pg_loss = torch.max(
                    -mb_adv * ratio,
                    -mb_adv * torch.clamp(ratio, 1 - EPSILON, 1 + EPSILON)
                ).mean()
                #newvalue = newvalue.view(-1)
                if CLIP_VLOSS:
                    v_loss_unclipped = (newvalue - b_returns[mb]) ** 2
                    v_clipped = b_values[mb] + torch.clamp(newvalue - b_values[mb], -EPSILON, EPSILON)
                    v_loss_clipped = (v_clipped - b_returns[mb]) ** 2
                    v_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb]) ** 2).mean()
                entropy_loss = entropy.mean()
                loss = pg_loss - ENT_COEF * entropy_loss + v_loss * VALUE_COEF
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(itertools.chain(actor.parameters(), critic.parameters()), MAX_GRAD_NORM)
                optimizer.step()
            epoch_kl = np.mean(minibatch_kls)
            if KL_EARLY_STOPPING and epoch_kl > KL_THRESHOLD:
                break
        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
        
        # 6) Evaluation episode
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

        stats = {
            "charts/learning_rate": optimizer.param_groups[0]["lr"],
            "losses/value_loss": v_loss.item(),
            "losses/policy_loss": pg_loss.item(),
            "losses/entropy": entropy_loss.item(),
            "losses/old_approx_kl": old_approx_kl.item(),
            "losses/approx_kl": approx_kl.item(),
            "losses/clipfrac": np.mean(clipfracs),
            "losses/explained_variance": explained_var,
            "eval/eval_ep_reward": eval_ep_rewards,
            "eval/eval_ep_length": eval_ep_length
        }

        for key, value in stats.items():
            writer.add_scalar(key, value, i)
    dir = "./models/ppo/"
    if not os.path.exists(dir):
        os.mkdir(dir)
    path = os.path.join(dir, NEW_POLICY_NAME)
    print(f"[INFO] Saving actor parameters to {path}...")
    torch.save(actor.state_dict(), path)