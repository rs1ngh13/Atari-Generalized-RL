import gymnasium as gym
import ale_py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
import matplotlib.pyplot as plt
import multiprocessing
from collections import deque

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ParallelEnv:
    def __init__(self, env_name, num_envs=8, render_mode=None):
        self.envs = []
        for _ in range(num_envs):
            env = gym.make(env_name, render_mode=render_mode)
            env = gym.wrappers.ResizeObservation(env, (84, 84))
            env = gym.wrappers.GrayscaleObservation(env)
            env = gym.wrappers.FrameStackObservation(env, 4)
            env = gym.wrappers.RecordEpisodeStatistics(env)
            self.envs.append(env)

        self.num_envs = num_envs
        self.observation_space = self.envs[0].observation_space
        self.action_space = self.envs[0].action_space

    def reset(self):
        observations = []
        infos = []
        for env in self.envs:
            obs, info = env.reset()
            observations.append(obs)
            infos.append(info)
        return np.array(observations), infos

    def step(self, actions):
        next_obs_list = []
        reward_list = []
        terminated_list = []
        truncated_list = []
        info_list = []

        for env_idx, env in enumerate(self.envs):
            next_obs, reward, terminated, truncated, info = env.step(actions[env_idx])
            if terminated or truncated:
                next_obs, _ = env.reset()
            next_obs_list.append(next_obs)
            reward_list.append(reward)
            terminated_list.append(terminated)
            truncated_list.append(truncated)
            info_list.append(info)

        return np.array(next_obs_list), np.array(reward_list), np.array(terminated_list), np.array(truncated_list), info_list

    def close(self):
        for env in self.envs:
            env.close()


class ActorCritic(nn.Module):
    def __init__(self, in_channels=4, num_actions=6):
        super(ActorCritic, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.ReLU()
        )

        for layer in self.conv:
            if isinstance(layer, nn.Conv2d):
                nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
                nn.init.zeros_(layer.bias)

        with torch.no_grad():
            dummy_input = torch.zeros(1, in_channels, 84, 84)
            conv_out_size = self.conv(dummy_input).flatten(1).shape[1]

        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU()
        )

        self.policy = nn.Linear(512, num_actions)
        self.value = nn.Linear(512, 1)

        for layer in self.fc:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
                nn.init.zeros_(layer.bias)

        nn.init.orthogonal_(self.policy.weight, gain=0.01)
        nn.init.zeros_(self.policy.bias)
        nn.init.orthogonal_(self.value.weight, gain=1.0)
        nn.init.zeros_(self.value.bias)

    def forward(self, x):
        x = self.conv(x)
        x = x.flatten(1)
        x = self.fc(x)
        policy_logits = self.policy(x)
        value = self.value(x)
        return policy_logits, value


def preprocess_obs(obs):
    if isinstance(obs, np.ndarray):
        obs = torch.FloatTensor(obs)
    if obs.max() > 1.0:
        obs = obs / 255.0
    return obs.to(device)


def collect_rollouts(envs, model, n_steps=128):
    num_envs = envs.num_envs
    obs, _ = envs.reset()
    obs = preprocess_obs(obs)

    observations = torch.zeros((n_steps, num_envs) + obs.shape[1:], dtype=torch.float32, device=device)
    actions = torch.zeros((n_steps, num_envs), dtype=torch.long, device=device)
    logprobs = torch.zeros((n_steps, num_envs), dtype=torch.float32, device=device)
    rewards = torch.zeros((n_steps, num_envs), dtype=torch.float32, device=device)
    dones = torch.zeros((n_steps, num_envs), dtype=torch.float32, device=device)
    values = torch.zeros((n_steps, num_envs), dtype=torch.float32, device=device)

    episode_rewards = []
    episode_lengths = []
    current_episode_rewards = torch.zeros(num_envs, dtype=torch.float32, device=device)
    current_episode_lengths = torch.zeros(num_envs, dtype=torch.int, device=device)

    for step in range(n_steps):
        observations[step] = obs

        with torch.no_grad():
            logits, value = model(obs)
            dist = Categorical(logits=logits)
            action = dist.sample()
            logprob = dist.log_prob(action)

        values[step] = value.squeeze()
        actions[step] = action
        logprobs[step] = logprob

        next_obs, reward, terminated, truncated, infos = envs.step(action.cpu().numpy())
        done = np.logical_or(terminated, truncated)

        if np.any(done):
            print(f"[DEBUG] Step {step} — Done detected in environments: {np.where(done)[0]}")

        rewards[step] = torch.tensor(reward, dtype=torch.float32, device=device)
        dones[step] = torch.tensor(done, dtype=torch.float32, device=device)
        current_episode_rewards += torch.tensor(reward, dtype=torch.float32, device=device)
        current_episode_lengths += 1

        for i, d in enumerate(done):
            if d:
                print(f"[DEBUG] Episode completed in env {i} | Reward: {current_episode_rewards[i].item()} | Length: {current_episode_lengths[i].item()}")
                episode_rewards.append(current_episode_rewards[i].item())
                episode_lengths.append(current_episode_lengths[i].item())
                current_episode_rewards[i] = 0
                current_episode_lengths[i] = 0

        obs = preprocess_obs(next_obs)

    with torch.no_grad():
        next_value = model(obs)[1].squeeze()

    returns = torch.zeros_like(rewards)
    advantages = torch.zeros_like(rewards)
    gae = torch.zeros(num_envs, device=device)

    for t in reversed(range(n_steps)):
        if t == n_steps - 1:
            next_val = next_value
            next_non_terminal = 1.0 - torch.tensor(done, dtype=torch.float32, device=device)
        else:
            next_val = values[t + 1]
            next_non_terminal = 1.0 - dones[t + 1]

        delta = rewards[t] + 0.99 * next_val * next_non_terminal - values[t]
        gae = delta + 0.99 * 0.95 * next_non_terminal * gae
        advantages[t] = gae
        returns[t] = gae + values[t]

    batch = {
        "observations": observations.reshape(-1, *observations.shape[2:]),
        "actions": actions.reshape(-1),
        "logprobs": logprobs.reshape(-1),
        "returns": returns.reshape(-1),
        "advantages": advantages.reshape(-1),
        "values": values.reshape(-1),
        "episode_rewards": episode_rewards,
        "episode_lengths": episode_lengths
    }

    return batch


def update_policy(model, optimizer, batch, value_coef=0.5, entropy_coef=0.01):
    observations = batch["observations"]
    actions = batch["actions"]
    returns = batch["returns"]
    advantages = batch["advantages"]

    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    logits, values = model(observations)
    dist = Categorical(logits=logits)
    logprobs = dist.log_prob(actions)
    entropy = dist.entropy().mean()

    policy_loss = -(logprobs * advantages).mean()
    value_loss = 0.5 * ((values.squeeze() - returns) ** 2).mean()
    loss = policy_loss + value_coef * value_loss - entropy_coef * entropy

    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.6)
    optimizer.step()

    return policy_loss.item(), value_loss.item(), entropy.item()


def linear_schedule(initial_value, final_value=0.0):
    def scheduler(progress):
        return final_value + (initial_value - final_value) * (1 - progress)
    return scheduler


def train():
    num_envs = 8
    n_steps = 2048
    total_updates = 10000

    envs = ParallelEnv("ALE/Pong-v5", num_envs=num_envs)
    in_channels = 4
    num_actions = envs.action_space.n

    model = ActorCritic(in_channels, num_actions)
    model = nn.DataParallel(model)
    model = model.to(device)
    initial_lr = 7e-4
    final_lr = 7e-4
    lr_scheduler = linear_schedule(initial_lr, final_lr)
    optimizer = optim.RMSprop(model.parameters(), lr=initial_lr, eps=1e-5, alpha=0.99)
    entropy_coef = 0.01

    training_returns = []
    episode_returns = []
    episode_lengths = []
    moving_avg_return = deque(maxlen=100)

    for update in range(1, total_updates + 1):
        progress = update / total_updates
        lr = lr_scheduler(progress)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        batch = collect_rollouts(envs, model, n_steps)

        policy_loss, value_loss, entropy = update_policy(model, optimizer, batch, entropy_coef=entropy_coef)

        if batch["episode_rewards"]:
            episode_returns.extend(batch["episode_rewards"])
            episode_lengths.extend(batch["episode_lengths"])
            moving_avg_return.extend(batch["episode_rewards"])
        else:
            print("[WARNING] No episodes completed during this update.")

        avg_return = batch["returns"].mean().item()
        training_returns.append(avg_return)

        print(f"[INFO] Update {update} — Avg reward: {avg_return:.3f}, Completed episodes: {len(batch['episode_rewards'])}")

        if update % 10 == 0:
            moving_avg = np.mean(moving_avg_return) if moving_avg_return else 0
            print(f"Update {update:04d} | Frames: {update * num_envs * n_steps:,} | "
                  f"Loss: {policy_loss:.3f} | Value Loss: {value_loss:.3f} | Entropy: {entropy:.3f} | "
                  f"Rolling Ep Return: {moving_avg:.3f} | LR: {lr:.6f}")

        if update % 100 == 0:
            plt.figure(figsize=(15, 5))

            plt.subplot(1, 3, 1)
            plt.plot(training_returns)
            plt.title("Recent Training Returns")
            plt.xlabel("Updates")
            plt.ylabel("Average Return")

            plt.subplot(1, 3, 2)
            if episode_returns:
                if len(episode_returns) > 100:
                    moving_avgs = [np.mean(episode_returns[max(0, i - 100):i]) for i in range(1, len(episode_returns) + 1)]
                    plt.plot(moving_avgs)
                else:
                    plt.plot(episode_returns)
                plt.title("Episode Returns (100-ep moving avg if available)")
                plt.xlabel("Episodes")
                plt.ylabel("Return")
            else:
                plt.text(0.5, 0.5, "No episodes completed yet", horizontalalignment='center', verticalalignment='center')

            plt.subplot(1, 3, 3)
            if episode_lengths:
                plt.plot(episode_lengths[-1000:] if len(episode_lengths) > 1000 else episode_lengths)
                plt.title("Episode Lengths")
                plt.xlabel("Episodes")
                plt.ylabel("Length")
            else:
                plt.text(0.5, 0.5, "No episodes completed yet", horizontalalignment='center', verticalalignment='center')

            plt.tight_layout()
            plt.savefig(f"a2c_parallel_training_plot.png")
            plt.close()

        if update % 1000 == 0:
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'update': update,
                'frames': update * num_envs * n_steps,
                'returns': training_returns,
                'episode_returns': episode_returns
            }, f"a2c_parallel_checkpoint.pt")

    torch.save(model.state_dict(), "a2c_parallel_final_model.pt")
    envs.close()
    return model, training_returns, episode_returns


if __name__ == "__main__":
    if torch.cuda.is_available():
        torch.set_num_threads(1)
    else:
        num_threads = max(1, multiprocessing.cpu_count() - 2)
        torch.set_num_threads(num_threads)

    print(f"Training on device: {device}")
    print(f"Number of CPU threads: {torch.get_num_threads()}")
    model, training_returns, episode_returns = train()
