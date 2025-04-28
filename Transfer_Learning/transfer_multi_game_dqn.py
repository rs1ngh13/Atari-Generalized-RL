import os
import imageio
import ale_py
import gymnasium as gym
from gymnasium import spaces
import torch
import numpy as np
import random
import matplotlib.pyplot as plt
from wrapper import NoopResetEnv, MaxAndSkipEnv, WarpFrame, FrameStack, ClipRewardEnv, FireResetEnv, EpisodicLifeEnv
from transfer_learning_dqn import DQN

# choose device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# set seed for reproducibility
seed = 42
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)

# create video output folder
os.makedirs("videos", exist_ok=True)

def record_video_episode(env, agent, game, tag):
    out_dir = f"videos/{game}_{tag}"
    os.makedirs(out_dir, exist_ok=True)
    frames = []
    obs, _ = env.reset()
    done = False
    while not done:
        frames.append(env.render())
        action = agent.decide_action(np.array(obs), game)
        obs, _, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
    video_path = os.path.join(out_dir, f"{game}_step{tag}.mp4")
    imageio.mimsave(video_path, frames, fps=30)
    print(f"saved video to {video_path}")

# register ALE environments
gym.register_envs(ale_py)

# define and wrap environments
env_ids = {
    "pong":      "ALE/Pong-v5",
    "breakout":  "ALE/Breakout-v5",
    "beamrider": "ALE/BeamRider-v5",
}
envs = {}
for name, game_id in env_ids.items():
    env = gym.make(game_id, render_mode="rgb_array")
    env = NoopResetEnv(env, noop_max=30)
    env = MaxAndSkipEnv(env, skip=4)
    env = EpisodicLifeEnv(env)
    env = FireResetEnv(env)
    env = WarpFrame(env)
    env = ClipRewardEnv(env)
    env = FrameStack(env, k=4)
    envs[name] = env

# hyperparameters for multi-task regular DQN
buffer_size        = 250000  # per-game capacity
learning_rate      = 1e-4      # lower LR for stability
gamma              = 0.99
total_steps        = 5000000
batch_size         = 64        # larger batch
begin_learning     = 15000     # warm up each buffer
update_target      = 10000
epsilon_start      = 1.0
epsilon_end        = 0.05      # explore longer
epsilon_decay_rate = 0.5
print_every        = 500

epsilon_decay = epsilon_decay_rate * total_steps

# build agent
action_space_dict = {g: envs[g].action_space for g in envs}
agent = DQN(
    observation_space=envs['pong'].observation_space,
    action_space_dict=action_space_dict,
    memory_capacity=buffer_size,
    learning=learning_rate,
    batch_size=batch_size,
    gamma=gamma,
    cpu=device
)

# trackers
rewards_per_episode = {g: [] for g in envs}
total_reward        = {g: 0  for g in envs}
episodes            = {g: 0  for g in envs}
observations        = {g: env.reset()[0] for g, env in envs.items()}

# training loop with proportional scheduling
episode_counts = episodes.copy()
for step in range(1, total_steps + 1):
    # dynamic sampling weights inverse to episodes played
    weights = [1.0/(episode_counts[g] + 1) for g in envs]
    game = random.choices(list(envs.keys()), weights=weights)[0]
    env = envs[game]
    obs = observations[game]

    # epsilon decay
    progress = min(1.0, step / epsilon_decay)
    epsilon = epsilon_start + progress * (epsilon_end - epsilon_start)

    # select action
    if random.random() < epsilon:
        action = env.action_space.sample()
    else:
        action = agent.decide_action(np.array(obs), game)

    # step environment
    next_obs, reward, terminated, truncated, _ = env.step(action)
    done = terminated or truncated

    # store transition
    agent.memories[game].push(obs, action, reward, next_obs, done)
    total_reward[game] += reward
    observations[game] = next_obs

    # learning and target update
    if step >= begin_learning and len(agent.memories[game]) >= batch_size:
        agent.update_q_network(game)
        if step % update_target == 0:
            agent.update_target_network()

    # record videos at milestones
    if step in [500000,1000000,1500000,2000000,2500000,3000000,3500000,4000000,4500000,5000000]:
        for g in envs:
            record_video_episode(envs[g], agent, g, f"step_{step}")

    # end of episode
    if done:
        rewards_per_episode[game].append(total_reward[game])
        episode_counts[game] += 1
        episodes[game] += 1
        total_reward[game] = 0
        observations[game], _ = env.reset()

        if episodes[game] % print_every == 0:
            recent = rewards_per_episode[game][-100:]
            avg    = np.mean(recent) if recent else float('nan')
            print("-------------------------------------------------------")
            print(f"Step: {step}, {game.capitalize()} Episode: {episodes[game]}")
            print(f"Avg Reward (last 100): {avg:.2f}")
            print(f"Exploration: {epsilon*100:.1f}%")
            print("-------------------------------------------------------")

# cleanup
for env in envs.values():
    env.close()

# save model
os.makedirs("models", exist_ok=True)
agent.save("models/multihead_transfer_proportional_dqn.pth")
print("Saved proportional-scheduled multi-task DQN to models/multihead_transfer_proportional_dqn.pth")

# plot rewards
for g, r in rewards_per_episode.items():
    plt.plot(range(1, len(r)+1), r, label=g)
plt.title("Episode Rewards Multi-Game Proportional Scheduling (DQN)")
plt.xlabel("Episode count per game")
plt.ylabel("Clipped Reward")
plt.legend()
plt.tight_layout()
plt.savefig("episode_rewards_proportional_dqn.png")
plt.show()
