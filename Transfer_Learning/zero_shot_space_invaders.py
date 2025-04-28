import os
import imageio
import ale_py
import gymnasium as gym
import numpy as np
import torch
import matplotlib.pyplot as plt
from transfer_learning_dqn import DQN, ReplayMemory
from wrapper import NoopResetEnv, MaxAndSkipEnv, WarpFrame, FrameStack, ClipRewardEnv, FireResetEnv, EpisodicLifeEnv

# 1) Register and build wrapped env
gym.register_envs(ale_py)
def make_env(env_id, seed=None):
    env = gym.make(env_id, render_mode="rgb_array")
    env = NoopResetEnv(env, noop_max=30)
    env = MaxAndSkipEnv(env, skip=4)
    env = EpisodicLifeEnv(env)
    env = FireResetEnv(env)
    env = WarpFrame(env)
    env = ClipRewardEnv(env)
    env = FrameStack(env, k=4)
    if seed is not None:
        env.reset(seed=seed)
    return env

# 2) Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
env = make_env("ALE/SpaceInvaders-v5", seed=123)

# 3) Dummy memory + agent instantiation
memory = ReplayMemory(capacity=1)
agent = DQN(
    observation_space=env.observation_space,
    action_space_dict={ "spaceinvaders": env.action_space },
    memory_capacity=1,
    learning=1e-4,
    batch_size=1,
    gamma=0.99,
    cpu=device
)

# 4) Load pre-trained shared-conv weights
checkpoint = "/projectnb/dl4ds/students/rsingh13/models/multihead_transfer_proportional_dqn.pth"
loaded = torch.load(checkpoint, map_location=device)
model_dict = agent.policy_net.state_dict()
filtered = {k: v for k, v in loaded.items()
            if k in model_dict and model_dict[k].shape == v.shape}
model_dict.update(filtered)
agent.policy_net.load_state_dict(model_dict)
agent.update_target_network()
print(f"✔️ Loaded {len(filtered)}/{len(loaded)} params")

# 5) Zero-shot evaluation + video every 5 eps
num_episodes = 100
returns = []
os.makedirs("videos", exist_ok=True)

def record_video_episode(env, agent, ep):
    out_dir = "videos"
    frames = []
    obs, _ = env.reset()
    frames.append(env.render())
    done = False
    while not done:
        a = agent.decide_action(np.array(obs), "spaceinvaders")
        obs, r, term, trunc, _ = env.step(a)
        done = term or trunc
        frames.append(env.render())
    path = os.path.join(out_dir, f"spaceinvaders_ep{ep:02d}.mp4")
    imageio.mimsave(path, frames, fps=30)
    print(f"  ↳ recorded zero-shot video: {path}")

for ep in range(1, num_episodes+1):
    obs, _ = env.reset()
    total_r, done = 0.0, False
    while not done:
        a = agent.decide_action(np.array(obs), "spaceinvaders")
        obs, r, term, trunc, _ = env.step(a)
        done = term or trunc
        total_r += r
    returns.append(total_r)
    print(f"Zero-shot Ep {ep:2d}: Return = {total_r:.1f}")
    if ep % 10 == 0:
        record_video_episode(env, agent, ep)

env.close()

# 6) Summary
mean_return = np.mean(returns)
print("─"*30)
print(f"Average zero-shot return: {mean_return:.2f}")
print("─"*30)

# 7) Plot episode returns
plt.figure(figsize=(8,4))
plt.plot(range(1, num_episodes+1), returns, marker='o')
plt.title("Zero-Shot SpaceInvaders Returns")
plt.xlabel("Episode")
plt.ylabel("Return")
plt.grid(True)
plt.tight_layout()
plt.savefig("zero_shot_spaceinvaders_returns.png")
plt.show()
