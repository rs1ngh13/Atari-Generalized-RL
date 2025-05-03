import torch
import numpy as np
import gymnasium as gym
import ale_py
from collections import deque
from multi_game_PPO import (
    UnifiedActorCritic,
    make_env,
    preprocess_obs,
)
from multi_game_PPO import record_video_episode
import matplotlib.pyplot as plt
import os
import imageio


# 1. Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2. Recompute max_action_dim or hardcode from training
#    (e.g., you found max_action_dim = 9 when training on Pong, Breakout, BeamRider)
max_action_dim = 9  

# 3. Instantiate and load model
model = UnifiedActorCritic(in_channels=4, action_dim=max_action_dim).to(device)
ckpt = torch.load("multi_game_ppo_output/models/multi_game_ppo_final.pt", map_location=device)
# if you saved a dict with 'model_state_dict', use that key:
state_dict = ckpt.get("Code/Python_Files/temp/multi_game_ppo_output/models/multi_game_ppo_final.pt", ckpt)
model.load_state_dict(state_dict)
model.eval()

# 4. Build SpaceInvaders env with identical wrappers
gym.register_envs(ale_py)
env = make_env("ALE/SpaceInvaders-v5", render_mode=None)
action_dim = env.action_space.n  # e.g. 6

def run_zero_shot_episodes(model, env, action_dim, n_episodes=30):
    rewards = []
    for ep in range(n_episodes):
        obs, _ = env.reset()
        obs = preprocess_obs(obs)
        done = False
        total_reward = 0
        while not done:
            with torch.no_grad():
                logits, _ = model(obs, game="space_invaders", valid_actions=action_dim)
                action = torch.argmax(logits, dim=1).item()
            obs_next, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
            obs = preprocess_obs(obs_next)
        rewards.append(total_reward)
        print(f"Episode {ep+1}: reward = {total_reward}")
    avg = np.mean(rewards)
    print(f"Average zero-shot reward over {n_episodes} episodes: {avg:.2f}")


def record_best_zero_shot_video_and_plot(
    model,
    game: str,
    env_id: str,
    action_dim: int,
    n_episodes: int = 100,
    tag: str = "best_zero_shot",
    output_dir: str = "multi_game_ppo_output"
):
    """
    Runs n_episodes zero-shot in `game`, tracks raw & clipped returns,
    saves the highest-raw-reward episode as a video, and plots both series
    side-by-side.
    """
    # 1) RGB env for recording
    env = make_env(env_id, render_mode="rgb_array")

    best_reward = -float("inf")
    best_frames = None

    raw_returns = []
    clipped_returns = []

    for ep in range(n_episodes):
        obs, _ = env.reset()
        obs = preprocess_obs(obs)
        done = False

        total_raw = 0
        total_clipped = 0
        frames = []

        while not done:
            frames.append(env.render())
            with torch.no_grad():
                logits, _ = model(obs, game, valid_actions=action_dim)
                action = torch.argmax(logits, dim=1).item()

            obs, reward, term, trunc, _ = env.step(action)
            obs = preprocess_obs(obs)

            total_raw += reward
            total_clipped += np.sign(reward)
            done = term or trunc

        raw_returns.append(total_raw)
        clipped_returns.append(total_clipped)
        print(f"[{game}] Ep {ep+1:2d} raw={total_raw:4.0f} clipped={total_clipped:4.0f}")

        if total_raw > best_reward:
            best_reward = total_raw
            best_frames = frames

    # 2) Plot raw vs clipped in subplots
    fig, (ax_raw, ax_clip) = plt.subplots(1, 2, figsize=(12, 4), sharex=True)
    episodes = np.arange(1, n_episodes + 1)

    ax_raw.plot(episodes, raw_returns, marker="o", linestyle="-")
    ax_raw.set_title("Raw Returns")
    ax_raw.set_xlabel("Episode")
    ax_raw.set_ylabel("Total Reward")

    ax_clip.plot(episodes, clipped_returns, marker="s", linestyle="--")
    ax_clip.set_title("Clipped Returns")
    ax_clip.set_xlabel("Episode")
    # share y-label only on first subplot

    plt.suptitle(f"Zero‐Shot Performance: {game}", y=1.02)
    plt.tight_layout()

    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    plot_path = os.path.join(plots_dir, f"{game}_{tag}_subplots.png")
    plt.savefig(plot_path)
    plt.close()
    print(f"→ Saved subplot figure to {plot_path}")

    # 3) Write best-episode video
    if best_frames:
        vid_dir = os.path.join(output_dir, "videos", f"{game}_{tag}")
        os.makedirs(vid_dir, exist_ok=True)
        vid_path = os.path.join(vid_dir, f"ppo_{game}_{tag}.mp4")
        imageio.mimsave(vid_path, best_frames, fps=30)
        print(f"→ Saved best‐reward video ({best_reward:.0f} pts) to {vid_path}")
    else:
        print("No frames captured—did something go wrong?")

    env.close()



if __name__ == "__main__":
    # run_zero_shot_episodes(model, env, action_dim, n_episodes=20)
    record_best_zero_shot_video_and_plot(
        model=model,
        game="space_invaders",
        env_id="ALE/SpaceInvaders-v5",
        action_dim=action_dim,
        n_episodes=100,
        tag="best_zero_shot"
    )
    
    # env.close()
