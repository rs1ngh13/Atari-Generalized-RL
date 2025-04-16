import gymnasium as gym
from gymnasium import spaces
import ale_py
import torch
import numpy as np
import random
import matplotlib.pyplot as plt
from wrapper import RandomNoopWrapper, FrameSkipMaxPoolWrapper, FrameWarper, FrameStacker, RewardClipperWrapper, FireResetEnv, EpisodicLifeEnv
from dqn import DQN, ReplayMemory
import argparse

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

hyper_params = {
    "seed": 42,
    "env": "ALE/Pong-v5",
    "replay-buffer-size": 100000,          # Maximum capacity of Replay Memory
    "learning-rate": 1e-4,
    "discount-factor": 0.99,
    "num-steps": int(1.5e6),                # Total number of environment steps
    "batch-size": 32,
    "learning-starts": 10000,             # Start updates only after these many steps
    "learning-freq": 1,                   # Optimize every step once learning starts
    "target-update-freq": 1000,           # Update target network every these many steps
    "eps-start": 1.0,                     # Initial epsilon for exploration
    "eps-end": 0.01,                      # Final epsilon after decay
    "eps-fraction": 0.1,                  # Fraction of total steps over which to decay epsilon
    "print-freq": 10                      # Print stats every print-freq episodes
}

# Set random seeds.
np.random.seed(hyper_params["seed"])
random.seed(hyper_params["seed"])
torch.manual_seed(hyper_params["seed"])

# Register ALE environments and create the environment.
gym.register_envs(ale_py)
env = gym.make(hyper_params["env"], render_mode="rgb_array")
env = RandomNoopWrapper(env, noop_max=30)
env = FrameSkipMaxPoolWrapper(env, skip=4)
env = EpisodicLifeEnv(env)
env = FireResetEnv(env)          #
env = FrameWarper(env)
env = RewardClipperWrapper(env)
env = FrameStacker(env, k=4)

# Initialize replay memory and the DQN agent.
memory = ReplayMemory(capacity=hyper_params["replay-buffer-size"])
agent = DQN(
    observation_space=env.observation_space,
    action_space=env.action_space,
    memory=memory,
    learning=hyper_params["learning-rate"],
    batch_size=hyper_params["batch-size"],
    gamma=hyper_params["discount-factor"],
    cpu=device
)

# Determine number of steps for epsilon decay.
eps_decay_steps = hyper_params["eps-fraction"] * hyper_params["num-steps"]

# Lists to record episode rewards.
episode_rewards = []
episode_reward = 0.0
episode_count = 0

# Reset the environment.
obs, info = env.reset()

# Main training loop, based on total steps.
for t in range(1, hyper_params["num-steps"] + 1):
    # Compute current epsilon threshold.
    fraction = min(1.0, t / eps_decay_steps)
    eps_threshold = hyper_params["eps-start"] + fraction * (hyper_params["eps-end"] - hyper_params["eps-start"])

    # Choose action: exploit with probability (1 - eps_threshold) otherwise random.
    if random.random() > eps_threshold:
        action = agent.decide_action(np.array(obs))
    else:
        action = env.action_space.sample()

    # Take a step in the environment.
    next_obs, reward, term, trunc, info = env.step(action)
    done = term or trunc

    # Store the transition in replay memory.
    memory.push(obs, action, reward, next_obs, done)

    # Update the state and accumulate reward.
    obs = next_obs
    episode_reward += reward

     # Start learning after a certain number of steps.
    if t >= hyper_params["learning-starts"] and t % hyper_params["learning-freq"] == 0:
        loss = agent.update_q_network()

    # Update the target network every target_update_freq steps.
    if t >= hyper_params["learning-starts"] and t % hyper_params["target-update-freq"] == 0:
        agent.update_target_network()

    # If the episode has ended, reset the environment.
    if done:
        episode_rewards.append(episode_reward)
        episode_count += 1
        obs, info = env.reset()
        episode_reward = 0.0

        # Print episode summary every print_frequent episodes.
        if episode_count % hyper_params["print-freq"] == 0:
            mean_reward = np.mean(episode_rewards[-100:]) if len(episode_rewards) >= 100 else np.mean(episode_rewards)
            print("********************************************************")
            print("Step: {}, Episodes: {}".format(t, episode_count))
            print("Mean 100 episode reward: {:.1f}".format(mean_reward))
            print("% Time spent exploring: {:.1f}".format(100 * eps_threshold))
            print("********************************************************")
            # Optionally, save a checkpoint of the model:    
env.close()

# Plot the episode rewards.
plt.figure(figsize=(8, 5))
plt.plot(range(1, len(episode_rewards) + 1), episode_rewards)
plt.title("Episode Reward over Time (Pong)")
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.tight_layout()
plt.savefig("episode_rewards.png")
plt.show()
