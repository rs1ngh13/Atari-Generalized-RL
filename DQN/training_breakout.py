import gymnasium as gym
from gymnasium import spaces
import ale_py
import torch
import numpy as np
import random
import matplotlib.pyplot as plt
from wrapper import RandomNoopWrapper, FrameSkipMaxPoolWrapper, FrameWarper, FrameStacker, RewardClipperWrapper, FireResetEnv, EpisodicLifeEnv
from dqn import DQN, ReplayMemory

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hardcoded hyperparameters
seed = 42
env_id = "ALE/Breakout-v5"
replay_buffer_size = 100000        # Maximum capacity of Replay Memory
learning_rate = 1e-4
discount_factor = 0.99
num_steps = int(1.5e6)               # Total number of environment steps
batch_size = 32
learning_starts = 10000            # Steps before learning starts
learning_freq = 1                  # Optimize every step once learning starts
target_update_freq = 1000          # Update target network every these many steps
eps_start = 1.0                    # Initial epsilon for exploration
eps_end = 0.01                     # Final epsilon after decay
eps_fraction = 0.1                 # Fraction of total steps over which to decay epsilon
print_freq = 10                    # Print stats every print_freq episodes

# Set random seeds.
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)

# Register ALE environments and create the environment.
gym.register_envs(ale_py)
env = gym.make(env_id, render_mode="rgb_array")
env = EpisodicLifeEnv(env)                       # Makes life loss = terminal
env = RandomNoopWrapper(env, noop_max=30)        # Randomizes initial state
env = FireResetEnv(env)                           # Ensures FIRE action on reset
env = FrameSkipMaxPoolWrapper(env, skip=4)        # Frame skipping and max-pooling
env = FrameWarper(env)                            # Grayscale + resize to 84x84
env = FrameStacker(env, k=4)                      # Stack 4 frames
env = RewardClipperWrapper(env)                   # Clip rewards to {-1, 0, 1}


# Initialize replay memory and the DQN agent.
memory = ReplayMemory(capacity=replay_buffer_size)
agent = DQN(
    observation_space=env.observation_space,
    action_space=env.action_space,
    memory=memory,
    learning=learning_rate,
    batch_size=batch_size,
    gamma=discount_factor,
    cpu=device
)

# Calculate the number of steps to decay epsilon.
eps_decay_steps = eps_fraction * num_steps

# Training control variables.
episode_rewards = []
episode_reward = 0.0
episode_count = 0

# Reset the environment to get the initial observation.
obs, info = env.reset()

# Main training loop (step-based).
for t in range(1, num_steps + 1):
    # Compute the current epsilon threshold (linear decay).
    fraction = min(1.0, t / eps_decay_steps)
    eps_threshold = eps_start + fraction * (eps_end - eps_start)

    # Choose an action: exploit with probability (1 - eps_threshold) or explore (random action).
    if random.random() > eps_threshold:
        action = agent.decide_action(np.array(obs))
    else:
        action = env.action_space.sample()

    # Take a step in the environment.
    next_obs, reward, term, trunc, info = env.step(action)
    done = term or trunc

    # Store the transition in replay memory.
    memory.push(obs, action, reward, next_obs, done)

    # Update the current state and accumulate the episode reward.
    obs = next_obs
    episode_reward += reward

    # Start learning after a certain number of steps.
    if t >= learning_starts and t % learning_freq == 0:
        loss = agent.update_q_network()

    # Update the target network at the specified frequency.
    if t >= learning_starts and t % target_update_freq == 0:
        agent.update_target_network()

    # If the episode has ended, reset the environment and log the reward.
    if done:
        episode_rewards.append(episode_reward)
        episode_count += 1
        obs, info = env.reset()
        episode_reward = 0.0

        # Print a summary every print_freq episodes.
        if episode_count % print_freq == 0:
            mean_reward = np.mean(episode_rewards[-100:]) if len(episode_rewards) >= 100 else np.mean(episode_rewards)
            print("-------------------------------------------------------")
            print(f"Step: {t}, Episodes: {episode_count}")
            print(f"Mean 100 episode reward: {mean_reward:.1f}")
            print(f"% Time spent exploring: {100 * eps_threshold:.1f}")
            print("-------------------------------------------------------")
env.close()

# Plot the episode rewards.
plt.figure(figsize=(8, 5))
plt.plot(range(1, len(episode_rewards) + 1), episode_rewards)
plt.title("Episode Reward over Time (Breakout)")
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.tight_layout()
plt.savefig("episode_rewards_breakout.png")
plt.show()
