import os
import imageio
import gymnasium as gym
from gymnasium import spaces
import ale_py
import torch
import numpy as np
import random
import matplotlib.pyplot as plt
from wrapper import NoopResetEnv, MaxAndSkipEnv, WarpFrame, FrameStack, ClipRewardEnv, FireResetEnv, EpisodicLifeEnv
from double_dqn import Double_DQN, ReplayMemory

#choose device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#set seed
seed = 42
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)

#create video output folder
os.makedirs("videos", exist_ok=True)

#unable to use record wrapper so had to use this approach instead
def record_video_episode(env, agent, tag):
    out_dir = f"videos/{tag}"
    os.makedirs(out_dir, exist_ok=True)
    frames = []
    obs, _ = env.reset()
    done = False
    while not done:
        frames.append(env.render())  # returns rgb_array
        action = agent.decide_action(np.array(obs))
        obs, _, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
    video_path = os.path.join(out_dir, f"dd_breakout_step{tag}.mp4")
    imageio.mimsave(video_path, frames, fps=30)
    print(f"saved video to {video_path}")

#initialize environment with wrappers
breakout = "ALE/Breakout-v5"
gym.register_envs(ale_py)
gym_env = gym.make(breakout, render_mode="rgb_array")
gym_env = NoopResetEnv(gym_env, noop_max=30)
gym_env = MaxAndSkipEnv(gym_env, skip=4)
gym_env = EpisodicLifeEnv(gym_env)
gym_env = FireResetEnv(gym_env)
gym_env = WarpFrame(gym_env)
gym_env = ClipRewardEnv(gym_env)
gym_env = FrameStack(gym_env, k=4)

#parameters for training
buffer_size = 100000               
learning_rate = 1e-4          
gamma = 0.99                      
steps = 1000000                  
batch_size = 32                    
begin_learning = 20000           
train_increment = 1               
update_target = 1000             
epsilon_start = 1.0                
epsilon_end = 0.05                  
epsilon_decay_rate = 0.5           
print_every = 10     

epsilon_decay = epsilon_decay_rate * steps

replaymemory = ReplayMemory(buffer_size)
DQNAgent = Double_DQN(
    observation_space=gym_env.observation_space,
    action_space=gym_env.action_space,
    memory=replaymemory,
    learning=learning_rate,
    batch_size=batch_size,
    gamma=gamma,
    cpu=device
)

#initializations for loop setup
rewards_per_episode = []
total_reward = 0
episodes = 0
observation, _ = gym_env.reset()

#training loop
for step in range(1, steps + 1):
    progress = min(1.0, step / epsilon_decay)
    epsilon = epsilon_start + progress * (epsilon_end - epsilon_start)

    if random.random() < epsilon:
        action = gym_env.action_space.sample()
    else:
        action = DQNAgent.decide_action(np.array(observation))

    next_observation, reward, terminated, truncated, _ = gym_env.step(action)
    done = terminated or truncated

    replaymemory.push(observation, action, reward, next_observation, done)
    total_reward += reward
    observation = next_observation

    if step >= begin_learning and step % train_increment == 0:
        DQNAgent.update_q_network()

    if step >= begin_learning and step % update_target == 0:
        DQNAgent.update_target_network()

    if step in [100000, 250000, 500000, 750000, 1000000]:
        record_video_episode(gym_env, DQNAgent, f"step_{step}")

    if done:
        rewards_per_episode.append(total_reward)
        episodes += 1
        total_reward = 0
        observation, _ = gym_env.reset()

        if episodes % print_every == 0:
            hundred_recent_episodes = rewards_per_episode[-100:]
            average_reward = np.mean(hundred_recent_episodes)
            print("-------------------------------------------------------")
            print(f"Step: {step}, Episode: {episodes}")
            print(f"Avg Reward (last 100): {average_reward:.1f}")
            print(f"Exploration: {epsilon*100:.1f}%")
            print("-------------------------------------------------------")

gym_env.close()

#plot rewards
plt.plot(range(1, len(rewards_per_episode) + 1), rewards_per_episode)
plt.title("Episode Rewards Breakout (Double_DQN)")
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.tight_layout()
plt.savefig("episode_rewards_breakout_double_dqn.png")
plt.show()
