import gymnasium as gym
import ale_py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
import matplotlib.pyplot as plt

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Environment setup with proper wrappers
def make_env():
    env = gym.make("ALE/Pong-v5", render_mode=None)
    env = gym.wrappers.ResizeObservation(env, (84, 84))
    env = gym.wrappers.GrayscaleObservation(env)  # Convert to grayscale
    env = gym.wrappers.FrameStackObservation(env, 4)
    env = gym.wrappers.RecordEpisodeStatistics(env)  # Track complete episode returns
    return env


class ActorCritic(nn.Module):
    def __init__(self, in_channels=4, num_actions=6):  # 4 grayscale frames
        super(ActorCritic, self).__init__()
        
        # Convolutional layers with proper initialization
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.ReLU()
        )
        
        # Apply proper weight initialization
        for layer in self.conv:
            if isinstance(layer, nn.Conv2d):
                nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
                nn.init.zeros_(layer.bias)
        
        # Determine conv output size dynamically
        with torch.no_grad():
            dummy_input = torch.zeros(1, in_channels, 84, 84)
            conv_out_size = self.conv(dummy_input).flatten(1).shape[1]
        
        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU()
        )
        
        # Policy and value heads
        self.policy = nn.Linear(512, num_actions)
        self.value = nn.Linear(512, 1)
        
        # Initialize fc layers
        for layer in self.fc:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
                nn.init.zeros_(layer.bias)
        
        # Initialize output layers with appropriate scaling
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
    """Process observations for input to the model"""
    if isinstance(obs, np.ndarray):
        obs = torch.FloatTensor(obs)
    
    # Normalize to [0, 1] if not already
    if obs.max() > 1.0:
        obs = obs / 255.0
    
    # Add batch dimension if needed
    if obs.dim() == 3:
        obs = obs.unsqueeze(0)
    
    return obs.to(device)


def collect_rollouts(env, model, rollout_length=2048):
    """Collect experience using the current policy"""
    obs, _ = env.reset()
    obs = preprocess_obs(obs)
    
    # Storage
    observations = []
    actions = []
    logprobs = []
    rewards = []
    dones = []
    values = []
    episode_rewards = []
    episode_lengths = []
    
    for step in range(rollout_length):
        # Get action and value
        with torch.no_grad():
            logits, value = model(obs)
            dist = Categorical(logits=logits)
            action = dist.sample()
            logprob = dist.log_prob(action)
        
        # Execute action
        next_obs, reward, terminated, truncated, info = env.step(action.item())
        done = terminated or truncated
        
        # Store data
        observations.append(obs)
        actions.append(action)
        logprobs.append(logprob)
        rewards.append(reward)
        dones.append(done)
        values.append(value)
        
        # Move to next state
        obs = preprocess_obs(next_obs)
        
        # Track episode statistics
        if "episode" in info:
            episode_rewards.append(info["episode"]["r"])
            episode_lengths.append(info["episode"]["l"])
        
        # Reset if needed
        if done:
            obs, _ = env.reset()
            obs = preprocess_obs(obs)
    
    # Get final value for bootstrapping
    with torch.no_grad():
        _, next_value = model(obs)
    
    # Calculate returns with GAE
    advantages = []
    returns = []
    gae = 0
    
    for i in reversed(range(len(rewards))):
        if i == len(rewards) - 1:
            next_val = next_value.item()
        else:
            next_val = values[i+1].item()
        
        # Handle episode boundaries
        mask = 1.0 - dones[i]
        
        # TD error and GAE calculation
        delta = rewards[i] + 0.99 * next_val * mask - values[i].item()
        gae = delta + 0.99 * 0.95 * mask * gae
        
        returns.insert(0, gae + values[i].item())
        advantages.insert(0, gae)
    
    # Convert to tensors
    batch = {
        "observations": torch.cat(observations),
        "actions": torch.cat(actions),
        "old_logprobs": torch.cat(logprobs),
        "returns": torch.FloatTensor(returns).to(device),
        "advantages": torch.FloatTensor(advantages).to(device),
        "values": torch.cat(values),
        "episode_rewards": episode_rewards,
        "episode_lengths": episode_lengths
    }
    
    # Normalize advantages
    batch["advantages"] = (batch["advantages"] - batch["advantages"].mean()) / (batch["advantages"].std() + 1e-8)
    
    return batch


def update_policy(model, optimizer, batch, clip_param=0.2, value_coef=0.5, entropy_coef=0.1, epochs=4):
    """Update policy using PPO algorithm"""
    # Prepare data
    observations = batch["observations"]
    actions = batch["actions"]
    old_logprobs = batch["old_logprobs"]
    returns = batch["returns"]
    advantages = batch["advantages"]
    
    # Training
    total_policy_loss = 0
    total_value_loss = 0
    total_entropy = 0
    
    # Calculate batch size and number of batches
    batch_size = len(observations)
    mini_batch_size = batch_size // 4  # 4 mini-batches per epoch
    
    for _ in range(epochs):
        # Generate random indices
        indices = np.random.permutation(batch_size)
        
        # Process mini-batches
        for start in range(0, batch_size, mini_batch_size):
            end = start + mini_batch_size
            idx = indices[start:end]
            
            # Get mini-batch data
            mb_obs = observations[idx]
            mb_actions = actions[idx]
            mb_old_logprobs = old_logprobs[idx]
            mb_returns = returns[idx]
            mb_advantages = advantages[idx]
            
            # Forward pass
            logits, values = model(mb_obs)
            dist = Categorical(logits=logits)
            new_logprobs = dist.log_prob(mb_actions)
            entropy = dist.entropy().mean()
            
            # Policy loss with clipping
            ratio = torch.exp(new_logprobs - mb_old_logprobs)
            surr1 = ratio * mb_advantages
            surr2 = torch.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param) * mb_advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # Value loss
            value_loss = 0.5 * ((values.squeeze() - mb_returns) ** 2).mean()
            
            # Total loss
            loss = policy_loss + value_coef * value_loss - entropy_coef * entropy
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            # Clip gradients (optional but recommended)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            optimizer.step()
            
            # Record losses
            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            total_entropy += entropy.item()
    
    # Average losses
    avg_policy_loss = total_policy_loss / (epochs * (batch_size // mini_batch_size))
    avg_value_loss = total_value_loss / (epochs * (batch_size // mini_batch_size))
    avg_entropy = total_entropy / (epochs * (batch_size // mini_batch_size))
    
    return avg_policy_loss, avg_value_loss, avg_entropy


def linear_schedule(initial_value, final_value=0.0):
    """Linear learning rate schedule"""
    def scheduler(progress):
        return final_value + (initial_value - final_value) * (1 - progress)
    return scheduler

def reward_shaping(info, reward):
    """Add shaped rewards to sparse environment rewards"""
    shaped_reward = reward
    
    # Add small reward for keeping ball in play (accessing ball position from info)
    if "ball_position" in info and info["ball_position"] is not None:
        # Pong-specific reward shaping (hypothetical implementation)
        if info.get("hit_ball", False):
            shaped_reward += 0.1  # Small reward for hitting the ball
    
    return shaped_reward

def train():
    # Environment
    env = make_env()
    in_channels = 4  # Grayscale, 4 frames
    num_actions = env.action_space.n
    
    # Model and optimizer
    model = ActorCritic(in_channels, num_actions).to(device)
    initial_lr = 5e-4
    final_lr = 1e-4
    lr_scheduler = linear_schedule(initial_lr, final_lr)
    optimizer = optim.Adam(model.parameters(), lr=initial_lr)
    
    # Training parameters
    total_updates = 10000
    rollout_length = 2048
    
    # High entropy coefficient for better exploration
    entropy_coef = 0.1
    
    # Training metrics
    training_returns = []
    episode_returns = []
    
    # Progress tracking
    for update in range(1, total_updates + 1):
        # Adjust learning rate
        progress = update / total_updates
        lr = lr_scheduler(progress)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        
        # Collect rollouts
        batch = collect_rollouts(env, model, rollout_length)
        
        # Update policy
        policy_loss, value_loss, entropy = update_policy(
            model, 
            optimizer, 
            batch, 
            entropy_coef=entropy_coef
        )
        
        # Record metrics
        if batch["episode_rewards"]:
            episode_returns.extend(batch["episode_rewards"])
        
        # Calculate average return from this batch
        avg_return = batch["returns"].mean().item()
        training_returns.append(avg_return)
        
        # Logging
        if update % 2 == 0:
            recent_ep_returns = episode_returns[-10:] if episode_returns else [0]
            print(f"Update {update:04d} | "
                  f"Loss: {policy_loss:.3f} | "
                  f"Value Loss: {value_loss:.3f} | "
                  f"Entropy: {entropy:.3f} | "
                  f"Return: {avg_return:.3f} | "
                  f"Recent Ep Return: {np.mean(recent_ep_returns):.3f} | "
                  f"LR: {lr:.6f}")
            
            # Save model checkpoint
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'update': update,
                'returns': training_returns,
                'episode_returns': episode_returns
            }, f"ppo_checkpoint.pt")
            
        # Visualization every 1000 updates
        if update % 1000 == 0:
            plt.figure(figsize=(12, 5))
            plt.subplot(1, 2, 1)
            plt.plot(training_returns)
            plt.title("Training Returns")
            plt.xlabel("Updates")
            plt.ylabel("Average Return")
            
            plt.subplot(1, 2, 2)
            plt.plot(episode_returns)
            plt.title("Episode Returns")
            plt.xlabel("Episodes")
            plt.ylabel("Return")
            
            plt.tight_layout()
            plt.savefig(f"ppo_training_plot.png")
            plt.close()
    
    # Final save
    torch.save(model.state_dict(), "ppo_final_model.pt")
    
    return model, training_returns, episode_returns


model, training_returns, episode_returns = train()
