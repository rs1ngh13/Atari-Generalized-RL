import os
import gymnasium as gym
import ale_py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
import matplotlib.pyplot as plt
import imageio
import random
from collections import deque

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set seed for reproducibility
seed = 42
np.random.seed(seed)
torch.manual_seed(seed)
random.seed(seed)

# Create output directories
output_dir = "multi_game_ppo_output"
os.makedirs(f"{output_dir}/models", exist_ok=True)
os.makedirs(f"{output_dir}/videos", exist_ok=True)
os.makedirs(f"{output_dir}/plots", exist_ok=True)

# Register ALE environments
gym.register_envs(ale_py)

# Environment setup with proper wrappers
def make_env(env_id, render_mode=None):
    env = gym.make(env_id, render_mode=render_mode)
    env = gym.wrappers.ResizeObservation(env, (84, 84))
    env = gym.wrappers.GrayscaleObservation(env)  # Convert to grayscale
    env = gym.wrappers.FrameStackObservation(env, 4)
    env = gym.wrappers.RecordEpisodeStatistics(env)  # Track complete episode returns
    return env

# Define training environments
env_ids = {
    "pong": "ALE/Pong-v5",
    "breakout": "ALE/Breakout-v5", 
    "beamrider": "ALE/BeamRider-v5",
}
env_ids["space_invaders"] = "ALE/SpaceInvaders-v5"
env_ids["river_raid"] = "ALE/Riverraid-v5"


# Find the maximum action space dimension across all environments
def get_max_action_dim(env_ids_dict):
    max_dim = 0
    for env_id in env_ids_dict.values():
        env = gym.make(env_id)
        max_dim = max(max_dim, env.action_space.n)
        env.close()
    return max_dim

max_action_dim = get_max_action_dim(env_ids)
print(f"Maximum action dimension across all games: {max_action_dim}")

class UnifiedActorCritic(nn.Module):
    def __init__(self, in_channels=4, action_dim=18):
        """
        Unified Actor-Critic model for all games
        
        Args:
            in_channels: Number of input channels (4 for stacked frames)
            action_dim: Maximum dimension of action space across all games
        """
        super(UnifiedActorCritic, self).__init__()
        
        # Shared convolutional backbone
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.ReLU()
        )
        
        # Apply proper weight initialization to conv layers
        for layer in self.conv:
            if isinstance(layer, nn.Conv2d):
                nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
                nn.init.zeros_(layer.bias)
        
        # Determine conv output size dynamically
        with torch.no_grad():
            dummy_input = torch.zeros(1, in_channels, 84, 84)
            conv_out_size = self.conv(dummy_input).flatten(1).shape[1]
        
        # Shared feature extractor
        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU()
        )
        
        # Initialize fc layers
        for layer in self.fc:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
                nn.init.zeros_(layer.bias)
        
        # Single unified policy and value heads
        self.policy = nn.Linear(512, action_dim)
        self.value = nn.Linear(512, 1)
        
        # Initialize output layers with appropriate scaling
        nn.init.orthogonal_(self.policy.weight, gain=0.01)
        nn.init.zeros_(self.policy.bias)
        nn.init.orthogonal_(self.value.weight, gain=1.0)
        nn.init.zeros_(self.value.bias)
        
        # Track valid action dimensions for each game
        self.game_action_dims = {}
    
    def forward(self, x, game=None, valid_actions=None):
        """
        Forward pass with optional masking for valid actions
        
        Args:
            x: Input observation tensor
            game: Name of the game (for logging purposes)
            valid_actions: Number of valid actions for the current game
        """
        x = self.conv(x)
        x = x.flatten(1)
        x = self.fc(x)
        
        policy_logits = self.policy(x)
        
        # Mask invalid actions if valid_actions is provided
        if valid_actions is not None:
            # Create mask for valid actions
            mask = torch.zeros_like(policy_logits)
            mask[:, :valid_actions] = 1
            # Apply mask by setting invalid actions to large negative values
            policy_logits = policy_logits + (mask - 1) * 1e10
        
        value = self.value(x)
        
        return policy_logits, value
    
    def register_game(self, game, action_dim):
        """Register a game's action space dimension"""
        self.game_action_dims[game] = action_dim


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


def collect_rollout_for_game(env, model, game, action_dim, rollout_length=128):
    """Collect experience for a specific game"""
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
    raw_rewards = []  # For tracking unclipped rewards
    
    steps_completed = 0
    episode_reward = 0  # Raw episode reward
    
    while steps_completed < rollout_length:
        # Get action and value
        with torch.no_grad():
            logits, value = model(obs, game, action_dim)
            dist = Categorical(logits=logits)
            action = dist.sample()
            logprob = dist.log_prob(action)
        
        # Execute action
        next_obs, reward, terminated, truncated, info = env.step(action.item())
        done = terminated or truncated
        
        # Save raw reward for logging
        raw_rewards.append(reward)
        episode_reward += reward
        
        # Clip rewards for stability
        clipped_reward = np.sign(reward)
        
        # Store data
        observations.append(obs)
        actions.append(action)
        logprobs.append(logprob)
        rewards.append(clipped_reward)  # Store clipped reward
        dones.append(done)
        values.append(value)
        
        # Move to next state
        obs = preprocess_obs(next_obs)
        
        # Track episode statistics
        if done:
            episode_rewards.append(episode_reward)  # Store raw episode reward
            if "episode" in info:
                episode_lengths.append(info["episode"]["l"])
            else:
                episode_lengths.append(0)
            episode_reward = 0  # Reset episode reward
            
            # Reset environment
            obs, _ = env.reset()
            obs = preprocess_obs(obs)
        
        steps_completed += 1
    
    # Get final value for bootstrapping
    with torch.no_grad():
        _, next_value = model(obs, game, action_dim)
    
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
    if observations:  # Check if we collected any data
        batch = {
            "observations": torch.cat(observations),
            "actions": torch.cat(actions),
            "old_logprobs": torch.cat(logprobs),
            "returns": torch.FloatTensor(returns).to(device),
            "advantages": torch.FloatTensor(advantages).to(device),
            "values": torch.cat(values),
            "episode_rewards": episode_rewards,
            "episode_lengths": episode_lengths,
            "raw_rewards": raw_rewards  # Include raw rewards for logging
        }
        
        # Normalize advantages
        if len(batch["advantages"]) > 1:  # Only normalize if we have more than one sample
            batch["advantages"] = (batch["advantages"] - batch["advantages"].mean()) / (batch["advantages"].std() + 1e-8)
        
        return batch
    else:
        return None


def update_policy(model, optimizer, batch, clip_param=0.2, value_coef=0.5, entropy_coef=0.01, epochs=4):
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
    mini_batch_size = max(batch_size // 4, 1)  # Ensure at least 1 sample per mini-batch
    
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
            
            # Forward pass - no need for game-specific handling
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
            # Clip gradients
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            optimizer.step()
            
            # Record losses
            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            total_entropy += entropy.item()
    
    # Average losses
    n_updates = epochs * (batch_size // mini_batch_size) or 1  # Ensure no division by zero
    avg_policy_loss = total_policy_loss / n_updates
    avg_value_loss = total_value_loss / n_updates
    avg_entropy = total_entropy / n_updates
    
    return avg_policy_loss, avg_value_loss, avg_entropy


def linear_schedule(initial_value, final_value=0.0):
    """Linear learning rate schedule"""
    def scheduler(progress):
        return final_value + (initial_value - final_value) * (1 - progress)
    return scheduler


def record_video_episode(env, model, game, action_dim, tag):
    """Record a video of the agent playing"""
    out_dir = f"{output_dir}/videos/{game}_{tag}"
    os.makedirs(out_dir, exist_ok=True)
    
    env_with_render = make_env(env_ids[game], render_mode="rgb_array")
    frames = []
    
    obs, _ = env_with_render.reset()
    obs = preprocess_obs(obs)
    done = False
    episode_reward = 0
    
    while not done:
        frames.append(env_with_render.render())
        
        # Choose action
        with torch.no_grad():
            logits, _ = model(obs, game, action_dim)
            dist = Categorical(logits=logits)
            action = dist.sample()
        
        # Step environment
        next_obs, reward, terminated, truncated, _ = env_with_render.step(action.item())
        done = terminated or truncated
        episode_reward += reward
        
        # Next state
        obs = preprocess_obs(next_obs)
    
    # Save video
    video_path = os.path.join(out_dir, f"ppo_{game}_step{tag}.mp4")
    imageio.mimsave(video_path, frames, fps=30)
    env_with_render.close()
    
    print(f"Saved video to {video_path} with reward {episode_reward}")
    return episode_reward




def train_multi_game_ppo():
    # Create environments
    envs = {game: make_env(env_id) for game, env_id in env_ids.items()}
    
    # Get action dimensions for each game
    action_dims = {game: env.action_space.n for game, env in envs.items()}
    
    # Initialize unified model
    model = UnifiedActorCritic(in_channels=4, action_dim=max_action_dim).to(device)
    
    # Register games with the model
    for game, action_dim in action_dims.items():
        model.register_game(game, action_dim)
    
    # Learning rate scheduling
    initial_lr = 3e-4
    final_lr = 5e-5
    lr_scheduler = linear_schedule(initial_lr, final_lr)
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=initial_lr)
    
    # Training parameters
    total_steps = 10000000
    update_steps_per_game = 1024  # Steps to collect before updating for each game
    print_every = 10
    record_video_steps = [100000, 500000, 1000000, 2000000, 3000000, 4000000, 5000000,7500000,10000000]
    
    # Hyperparameters
    clip_param = 0.1  # Smaller clip parameter for more conservative updates
    value_coef = 0.5
    entropy_coef = 0.11  # Encourage exploration
    ppo_epochs = 3
    
    # State tracking
    steps_completed = 0
    episodes = {game: 0 for game in envs}
    rewards_per_episode = {game: [] for game in envs}
    recent_rewards = {game: deque(maxlen=100) for game in envs}
    
    # Training loop
    while steps_completed < total_steps:
        # Compute dynamic weights inversely proportional to episodes completed
        weights = [1.0/(episodes[g]+1) for g in envs]
        game = random.choices(list(envs.keys()), weights=weights)[0]
        action_dim = action_dims[game]
        
        # Adjust learning rate
        progress = min(1.0, steps_completed / total_steps)
        lr = lr_scheduler(progress)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        
        # Collect rollout for this game
        batch = collect_rollout_for_game(envs[game], model, game, action_dim, rollout_length=update_steps_per_game)
        
        if batch:
            # Update policy
            policy_loss, value_loss, entropy = update_policy(
                model, 
                optimizer, 
                batch,
                clip_param=clip_param,
                value_coef=value_coef,
                entropy_coef=entropy_coef,
                epochs=ppo_epochs
            )
            
            # Update episode statistics
            for ep_reward in batch["episode_rewards"]:
                rewards_per_episode[game].append(ep_reward)
                recent_rewards[game].append(ep_reward)
                episodes[game] += 1
            
            # Reporting
            if episodes[game] % print_every == 0 and episodes[game] > 0:
                avg_reward = np.mean(list(recent_rewards[game])) if recent_rewards[game] else float('nan')
                print("-------------------------------------------------------")
                print(f"Step: {steps_completed}, {game.capitalize()} Episodes: {episodes[game]}")
                print(f"Avg Reward (last100): {avg_reward:.2f}")
                print(f"Policy Loss: {policy_loss:.4f}, Value Loss: {value_loss:.4f}, Entropy: {entropy:.4f}")
                print(f"Learning Rate: {lr:.6f}")
                print("-------------------------------------------------------")
            
            # Update step counter
            steps_completed += update_steps_per_game
            
            # Video recording and checkpoint saving at milestones
            if any(abs(steps_completed - milestone) < update_steps_per_game for milestone in record_video_steps):
                for g in envs:
                    record_video_episode(envs[g], model, g, action_dims[g], f"step_{steps_completed}")
                
                # Save checkpoint
                checkpoint_path = f"{output_dir}/models/multi_game_ppo_step_{steps_completed}.pt"
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'steps': steps_completed,
                    'episodes': episodes,
                    'rewards': rewards_per_episode,
                }, checkpoint_path)
                print(f"Saved checkpoint to {checkpoint_path}")
                
                # Plot learning curves
                for g, rewards in rewards_per_episode.items():
                    if rewards:  # Only plot if we have data
                        plt.figure(figsize=(10, 5))
                        plt.plot(rewards)
                        plt.title(f"Episode Rewards - {g.capitalize()}")
                        plt.xlabel("Episode")
                        plt.ylabel("Reward")
                        plt.savefig(f"{output_dir}/plots/rewards_{g}_step_{steps_completed}.png")
                        plt.close()
    
    # Clean up
    for env in envs.values():
        env.close()
    
    # Save final model
    torch.save(model.state_dict(), f"{output_dir}/models/multi_game_ppo_final.pt")
    
    # Plot final learning curves
    plt.figure(figsize=(12, 8))
    for g, rewards in rewards_per_episode.items():
        if rewards:  # Only plot if we have data
            plt.plot(rewards, label=g)
    plt.title("Multi-Game PPO Training - Episode Rewards")
    plt.xlabel("Episodes")
    plt.ylabel("Reward")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{output_dir}/plots/multi_game_ppo_final_rewards.png")
    plt.close()
    
    return model, rewards_per_episode


if __name__ == "__main__":
    model, rewards = train_multi_game_ppo()
    print("Training complete!")