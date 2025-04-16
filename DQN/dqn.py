import gymnasium as gym
from gymnasium import spaces
import torch
import torch.nn as nn 
import torch.nn.functional as F
import numpy as np
from collections import deque
import random

class QNetwork(nn.Module):
    def __init__(self, observation_space: spaces.Box, action_space: spaces.Discrete):
        super().__init__()
        self.observation_space = observation_space
        self.action_space = action_space

        input_channels = observation_space.shape[0]
        num_actions = self.action_space.n

        self.convolution = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        
        self.fully_connected = nn.Sequential(
            nn.Linear(64 * 7 * 7, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions)
        )

    def forward(self, x):
        convolutional_out = self.convolution(x).view(x.size()[0], -1)
        return self.fully_connected(convolutional_out)


class ReplayMemory: 
    def __init__(self, capacity=10000):
        self.memory = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def sample(self, batch_size=32):
        # Assumes that the caller checks len(self.memory) >= batch_size.
        transitions = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*transitions)
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.memory)
    

class DQN:
    def __init__(self, observation_space: spaces.Box, action_space: spaces.Discrete, memory: ReplayMemory, learning, batch_size, gamma, cpu=torch.device("cpu")):
        self.memory = memory
        self.batch_size = batch_size
        self.gamma = gamma
        self.cpu = cpu

        self.policy_net = QNetwork(observation_space, action_space).to(cpu)
        self.target_net = QNetwork(observation_space, action_space).to(cpu)
        self.update_target_network()  
        self.target_net.eval()         
        self.optimizer = torch.optim.RMSprop(self.policy_net.parameters(), lr=learning)

    def update_q_network(self):
        cpu = self.cpu
        
        sample_states, sample_actions, sample_rewards, sample_next_states, sample_dones = self.memory.sample(self.batch_size)
        sample_states = np.array(sample_states) / 255.0
        sample_next_states = np.array(sample_next_states) / 255.0 
        
        sample_states = torch.from_numpy(sample_states).float().to(cpu)
        sample_actions = torch.from_numpy(np.array(sample_actions)).long().to(cpu)
        sample_rewards = torch.from_numpy(np.array(sample_rewards)).float().to(cpu)
        sample_next_states = torch.from_numpy(sample_next_states).float().to(cpu)
        sample_dones = torch.from_numpy(np.array(sample_dones)).float().to(cpu)

        with torch.no_grad():
            q_vals_next = self.target_net(sample_next_states)
            max_q_vals, _ = q_vals_next.max(dim=1)
            target_q_vals = sample_rewards + (1 - sample_dones) * self.gamma * max_q_vals

        current_q_vals = self.policy_net(sample_states)
        current_q_vals = current_q_vals.gather(1, sample_actions.unsqueeze(1)).squeeze()
        loss = F.smooth_l1_loss(current_q_vals, target_q_vals)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        del sample_states, sample_next_states
        return loss.item()
    
    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def decide_action(self, state: np.ndarray):
        cpu = self.cpu
        state = np.array(state) / 255.0
        state = torch.from_numpy(state).float().unsqueeze(0).to(cpu)
        with torch.no_grad():
            q_vals = self.policy_net(state)
            _, action = q_vals.max(dim=1)
            return action.item()
