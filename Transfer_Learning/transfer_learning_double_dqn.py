import gymnasium as gym
from gymnasium import spaces
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import deque
import random

class QNetwork(nn.Module):
    def __init__(self, observation_space: spaces.Box, action_space_dict: dict):
        super().__init__()
        input_channels = observation_space.shape[0]

        self.convolution = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        self.feature_size = 64 * 7 * 7

        self.heads = nn.ModuleDict({
            game: nn.Sequential(
                nn.Linear(self.feature_size, 512),
                nn.ReLU(),
                nn.Linear(512, action_space.n)
            )
            for game, action_space in action_space_dict.items()
        })

    def forward(self, x, game: str):
        c = self.convolution(x)
        return self.heads[game](c.view(c.size(0), -1))


class ReplayMemory:
    def __init__(self, capacity=10000):
        self.memory = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def sample(self, batch_size=32):
        transitions = random.sample(self.memory, batch_size)
        return zip(*transitions)

    def __len__(self):
        return len(self.memory)


class DoubleDQN:
    def __init__(self, observation_space: spaces.Box, action_space_dict: dict, memory_capacity: int, learning: float, batch_size: int, gamma: float, cpu=torch.device("cpu")):
        self.batch_size = batch_size
        self.gamma = gamma
        self.device = cpu

        self.memories = {
            game: ReplayMemory(memory_capacity)
            for game in action_space_dict
        }

        self.policy_net = QNetwork(observation_space, action_space_dict).to(cpu)
        self.target_net = QNetwork(observation_space, action_space_dict).to(cpu)
        self.update_target_network()
        self.target_net.eval()

        self.optimizer = torch.optim.RMSprop(self.policy_net.parameters(), lr=learning)

    def update_q_network(self, game: str):
        states, actions, rewards, next_states, dones = self.memories[game].sample(self.batch_size)
        S  = torch.from_numpy(np.array(states)      / 255.0).float().to(self.device)
        A  = torch.from_numpy(np.array(actions)).long().to(self.device)
        R  = torch.from_numpy(np.array(rewards)).float().to(self.device)
        S2 = torch.from_numpy(np.array(next_states) / 255.0).float().to(self.device)
        D  = torch.from_numpy(np.array(dones)).float().to(self.device)

        with torch.no_grad():
            next_actions = self.policy_net(S2, game).argmax(1)
            q_next = self.target_net(S2, game).gather(1, next_actions.unsqueeze(1)).squeeze()
            q_target = R + (1 - D) * self.gamma * q_next

        q_current = self.policy_net(S, game).gather(1, A.unsqueeze(1)).squeeze()
        loss = F.smooth_l1_loss(q_current, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def decide_action(self, state: np.ndarray, game: str):
        s = torch.from_numpy(state / 255.0).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            return int(self.policy_net(s, game).argmax(1).item())

    # save and load
    def save(self, path):
        torch.save(self.policy_net.state_dict(), path)

    def load(self, path, strict=True):
        sd = torch.load(path, map_location=self.device)
        self.policy_net.load_state_dict(sd, strict=strict)
        self.update_target_network()