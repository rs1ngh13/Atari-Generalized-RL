import gymnasium
from gymnasium import spaces
import torch
import torch.nn as nn 
import torch.nn.functional as F
import numpy as np

#Based off of the Nature Paper
class QNetwork(nn.Module):
    def __init__(self, observation_space: spaces.Box, action_space: spaces.Discrete):
        super().__init__()
        self.observation_space = observation_space
        self.action_space = action_space

        input_channels = observation_space.shape[0]
        num_actions = self.action_space.n

        self.convolution = nn.Sequential(
            nn.Conv2d(input_channels, 32, 8, 4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1),
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

#Revised Version from Homework 2 
class ReplayMemory: 
    def __init__(self, length):
        self.memory = []
        self.memory_length = length
        self.next_index = 0

    def push(self, state, action, reward, next_state, done):
        transition = (state, action, reward, next_state, done)
        if self.next_index >= len(self.memory):
            self.memory.append(transition)
        else:
            self.memory[self.next_index] = transition
        self.next_index = (self.next_index + 1) % self.memory_length

    def sample(self, batch):
        states = []
        actions = []
        rewards = []
        next_states = []
        dones = []
        
        index = np.random.randint(0, len(self.memory) - 1, size = batch)
        for i in index:
            state, action, reward, next_state, done = self.memory[i]
            states.append(np.array(state, copy = False))
            actions.append(action)
            rewards.append(reward)
            next_states.append(np.array(next_state, copy = False))
            dones.append(done)

        return(np.array(states), np.array(actions), np.array(rewards), np.array(next_states), np.array(dones))

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
