import gymnasium as gym
import math
import random
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import h5py
import os

# DUELING DEEP Q-NETWORK
# Simple neural network with three fully connected layers:
class DuelingDQN(nn.Module):
    def __init__(self, n_observations, n_actions):
        super(DuelingDQN, self).__init__()
        self.feature_layer = nn.Sequential(
            nn.Linear(n_observations, 512),
            nn.ReLU()
        )
        # Value stream
        self.value_layer = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
        # Advantage stream
        self.advantage_layer = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, n_actions)
        )

    def forward(self, x):
        features = self.feature_layer(x)
        value = self.value_layer(features)
        advantage = self.advantage_layer(features)
        # Combine Value and Advantage
        q_values = value + advantage - advantage.mean(dim=1, keepdim=True)
        return q_values

###################################################################################################

# NOISY DUELING DEEP Q-NETWORK
class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, std_init=0.5):
        super(NoisyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.register_buffer('weight_epsilon', torch.empty(out_features, in_features))

        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        self.register_buffer('bias_epsilon', torch.empty(out_features))

        self.std_init = std_init
        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        mu_range = 1 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.out_features))

    def reset_noise(self):
        epsilon_in = torch.randn(self.in_features)
        epsilon_out = torch.randn(self.out_features)
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    def forward(self, x):
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        return F.linear(x, weight, bias)

# Dueling DQN with Noisy Layers
class NoisyDuelingDQN(nn.Module):
    def __init__(self, n_observations, n_actions, std_init=0.5):
        super(NoisyDuelingDQN, self).__init__()
        self.feature_layer = nn.Sequential(
            nn.Linear(n_observations, 512),
            nn.ReLU()
        )
        # Value stream with NoisyLinear
        self.value_layer = nn.Sequential(
            NoisyLinear(512, 256, std_init),
            nn.ReLU(),
            NoisyLinear(256, 1, std_init)
        )
        # Advantage stream with NoisyLinear
        self.advantage_layer = nn.Sequential(
            NoisyLinear(512, 256, std_init),
            nn.ReLU(),
            NoisyLinear(256, n_actions, std_init)
        )

    def reset_noise(self):
        """Resets noise in all NoisyLinear layers."""
        for layer in self.value_layer:
            if isinstance(layer, NoisyLinear):
                layer.reset_noise()
        for layer in self.advantage_layer:
            if isinstance(layer, NoisyLinear):
                layer.reset_noise()

    def forward(self, x):
        features = self.feature_layer(x)
        value = self.value_layer(features)
        advantage = self.advantage_layer(features)
        # Combine Value and Advantage
        q_values = value + advantage - advantage.mean(dim=1, keepdim=True)
        return q_values
    
###################################################################################################

# DOUBLE DUELING DEEP Q-NETWORK    
# Simple neural network with three fully connected layers:
class DoubleDuelingDQN(nn.Module):
    def __init__(self, n_observations, n_actions):
        super(DoubleDuelingDQN, self).__init__()
        self.feature_layer = nn.Sequential(
            nn.Linear(n_observations, 512),
            nn.ReLU()
        )
        # Value stream
        self.value_layer = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
        # Advantage stream
        self.advantage_layer = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, n_actions)
        )

    def forward(self, x):
        features = self.feature_layer(x)
        value = self.value_layer(features)
        advantage = self.advantage_layer(features)
        # Combine Value and Advantage
        q_values = value + advantage - advantage.mean(dim=1, keepdim=True)
        return q_values
    
###################################################################################################

# Distributional Dueling DEEP Q-NETWORK
# Simple neural network with three fully connected layers:
class DistributionalDuelingDQN(nn.Module):
    def __init__(self, n_observations, n_actions, n_atoms=51, v_min=-10, v_max=10):
        super(DistributionalDuelingDQN, self).__init__()
        self.n_actions = n_actions
        self.n_atoms = n_atoms
        self.v_min = v_min
        self.v_max = v_max

        self.feature_layer = nn.Sequential(
            nn.Linear(n_observations, 512),
            nn.ReLU()
        )
        # Value stream
        self.value_layer = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, n_atoms)
        )
        # Advantage stream
        self.advantage_layer = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, n_actions * n_atoms)
        )

    def forward(self, x):
        features = self.feature_layer(x)
        value = self.value_layer(features).view(-1, 1, self.n_atoms)
        advantage = self.advantage_layer(features).view(-1, self.n_actions, self.n_atoms)
        q_atoms = value + advantage - advantage.mean(dim=1, keepdim=True)
        q_distribution = F.softmax(q_atoms, dim=-1)  # Ensure it's a distribution
        return q_distribution

    def get_q_values(self, x):
        distribution = self(x)
        z = torch.linspace(self.v_min, self.v_max, self.n_atoms).to(x.device)
        q_values = torch.sum(distribution * z, dim=2)
        return q_values