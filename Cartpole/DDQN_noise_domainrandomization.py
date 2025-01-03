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
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import h5py
import os
from Agents import DuelingDQN
from plot_graph import plot_rewards
from save_model import save_model

class RandomizedCartPole(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.env = env
        self.randomize_environment()
        
        # Initialize friction and wind attributes with random values
        self.friction = np.random.uniform(0.1, 0.3)
        self.wind_force = np.random.uniform(-0.1, 0.1)

    def randomize_environment(self):
        # Randomize only parameters that should remain constant within an episode
        self.env.unwrapped.length = np.random.uniform(0.4, 0.6)  # Random pole length
        self.env.unwrapped.masscart = np.random.uniform(0.8, 1.0)  # Random cart mass
        self.env.unwrapped.masspole = np.random.uniform(0.05, 0.10)  # Random pole mass

    def step(self, action):
        # Original step function to get initial observation, reward, etc.
        observation, reward, terminated, truncated, info = self.env.step(action)

        # Extract state variables from observation
        cart_position, cart_velocity, pole_angle, pole_velocity = observation

        # Apply dynamic step-wise friction and wind
        cart_acceleration = -self.friction * cart_velocity + self.wind_force
        cart_velocity += cart_acceleration

        # Create a new observation incorporating the modified cart_velocity
        new_observation = np.array([cart_position, cart_velocity, pole_angle, pole_velocity])

        # Adding noise to the new observation
        noise = np.random.normal(0, 0.5, size=new_observation.shape)  # Gaussian noise
        new_observation = new_observation + noise

        return new_observation, reward, terminated, truncated, info

    def reset(self, **kwargs):
        # Randomize episode-level parameters (e.g., pole length, mass)
        self.randomize_environment()
        # Call the base environment's reset method and return the initial state
        return self.env.reset(**kwargs)

# Use the modified environment with wrapped environment
env = RandomizedCartPole(gym.make("CartPole-v1"))
env.metadata['render_fps'] = 240

# if GPU is to be used
device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
    
# BATCH_SIZE is the number of transitions sampled from the replay buffer
# GAMMA is the discount factor as mentioned in the previous section
# EPS_START is the starting value of epsilon
# EPS_END is the final value of epsilon
# EPS_DECAY controls the rate of exponential decay of epsilon, higher means a slower decay
# TAU is the update rate of the target network
# LR is the learning rate of the AdamW optimizer
BATCH_SIZE = 256
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 1000
TAU = 0.005
LR = 1e-4

# Get number of actions from gym action space
n_actions = env.action_space.n
# Get the number of state observations
state, info = env.reset()
n_observations = len(state)

policy_net = DuelingDQN(n_observations, n_actions).to(device)
target_net = DuelingDQN(n_observations, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())

optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
memory = ReplayMemory(100000)


steps_done = 0


# Modify the select_action function to stop exploration once the threshold is met
def select_action(state):
    global steps_done, eps_threshold
    sample = random.random()
    
    # If threshold is reached, set epsilon to 0 (fully exploit)
    if consecutive_max_reward_episodes >= CONSECUTIVE_OPTIMAL_THRESHOLD:
        eps_threshold = 0.0
    else:
        eps_threshold = EPS_END + (EPS_START - EPS_END) * \
            math.exp(-1. * steps_done / EPS_DECAY)
        
    steps_done += 1

    if sample > eps_threshold:
        with torch.no_grad():
            return policy_net(state).max(1).indices.view(1, 1)
    else:
        return torch.tensor([[env.action_space.sample()]], device=device, dtype=torch.long)

def optimize_model():
    # Return if there are not enough samples in memory
    if len(memory) < BATCH_SIZE:
        return

    # Sample a batch of transitions from the replay memory
    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))  # Convert batch to Transition of batch-arrays

    # Create mask for non-final states (states after which episode ends)
    non_final_mask = torch.tensor(
        [s is not None for s in batch.next_state], device=device, dtype=torch.bool
    )
    
    # Stack non-final next states and batch states
    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute predicted Q-values for the batch: Q(s_t, a)
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Initialize next_state_values to zero for non-final states
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    
    # Compute the expected Q-values for non-final next states using the target network
    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1).values

    # Compute expected Q-values using the Bellman equation
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute the Huber loss (SmoothL1Loss) between predicted and expected Q-values
    loss = nn.SmoothL1Loss()(state_action_values, expected_state_action_values.unsqueeze(1))

    # Backpropagation and optimization
    optimizer.zero_grad()  # Clear previous gradients
    loss.backward()  # Compute gradients
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)  # Clip gradients to avoid explosion
    optimizer.step()  # Update model parameters


# Initialize a list to track total rewards per episode
episode_rewards = []

# Main training loop
if torch.cuda.is_available() or torch.backends.mps.is_available():
    num_episodes = 2000
else:
    num_episodes = 600

# Define the threshold for consecutive optimal rewards
CONSECUTIVE_OPTIMAL_THRESHOLD = 50
max_reward = 500

# Track the number of consecutive episodes achieving the max reward
consecutive_max_reward_episodes = 0

# Modify the select_action function to stop exploration once the threshold is met
def select_action(state):
    global steps_done, eps_threshold
    sample = random.random()
    
    # If threshold is reached, set epsilon to 0 (fully exploit)
    if consecutive_max_reward_episodes >= CONSECUTIVE_OPTIMAL_THRESHOLD:
        eps_threshold = 0.0
    else:
        eps_threshold = EPS_END + (EPS_START - EPS_END) * \
            math.exp(-1. * steps_done / EPS_DECAY)
        
    steps_done += 1

    if sample > eps_threshold:
        with torch.no_grad():
            return policy_net(state).max(1).indices.view(1, 1)
    else:
        return torch.tensor([[env.action_space.sample()]], device=device, dtype=torch.long)

# Main training loop
for i_episode in range(num_episodes):
    # Initialize the environment and get its state
    state, info = env.reset()
    state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    total_reward = 0  # Track total reward for the episode
    
    for t in count():
        action = select_action(state)
        observation, reward, terminated, truncated, _ = env.step(action.item())
        total_reward += reward  # Add the reward to the total
        reward = torch.tensor([reward], device=device)
        done = terminated or truncated

        if terminated:
            next_state = None
        else:
            next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

        # Store the transition in memory
        memory.push(state, action, next_state, reward)

        # Move to the next state
        state = next_state

        # Perform one step of the optimization (on the policy network)
        optimize_model()

        # Soft update of the target network's weights
        target_net_state_dict = target_net.state_dict()
        policy_net_state_dict = policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key] * TAU + target_net_state_dict[key] * (1 - TAU)
        target_net.load_state_dict(target_net_state_dict)

        if done:
            # Append total reward for this episode
            episode_rewards.append(total_reward)
            plot_rewards(episode_rewards, noise_type="Random_Noise")  # Plot rewards instead of durations
            break

# After the training loop
print('Training Complete')
plot_rewards(episode_rewards, show_result=True, noise_type="Random_Noise", save_path="training_rewards_noiseType.png")
save_model(policy_net, noise_type="Random_Noise", save_dir="models/", file_name="DDQN_cartpole_noiseType.h5")
plt.ioff()
plt.show()

# Close the environment
env.close()