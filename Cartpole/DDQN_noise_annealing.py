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
        self.episode_count = 0  # Track the current episode
        self.noise_stage = 0  # Track which noise component is active

        # Initialize friction and wind attributes with random values
        self.friction = np.random.uniform(0.1, 0.3)
        self.wind_force = np.random.uniform(-0.1, 0.1)
        self.noise_enabled = [False, False, False]  # Gradual noise activation
        
    def enable_next_noise(self):
        """Enable the next noise component and log the activation."""
        noise_types = ["Wind Force Noise", "Friction Noise", "Gaussian Noise"]
        if self.noise_stage < len(self.noise_enabled):
            self.noise_enabled[self.noise_stage] = True
            print(f"Noise Stage {self.noise_stage + 1} Enabled: {noise_types[self.noise_stage]}")
            self.noise_stage += 1

    def get_noise_scale(self):
        """Calculate noise scale based on episode count, starting large and shrinking."""
        max_episodes = 2000  # Total episodes (adjust based on training loop)
        scale_start = 0.05  # Initial noise scale (high)
        scale_end = 0.01  # Final noise scale (low)
        progress = min(self.episode_count / max_episodes, 1.0)

        # Reverse the scaling: start with scale_start and shrink to scale_end
        return scale_start * (1 - progress) + scale_end * progress

    def randomize_environment(self):
        """Randomize the environment parameters."""
        self.env.unwrapped.length = np.random.uniform(0.4, 0.6)  # Random pole length
        self.env.unwrapped.masscart = np.random.uniform(0.8, 1.0)  # Random cart mass
        self.env.unwrapped.masspole = np.random.uniform(0.05, 0.10)  # Random pole mass
        
    def get_dynamic_range(self, base_min, base_max, noise_type="generic"):
        """Calculate the range for the noise, shrinking over episodes."""
        max_episodes = 3000  # Total episodes (adjust based on training loop)
        progress = min(self.episode_count / max_episodes, 1.0)

        # Gradually shrink the range from (base_min, base_max) to (0, 0)
        min_range = (1 - progress) * base_min + progress * 0
        max_range = (1 - progress) * base_max + progress * 0

        # Apply specific constraints for each noise type
        if noise_type == "wind":
            min_range = np.clip(min_range, -0.5, 0.5)
            max_range = np.clip(max_range, -0.5, 0.5)
        elif noise_type == "friction":
            min_range = np.clip(min_range, 0.1, 0.3)
            max_range = np.clip(max_range, 0.1, 0.3)

        return min_range, max_range



    def step(self, action):
        """Modify step method to include dynamic wind, friction, and Gaussian noise."""
        # Perform the step
        observation, reward, terminated, truncated, info = self.env.step(action)

        # Extract observation variables
        cart_position, cart_velocity, pole_angle, pole_velocity = observation

        # Apply active noise components with dynamic ranges
        if self.noise_enabled[0]:  # Wind force noise
            min_wind, max_wind = self.get_dynamic_range(-0.1, 0.1, noise_type="wind")
            wind_force = np.random.uniform(min_wind, max_wind)
            cart_velocity += wind_force
            print(f"Wind Force Range: ({min_wind}, {max_wind}), Applied: {wind_force}")

        if self.noise_enabled[1]:  # Friction noise
            min_friction, max_friction = self.get_dynamic_range(0.1, 0.3, noise_type="friction")
            friction = np.random.uniform(min_friction, max_friction)
            cart_velocity -= friction * cart_velocity
            print(f"Friction Range: ({min_friction}, {max_friction}), Applied: {friction}")

        if self.noise_enabled[2]:  # Gaussian noise
            min_gauss, max_gauss = self.get_dynamic_range(0.0, 0.5)
            observation += np.random.normal(0, np.random.uniform(min_gauss, max_gauss), size=observation.shape)
            print(f"Gaussian Noise Range: ({min_gauss}, {max_gauss})")


        # Update the observation and compute the reward
        new_observation = np.array([cart_position, cart_velocity, pole_angle, pole_velocity])
        stability_bonus = 1.0 - abs(pole_angle / (2 * math.pi))
        reward += stability_bonus
        return new_observation, reward, terminated, truncated, info



    def reset(self, **kwargs):
        """Randomize episode-level parameters (e.g., pole length, mass) and reset."""
        self.randomize_environment()
        self.episode_count += 1  # Increment the episode count
        return self.env.reset(**kwargs)

env = RandomizedCartPole(gym.make("CartPole-v1"))

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
# LR is the learning rate of the ``AdamW`` optimizer
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

# Uses epsilon-greedy action which balances exploration and exploitation
def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return the largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            return policy_net(state).max(1).indices.view(1, 1)
    else:
        return torch.tensor([[env.action_space.sample()]], device=device, dtype=torch.long)
    
episode_rewards = []
       
# Updates the DQN's weights using experiences from the replay buffer
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
        
if torch.cuda.is_available() or torch.backends.mps.is_available():
    num_episodes = 2000
else:
    num_episodes = 50

consecutive_max_reward_episodes = 0  # Track consecutive max-reward episodes
max_consecutive_max_reward_episodes = 5  # Threshold to enable next noise
max_reward = 500  # Define what constitutes a max reward

for i_episode in range(num_episodes):
    # Initialize the environment and get its state
    state, info = env.reset()
    state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    total_reward = 0
    
    # Select action, receive rewards
    for t in count():
        action = select_action(state)
        observation, reward, terminated, truncated, _ = env.step(action.item())
        reward = max(min(reward, 1), -1)
        reward = torch.tensor([reward], device=device)
        total_reward += reward.item()  # Accumulate reward
        done = terminated or truncated

        # Observing next state
        if terminated:
            next_state = None
        else:
            next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

        # Store the transition in replay memory
        priority=1.0
        memory.push(state, action, next_state, reward)

        # Move to the next state
        state = next_state

        # Perform one step of the optimization (on the policy network)
        optimize_model()

        # Soft update of the target network's weights
        # θ′ ← τ θ + (1 −τ )θ′
        target_net_state_dict = target_net.state_dict()
        policy_net_state_dict = policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
        target_net.load_state_dict(target_net_state_dict)

        if done:
            # Check if the reward for this episode reaches the max reward
            if total_reward >= max_reward:
                consecutive_max_reward_episodes += 1
            else:
                consecutive_max_reward_episodes = 0

            # Gradually enable new noise components
            if consecutive_max_reward_episodes >= max_consecutive_max_reward_episodes:
                env.enable_next_noise()
                consecutive_max_reward_episodes = 0

            episode_rewards.append(total_reward)  # Store total reward for the episode
            plot_rewards(episode_rewards, noise_type="Annealing_Noise")  # Plot rewards instead of durations
            break

# After the training loop
print('Training Complete')
plot_rewards(episode_rewards, show_result=True, noise_type="Annealing_Noise", save_path="training_rewards_noiseType.png")
save_model(policy_net, noise_type="Annealing_noise", save_dir="models/", file_name="DDQN_cartpole_noiseType.h5")
plt.ioff()
plt.show()

# Close the environment
env.close()