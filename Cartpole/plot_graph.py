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

# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()

def plot_rewards(episode_rewards, show_result=False, noise_type="Noise", save_dir="images/", save_path="training_rewards_noiseType.png"):
    """
    Plot rewards for training or testing.
    
    Args:
        episode_rewards (list): List of episode rewards to plot.
        show_result (bool): Whether to show the final result plot.
        noise_type (str): Noise type to include in the title.
        save_dir (str): Directory to save the plot.
        save_path (str): Path template for saving the plot.
    """
    # Format the save path
    save_path_with_noise = save_path.replace("noiseType", noise_type)

    plt.figure(1)
    rewards_t = torch.tensor(episode_rewards, dtype=torch.float)
    if show_result:
        plt.title(f'Result ({noise_type})')
    else:
        plt.clf()
        plt.title(f'Training {noise_type}...')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.plot(rewards_t.numpy())

    # Take 100-episode averages and plot them too
    if len(rewards_t) >= 100:
        means = rewards_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy(), label='100-Episode Avg')

    plt.legend()
    plt.pause(0.001)  # pause a bit so that plots are updated
    
    # Save the plot to the specified directory
    os.makedirs(save_dir, exist_ok=True)
    file_path = os.path.join(save_dir, save_path_with_noise)
    plt.savefig(file_path)
    
