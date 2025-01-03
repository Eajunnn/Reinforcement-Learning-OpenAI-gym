import random
import numpy as np
import gym

import cv2
from scipy import stats

# Tensorflow training imports
from collections import deque
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam
import tensorflow as tf

# Eager execution to speed up training speeds (A LOT!)
# tf.compat.v1.disable_eager_execution()

# Training monitoring imports
import datetime, os
from tqdm import tqdm
import time
from plot_results import plot_rewards

class RandomizedCarRacingTesting(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.env = env
        self.randomize_environment()

        # Initialize dynamic noise parameters
        self.friction = np.random.uniform(0.05, 1)  # Affects braking
        self.wind_force = np.random.uniform(-0.8, 0.8)  # Affects steering
        self.gaussian_noise_std = 0.5  # Affects observation noise

    def randomize_environment(self):
        """Randomize environment parameters at the start of an episode."""
        self.env.unwrapped.gravity = np.random.uniform(8.0, 10.0)  # Adjust gravity

    def apply_friction(self, brake_action):
        """Modify braking based on friction."""
        modified_brake = brake_action * (1 - self.friction)  # Reduces brake effectiveness
        print("Friction Force", self.friction)
        return np.clip(modified_brake, 0.0, 1.0)

    def apply_wind(self, steering_action):
        """Modify steering based on wind force."""
        modified_steering = steering_action + self.wind_force
        print("Wind Force", self.wind_force)
        return np.clip(modified_steering, -1.0, 1.0)

    def add_gaussian_noise(self, observation):
        """Add Gaussian noise to the observation."""
        noise = np.random.normal(0, self.gaussian_noise_std, size=observation.shape)
        print("Gaussian noise", self.gaussian_noise_std)
        return observation + noise

    def step(self, action):
        """
        Modify actions dynamically:
        - Apply wind to steering
        - Apply friction to braking
        - Add Gaussian noise to observations
        """
        steering, gas, brake = action

        # Modify actions
        modified_steering = self.apply_wind(steering)
        modified_brake = self.apply_friction(brake)

        # Create modified action
        modified_action = [modified_steering, gas, modified_brake]

        # Take a step in the environment
        observation, reward, terminated, truncated, info = self.env.step(modified_action)

        # Add Gaussian noise to the observation
        noisy_observation = self.add_gaussian_noise(observation)

        # Ensure the state is in the correct type (uint8) before returning it
        noisy_observation = np.clip(noisy_observation, 0, 255).astype(np.uint8)

        return noisy_observation, reward, terminated, truncated, info

    def reset(self, **kwargs):
        """Randomize environment parameters and reset the environment."""
        self.randomize_environment()
        self.friction = np.random.uniform(0.1, 1)  # Reset friction in range [0.1, 0.5]
        self.wind_force = np.random.uniform(-0.8, 0.8)   # Reset wind force in range [-1, 1]
        observation, info = self.env.reset(**kwargs)  # Capture both observation and info

        # Ensure the observation is in the correct type (uint8) before returning it
        observation = np.clip(observation, 0, 255).astype(np.uint8)

        return observation, info  # Return both observation and info