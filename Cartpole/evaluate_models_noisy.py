import os
import torch
import gymnasium as gym
import h5py
import matplotlib.pyplot as plt
import torch.nn as nn
import numpy as np
import os
from  Agents import DuelingDQN

# Load the model weights from the .h5 file
def load_model(file_name, model):
    with h5py.File(file_name, 'r') as h5file:
        model_weights = {key: torch.tensor(h5file[key][:]) for key in h5file.keys()}
    model.load_state_dict(model_weights)

# Define the noisy environment wrapper
class RandomizedCartPole(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.env = env
        self.randomize_environment()
        
        # Initialize friction and wind attributes with random values
        self.friction = np.random.uniform(0.1, 0.5)
        self.wind_force = np.random.uniform(-1, 1)

    def randomize_environment(self):
        # Randomize only parameters that should remain constant within an episode
        self.env.unwrapped.gravity = np.random.uniform(0.6, 1.0)  # Random gravity
        self.env.unwrapped.length = np.random.uniform(0.2, 0.6)  # Random pole length
        self.env.unwrapped.masscart = np.random.uniform(0.6, 1.0)  # Random cart mass
        self.env.unwrapped.masspole = np.random.uniform(0.05, 0.2)  # Random pole mass

    def step(self, action):
        # Original step function to get initial observation, reward, etc.
        observation, reward, terminated, truncated, info = self.env.step(action)

        # Apply dynamic step-wise friction and wind
        cart_position, cart_velocity, pole_angle, pole_velocity = observation
        cart_acceleration = -self.friction * cart_velocity + self.wind_force
        # print("Friction", self.friction, "wind force", self.wind_force)
        cart_velocity += cart_acceleration

        # Create a new observation incorporating the modified cart_velocity
        new_observation = np.array([cart_position, cart_velocity, pole_angle, pole_velocity])

        # Adding noise to the new observation
        noise = np.random.normal(0, 0.02, size=new_observation.shape)  # Gaussian noise
        new_observation = new_observation + noise

        return new_observation, reward, terminated, truncated, info

    def reset(self, **kwargs):
        # Randomize episode-level parameters (e.g., pole length, mass)
        self.randomize_environment()
        # Re-randomize noise parameters explicitly for each episode
        self.friction = np.random.uniform(0.1, 0.5)  # Reset friction in range [0.1, 0.5]
        self.wind_force = np.random.uniform(-1, 1)   # Reset wind force in range [-1, 1]
        return self.env.reset(**kwargs)

# Initialize the environment
env = RandomizedCartPole(gym.make("CartPole-v1"))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Get the number of actions and observations
state, info = env.reset()
n_observations = len(state)
n_actions = env.action_space.n

def compute_convergence_rate(rewards, threshold, stability_episodes=5):
    """
    Compute the episode at which convergence is achieved.
    """
    for i in range(len(rewards) - stability_episodes):
        if all(r >= threshold for r in rewards[i:i + stability_episodes]):
            return i + stability_episodes  # Episode at convergence
    return len(rewards)  # If not converged, return total episodes

def compute_stability(rewards):
    """
    Compute the standard deviation of rewards.
    """
    return np.std(rewards)

def compute_exploration_efficiency(env, model, device, num_episodes=100):
    """
    Count the number of unique states visited during evaluation.
    """
    unique_states = set()
    for _ in range(num_episodes):
        state, info = env.reset()
        state = tuple(np.round(state, decimals=2))  # Round to reduce sensitivity
        unique_states.add(state)
        while True:
            with torch.no_grad():
                action = model(torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)).max(1).indices.item()
            next_state, _, terminated, truncated, _ = env.step(action)
            state = tuple(np.round(next_state, decimals=2))
            unique_states.add(state)
            if terminated or truncated:
                break
    return len(unique_states)

def compute_sample_efficiency(rewards, total_steps):
    """
    Compute the average reward per step.
    """
    return sum(rewards) / total_steps


# Function to test a single model and return total reward for the episode
def test_model(model, env, device, num_episodes=100, reward_threshold=500):
    total_rewards = []
    total_steps = 0
    unique_states = set()

    for i_episode in range(num_episodes):
        state, info = env.reset()
        state = tuple(np.round(state, decimals=2))
        unique_states.add(state)
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        total_reward = 0
        steps = 0

        while True:
            with torch.no_grad():
                action = model(state).max(1).indices.item()  # Select action
            state, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            steps += 1
            total_steps += 1

            # Add the state to the unique set
            state_tuple = tuple(np.round(state, decimals=2))
            unique_states.add(state_tuple)

            if terminated or truncated:
                break
            state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)

        total_rewards.append(total_reward)
        print(f"Episode {i_episode + 1}: Total Reward = {total_reward}")
    
    # Calculate the average reward for this model
    avg_reward = sum(total_rewards) / num_episodes
    print(f"Average Reward for this model: {avg_reward}")
    # Compute metrics
    avg_reward = np.mean(total_rewards)
    convergence_rate = compute_convergence_rate(total_rewards, reward_threshold)
    stability = compute_stability(total_rewards)
    exploration_efficiency = len(unique_states)
    sample_efficiency = compute_sample_efficiency(total_rewards, total_steps)

    return total_rewards, avg_reward, convergence_rate, stability, exploration_efficiency, sample_efficiency

# Function to load multiple models and test
def test_multiple_models(model_files, env, device):
    episode_rewards_by_model = []
    metrics_by_model = {
        "Avg Reward": [],
        "Convergence Rate": [],
        "Stability": [],
        "Exploration Efficiency": [],
        "Sample Efficiency": [],
    }

    for model_file in model_files:
        model = DuelingDQN(n_observations, n_actions).to(device)
        load_model(model_file, model)
        model.eval()
        
        # Test the model and get all metrics
        rewards, avg_reward, convergence_rate, stability, exploration_efficiency, sample_efficiency = test_model(model, env, device)
        episode_rewards_by_model.append(rewards)  # Collect episode-level rewards
        metrics_by_model["Avg Reward"].append(avg_reward)
        metrics_by_model["Convergence Rate"].append(convergence_rate)
        metrics_by_model["Stability"].append(stability)
        metrics_by_model["Exploration Efficiency"].append(exploration_efficiency)
        metrics_by_model["Sample Efficiency"].append(sample_efficiency)

    # Aggregate metrics across all models
    aggregated_metrics = {key: np.mean(values) for key, values in metrics_by_model.items()}
    
    # Aggregate episode-level rewards across models
    avg_rewards_per_episode = np.mean(episode_rewards_by_model, axis=0)  # Average rewards per episode across models
    return avg_rewards_per_episode, aggregated_metrics



# Group model files by noise type
model_files_grouped = {
    "No Noise": [
        "DDQN_cartpole_No_Noise_1.h5",
        "DDQN_cartpole_No_Noise_2.h5",
        "DDQN_cartpole_No_Noise_3.h5",
        "DDQN_cartpole_No_Noise_4.h5",
        "DDQN_cartpole_No_Noise_5.h5",
    ],
    "Random Noise": [
        "DDQN_cartpole_Random_Noise_1.h5",
        "DDQN_cartpole_Random_Noise_2.h5",
        "DDQN_cartpole_Random_Noise_3.h5",
        "DDQN_cartpole_Random_Noise_4.h5",
        "DDQN_cartpole_Random_Noise_5.h5",
    ],
    "Stochastic Noise": [
        "DDQN_cartpole_Stochastic_Noise_1.h5",
        "DDQN_cartpole_Stochastic_Noise_2.h5",
        "DDQN_cartpole_Stochastic_Noise_3.h5",
        "DDQN_cartpole_Stochastic_Noise_4.h5",
        "DDQN_cartpole_Stochastic_Noise_5.h5",
    ],
    "Curriculum Learning": [
        "DDQN_cartpole_Curriculum_1.h5",
        "DDQN_cartpole_Curriculum_2.h5",
        "DDQN_cartpole_Curriculum_3.h5",
        "DDQN_cartpole_Curriculum_4.h5",
        "DDQN_cartpole_Curriculum_5.h5",
    ],
    "Annealing Noise": [
        "DDQN_cartpole_Annealing_noise_1.h5",
        "DDQN_cartpole_Annealing_noise_2.h5",
        "DDQN_cartpole_Annealing_noise_3.h5",
        "DDQN_cartpole_Annealing_noise_4.h5",
        "DDQN_cartpole_Annealing_noise_5.h5",
    ],
}

# Initialize the environment
env = RandomizedCartPole(gym.make("CartPole-v1"))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Get the number of actions and observations
state, info = env.reset()
n_observations = len(state)
n_actions = env.action_space.n

# Plot the results for all models
# Test each group of models and compute the metrics
# Updated loop for testing and plotting
metrics_by_group = {}
avg_rewards_per_noise_type = {}

for noise_type, model_files in model_files_grouped.items():
    print(f"Testing models with {noise_type}...")
    model_paths = [os.path.join("models", file) for file in model_files]  # Adjust to your path
    avg_rewards_per_episode, metrics = test_multiple_models(model_paths, env, device)
    avg_rewards_per_noise_type[noise_type] = avg_rewards_per_episode  # Store episode-level rewards for plotting
    metrics_by_group[noise_type] = metrics

# Plot the average rewards per episode for all noise types
# Ensure the directory exists
save_dir = "images/"
os.makedirs(save_dir, exist_ok=True)

# Save path for the plot
plot_save_path = os.path.join(save_dir, "testing_result_noise_3.png")

# Plot the graph and save it
plt.figure()
for noise_type, avg_rewards in avg_rewards_per_noise_type.items():
    plt.plot(avg_rewards, label=noise_type)

plt.title("Average Rewards Per Episode (Across Models)")
plt.xlabel("Episode")
plt.ylabel("Average Reward")
plt.legend()

# Save the plot
plt.savefig(plot_save_path)
print(f"Plot saved to {plot_save_path}")

# Show the plot
plt.show()


# Print the aggregated metrics for each noise type
for noise_type, metrics in metrics_by_group.items():
    print(f"Noise Type: {noise_type}")
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.2f}")