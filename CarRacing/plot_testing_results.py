import numpy as np
import matplotlib.pyplot as plt
import os

def compute_metrics(rewards):
    """Compute convergence rate, stability, and exploration efficiency."""
    # Convergence rate: Identify the number of episodes to reach 90% of max reward
    max_reward = np.max(rewards)
    threshold = 0.9 * max_reward
    convergence_episode = next((i + 1 for i, r in enumerate(rewards) if r >= threshold), len(rewards))
    
    # Stability: Standard deviation of rewards after convergence
    stable_rewards = rewards[convergence_episode - 1:]
    stability = np.std(stable_rewards) if len(stable_rewards) > 0 else np.nan
    
    # Exploration efficiency: Improvement in rewards over the first half
    first_half = rewards[:len(rewards) // 2]
    exploration_efficiency = np.mean(first_half) if len(first_half) > 0 else np.nan
    
    return convergence_episode, stability, exploration_efficiency

def plot_multiple_csv_with_metrics(csv_paths, labels, save_dir="images"):
    """
    Plot run rewards from multiple CSV files on a single graph and compute metrics.
    
    Args:
        csv_paths (list of str): List of file paths to the CSV files.
        labels (list of str): List of labels corresponding to each CSV file.
    """
    plt.figure(figsize=(10, 6))
    
    for csv_path, label in zip(csv_paths, labels):
        # Load the CSV file
        data = np.loadtxt(csv_path, delimiter=",")
        
        # Extract run rewards (all rows except the last row, column 0)
        run_rewards = data[:-1, 0]
        avg_row = data[-1]  # Last row contains stats

        # Compute average, max, and min rewards
        r_avg, _, _, r_max, r_min, _ = avg_row

        # Compute additional metrics
        convergence_episode, stability, exploration_efficiency = compute_metrics(run_rewards)
        
        # Print metrics
        print(f"{label}:")
        print(f"  Avg Reward: {r_avg:.2f}, Max Reward: {r_max:.2f}, Min Reward: {r_min:.2f}")
        print(f"  Convergence Episode: {convergence_episode}, Stability: {stability:.2f}, Exploration Efficiency: {exploration_efficiency:.2f}\n")
        
        # Plot the rewards
        plt.plot(range(1, len(run_rewards) + 1), run_rewards, label=label, marker='o')

   # Graph formatting
    plt.title("Run Rewards from Multiple Tests")
    plt.xlabel("Test Runs")
    plt.ylabel("Rewards")
    plt.legend(loc="best")
    plt.grid(True)

    # Ensure the output directory exists
    os.makedirs(save_dir, exist_ok=True)

    # Save the plot
    output_path = os.path.join(save_dir, "run_rewards_comparison_3.png")
    plt.savefig(output_path)
    print(f"Plot saved to {output_path}")
    
    # Show the plot
    plt.show()

# Example usage
csv_paths = [
    "test_rewards/Eajun/DDQN2/NONOISE/20241211-225510/episode_400.weights_run_rewards.csv",
    "test_rewards/Eajun/DDQN2/ANNEALING/20241211-234316/episode_400.weights_run_rewards.csv",
    "test_rewards/Eajun/DDQN2/CURRICULUM/20241211-225454/episode_400.weights_run_rewards.csv",
    "test_rewards/Eajun/DDQN2/RANDOM/20241211-231427/episode_400.weights_run_rewards.csv",
    "test_rewards/Eajun/DDQN2/STOCHASTIC/20241211-225503/episode_400.weights_run_rewards.csv",
    "test_rewards/Eajun/DDQN2/CURRICULUMDUELING/20241211-225515/episode_300.weights_run_rewards.csv",
]
labels = ["Base", "Annealing", "Curriculum", "Random", "Stochastic", "Curriculum Dueling"]

plot_multiple_csv_with_metrics(csv_paths, labels)
