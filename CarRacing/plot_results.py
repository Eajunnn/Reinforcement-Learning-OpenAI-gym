import pandas as pd
import matplotlib.pyplot as plt
import os

# Define a mapping of noise types to file paths
noise_paths = {
    'Curriculum': 'rewards/Eajun/DDQN2/CURRICULUM/20241208-130335/episode_400.weights.csv',
    'Random': 'rewards/Eajun/DDQN2/RANDOM/20241206-013930/episode_400.weights.csv',
    'Annealing': 'rewards/Eajun/DDQN2/ANNEALING/20241206-014005/episode_400.weights.csv',
    'Stochastic': 'rewards/Eajun/DDQN2/STOCHASTIC/20241206-014040/episode_400.weights.csv',
    'DuelingCurriculum': 'rewards/Eajun/DDQN2/CURRICULUMDUELING/20241210-004430/episode_300.weights.csv',
    'Base': 'rewards/Eajun/DDQN2/NONOISE/20241208-130704/episode_400.weights.csv'

}

def plot_rewards(csv_file, save_dir='images', noise_type='Custom'):
    """Plot rewards for a given CSV file and save the graph."""
    # Load the CSV file for the specified noise type
    if not os.path.exists(csv_file):
        raise FileNotFoundError(f"CSV file not found: {csv_file}")

    data = pd.read_csv(csv_file, header=None, names=['rewards'])

    # Create an episode column (assuming the rewards are in episode order)
    data['episode'] = data.index + 1  # Starting episodes from 1

    # Plotting the graph based on the episode and rewards
    plt.figure(figsize=(10, 6))
    plt.plot(data['episode'], data['rewards'], marker='o', label=f'{noise_type} Rewards')

    # Graph formatting
    plt.title(f"Episode vs Rewards ({noise_type})")
    plt.xlabel("Episode")
    plt.ylabel("Rewards")
    plt.legend(loc="best")
    plt.grid(True)

    # Ensure the output directory exists
    os.makedirs(save_dir, exist_ok=True)

    # Save the plot to the specified directory
    output_path = os.path.join(save_dir, f'training_rewards_{noise_type.lower()}.png')
    plt.savefig(output_path)

    # Show the plot
    plt.show()

    print(f"Plot saved to {output_path}")
    


