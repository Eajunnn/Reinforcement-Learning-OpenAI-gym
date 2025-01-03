import os
import h5py

# Define a directory for saving the model and plot
save_dir = "models/"  # Change this to your desired directory path
os.makedirs(save_dir, exist_ok=True)  # Create directory if it doesn't exist

# Save the trained policy_net model
def save_model(model, noise_type, save_dir, file_name="dqn_cartpole_noiseType.h5"):
    """
    Save the model's weights to an h5 file, including noise_type in the file name.

    Parameters:
        model (nn.Module): The trained model to be saved.
        noise_type (str): The noise type to be included in the file name.
        save_dir (str): The directory to save the model.
        file_name (str): The base name for the file. Defaults to "dqn_cartpole_nonoise_5_bla.h5".

    Returns:
        None
    """
    # Ensure the directory exists
    os.makedirs(save_dir, exist_ok=True)

    # Update the file name to include the noise type
    file_name_with_noise = file_name.replace("noiseType", noise_type)  # Replace placeholder with noise type

    # Create the full path to save the model
    file_path = os.path.join(save_dir, file_name_with_noise)

    # Convert the model's state_dict to numpy arrays
    model_weights = {k: v.cpu().numpy() for k, v in model.state_dict().items()}

    # Save to .h5 file using h5py
    with h5py.File(file_path, 'w') as h5file:
        for key, value in model_weights.items():
            h5file.create_dataset(key, data=value)

    print(f"Model saved as {file_path}")
