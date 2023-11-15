import os
import subprocess
import csv
import shutil

# Define your configurations
configs = [
    {'batch_size': 32, 'num_epochs': 100, 'loss_function': 'mse'},
    {'batch_size': 64, 'num_epochs': 200, 'loss_function': 'mae'},
    # Add more configurations as needed
]

# Path to the other script
training_script = './deepQsolver.py'

# Get the number of existing subdirectories
num_dirs = len([name for name in os.listdir('.') if os.path.isdir(name)])

for i, config in enumerate(configs):
    # Create a new directory for this configuration
    dir_name = f'config_{num_dirs + i}'
    os.makedirs(dir_name, exist_ok=True)

    # Write the configuration to a Python file
    with open(os.path.join(dir_name, 'config.py'), 'w') as f:
        for key, value in config.items():
            f.write(f'{key.upper()} = {value}\n')

    # Write the configuration to a CSV file
    with open(os.path.join(dir_name, 'config.csv'), 'w') as f:
        writer = csv.writer(f)
        for key, value in config.items():
            writer.writerow([key, value])

    # Call the other script with this configuration
    subprocess.run(['python', training_script, '--config', os.path.join(dir_name, 'config.py')])

    # Move the generated neural network to the new directory
    shutil.move('neural_network.h5', os.path.join(dir_name, 'neural_network.h5'))
