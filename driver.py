import os
import subprocess
import csv
import shutil

base_path = '../dq_runs/'

# Define your configurations
configs = [
    {'batch_size': 4,
     'num_epochs': 5000,
     "DECAY": 0.88,
     "LEARNING_RATE": 1e-3,
     "BATCH_SIZE": 3,
     'loss_function': 'mse',
     "DYN_LEARNING_RATE" : False,
     "STEP_LEARN_RATE"   : True,
     "NUM_TUBES"  : 4,
     "NUM_COLORS" : 2,
     "WRITE_LOG" : True,
     "EXHAUSTIVE" : False,
     "SQUARED_OUTPUT" : True,
     }
]

configs.append(configs[0]) # make a 2nd copy
configs.append(configs[0]) # make another copy
configs.append(configs[0]) # make another copy




# Path to the other script
training_script = './deepQsolver.py'

# Get the number of existing subdirectories
num_dirs = len(next(os.walk(base_path))[1])



for i, config in enumerate(configs):    
    # Create a new directory for this configuration
    dir_name = f'config_{num_dirs + i}'
    os.makedirs(base_path + dir_name, exist_ok=False)
    config['dir_name'] = dir_name


    # Write the configuration to a Python file
    with open(os.path.join(base_path + dir_name, 'config.py'), 'w') as f:
        for key, value in config.items():
            f.write(f'{key.upper()} = {value}\n')

    # Write the configuration to a CSV file
    with open(os.path.join(base_path + dir_name, 'config.csv'), 'w') as f:
        writer = csv.writer(f)
        for key, value in config.items():
            writer.writerow([key, value])

    # Call the other script with this configuration
    #subprocess.run(['python3', training_script, '--config', os.path.join(dir_name, 'config.py')])

    # Move the generated neural network to the new directory
    # shutil.move('neural_network.h5', os.path.join(dir_name, 'neural_network.h5'))
