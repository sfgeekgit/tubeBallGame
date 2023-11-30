import os
import subprocess
import csv
import shutil

base_path = '../dq_runs/'


configs = [
    {'batch_size': 10,
     'num_epochs': 500000, # 5e5
     #'num_epochs': 2.5e6, # 1000000

     "DECAY": 0.7,
     "LEARNING_RATE": 1e-3,
     'loss_function': 'MSE',
     #'loss_function': 'MAE',
     #'loss_function': 'Huber',
     "DYN_LEARNING_RATE" : False,
     "STEP_LEARN_RATE"   : False,
     #"STEP_LEARN_RATE"   : True,
     "NUM_TUBES"  : 4,
     "NUM_COLORS" : 2,
     #"TRAIN_LEVEL_TYPE":'scramble8',
     "TRAIN_LEVEL_TYPE":'scram_ceil',
     "TRAIN_LEVEL_PARAM": 12,

     "WRITE_LOG" : True,
     "EXHAUSTIVE" : False,
     "SQUARED_OUTPUT" : True,
     "WIN_REWARD" : 20
     }
]

configs.append(configs[0]) # make a 2nd copy
configs.append(configs[0]) # make another copy
#configs.append(configs[0]) # make another copy




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
            if isinstance(value, str):
                f.write(f'{key.upper()} = \"{value}\"\n')
            else:
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
