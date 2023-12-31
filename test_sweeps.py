#import torch
#import level_gen
import test_models_lib as lib
from test_tube import TestTube, TUBE_LENGTH, move_allowed, show_tubes_up
#from colors import to_colored_ball
#from bfs import breadth_first_search, a_star_search
#import sys




NUM_TUBES = 4
NUM_COLORS = 2
SQUARED_OUTPUT = True



import glob


model_paths = glob.glob('../py/wandb_ball_runs/try_bayes/sweep_con_*/model.pth')
model_paths.sort() 


results = {}
extra_moves = {}

for idx, model_path in enumerate(model_paths):
    config_id = model_path.split('/')[-2]

    id_num = int(config_id.split('_')[-1])
    #if not (id_num in (31,33,34)  or id_num >= 198 or (id_num >= 124 and id_num <= 137)):
    #    continue
    #if id_num > 111:
    #    continue


    #res = lib.run_2x4_tests(model_path)
    res = lib.run_2x4_tests(model_path)
    print(config_id , res[0])
    
    results[config_id] = res[0]
    extra_moves[config_id] = res[1]

sorted_results = dict(sorted(results.items(), key=lambda item: item[1], reverse=True))

#import csv

config_dict = {}

for idx, config_id in enumerate(sorted_results.keys()):
#for idx, config_id in enumerate(results.keys()):
    if idx >= 9999:
        break
#    config_file_path = f'../dq_runs/{config_id}/config.csv'
    config_file_path = f'../py/wandb_ball_runs/try_bayes/{config_id}/config.py'

    #open the file and read the config each line is a key value pair
    with open(config_file_path, 'r') as f:
        # read a line of the file
        line = f.readline()
        # while there are still lines to read
        while line:
            # if the line is not a comment
            if not line.startswith('#'):
                # split the line into a key value pair
                key, value = line.split('=')
                # strip the whitespace
                key = key.strip()
                value = value.strip()

                # if the value is a string
                if value.startswith('"'):
                    # strip the quotes
                    value = value.strip('"')
                # otherwise
                else:
                    # strip the newline
                    value = value.strip('\n')
                # add the key value pair to the config dictionary
                config_dict[key] = value
            # read the next line
            line = f.readline()


    num_epochs = config_dict.get('NUM_EPOCHS', 'Not found')
    batch_size = int(float(config_dict.get('BATCH_SIZE', 'Not found')))
    win_reward = float(config_dict.get('WIN_REWARD', 10))
    decay = float(config_dict.get('DECAY', .95))
    loss_fun = config_dict.get('LOSS_FUNCTION', 'mSe')
    lvl_type = config_dict.get('TRAIN_LEVEL_TYPE', 'one_or_two')
    nn_shape = config_dict.get('NN_SHAPE', "['I', 'I', 'I', 'O']")
    learning_rate = float(config_dict.get('LEARNING_RATE', .001))


    # sorted_results[config_id] as a 2 digit string with leading zeros if needed
    pass_perc = str(sorted_results[config_id]).zfill(2)

    #decay with zeros filled to 2 decimal places
    decay = str(decay).zfill(4)
    
    # {extra_moves[config_id]} avg extra moves,
    #print(f"{config_id}: {pass_perc}% pass, dec={decay}, w_rew={win_reward}, batch={batch_size}, \t{lvl_type=} \tnn={nn_shape} \tep*batch: {int(num_epochs) * int(batch_size)}")
    print(f"{config_id}: {pass_perc}% pass, dec={decay}, learn={learning_rate}, batch={batch_size},\tnn={nn_shape}     \tep*batch: {int(num_epochs) * int(batch_size)}")
