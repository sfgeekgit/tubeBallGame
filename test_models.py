import torch
import level_gen
import test_models_lib as lib
from test_tube import TestTube, TUBE_LENGTH, move_allowed, show_tubes_up
#from colors import to_colored_ball
from bfs import breadth_first_search, a_star_search
import sys




NUM_TUBES = 4
NUM_COLORS = 2
SQUARED_OUTPUT = True



import glob

model_paths = glob.glob('../dq_runs/config_*/model.pth')
model_paths.sort() 
#model_paths = model_paths[15:]
# remove the first few models they are garbage
#model_paths = model_paths[:48] + model_paths[66:]

results = {}
extra_moves = {}

for idx, model_path in enumerate(model_paths):
    config_id = model_path.split('/')[-2]

    id_num = int(config_id.split('_')[-1])
    if not (id_num in (31,33,34)  or id_num >= 198 or (id_num >= 124 and id_num <= 137)):
        continue



    #res = lib.run_2x4_tests(model_path)
    res = lib.run_2x4_tests(model_path)

    results[config_id] = res[0]
    extra_moves[config_id] = res[1]

sorted_results = dict(sorted(results.items(), key=lambda item: item[1], reverse=True))

import csv

#for idx, config_id in enumerate(sorted_results.keys()):
for idx, config_id in enumerate(results.keys()):
    if idx >= 9999:
        break
    config_file_path = f'../dq_runs/{config_id}/config.csv'
    with open(config_file_path, 'r') as f:
        reader = csv.reader(f)
        config_dict = {rows[0]:rows[1] for rows in reader}
    num_epochs = config_dict.get('num_epochs', 'Not found')
    num_epochs = int(float(num_epochs))
    batch_size = int(float(config_dict.get('batch_size', 'Not found')))
    win_reward = float(config_dict.get('WIN_REWARD', 10))
    decay = float(config_dict.get('DECAY', .95))
    loss_fun = config_dict.get('loss_function', 'mSe')
    lvl_type = config_dict.get('TRAIN_LEVEL_TYPE', 'one_or_two')
    nn_shape = config_dict.get('NN_SHAPE', "['I', 'I', 'I', 'O']")



    # {extra_moves[config_id]} avg extra moves,
    print(f"{config_id}: {sorted_results[config_id]}% pass, dec={decay}, w_rew={win_reward}, batch={batch_size}, \t{lvl_type=} \tnn={nn_shape} \tep*batch: {num_epochs * batch_size}")


