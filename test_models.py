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


def run_2x4_tests(model_file_path):
    mynet = torch.jit.load(model_file_path)

    fail_cnt = 0
    pass_cnt = 0
    xtra_move_cnt = 0

    num_test_levels = 100
    for i in range(1,num_test_levels):

        level_path = '../tubeballgame_stuff/easy_24_lvls/' + str(i) + '.lvl'
        level = level_gen.GameLevel()
        
        level.load_from_disk(level_path)
        
        run_res = lib.run_test(mynet, level)
        if run_res == -1:
            fail_cnt += 1
        else:
            pass_cnt += 1
            xtra_move_cnt += run_res


    pass_perc = int(100 * pass_cnt / num_test_levels)
    #print (f"{pass_perc} perecnt passed!")
    avg_xtra_moves = 0
    if pass_cnt > 1:
        avg_xtra_moves = xtra_move_cnt / pass_cnt # round this to 2 decimal places
        avg_xtra_moves = round(avg_xtra_moves, 2)
    #    print (f"{xtra_move_cnt} above optimal (lower is better)")
    
    return (pass_perc, avg_xtra_moves)




import glob

model_paths = glob.glob('../dq_runs/config_*/model.pth')
model_paths.sort() 
#model_paths = model_paths[15:]
# remove the first few models they are garbage


results = {}
extra_moves = {}

for idx, model_path in enumerate(model_paths):
    config_id = model_path.split('/')[-2]
    if len(config_id) < 9: # single digit config id, skip these
        continue
    res = run_2x4_tests(model_path)
    results[config_id] = res[0]
    extra_moves[config_id] = res[1]

sorted_results = dict(sorted(results.items(), key=lambda item: item[1], reverse=True))

import csv

#for idx, config_id in enumerate(sorted_results.keys()):
for idx, config_id in enumerate(results.keys()):
    if idx >= 60:
        break
    config_file_path = f'../dq_runs/{config_id}/config.csv'
    with open(config_file_path, 'r') as f:
        reader = csv.reader(f)
        config_dict = {rows[0]:rows[1] for rows in reader}
    num_epochs = config_dict.get('num_epochs', 'Not found')
    num_epochs = int(float(num_epochs))
    batch_size = int(float(config_dict.get('batch_size', 'Not found')))
    win_reward = float(config_dict.get('WIN_REWARD', 10))
    loss_fun = config_dict.get('loss_function', 'mSe')
    lvl_type = config_dict.get('TRAIN_LEVEL_TYPE', 'one_or_two')



    # {extra_moves[config_id]} avg extra moves,
    print(f"{config_id}: {sorted_results[config_id]}% passed, {config_dict['DECAY']=} {win_reward=} \t{lvl_type=} \tep*batch: {num_epochs * batch_size}")


