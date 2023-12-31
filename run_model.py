import torch
import level_gen
from test_tube import TestTube, TUBE_LENGTH, move_allowed, show_tubes_up
from colors import to_colored_ball
from bfs import breadth_first_search, a_star_search
import deepQlib
import sys
import time


if len(sys.argv) > 1:
    model_file_path = '../dq_runs/config_' + sys.argv[1] + '/model.pth'
else:
    model_file_path = '../dq_runs/config_14/model.pth'

level_path = False
#level_path = '../tubeballgame_stuff/easy_24_lvls/1.lvl'

mynet = torch.jit.load(model_file_path)


NUM_TUBES = 4
NUM_COLORS = 2
SQUARED_OUTPUT = True


def tube_list_to_tensor(tubes):  # this should be in another file...
    dic = {}
    for i in range(NUM_COLORS +1):
        dic[i] = [0]*(NUM_COLORS+1)
        dic[i][i] = 1

    t_input = []

    for ball in tubes:
        t_input.extend(dic[ball])

    T = torch.tensor(t_input)
    return T.float()




def next_state(state, move):  # move is to,from
    test_tubes = state
    move_to    = move[0]
    move_from  = move[1]

    if move_from == move_to:
        return state
    
    allowed, _ = move_allowed(test_tubes, move_from, move_to)
    if not allowed:
        return state

    new_tubes = [tt.copy() for tt in test_tubes]
    new_tubes[move_to].add_ball(new_tubes[move_from].pop_top())
    return new_tubes




level = level_gen.GameLevel()

if level_path:
    level.load_from_disk(level_path)
else:
    #level.load_demo_one_or_two_move_rand(NUM_TUBES)
    level.load_demo_easy()

test_tubes = level.get_tubes()
show_tubes_up(test_tubes, False)

a_star_path = a_star_search(test_tubes, quiet=True)
a_star_len = len(a_star_path) -1  # -1 because the path includes the initial state
print(f"A star steps to solve: {a_star_len}")





steps = 0
reward = 0
while (reward == 0) and steps < 4+ (a_star_len * 2):
    steps += 1
    print ("Step ", steps)
    net_input = level_gen.tubes_to_list(test_tubes, NUM_TUBES)  
    T = tube_list_to_tensor(net_input)

  
    logits = mynet(T)  # that calls forward because __call__ is coded magic backend
    #print(f"{logits=}")
    logits_mv = logits.max(dim=0).values
    print(f"{logits_mv=}")
    to_from_logs = logits.argmax()  # do this after training

    move_to = to_from_logs // NUM_TUBES
    move_from = to_from_logs - move_to * NUM_TUBES
    to_from = [move_to, move_from]
    

    print(f"{to_from=}")
    
    reward = deepQlib.reward_f(test_tubes, to_from)
    print(f"{reward=}")

    new_state = next_state(test_tubes, to_from)    
    show_tubes_up(new_state, False)
    test_tubes = new_state

    if (steps > 3):
        time.sleep(.05)
        
    if reward == -3:
        print (f"Fail. Invalid move in step {steps}\n\n")
        
    elif reward == 10:
        print (f"You did it! Solved in {steps} steps. (Best possible is {a_star_len})\n\nFIREWORKS\n\n\n")
        



#output = loaded_model(input_data)


#torch.save(mynet.state_dict(), model_file_path)
#model = TheModelClass(*args, **kwargs)
#the_model.load_state_dict(torch.load(PATH))
#model.eval()


