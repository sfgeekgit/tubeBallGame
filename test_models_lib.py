from test_tube import TestTube, TUBE_LENGTH, move_allowed, show_tubes_up
from bfs import breadth_first_search, a_star_search
import level_gen
import torch
import deepQlib
import test_models_lib



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
        
        run_res = run_test(mynet, level)
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

def run_test(model, level):
    test_tubes = level.get_tubes()
    #show_tubes_up(test_tubes, False)

    a_star_path = a_star_search(test_tubes, quiet=True)
    a_star_len = len(a_star_path) -1  # -1 because the path includes the initial state
    #print(f"A star steps to solve: {a_star_len}")



    steps = 0
    reward = 0
    while (reward == 0) and steps < 4+ (a_star_len * 2):
        steps += 1
        #print ("Step ", steps)
        net_input = level_gen.tubes_to_list(test_tubes, NUM_TUBES)  
        T = tube_list_to_tensor(net_input)

    
        #logits = mynet(T)  # that calls forward because __call__ is coded magic backend
        logits = model(T)  
        #print(f"{logits=}")
        logits_mv = logits.max(dim=0).values
        #print(f"{logits_mv=}")
        to_from_logs = logits.argmax()  # do this after training

        move_to = to_from_logs // NUM_TUBES
        move_from = to_from_logs - move_to * NUM_TUBES
        to_from = [move_to, move_from]
        

        #print(f"{to_from=}")
        
        reward = deepQlib.reward_f(test_tubes, to_from)
        #print(f"{reward=}")

        new_state = next_state(test_tubes, to_from)    
        #show_tubes_up(new_state, False)
        test_tubes = new_state
            
        if reward == -3:
            #print (f"Fail. Invalid move in step {steps}\n\n")
            return -1
            
        elif reward == 10:
            #print (f"You did it! Solved in {steps} steps. (Best possible is {a_star_len})\n\nFIREWORKS\n\n\n")
            return steps - a_star_len

    # still here? Failed
    return -1
