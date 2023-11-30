from test_tube import TestTube, TUBE_LENGTH, move_allowed, show_tubes_up


def reward_f (state, move):   # state is list of tubes, move is tuple {to, from} 
    
    reward = {}
    reward['invalid_move']    = -3
    reward['winning_move']    = 10
#    reward['winning_move']    = WIN_REWARD
    reward['meh']             =  0

    test_tubes = state
    move_to    = move[0]
    move_from  = move[1]
    
    if move_from == move_to:
        return reward['invalid_move']

    allowed, _ = move_allowed(test_tubes, move_from, move_to)
    #print("allowed?", allowed)
    if not allowed:
        return reward['invalid_move']
    
    #if allowed:
    new_tubes = [tt.copy() for tt in test_tubes]
    new_tubes[move_to].add_ball(new_tubes[move_from].pop_top())
    

    if all(tt.is_complete() or tt.is_empty() for tt in new_tubes):
        #print ("Winning state!")
        return reward['winning_move']

    return reward['meh']


