import torch
import torch.nn as nn 

from typing import Type
from bfs import breadth_first_search, a_star_search
from test_tube import TestTube, TUBE_LENGTH, move_allowed, show_tubes_up
from colors import to_colored_ball
import time
import level_gen

## Work in progress....


NUM_TUBES  = 4
NUM_COLORS = 2

INPUT_SIZE = ( NUM_COLORS +1 ) * NUM_TUBES * TUBE_LENGTH

HIDDEN_SIZE = INPUT_SIZE  # why not...

OUTPUT_SIZE = NUM_TUBES * 2   # ball from and ball to



def tube_list_to_tensor(tubes):
    dic = {}
    dic[0] = [1,0,0]
    dic[1] = [0,1,0]
    dic[2] = [0,0,1]

    t_input = []

    for ball in tubes:
        t_input.extend(dic[ball])

    T = torch.tensor(t_input)
    #print(T)
    return T.float()




class NeuralNetwork(nn.Module):

    # might add more layers... Just get prototype working first

    def __init__(self):
        super().__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(INPUT_SIZE, HIDDEN_SIZE),
            nn.ReLU(),
            nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE),
            nn.ReLU(),
            nn.Linear(HIDDEN_SIZE, OUTPUT_SIZE),
        )

    def forward(self, x):
        logits = self.linear_relu_stack(x)        
        return logits




def reward_f (state, move):   # stat is list of tubes, move is tuple {to, from}
    reward = {}
    reward['invalid_move']    = -3
    reward['winning_move']    = 10
    reward['meh']             =  0

    test_tubes = state
    move_from  = move[0]
    move_to    = move[1]
    show_tubes_up(test_tubes, False)
    print("from to", move_from, move_to)

    
    if move_from == move_to:
        print ("no way")
        return reward['invalid_move']

    allowed, _ = move_allowed(test_tubes, move_from, move_to)
    print("allowed?", allowed)
    if not allowed:
        return reward['invalid_move']
    
    #if allowed:
    new_tubes = [tt.copy() for tt in test_tubes]
    new_tubes[move_to].add_ball(new_tubes[move_from].pop_top())
    print("new tubes")
    show_tubes_up(new_tubes, False)    
    if all(tt.is_complete() or tt.is_empty() for tt in test_tubes):
        print ("Winning state!")
        return reward['winning_move']
        
    
    return reward['meh']
    

level = level_gen.GameLevel()

level.load_demo_easy()

test_tubes = level.get_tubes()

net_input = level_gen.tubes_to_list(test_tubes)
#print("net input", net_input)
T = tube_list_to_tensor(net_input)


mynet  = NeuralNetwork()
logits = mynet(T)  # that calls forward because __call__ is coded magic backend
#print(logits)
logits = logits.view(2,4)
#print(logits)
to_from = logits.argmax(1).tolist()
#print(to_from)

reward_f(test_tubes, to_from)

# move from is first half of logits,  NUM_TUBES, 
# move to   is next  half of logits,  NUM_TUBES, 
