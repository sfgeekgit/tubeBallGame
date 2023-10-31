import torch
import torch.nn as nn 
import torch.nn.functional as F



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


def next_state(state, move):
    test_tubes = state
    move_from  = move[0]
    move_to    = move[1]

    if move_from == move_to:
        return state
    
    allowed, _ = move_allowed(test_tubes, move_from, move_to)
    if not allowed:
        return state

    new_tubes = [tt.copy() for tt in test_tubes]
    new_tubes[move_to].add_ball(new_tubes[move_from].pop_top())
    return new_tubes

    

level = level_gen.GameLevel()

level.load_demo_easy()
level.load_demo_one_move()

test_tubes = level.get_tubes()

net_input = level_gen.tubes_to_list(test_tubes)
#print("net input", net_input)
T = tube_list_to_tensor(net_input)


mynet  = NeuralNetwork()
logits = mynet(T)  # that calls forward because __call__ is coded magic backend
print("logits" , logits)
logits = logits.view(2,4)
print("logits" , logits)
### to_from = logits.argmax(1).tolist()  # do this after training
### print(to_from)


# move from is first half of logits,  NUM_TUBES, 
# move to   is next  half of logits,  NUM_TUBES, 

#these "rand" should be actual rands...
# these are the values we are randomly checking with bellman
rand_from = torch.tensor(1)  # do move, and compute right side
rand_to   = torch.tensor(3)  # do move, and compute right side


one_hots = torch.stack((F.one_hot(rand_to, NUM_TUBES), F.one_hot(rand_from, NUM_TUBES)))


print(logits)
logits = logits * one_hots
print(logits)


logits = logits.sum(1)   # This is the left bellman!!  (need to turn this into one number, sum them or whatever)
# question, what is the best way to combine the 2 numbers (to and from) into 1 number for loss function. Start with just a sum, will work well enough, but maybe get something better. Whatever it is, needs to be used consistantly.
print(logits)

quit()




#rand_to_from = (rand_to, rand_from)



reward    =   reward_f(test_tubes, rand_to_from)  # bellmen_left side (I think??)
new_state = next_state(test_tubes, rand_to_from)
# run network with this state, get back 8 values
# split into 2x4
# take the max of each (get 2 numbers)
# that's the right!! sum the 2 numbers add the reward, that is the right side of the bellman eq.
# bell_right = reward + (num1 + num2) * decay   (assuming that we "sum" the left side, do the same here)
# decay is something like 0.95
# do some magic so that an illegal move will make bell_right be ONLY the reward (nuke the num1+num2 stuff)

#loss = F.mse_loss(bell_left, bell_right)
#loss.backward() # or something...


# then pass these to a loss function and train on that.

# Q(s, a) = R + max(Q(s', a))
