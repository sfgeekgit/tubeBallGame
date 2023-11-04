import torch
import torch.nn as nn 
import torch.nn.functional as F
import random

from typing import Type
from bfs import breadth_first_search, a_star_search
from test_tube import TestTube, TUBE_LENGTH, move_allowed, show_tubes_up
from colors import to_colored_ball
import time
import level_gen


# work in progress


NUM_TUBES  = 4
NUM_COLORS = 2

INPUT_SIZE = ( NUM_COLORS +1 ) * NUM_TUBES * TUBE_LENGTH

HIDDEN_SIZE = INPUT_SIZE  # why not...


OUTPUT_SIZE = NUM_TUBES * 2   # ball from and ball to

DECAY = 0.95
LEARNING_RATE = 1e-3


NUM_EPOCHS = 5000


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



#####    reward    =   reward_f(test_tubes, rand_to_from)  
def reward_f (state, move):   # stat is list of tubes, move is tuple {to, from}  # todo, is this wrong?? {from, to} maybe?
    #print(f"{move=}")
    #if (move == (2,1)):
    #    print ("hard coded win!")
    #    return 10
    
    reward = {}
    reward['invalid_move']    = -3
    reward['winning_move']    = 10
    reward['meh']             =  0

    test_tubes = state

    #move_from  = move[0]
    #move_to    = move[1]

    move_to    = move[0]
    move_from  = move[1]


    #show_tubes_up(test_tubes, False)
    #print("from to", move_from, move_to)

    
    if move_from == move_to:
        #print ("no way")
        return reward['invalid_move']

    allowed, _ = move_allowed(test_tubes, move_from, move_to)
    #print("allowed?", allowed)
    if not allowed:
        return reward['invalid_move']
    
    #if allowed:
    new_tubes = [tt.copy() for tt in test_tubes]
    new_tubes[move_to].add_ball(new_tubes[move_from].pop_top())
    #print("new tubes")
    #show_tubes_up(new_tubes, False)
    
    # to do!!! bug??? This winning is never called??
    if all(tt.is_complete() or tt.is_empty() for tt in new_tubes):
        #print ("Winning state!")
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



net_input = level_gen.tubes_to_list(test_tubes)  # net_input is the state # todo, move this inside the loop
T = tube_list_to_tensor(net_input)


mynet  = NeuralNetwork()
optimizer = torch.optim.AdamW(mynet.parameters(), lr=LEARNING_RATE)


#######
loss_rec = []

for stepnum in range(NUM_EPOCHS):

    #print("\n\n\n\n\n\n--\n\nstepnum" , stepnum)
    
    logits = mynet(T)  # that calls forward because __call__ is coded magic backend
    #print("logits" , logits)
    logits = logits.view(2,4)
    #print("logits" , logits)
    #to_from = logits.argmax(1).tolist()  # do this after training
    ### print(to_from)

    # Note .. will these be off by one??  Game expects 1-4 (does it?)  but this is 0-3 

    # move from is first half of logits,  NUM_TUBES, 
    # move to   is next  half of logits,  NUM_TUBES, 

	
    # these are the values we are randomly checking with bellman
    rfrom = random_int = random.randint(0, 3)
    rto = random_int = random.randint(0, 3)


    rand_from = torch.tensor(rfrom)  # do move, and compute right side
    rand_to   = torch.tensor(rto)  # do move, and compute right side
    
    one_hots = torch.stack((F.one_hot(rand_to, NUM_TUBES), F.one_hot(rand_from, NUM_TUBES)))
    
	
    logits = logits * one_hots
	
	

    logits = logits.sum(1)   # This is the left bellman!!  (need to turn this into one number, sum them or whatever)
    # question, what is the best way to combine the 2 numbers (to and from) into 1 number for loss function. Start with just a sum, will work well enough, but maybe get something better. Whatever it is, needs to be used consistantly.
    #print(f"{logits=}")

    bellman_left = logits.sum()
    #print(f"{bellman_left=}")
    
    rand_to_from = (rto, rfrom)
    
    reward    =   reward_f(test_tubes, rand_to_from)  



    if reward == 0:
        keep_playing = torch.tensor(1)
    else:
        keep_playing = torch.tensor(0)
    

        
    new_state = next_state(test_tubes, rand_to_from)
    right_input = level_gen.tubes_to_list(new_state)
    T = tube_list_to_tensor(right_input)
    right_logits = mynet(T) 

    right_logits = right_logits.view(2,4)
    #print(f"{right_logits=}")
    right_logits = right_logits.max(dim=1).values
    #print(f"{right_logits=}")
    
    bellman_right = reward + keep_playing * DECAY * right_logits.sum()
    #print(f"{bellman_left=}")
    #print(f"{bellman_right=}")
    

    loss = F.mse_loss(bellman_left, bellman_right)
    #print(f"{loss=}")
    loss_rec.append(loss.item())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if stepnum %100==0:
        print(stepnum , loss.item())

    if stepnum > (NUM_EPOCHS - 100) and stepnum %5==0:
        print(stepnum , loss.item())

        
        

    #if loss.item() >= 50 and stepnum > 40:
    #    print ("huge loss, exit", loss.item)
    #    quit()


#print(f"{loss_rec}")

'''
for idx, val in enumerate(loss_rec):
    if val > 20 or  (idx%100==0):
        print(idx, val)
'''




print("\n\n--\nNow run it again and see what it does\n")

net_input = level_gen.tubes_to_list(test_tubes)  # net_input is the state # todo, move this inside the loop

show_tubes_up(test_tubes, False)
T = tube_list_to_tensor(net_input)


  
logits = mynet(T)  # that calls forward because __call__ is coded magic backend
#print("logits" , logits)
logits = logits.view(2,4)
#print("logits" , logits)
to_from = logits.argmax(1).tolist()  # do this after training

print(f"{logits=}")
print(f"{to_from=}")


quit()













#bellman_right

'''

From maze, if the next move is a "terminal" state then only use the reward, nuke the rest
Terminal can be win OR lose.
terminal is either zero or one


        TERMINAL = torch.tensor(terminal).to(device).view(-1, 1)
        bellman_left = (model(XS) * MS).sum(dim=1, keepdim=True)
        qqs = model(YS).max(dim=1, keepdim=True).values
        bellman_right = RS + qqs * TERMINAL * GAMMA_DECAY
'''

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
