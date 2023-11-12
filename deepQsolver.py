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

#DECAY = 0.95
#DECAY = 0.6
DECAY = 0.7
# interesting. Training this on puzzles that can be sloved in one move was not learning with a decay=.95 I think because the next move would still be so close to a win. Similar for training on puzzles solveable in 1 or 2 moves. But lowering the decay rate for these easy puzzles seems to have worked!



LEARNING_RATE = 1e-3
#LEARNING_RATE = 1e-4



NUM_EPOCHS =    5000  
NUM_EPOCHS = 2500000



NUM_TUBES  = 4
#NUM_TUBES  = 5
NUM_COLORS = 2


INPUT_SIZE = ( NUM_COLORS +1 ) * NUM_TUBES * TUBE_LENGTH
HIDDEN_SIZE = INPUT_SIZE  # why not...



'''
Plays the tube ball game. 
Every step of the game needs two values.
-- What tube to move from
-- What tube to move to

This code has two differnt settings for how to build the neural network output.
If SQUARED_OUTPUT == False then the final output layer of the network will an
output size of  NUM_TUBES * 2 
this output will be split in half. The largest logit from the first half will be move_to and the largest logit of the second half will be move_from

But if SQUARED_OUTPUT == True, then the final output layer will be of 
size NUM_TUBES * NUM_TUBES
and the single biggest logit will be taken as the answer. For example if the largest logit is 0, then to_from will be assumed to be (0,0) and if the network outputs a 1, then to_from will be (0,1) etc
'''


EXHAUSTIVE = False

#SQUARED_OUTPUT = False
SQUARED_OUTPUT = True

if SQUARED_OUTPUT:
    # get ONE output, take the to and from (so 7 tubes would need 49 outputs)
    OUTPUT_SIZE = NUM_TUBES * NUM_TUBES   # 

else:
    # get TWO outputs, take the to from first half and from from next half(so 7 tubes would need 14 outputs)
    OUTPUT_SIZE = NUM_TUBES * 2   # ball to and ball from



    ### To do...
    ### currently only recodrind final loss rate, but that isn't reall what maters
    ### end of training, automate 100 actual tests and see if it gets it right.
    ### and/or what are the final logits, are they correct, how confident is the system of it's next move. (shoule be very confident with these simple puzzles

    # also to do, have it step though REAL puzzels, that's a soon to-do (and maybe exhausitve search)
    
loss_function = 'MSE'
#loss_function = nn.L1Loss()  # MAE  mean absolute error
#loss_function = nn.SmoothL1Loss()  #huber


    
DYN_LEARNING_RATE = False
STEP_LEARN_RATE   = True
##STEP_LEARN_RATE   = False

    
NN_SIZE = [INPUT_SIZE, HIDDEN_SIZE, HIDDEN_SIZE, OUTPUT_SIZE]


def tube_list_to_tensor(tubes):
    dic = {}
    dic[0] = [1,0,0]
    dic[1] = [0,1,0]
    dic[2] = [0,0,1]

    t_input = []

    for ball in tubes:
        t_input.extend(dic[ball])

    T = torch.tensor(t_input)
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
            nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE),
            nn.ReLU(),
            nn.Linear(HIDDEN_SIZE, OUTPUT_SIZE),
        )

    def forward(self, x):
        logits = self.linear_relu_stack(x)        
        return logits




def reward_f (state, move):   # state is list of tubes, move is tuple {to, from} 
    
    reward = {}
    reward['invalid_move']    = -3
    reward['winning_move']    = 10
    reward['meh']             =  0

    test_tubes = state
    move_to    = move[0]
    move_from  = move[1]

    #print("\n\n--\n\n")
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
    

    if all(tt.is_complete() or tt.is_empty() for tt in new_tubes):
        #print ("Winning state!")
        return reward['winning_move']

    #print("meh")
    return reward['meh']


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

    
def exhaustive_search():
    all_moves = []
    for j in range(NUM_TUBES):
        for k in range(NUM_TUBES):
            all_moves.append((j,k))


    # Calculate the next state and the reward for each move
    all_next_states = [next_state(test_tubes, move) for move in all_moves]
    all_rewards = [reward_f(test_tubes, move) for move in all_moves]

    #for foo in all_next_states:
    #    print(f"{foo=}")
    #    show_tubes_up(foo, False)

    #print(f"{len(all_next_states)=}")
    #print(f"{len(all_rewards)=}")

    #print(f"{all_next_states=}") 
    #print(f"{all_rewards=}") 
    all_next_states_tensor = torch.stack([tube_list_to_tensor(level_gen.tubes_to_list(state, NUM_TUBES)) for state in all_next_states])
    #print(f"{all_next_states_tensor=}")

    all_logits = mynet(all_next_states_tensor)

    all_next_q_values = mynet(all_next_states_tensor)

    # Calculate the maximum Q-value for each next state
    if SQUARED_OUTPUT:
        all_next_max_q_values = all_next_q_values.max(dim=1).values
    else:
        all_next_max_q_values = all_next_q_values.view(-1, 2, NUM_TUBES).max(dim=2).values




    # Calculate the target Q-values for the Bellman equation
    all_rewards_tensor = torch.tensor(all_rewards)
    keep_playing_tensor = (all_rewards_tensor == 0).float()
    all_bellman_targets = all_rewards_tensor + keep_playing_tensor * DECAY * all_next_max_q_values.sum(dim=1)
    
    # Calculate the predicted Q-values for the moves that were actually taken
    all_moves_tensor = torch.tensor(all_moves)
    if SQUARED_OUTPUT:
        all_predicted_q_values = all_logits.gather(1, all_moves_tensor.prod(dim=1).view(-1, 1)).squeeze()
    else:
        all_predicted_q_values = all_logits.view(-1, 2, NUM_TUBES).gather(2, all_moves_tensor.view(-1, 2, 1)).sum(dim=1)
        
    # Calculate the loss
    loss_function = nn.SmoothL1Loss()
    loss = loss_function(all_predicted_q_values, all_bellman_targets)
    return loss





mynet  = NeuralNetwork()
optimizer = torch.optim.AdamW(mynet.parameters(), lr=LEARNING_RATE)
loss_rec = []


level = level_gen.GameLevel()
#level.load_demo_easy()
#level.load_demo_one_move()


for stepnum in range(NUM_EPOCHS):
    #level.load_demo_one_move_rand(NUM_TUBES)
    level.load_demo_one_or_two_move_rand(NUM_TUBES)
    test_tubes = level.get_tubes()
    #show_tubes_up(test_tubes, False)


    net_input = level_gen.tubes_to_list(test_tubes, NUM_TUBES)  # net_input is the state    
    T = tube_list_to_tensor(net_input)


    logits = mynet(T)  # that calls forward because __call__ is coded magic backend
    #print("logits" , logits)


    if SQUARED_OUTPUT:
        pass
    else:
        logits = logits.view(2,NUM_TUBES)
    #print("logits" , logits)


    # these are the values we are randomly checking with bellman
    rfrom = random_int = random.randint(0, NUM_TUBES-1)
    rto = random_int = random.randint(0, NUM_TUBES-1)




    if stepnum % 800 == 0:
        print(f"{LEARNING_RATE=}")
        print("4 layers (As of this writing)")
        print(f"{logits=}")
        if SQUARED_OUTPUT:
            to_from_logs = logits.argmax()  # do this after training
            move_to = to_from_logs // NUM_TUBES
            move_from = to_from_logs - move_to * NUM_TUBES
            to_from = [move_to, move_from]

            #if stepnum % 3 == 0:
            #   print("non-rand shenangans!")
            #   rfrom = move_from
            #   rto   = move_to



        else:
            to_from = logits.argmax(1).tolist()  # do this after training

        
        show_tubes_up(test_tubes, False)
        print(f"{to_from=}")
        new_state = next_state(test_tubes, to_from)    
        show_tubes_up(new_state, False)
        print("\n\n")
        

    # move to   is first half of logits,  NUM_TUBES, 
    # move from is other half of logits,  NUM_TUBES, 
    ################

    #print(f"{logits=}")
    

	

    rand_from = torch.tensor(rfrom)  # do move, and compute right side
    rand_to   = torch.tensor(rto)  # do move, and compute right side


    if SQUARED_OUTPUT:
        to_from_sq = rto * NUM_TUBES + rfrom
        one_hots = F.one_hot(torch.tensor(to_from_sq), NUM_TUBES * NUM_TUBES)
        logits = logits * one_hots

        
    else:
        one_hots = torch.stack((F.one_hot(rand_to, NUM_TUBES), F.one_hot(rand_from, NUM_TUBES)))
        logits = logits * one_hots	
        logits = logits.sum(1)   # This is the left bellman!!  (need to turn this into one number, sum them or whatever)
    # question, what is the best way to combine the 2 numbers (to and from) into 1 number for loss function. Start with just a sum, will work well enough, but maybe get something better. Whatever it is, needs to be used consistantly.
    #print(f"{logits=}")




    
    bellman_left = logits.sum()
    #print(f"{bellman_left=}")

    # test one random
    rand_to_from = (rto, rfrom)
    reward    =   reward_f(test_tubes, rand_to_from)  

    ## THIS REWARD IS WRONG?????



    '''
    if stepnum % 999800 == 0:
        print(f"{rand_to=}")
        print(f"{rand_from=}")
        print(f"{rand_to_from=}")
        print(f"{one_hots=}")
        print(f"{logits=}")
        print(f"{reward=}")
        print(f"{bellman_left=}")
    '''


    if EXHAUSTIVE:
        loss = exhaustive_search()
    else:
        # have we reached a terminal state?
        if reward == 0:
            keep_playing = torch.tensor(1)   # keep playing
        else:
            keep_playing = torch.tensor(0)   # terminal state, zero out the right logits, only use the reward
    

        
        new_state = next_state(test_tubes, rand_to_from)    
        right_input = level_gen.tubes_to_list(new_state, NUM_TUBES)
        T = tube_list_to_tensor(right_input)
        right_logits = mynet(T) 


        if SQUARED_OUTPUT:
            right_logits = right_logits.max(dim=0).values
        else:
            right_logits = right_logits.view(2,NUM_TUBES)
            #print(f"{right_logits=}")
            right_logits = right_logits.max(dim=1).values
            #print(f"{right_logits=}")
            #right_logits_max = right_logits.max(dim=1).values
            #print(f"{right_logits_max=}")

        
        bellman_right = reward + keep_playing * DECAY * right_logits.sum()   # should this be right_logits max??
        # for right_logits, this is using the highest value (the highest confidence) but should be the position of that???

        '''
        if stepnum % 800 == 0:
            print(f"{bellman_left=}")
            print(f"{bellman_right=}")
            print(f"{reward=}")
            print(f"{keep_playing=}")
            print(f"{right_logits=}")
            print(f"{right_logits.sum()=}")
            print(f"{DECAY=}")
        '''
        
        if loss_function == 'MSE':
            # MSE
            loss = F.mse_loss(bellman_left, bellman_right)

        else:
            loss = loss_function(bellman_left, bellman_right) 

            #  Huber Loss
            #loss_function = nn.SmoothL1Loss()
            #loss = loss_function(bellman_left, bellman_right)
        
            #Mean Absolute Error (MAE)
            #loss_function = nn.L1Loss()
            #loss = loss_function(bellman_left, bellman_right)
    

    #print(f"{loss=}")
    loss_rec.append(loss.item())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if stepnum %100==0:
        print(stepnum , loss.item())

        if DYN_LEARNING_RATE:
            if stepnum > 50:
                max_rec_loss = max(loss_rec[-8:])
                median_rec_loss = sorted(loss_rec[-20:])[len(loss_rec[-20:]) // 2]
                #print(f"{max_rec_loss=}")
                #print(f"{median_rec_loss=}")
                if median_rec_loss < LEARNING_RATE * 50 and max_rec_loss < LEARNING_RATE*500:
                    print(f"{LEARNING_RATE=}")
                    LEARNING_RATE = LEARNING_RATE / 5
                    print ("---------------------------------------\nLower Learning Rate \n==================================================================")
                    print(f"{LEARNING_RATE=}")


        
    if STEP_LEARN_RATE:
        for l_step in [.9, .95, .99]:
            if stepnum == NUM_EPOCHS * l_step:
                print(f"{LEARNING_RATE=}")
                print ("---------------------------------------\nLower Learning Rate \n==================================================================")
                LEARNING_RATE = LEARNING_RATE / 10
                print(f"{LEARNING_RATE=}")
                optimizer = torch.optim.AdamW(mynet.parameters(), lr=LEARNING_RATE)

                
                
                
avg_last = 500
#print("len rec" , len(loss_rec))
average_loss_end = sum(loss_rec[-avg_last:]) / avg_last
print("avg of last __ steps for end" , avg_last)
print(f"{average_loss_end=}")
print(f"{LEARNING_RATE=}")




current_time = time.strftime("%Y-%m-%d-%H:%M:%S")
log_content = (f"\n\n--------\nFinished run at {current_time}")
log_content += (f"\n {average_loss_end} average_loss_end")
log_content += (f"\n {NUM_EPOCHS} NUM_EPOCHS")
log_content += (f"\n    {SQUARED_OUTPUT=}  {loss_function=}")
log_content += (f"\n  {NUM_EPOCHS=} {DECAY=}  final learn: {LEARNING_RATE=} ")
log_content += (f"  {DYN_LEARNING_RATE=} {STEP_LEARN_RATE=} Step is Learn_rate /10 at 90% 95% and 99%")
log_content += (f"\n    Added a second hidden layer, so neural net now has 4 total layers 2 hidden are same size as input. Let's see how this does")
log_content += (f"\n    All above use MSE loss function. Now trying different loss on non-square to see. BUT different loss function might mean that the numbers here for loss are apples and oranges, so...")


log_file_path = './run_log.txt'
with open(log_file_path, 'a') as f:
    f.write(log_content)



print("\n\n--\nNow run it again and see what it does\n")

net_input = level_gen.tubes_to_list(test_tubes, NUM_TUBES)  
show_tubes_up(test_tubes, False)
T = tube_list_to_tensor(net_input)


  
logits = mynet(T)  # that calls forward because __call__ is coded magic backend
#print("logits" , logits)

if SQUARED_OUTPUT:
    print(f"{logits=}")
    logits_mv = logits.max(dim=0).values
    print(f"{logits_mv=}")
    to_from_logs = logits.argmax()  # do this after training
    print(f"{to_from_logs=}")
    move_to = to_from_logs // NUM_TUBES
    move_from = to_from_logs - move_to * NUM_TUBES
    to_from = [move_to, move_from]


    
else:
    logits = logits.view(2,NUM_TUBES)
    print(f"{logits=}")
    to_from = logits.argmax(1).tolist()  # do this after training

print(f"{to_from=}")
new_state = next_state(test_tubes, to_from)    
show_tubes_up(new_state, False)






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
