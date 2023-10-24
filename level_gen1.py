from test_tube import TestTube, TUBE_LENGTH, move_allowed, show_tubes_up
from colors import to_colored_ball
import random

# old, ignore this file, use level_gen.py


def gen_level_rand(num_colors, num_tubes):
    # As this stands now:
    # -- Some random boards will be un-solvable!
    # -- This fills the leftmost tubes and leaves the rest to the right empty
    #    (this could potentially be a limit to the "training data" as real world problems could conceivable not be like thi)

    if num_tubes <= num_colors:
        raise Exception("Num tubes must be more than num colors")

    all_balls = []
    for i in range(num_colors):
        for j in range(TUBE_LENGTH):
            all_balls.append(i+1) # no zero
    random.shuffle(all_balls)

    tube_list = [all_balls[i:i + TUBE_LENGTH] for i in range(0, len(all_balls), TUBE_LENGTH)]
    for k in range(num_tubes - num_colors):
        tube_list.append([])

    out = []
    for tt in tube_list:
        out.append(TestTube(tt))

    return out
        
              
def gen_solved_level(num_colors, num_tubes):
    if num_tubes < num_colors:
        raise Exception("Num tubes must be more than num colors")

    out = []
    for i in range(num_colors):
        this_list = []
        for j in range(TUBE_LENGTH):
            #this_list.append(i+1) # there is no color zero!
            this_list.append(i)    # now there is 
        out.append(TestTube(this_list))


    for k in range(num_tubes - num_colors):
        out.append(TestTube([]))

    return out



def gen_level_lame_dont_use(num_colors, num_tubes, scramble_itts = 1000):
        
    # Frist generate a solved level
    # then do a bunch of random legal moves to scramble it
    # solution will be to just play this backward.
    # no, this is lame
    # This code will only make VALID moves, but to scramble the puzzle
    # it puts it into a state that cannot be reached in reverse... try again
    # maybe just random?? But that can make impossible puzzles?
    # maybe make a random one, then test it with Astar... guess I'll do that 
    
    if num_tubes <= num_colors:
        raise Exception("Num tubes must be more than num colors")

    moves_made = 0
    moves_deny = 0
    tubes = gen_solved_level(num_colors, num_tubes)

    for _ in range(scramble_itts):
        move_fr, move_to = random.sample(range(0, num_tubes), 2)
        allowed, _ = move_allowed(tubes, move_fr, move_to)
        if allowed:
            moves_made += 1
            tubes[move_to].add_ball(tubes[move_fr].pop_top())
        else:
            moves_deny += 1

    #print ("moves: " , moves_made)
    #print ("deny: " , moves_deny)

    show_tubes_up(tubes, False)
    
    return tubes




#lev = gen_level_rand(5,8)
#print(lev)
#show_tubes_up(lev, False)

#lev = gen_solved_level(8,11)
#show_tubes_up(lev, False)
