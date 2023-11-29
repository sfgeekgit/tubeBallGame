from test_tube import TestTube, TUBE_LENGTH, move_allowed, show_tubes_up, move_allowed_physics    # , *
from random import shuffle

from bfs import a_star_search
import random
import os


class GameLevel:
    def __init__(self, tubes=[]) -> None:
        # tubes should be a list of TestTube objects, or a 2D list of ints
        self.tubes = []
        for tube in tubes:
            if isinstance(tube, TestTube):
                self.tubes.append(tube)
            elif isinstance(tube, list) and all(isinstance(x, int) for x in tube):
                self.tubes.append(TestTube(tube))
            else:
                raise ValueError("Invalid input: must be a TestTube, an empty list, or a list of ints")


    def get_tubes(self):
        return self.tubes

    def __str__(self):
        ret = []
        for tt in self.tubes:
            ret.append(tt.contents)
        return str(ret)

    def store_to_disk(self, dir_path='./lvls/'):
        content = self.__str__()
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        existing_files = [f for f in os.listdir(dir_path) if f.endswith('.lvl')]
        file_count = len(existing_files)
        new_file_name = f"{file_count + 1}.lvl"
        new_file_path = os.path.join(dir_path, new_file_name)
        with open(new_file_path, 'w') as f:
            f.write(content)
        return new_file_name

    def load_lvl(self, level):
        self.__init__(level)
    
    def load_from_disk(self, filename, dir_path='./lvls/'):
        file_path = dir_path + filename
        with open(file_path, 'r') as f:
            content = f.read()
        data = eval(content)
        self.__init__(data)
        
    def load_demo_easy(self):
        tubes = [
            TestTube([1, 1, 2, 1]),
            TestTube([2, 1, 2]),
            TestTube([2]),
            TestTube()
        ]
        self.__init__(tubes)


    def load_demo_one_move(self):
        tubes = [
            TestTube([1, 1, 1, 1]),
            TestTube([2, 2, 2]),
            TestTube([2]),
            TestTube()
        ]
        self.__init__(tubes)

    def load_demo_one_move_rand(self, num_tubes):
        coin = random.randint(0,1)
        x = coin +1 
        y = (coin +1) % 2 +1        
        tubes = [
            TestTube([x, x, x, x]),
            TestTube([y, y, y]),
            TestTube([y])            
        ]
        for _ in range(num_tubes - len(tubes)):
            tubes.append(TestTube())
        
        shuffle(tubes)
        self.__init__(tubes)

    def load_demo_two_move_rand(self, num_tubes):
        coin = random.randint(0,1)
        x = coin +1 
        y = (coin +1) % 2 +1
        coin2 = random.randint(0,1)
        x2 = coin2 +1 
        y2 = (coin2 +1) % 2 +1
        tubes = [
            TestTube([x, x, x]),
            TestTube([y, y, y]),
            TestTube([x2,y2])
        ]
        for _ in range(num_tubes - len(tubes)):
            tubes.append(TestTube())
        
        shuffle(tubes)
        self.__init__(tubes)


    def load_demo_one_or_two_move_rand(self, num_tubes):
        # not super random, not exhausitve
        coin = random.randint(0,1)
        if coin == 0:
            self.load_demo_one_move_rand(num_tubes)
        else:
            self.load_demo_two_move_rand(num_tubes)
        
    def load_demo_hard(self):
        tubes = [
            TestTube([1, 2, 3, 4]),
            TestTube([5, 5, 6, 5]),
            TestTube([1, 2, 7, 2]),
            TestTube([7, 4, 5, 6]),
            TestTube([4, 4, 3, 2]),
            TestTube([6, 1, 1, 3]),
            TestTube([3, 7, 7, 6]),
            TestTube([]),
            TestTube([])
        ]
        self.__init__(tubes)

    

    def load_level_rand(self, num_colors=4, num_tubes=6):
        lvl = self.gen_level_rand(num_colors, num_tubes)
        self.load_lvl(lvl)
        

    def gen_level_rand(self, num_colors=4, num_tubes=6, allow_impossible=False):

        if num_tubes <= num_colors:
            raise Exception("Num tubes must be more than num colors")

        out = False
        while not out:
            all_balls = []
            for i in range(num_colors):
                for j in range(TUBE_LENGTH):
                    all_balls.append(i+1) # no zero


            # add some temp "padding" to randomize the empty slots
            num_blank_spots = TUBE_LENGTH * (num_tubes - num_colors)
            for _ in range(num_blank_spots):
                all_balls.append(0)

            random.shuffle(all_balls)
            tube_list = [all_balls[i:i + TUBE_LENGTH] for i in range(0, len(all_balls), TUBE_LENGTH)]
            tube_list = [[ball for ball in tube if ball != 0] for tube in tube_list]  # remove the padding


            out = []
            for tt in tube_list:
                out.append(TestTube(tt))

                
            if allow_impossible:
                solution = True
            else: 
                solution = a_star_search(out)
                # solution will be False if there is none, list of path if it is possible
                #if solution: print ("A star path len " , len(solution))
            
            if solution:  # or allow_impossible:
                return out                
            else:
                out = False # loop again!
                


def tubes_to_list(tubes, max_tubes=4):
    # given a list of tubes, turn it into a list of floats to feed into a neural net.
    # if the # of tubes is less than the max, fill out with 9's
    if len(tubes) > max_tubes:
        print ("turning too many tubes into an input")
        return
    out = []
    for tube in tubes:
        tu = tube.contents
        for ball in tu:
            #out.append(float(ball))
            out.append(ball)
        for _ in range(TUBE_LENGTH - len(tu)):
            out.append(0)
    for _ in range(max_tubes - len(tubes)):
        for __ in range(TUBE_LENGTH):
            out.append(9)
    return out            


def scramble_level(lvl, num_steps):
    num_tu = len(lvl)
    for _ in range(num_steps):
        possible = False
        while not possible:
            move_from = random.randint(0,num_tu -1)
            move_to   = random.randint(0,num_tu -1)
            possible = move_allowed_physics(lvl, move_from, move_to)[0]
        lvl[move_to].add_any_ball(lvl[move_from].pop_top())
    shuffle(lvl)
    return lvl

              
def gen_solved_level(num_colors, num_tubes):
    if num_tubes < num_colors:
        raise Exception("Num tubes must be more than num colors")

    out = []
    for i in range(num_colors):
        this_list = []
        for j in range(TUBE_LENGTH):
            this_list.append(i+1) # avoid zero
        out.append(TestTube(this_list))


    for k in range(num_tubes - num_colors):
        out.append(TestTube([]))

    return out

#level = GameLevel()
#lvl = gen_solved_level(2,4)
#for _ in range(1):
#    #level.load_level_rand(2,4)
#    #lvl = level.gen_level_rand(2,4)
#    lvl = scramble_level(lvl, 2)
#    level.load_lvl(lvl)
#    test_tubes = level.get_tubes()
#    show_tubes_up(test_tubes, False)
#level.load_demo_one_or_two_move_rand(4)
#level.load_demo_one_or_two_move_rand(5)
