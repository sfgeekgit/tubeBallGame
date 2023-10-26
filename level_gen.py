from test_tube import TestTube, TUBE_LENGTH, move_allowed, show_tubes_up    # , *

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
        # As this stands now:
        # -- This fills the leftmost tubes and leaves the rest to the right empty
        #   (which could potentially be a limit to the "training data" as real world problems could conceivably not be like this)

        if num_tubes <= num_colors:
            raise Exception("Num tubes must be more than num colors")

        out = False
        while not out:
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

            solution = a_star_search(out)
            # solution will be False if there is none, list of path if it is possible
            
            if solution or allow_impossible:
                return out                
            else:
                out = False # loop again!
                



def tubes_to_input(tubes, max_tubes=7):
    # given a list of tubes, turn it into a list of floats to feed into a neural net.
    # if the # of tubes is less than the max, fill out with 9's
    if len(tubes) > max_tubes:
        print ("turning too many tubes into an input")
        return
    out = []
    for tube in tubes:
        tu = tube.contents
        for ball in tu:
            out.append(float(ball))
        for _ in range(TUBE_LENGTH - len(tu)):
            out.append(0.)
    for _ in range(max_tubes - len(tubes)):
        for __ in range(TUBE_LENGTH):
            out.append(9.)
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
