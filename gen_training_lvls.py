import level_gen



level = level_gen.GameLevel()

num_tubes = 5
num_colors = 3


out_dir = '../tubeballgame_stuff/test_lvls/' + str(num_tubes) + '_' + str(num_colors)

import os
if not os.path.exists(out_dir):
    os.makedirs(out_dir)    


for i in range(300):
    level.load_level_rand(num_colors, num_tubes)  # allow_impossible is False by default
    print("level" , level)
    #level.store_to_disk('../tubeballgame_stuff/easy_24_lvls/')
    level.store_to_disk(out_dir)


    # todo
    # get the number of moves to solve them (both a_star and bfs)
    # store that in a file? Where exactly? 



