import level_gen



level = level_gen.GameLevel()

num_tubes = 4
num_colors = 2


out_dir = '../tubeballgame_stuff/test_lvls/' + str(num_tubes) + '_' + str(num_colors)
for i in range(300):
    level.load_level_rand(num_colors, num_tubes)
    print("level" , level)
    #level.store_to_disk('../tubeballgame_stuff/easy_24_lvls/')
    level.store_to_disk(out_dir)


    # todo
    # test that these levels are solvable
    # get the number of moves to solve them (both a_star and bfs)


