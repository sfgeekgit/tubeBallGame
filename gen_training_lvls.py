import level_gen



level = level_gen.GameLevel()


for i in range(100):
    level.load_level_rand(2,4)
    print("level" , level)
    level.store_to_disk('../tubeballgame_stuff/easy_24_lvls/')
