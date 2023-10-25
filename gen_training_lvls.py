import level_gen




level = level_gen.GameLevel()


for i in range(500):
    level.load_level_rand(4,6)
    print("level" , level)
    level.store_to_disk()
