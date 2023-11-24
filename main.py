from typing import Type
from bfs import breadth_first_search, a_star_search
from test_tube import TestTube, TUBE_LENGTH, move_allowed, show_tubes_up
from colors import to_colored_ball
import time
import level_gen


SLEEP_TIME=0.2

def generate_level(hard_mode=True):
  level = level_gen.GameLevel()
  if hard_mode:
    level.load_demo_hard()
  else:
    level.load_demo_easy()
  return level.get_tubes()



def solver_solve(test_tubes):
  #path = breadth_first_search(test_tubes)
  path = a_star_search(test_tubes)
  if (path):
    for step in path:
      time.sleep(SLEEP_TIME)
      show_tubes_up(step, False)
    print("\nSteps to solve:", len(path))

def player_solve(test_tubes):
  while not all(tt.is_complete() or tt.is_empty() for tt in test_tubes):
    # present the current state of the test tubes
    show_tubes_up(test_tubes, False)
    move_from = int(input("move from: "))
    move_to = int(input("move to: "))
    allowed, reason = move_allowed(test_tubes, move_from, move_to)
    if allowed:
      test_tubes[move_to].add_ball(test_tubes[move_from].pop_top())
    else:
      print("Invalid move: {}".format(reason))

  show_tubes_up(test_tubes, False)
  print("congrats!!! you won the game!!!")

def main():
  #test_tubes = generate_level(True)

  level = level_gen.GameLevel()

  '''
  global SLEEP_TIME
  SLEEP_TIME=0
  for i in range(1,4): # 501
    f = str(i) + '.lvl'
    level.load_from_disk(f)    
    test_tubes = level.get_tubes()
    solver_solve(test_tubes)
  quit()
  '''

  level.load_demo_easy()
  #level.load_level_rand(4,6)
  #print("level" , level)
  
  #quit()
  
  #level.load_demo_hard()
  test_tubes = level.get_tubes()

  #net_input = level_gen.tubes_to_input(test_tubes)
  net_input = level_gen.tubes_to_list(test_tubes)
  print("net input", net_input)
  #print(test_tubes)
  #quit()

  
  solver_solve(test_tubes)
  #player_solve(test_tubes)

if __name__ == '__main__':
  main()
