from typing import Type
from bfs import breadth_first_search, a_star_search
from test_tube import TestTube, TUBE_LENGTH, move_allowed, show_tubes_up
from colors import to_colored_ball
import time


def generate_level(hard_mode=True):
  return [
      TestTube([1, 2, 3, 4]),
      TestTube([5, 5, 6, 5]),
      TestTube([1, 2, 7, 2]),
      TestTube([7, 4, 5, 6]),
      TestTube([4, 4, 3, 2]),
#      TestTube([3,2,4, 4]), # Not orignial
      TestTube([6, 1, 1, 3]),
      TestTube([3, 7, 7, 6]),
      TestTube([]),
      TestTube([])
  ] if hard_mode else [
      TestTube([1, 1, 2, 1]),
      TestTube([2, 1, 2]),
      TestTube([2]),
      TestTube()
  ]

def solver_solve(test_tubes):
  #path = breadth_first_search(test_tubes)
  path = a_star_search(test_tubes)
  for step in path:
    time.sleep(.3)
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
  test_tubes = generate_level()
  solver_solve(test_tubes)
  #player_solve(test_tubes)

if __name__ == '__main__':
  main()
