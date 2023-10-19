from colors import to_colored_ball
import os  # for clear screen

TUBE_LENGTH = 4

class TestTube:

  def __init__(self, contents=[]) -> None:
    if len(contents) > TUBE_LENGTH:
      raise Exception("trying to create an impossible level")
    self.contents = contents  # organized bottom to top

  def is_complete(self) -> bool:  # "-> bool" is an optional type-hint
    return len(set(self.contents)) == 1 and self.is_full()

  def is_empty(self) -> bool:
    return self.contents == []

  def is_full(self) -> bool:
    return len(self.contents) >= TUBE_LENGTH

  def top_ball(self) -> int | None:
    if len(self.contents) == 0:
      return
    return self.contents[-1]

  def pop_top(self) -> int:
    try:
      return self.contents.pop()
    except IndexError:
      return 0

  def can_accept(self, ball_color: int) -> bool:
    if self.is_empty():
      return True
    if self.is_full():
      return False
    return self.top_ball() == ball_color

  def add_ball(self, ball_color: int) -> bool:
    if self.can_accept(ball_color):
      self.contents.append(ball_color)
      return True
    return False

  def heuristic(self):  # average heur1 and heur2
    return self.heuristic1()
    # return (self.heuristic2() + self.heuristic1())/2
    
  def heuristic2(self):  # Min number of moves to make all in tube same color... bad heuristic!
    if len(set(self.contents)) <= 1:
      return 0
    first_color = self.contents[0]
    balls_to_remove = 0
    for color in self.contents:
      if color != first_color or balls_to_remove > 0:
        balls_to_remove += 1
    return balls_to_remove

  def heuristic1(self): # Min number of moves to complete this tube
    if self.is_complete() or self.is_empty():
      return 0
    if len(self.contents) <= 2:
      return len(self.contents)
    if len(self.contents) == 3:
      return 1 if len(set(self.contents)) == 1 else 3
    if len(set(self.contents[0:3])) == 1:
      return 2
    return 4

  def to_int(self) -> int:
    filled = self.contents + [0 for _ in range(TUBE_LENGTH - len(self.contents))]
    return sum([(10**i) * num for i, num in enumerate(reversed(filled))])

  def copy(self):
    return TestTube(list(self.contents))

  def __lt__(self, other):
    return self.to_int() < other.to_int()

  def __eq__(self, other):
    if len(self.contents) == len(other.contents):
      return all(color == other[idx] for idx, color in enumerate(self.contents))
    return False
    # return self.to_int() == other.to_int()

  def __str__(self):
    to_return = '--------\n|' + " ".join(
        to_colored_ball(i) for i in self.contents) + '\n--------'
    return to_return

def move_allowed(test_tubes, move_from: int, move_to: int):
  cnt_of_tubes = len(test_tubes)
  if move_from not in range(cnt_of_tubes):
    return (False, "{} is not in the right range!".format(move_from))
  if move_to not in range(cnt_of_tubes):
    return (False, "{} is not in the right range!".format(move_to))

  if test_tubes[move_to].is_full():
    return (False, "{} is already full".format(move_to))

  if test_tubes[move_from].is_empty():
    return (False, "{} is already empty".format(move_from))

  if not test_tubes[move_to].can_accept(test_tubes[move_from].top_ball()):
    return (False, "test tube {} can only accept {}s".format(
        move_to, to_colored_ball(test_tubes[move_to].top_ball())))

  return (True, "")


def show_tubes_up(test_tubes, clear=True):
  if clear:
    os.system('clear')
  ret    = ''
  bottom = ''
  ids    = ''
  for i in range(TUBE_LENGTH, 0, -1):
    for idx, tube in enumerate(test_tubes):
      if i == 1: # bottom row, also make the bottoms
        bottom = bottom + '----   '
        ids = ids + '  ' + str(idx) + '    '      
      if len(tube.contents) < i:
        this_ball = ' '
      else:
        this_ball = to_colored_ball(tube.contents[i -1])
      ret = ret + '|'
      ret = ret + this_ball
      ret = ret + ' |   ' 
    ret = ret + "\n"  
  ret = ret + bottom + "\n" + ids
  print (ret)%                                                           
