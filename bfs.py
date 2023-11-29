import heapq

from test_tube import TestTube, move_allowed, show_tubes_up


class Priority:
  # easy-to-use wrapper for python's priority queue (min)
  def __init__(self, data: list | None = None):
    self.data = data if data is not None else []    
    heapq.heapify(self.data)

  def push(self, obj):
    heapq.heappush(self.data, obj)

  def pop(self):
    return heapq.heappop(self.data)

  def is_empty(self):
    return len(self.data) == 0

  def __len__(self):
    return len(self.data)


class Node:

  def __init__(self, test_tubes: list[TestTube], parent=None) -> None:
    self.test_tubes = test_tubes
    self.parent = parent
    self.height = parent.height + 1 if parent else 0
    self.child_type = Node

  def is_winner(self):
    return all(tt.is_complete() or tt.is_empty() for tt in self.test_tubes)

  def children(self):
    possible_moves = [(i, j) for i in range(len(self.test_tubes)) for j in range(len(self.test_tubes)) if i !=j]
    valid_children = []
    for pm in possible_moves:
      allowed, _ = move_allowed(self.test_tubes, pm[0], pm[1])
      if allowed:
        new_tubes = [tt.copy() for tt in self.test_tubes]
        new_tubes[pm[1]].add_ball(new_tubes[pm[0]].pop_top())
        new_child = self.__class__(new_tubes, self)
        valid_children.append(new_child)
    return valid_children


  def __lt__(self, Node2):
    # need this operator to use this class in python3's heapq module
    return self.height < Node2.height

  def __bool__(self):
    # useful in __init__ when checking for parent and in bfs
    return True

  def __eq__(self, other):
    # need this operator to use this class is python3's set
    my_tubes = sorted(tube.to_int() for tube in self.test_tubes)
    their_tubes = sorted(tube.to_int() for tube in other.test_tubes)
    return my_tubes == their_tubes

  def __hash__(self):
    # need this operator to use this class is python3's set
    # sorted_tubes = sorted(self.test_tubes)
    # return hash((st.to_int() for st in sorted_tubes))
    return hash(tuple(sorted(tube.to_int() for tube in self.test_tubes)))


class AStarNode(Node):

  def __lt__(self, other):
    return self.height + self.heuristic() < other.height + other.heuristic()

  def heuristic(self):
    return sum([tt.heuristic() for tt in self.test_tubes])



def breadth_first_search(head: list[TestTube]) -> list[Node]:
  # create explored set, priority queue, and final path
  explored = set()
  priority = Priority([Node(head)])
  path = []
  # goal is a flag we use to see if we've found our goal
  goal = None
  # use breadth first search until we find our goal state
  while not goal:
    if priority.is_empty():
      print("I think I've gotten an impossible input.")
    # exploring is the node whose children we're checking, priority is
    # ranked based on __lt__ method in Node class
    exploring = priority.pop()
    # add this node to set of explored states if it hasn't been explored
    if exploring not in explored:
      # print(len(explored))
      # show_tubes_up(exploring.test_tubes, False)
      explored.add(exploring)
      if exploring.is_winner():
        goal = exploring
        break
      for child in exploring.children():
        if child not in explored:
          priority.push(child)
  print("Total nodes explored BFS: {}".format(len(explored)))
  while goal:
    path.append(goal.test_tubes)
    goal = goal.parent

  return path[::-1]

  
def a_star_search(head: list[TestTube], quiet=True) -> list[Node]:
  # create explored set, priority queue, and final path
  explored = set()
  priority = Priority([AStarNode(head)])
  path = []
  # goal is a flag we use to see if we've found our goal
  goal = None
  # use breadth first search until we find our goal state
  while not goal:
    if priority.is_empty():
      print("I think I've gotten an impossible input. A star not havin this.")
      return False
    # exploring is the node whose children we're checking, priority is
    # ranked based on __lt__ method in Node class
    exploring = priority.pop()
    # add this node to set of explored states if it hasn't been explored
    if exploring not in explored:
      # print(len(explored))
      # show_tubes_up(exploring.test_tubes, False)
      explored.add(exploring)
      if exploring.is_winner():
        goal = exploring
        break
      for child in exploring.children():
        if child not in explored:
          priority.push(child)
  if not quiet:
    print("Total nodes explored AStar: {}".format(len(explored)))
  while goal:
    path.append(goal.test_tubes)
    goal = goal.parent

  return path[::-1]
