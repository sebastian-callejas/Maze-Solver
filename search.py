from __future__ import annotations
from math import sqrt
from re import T
import sys
from enum import Enum
from typing import Callable, Dict, Generic, NamedTuple, List, Optional, Set, TypeVar
from heapq import heappush, heappop


T = TypeVar('T')


class Cell(str, Enum):
    EMPTY = " "
    BLOCKED = "%"
    START = "S"
    GOAL = "G"
    PATH = "*"


class Location(NamedTuple):
    row: int
    column: int


# node class for each node in our tree
class Node(Generic[T]):
    def __init__(self, state: T, parent: Optional[Node], cost: float = 0.0, heuristic: float = 0.0):
        # current location aka state
        self.state: T = state
        # node which brought us here to determine path later on
        self.parent: Optional[Node] = parent
        # cost and heuristic for A* and greedy
        self.cost: float = cost
        self.heuristic: float = heuristic

    def __lt__(self, other: Node) -> bool:
        return (self.cost + self.heuristic) < (other.cost + other.heuristic)


# stack to be used for DFS
class Stack(Generic[T]):
    def __init__(self) -> None:
        self._container: List[T] = []

    @property
    def empty(self) -> bool:
        return not self._container  # not is true for empty container

    def push(self, item: T) -> None:
        self._container.append(item)

    def pop(self) -> T:
        return self._container.pop()  # LIFO

    def __repr__(self) -> str:
        return repr(self._container)


# priority queue for use of A* and Greedy search
class PriorityQueue(Generic[T]):
    def __init__(self):
        self._container: List[T] = []

    @property
    def empty(self) -> bool:
        return not self._container

    def push(self, item: T) -> None:
        heappush(self._container, item)

    def pop(self) -> T:
        return heappop(self._container)

    def __repr__(self) -> str:
        return repr(self._container)


# using pythagorean theorem return the euclidean distance
def euclideanDistance(goal: Location) -> Callable[[Location], float]:
    def distance(node: Location) -> float:
        x: int = node.column - goal.column
        y: int = node.row - goal.row
        return sqrt((x * x) + (y * y))
    return distance


# return the abs(different) betweent he start and goal
def manhattanDistance(goal: Location) -> Callable[[Location], float]:
    def distance(node: Location) -> float:
        x: int = abs(node.column - goal.column)
        y: int = abs(node.row - goal.row)
        return (x + y)
    return distance


class Maze():
    def __init__(self, FILE):
        self.file = open(FILE, 'r')
        self.data = self.file.readlines()
        self.rows = len(self.data)
        self.columns = len(self.data[0]) - 1
        self.maze = self.readMaze()
        if (not hasattr(self, 'start') or not hasattr(self, 'goal')):
            sys.stderr.write('ERROR: missing start and/or goal node ')

    # read by line in data and convert to enumerated Cell
    # set start and goal while traversing (assuming one is given for each)
    def readMaze(self):
        mx: List[List[Cell]] = [[Cell.EMPTY for c in range(
            self.columns)] for r in range(self.rows)]
        row = 0
        for line in self.data:
            col = 0
            for cell in line:
                if cell == 'S':
                    mx[row][col] = (Cell.START)
                    self.start: Location = Location(row, col)
                elif cell == 'G':
                    mx[row][col] = (Cell.GOAL)
                    self.goal: Location = Location(row, col)
                elif cell == '%':
                    mx[row][col] = (Cell.BLOCKED)
                elif cell == ' ':
                    mx[row][col] = (Cell.EMPTY)
                col += 1
            row += 1
        return mx

    # clean print statement for maze (whether complete or not)
    def __str__(self):
        output: str = ""
        for row in self.maze:
            output += "".join([c.value for c in row]) + "\n"
        return output

    # check above, below, left right for current location on maze
    def branching(self, node: Location) -> List[Location]:
        # list of locations for branching
        branches: List[Location] = []

        # initialize row,col for current node and total rows,cols for faster access
        row, col = node.row, node.column
        rows, cols = self.rows, self.columns

        # check if inbounds and if cell isn't blocked
        # row down
        if row + 1 < rows and self.maze[row + 1][col] != Cell.BLOCKED:
            branches.append(Location(row + 1, col))
        # row up
        if row - 1 < rows and self.maze[row - 1][col] != Cell.BLOCKED:
            branches.append(Location(row - 1, col))
        # row right
        if col + 1 < cols and self.maze[row][col + 1] != Cell.BLOCKED:
            branches.append(Location(row, col + 1))
        # row left
        if col - 1 < cols and self.maze[row][col - 1] != Cell.BLOCKED:
            branches.append(Location(row, col - 1))
        # return all possibilities
        return branches

    # return bool for currentNode  == goalNode
    def checkComplete(self, node: Location):
        return node == self.goal

    # create path for maze once solved
    def createPath(self, path: List[Location]):
        for node in path:
            self.maze[node.row][node.column] = Cell.PATH

    # add start and goal nodes for better visualization
    def reMakeStart(self):
        self.maze[self.start.row][self.start.column] = Cell.START
        self.maze[self.goal.row][self.goal.column] = Cell.GOAL


# frontier: Stack - (first-in-last-out) to search through maze and find the goal
# explored: Set - holds the total number of nodes expanded
###########################################################################
# Path is similar to a singly linked list where every
#   node holds the value to its parent node which is used to generate path.
def dfs(initial: T, checkComplete: Callable[[T], bool], branches: Callable[[T], List[T]]) -> Optional[Node[T]]:
    # queue which determines our next moves in the maze
    frontier: Stack[Node[T]] = Stack()
    frontier.push(Node(initial, None))
    # list to hold explored nodes
    explored: Set[T] = {initial}

    # run until frontier is empty (goal found or maze finished)
    while not frontier.empty:
        # intialize current node and grab state -> location
        currentNode: Node[T] = frontier.pop()
        currentState: T = currentNode.state

        # check if goal has been found
        if checkComplete(currentState):
            print("Node Expanded: {}".format(len(explored)))
            return currentNode

        # if currentNode not goalNode
        else:
            # check if neighboring squares have been explored for each node in frontier
            for node in branches(currentState):
                # if explored, skip node
                if node in explored:
                    continue
                # else add to explored and frontier
                else:
                    explored.add(node)
                    frontier.push(Node(node, currentNode))

    # return None if goal node is unreachable
    return None


# frontier: Priority Queue - (best out) to search through maze and find the goal
# explored: Set - holds the total number of nodes expanded
###########################################################################
# Path is similar to a singly linked list where every
#   node holds the value to its parent node which is used to generate path.
def iddfs(initial: T, checkComplete: Callable[[T], bool], branches: Callable[[T], List[T]]) -> Optional[Node[T]]:
    # priority queue which determines our next moves in the maze
    frontier: PriorityQueue[Node[T]] = PriorityQueue()
    frontier.push(Node(initial, None, 0.0, 0.0))
    # dict of explored nodes with our current cost at said node
    explored: Dict[T, float] = {initial: 0.0}

    while not frontier.empty:
        currentNode: Node[T] = frontier.pop()
        currentState: T = currentNode.state

        if (checkComplete(currentState)):
            print("Node Expanded: {}".format(len(explored)))
            return currentNode
        else:
            for node in branches(currentState):
                tempCost: float = currentNode.cost + 1

                if node not in explored or explored[node] > tempCost:
                    explored[node] = tempCost
                    frontier.push(
                        Node(node, currentNode, tempCost, 0.0))
    return None


# uses 0 as a cost for the node and passes the selected heuristic
# fn: h(n)
def greedy(initial: T, checkComplete: Callable[[T], bool], branches: Callable[[T], List[T]], heuristic: Callable[[T], float]) -> Optional[Node[T]]:
    # priority queue which determines our next moves in the maze
    frontier: PriorityQueue[Node[T]] = PriorityQueue()
    frontier.push(Node(initial, None, 0.0, heuristic(initial)))
    # dict of explored nodes with our current cost at said node
    explored: Set[T] = {initial}

    while not frontier.empty:
        currentNode: Node[T] = frontier.pop()
        currentState: T = currentNode.state

        if (checkComplete(currentState)):
            print("Node Expanded: {}".format(len(explored)))
            return currentNode
        else:
            for node in branches(currentState):
                if node not in explored:
                    explored.add(node)
                    frontier.push(
                        Node(node, currentNode, 0, heuristic(node)))
    return None


# frotier: Priority Queue - based on selected heuristic -> highest h(n) out
# explored: Dictionary - maps between our location and our current cost
# f(n): c(n) + h(n)
def astar(initial: T, checkComplete: Callable[[T], bool], branches: Callable[[T], List[T]], heuristic: Callable[[T], float]) -> Optional[Node[T]]:
    # priority queue which determines our next moves in the maze
    frontier: PriorityQueue[Node[T]] = PriorityQueue()
    frontier.push(Node(initial, None, 0.0, heuristic(initial)))
    # dict of explored nodes with our current cost at said node
    explored: Dict[T, float] = {initial: 0.0}

    while not frontier.empty:
        currentNode: Node[T] = frontier.pop()
        currentState: T = currentNode.state

        if (checkComplete(currentState)):
            print("Node Expanded: {}".format(len(explored)))
            return currentNode
        else:
            for node in branches(currentState):
                tempCost: float = currentNode.cost + 1

                if node not in explored or explored[node] > tempCost:
                    explored[node] = tempCost
                    frontier.push(
                        Node(node, currentNode, tempCost, heuristic(node)))
    return None


# once found last node iteratively traverse using parent nodes to get the path taken
def getFinalPath(node: Node[T]) -> List[T]:
    path: List[T] = [node.state]
    cost = 0
    while node.parent is not None:
        cost += 1
        node = node.parent
        path.append(node.state)
    path.reverse()
    print("Total Cost: {}".format(cost))
    return path


def main():
    cmd = sys.argv
    if len(cmd) == 6:
        if (cmd[1] != '–method' or cmd[3] != '-heuristic'):
            sys.stderr.write("ERROR: Invalid command line statements")
            exit(1)
        if (cmd[2] == 'dfs' or cmd[2] == 'iddfs'):
            sys.stderr.write(
                "ERROR: Too many arguments for search strategy\nFIX: Remove -heuristic flag")

        # init maze
        m: Maze = Maze(cmd[5])

        # Greedy searching algorithm
        if (cmd[2] == "greedy"):
            if (cmd[4] == "euclidian"):
                distance: Callable[[Location],
                                   float] = euclideanDistance(m.goal)
            elif (cmd[4] == "manhattan"):
                distance: Callable[[Location],
                                   float] = manhattanDistance(m.goal)
            solution: Optional[Node[Location]] = greedy(
                m.start, m.checkComplete, m.branching, distance)
            if solution is not None:
                path: List[Location] = getFinalPath(solution)
                m.createPath(path)
                m.reMakeStart()
                print(m)
            else:
                sys.stderr.write('ERROR: No goal found using current solution')

        # A* searching algorithm
        elif (cmd[2] == "astar"):
            if (cmd[4] == "euclidian"):
                # get distance value using euclidian formula
                distance: Callable[[Location],
                                   float] = euclideanDistance(m.goal)
            elif (cmd[4] == "manhattan"):
                # get distance value using manhattan formula
                distance: Callable[[Location],
                                   float] = manhattanDistance(m.goal)
            # get solution node
            solution: Optional[Node[Location]] = astar(
                m.start, m.checkComplete, m.branching, distance)
            # check is solution was reachable
            if solution is not None:
                path: List[Location] = getFinalPath(solution)
                # mark path
                m.createPath(path)
                # bring back the S and G nodes
                m.reMakeStart()
                print(m)
            else:
                sys.stderr.write('ERROR: No goal found using current solution')

    # if dfs or greedy ignore -heuristic flag
    elif (len(cmd) == 4):
        if (cmd[1] != '–method'):
            sys.stderr.write("ERROR: Invalid command line statement")
            exit(1)

        # init maze
        m: Maze = Maze(cmd[3])

        # Depth first search
        if (cmd[2] == "dfs"):
            solution: Optional[Node[Location]] = dfs(
                m.start, m.checkComplete, m.branching)
            if solution is not None:
                path: List[Location] = getFinalPath(solution)
                # mark path
                m.createPath(path)
                # bring back the S and G nodes
                m.reMakeStart()
                print(m)
            else:
                sys.stderr.write('ERROR: No goal found using current solution')

        # IDDFS
        elif (cmd[2] == "iddfs"):
            solution: Optional[Node[Location]] = iddfs(
                m.start, m.checkComplete, m.branching)
            if solution is not None:
                path: List[Location] = getFinalPath(solution)
                m.createPath(path)
                m.reMakeStart()
                print(m)
            else:
                sys.stderr.write('ERROR: No goal found using current solution')
    else:
        sys.stderr.write('ERROR: Invalid number of command line statements')


if __name__ == "__main__":
    main()
