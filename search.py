from __future__ import annotations
from re import T
import sys
from enum import Enum
from typing import Callable, Generic, Deque, NamedTuple, List, Optional, Set, TypeVar


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


class Queue(Generic[T]):
    def __init__(self) -> None:
        self._container: Deque[T] = Deque()

    @property
    def empty(self) -> bool:
        return not self._container

    def push(self, item: T) -> None:
        self._container.append(item)

    def pop(self) -> T:
        return self._container.popleft()

    def __repr__(self) -> str:
        return repr(self._container)


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

    def reMakeStart(self):
        self.maze[self.start.row][self.start.column] = Cell.START
        self.maze[self.goal.row][self.goal.column] = Cell.GOAL


def bfs(initial: T, checkComplete: Callable[[T], bool], branches: Callable[[T], List[T]]) -> Optional[Node[T]]:
    # queue which determines our next moves in the maze
    frontier: Queue[Node[T]] = Queue()
    frontier.push(Node(initial, None))
    # list to hold explored nodes
    explored: Set[T] = {initial}

    while not frontier.empty:
        currentNode: Node[T] = frontier.pop()
        currentState: T = currentNode.state

        if checkComplete(currentState):
            return currentNode

        else:
            for node in branches(currentState):
                if node in explored:
                    continue
                else:
                    explored.add(node)
                    frontier.push(Node(node, currentNode))
    return None


# once found last node iteratively traverse using parent nodes to get the path taken


def getFinalPath(node: Node[T]) -> List[T]:
    path: List[T] = [node.state]
    while node.parent is not None:
        node = node.parent
        path.append(node.state)
    path.reverse()
    return path


def main():
    cmd = sys.argv
    if len(cmd) != 6:
        sys.stderr.write("ERROR: Invalid number of arguments")
        exit(1)
    if (cmd[1] != '–method' or cmd[3] != '-heuristic'):
        sys.stderr.write("ERROR: Invalid command line statement")
        exit(1)

    # init maze
    m: Maze = Maze(cmd[5])

    # Breadth first search
    if (cmd[2] == "bfs"):
        solution: Optional[Node[Location]] = bfs(
            m.start, m.checkComplete, m.branching)
        if solution is not None:
            path: List[Location] = getFinalPath(solution)
            m.createPath(path)
            m.reMakeStart()
            print(m)
        else:
            sys.stderr.write('ERROR: No goal found using current solution')


if __name__ == "__main__":
    main()
