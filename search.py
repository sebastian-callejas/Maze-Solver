import sys
from enum import Enum
from typing import NamedTuple, List


class Cell(str, Enum):
    EMPTY = " "
    BLOCKED = "%"
    START = "S"
    GOAL = "G"
    PATH = "*"


class Location(NamedTuple):
    row: int
    column: int


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


def main():
    cmd = sys.argv
    if len(cmd) != 6:
        sys.stderr.write("ERROR: Invalid number of arguments")
        exit(1)
    if (cmd[1] != '–method' or cmd[3] != '-heuristic'):
        sys.stderr.write("ERROR: Invalid command line statement")
        exit(1)
    m1: Maze = Maze(cmd[5])
    print(m1)


if __name__ == "__main__":
    main()
