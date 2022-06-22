from queue import PriorityQueue
import numpy as np
import sys

#---------------------------Algorithm Implementation---------------------------#

# A queue node used in BFS
class Node:
    # (x, y) represents coordinates of a cell in the matrix
    # maintain a parent node for the printing path
    def __init__(self, x, y, parent=None, g=0, h=0, f=0):
        self.x = x
        self.y = y
        self.parent = parent
        self.g = g
        self.h = h
        self.f = f
 
def getPath(node):
    path = []
    while(node):
        path.append((node.x, node.y))
        node = node.parent

    return path
 
def bfs(grid, start, end):
    dirs = ((-1, 0), (1, 0), (0,-1), (0, 1)) #up, down, left, right 

    queue = []
    visited = set()

    startNode = Node(start[0], start[1], None)
    queue.append(startNode)
    visited.add((start[0], start[1]))

    while(len(queue) != 0):
        cur_node = queue.pop(0)
        x = cur_node.x
        y = cur_node.y

        #Found path
        if (x, y) == end:
            return getPath(cur_node)

        #enumerate up, down, left, right
        for r, c in dirs:
            r += x
            c += y
            if 0<=r<len(grid) and 0<=c<len(grid[0]) and (r, c) not in visited and grid[r][c] != 'X':
                newNode = Node(r, c, cur_node)
                queue.append(newNode)
                visited.add((r, c))

    return []

def ucs(grid, start, end):
    dirs = ((-1, 0), (1, 0), (0,-1), (0, 1)) #up, down, left, right 

    q = PriorityQueue()    
    visited = set()

    startNode = Node(start[0], start[1], None)
    q.put((0, 0, startNode))
    visited.add((start[0], start[1]))

    freq = {}
    while not q.empty():
        cur_node = q.get()[2]
        x = cur_node.x
        y = cur_node.y
        
        #Found path
        if (x, y) == end:
            return getPath(cur_node)

        #enumerate up, down, left, right
        for r, c in dirs:
            r += x
            c += y
            if 0<=r<len(grid) and 0<=c<len(grid[0]) and (r, c) not in visited and grid[r][c] != 'X':

                #Calculate heuristic 
                elevation = grid[r][c]

                if elevation not in freq:
                    freq[elevation] = 1
                else:
                    freq[elevation] += 1
                
                #Add new node to priority queue
                newNode = Node(r, c, cur_node)
                q.put((elevation, freq[elevation], newNode))
                visited.add((r, c))

    return []

def astar(heuristic, grid, start, end):
    dirs = ((-1, 0), (1, 0), (0,-1), (0, 1)) #up, down, left, right 

    q = PriorityQueue()
    visited = set()

    est_g = 0
    if heuristic == "euclidean":
        est_h = abs(end[0] - start[0])**2 + abs(end[1] - start[1])**2
    elif heuristic == "manhattan":
        est_h = abs(end[0] - start[0]) + abs(end[1] - start[1])
    est_f = est_h + est_g
    startNode = Node(start[0], start[1], est_g, est_h, est_f)

    q.put((est_f, 0, startNode))
    visited.add((start[0], start[1]))

    freq = {}
    while not q.empty():
        cur_node = q.get()[2]
        x = cur_node.x
        y = cur_node.y

        #Found path
        if (x, y) == end:
            return getPath(cur_node)

        #enumerate up, down, left, right
        for r, c in dirs:
            r += x
            c += y
            if 0<=r<len(grid) and 0<=c<len(grid[0]) and (r, c) not in visited and grid[r][c] != 'X':
                
                #Dist between current node and the start node
                if int(grid[r][c]) - int(grid[x][y]) > 0:
                    est_g = cur_node.g + abs(int(grid[x][y]) - int(grid[r][c])) + 1
                else:
                    est_g = cur_node.g + 1

                #Calculate heuristic
                #Euclidean
                if heuristic == "euclidean":
                    est_h = (end[0] - r)**2 + (end[1] - c)**2

                #Manhatten
                elif heuristic == "manhattan":
                    est_h = abs(end[0] - r) + abs(end[1] - c)

                est_f = est_g + est_h

                if est_f not in freq:
                    freq[est_f] = 1
                else:
                    freq[est_f] += 1
                
                #Add new node the priority queue
                newNode = Node(r, c, cur_node, est_g, est_h, est_f)
                q.put((est_f, freq[est_f], newNode))
                visited.add((r, c))

    return []

#---------------------------PRINT SOLUTION---------------------------#

def print_answer(grid, path):


    if path == []:
        print("null")
    else:
        #Edit ans to include path
        for i in range(len(grid)):
            for j in range(len(grid[0])):
                if (i, j) in path:
                    grid[i][j] = "*"

        #Print ans
        for rows in grid:
            cur = ' '.join([str(val) for val in rows])
            print(cur)

        print()
#---------------------------Reading Input---------------------------#

def read_input():

    #Read arguments
    arguments = sys.argv

    if len(arguments) == 4:        
        map_path, algorithm, heuristic = arguments[1], arguments[2], arguments[3]
    else:
        map_path, algorithm, heuristic = arguments[1], arguments[2], None

    #Open txt file
    with open(map_path) as file:
        lines = file.readlines()

    #Read txt file
    R, C = lines[0].split()
    start_r, start_c = lines[1].split()
    end_r, end_c = lines[2].split()
    R, C, start_r, start_c, end_r, end_c = int(R), int(C), int(start_r)-1, int(start_c)-1, int(end_r)-1, int(end_c)-1

    grid = []
    for i in range(3, R+3):
        grid.append(lines[i].split())

    return grid, R, C, algorithm, heuristic, (start_r, start_c), (end_r, end_c)

#---------------------------Run algorithm---------------------------#

def run(grid, heuristic, algorithm, start, end):
    if algorithm == "bfs":
        return( bfs(grid, start, end) )
    elif algorithm == "ucs":
        return ( ucs(grid, start, end) )
    elif algorithm == "astar": 
    #also need to pass heurstic 
        return ( astar(heuristic, grid, start, end) )

#---------------------------Main code---------------------------#

grid, R, C, algorithm, heuristic, start, end = read_input()

path = run(grid, heuristic, algorithm, start, end)
print_answer(grid, path)