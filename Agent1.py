import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
from queue import PriorityQueue

# Maze Dimension
l = 53
b = 53 
n = 10 

start_node = [1, 1]
end_node = [l - 2, b - 2]

# creating and printing maze with block probablity of 28%
def maze_creation(l, b):
    bs=set()
    maze = np.zeros((int(l),int(b)))
    temp = 1
    blocks = int(0.28*(l-2)*(b-2))
    for i in range(l):
        maze[0][i] = 3
        maze[l-1][i] = 3
        maze[i][0] = 3
        maze[i][b-1] = 3
    while(temp <= blocks):
        i = np.random.randint(1,l-1)
        j = np.random.randint(1,b-1)
        if(i,j) not in bs:
            bs.add((i,j))
            maze[i][j] = 1
            temp += 1

    maze[1][1] = 0
    maze[l-2][b-2] = 0
    return maze

# this returns neighbour nodes with conditions that it should not be 
# an iron wall(value 3(outer wall)), a normal wall (value 1)
def find_neighbours(curr_node, maze):
    possible_move1 = [curr_node[0] + 1, curr_node[1]]
    possible_move2 = [curr_node[0] - 1, curr_node[1]]
    possible_move3 = [curr_node[0], curr_node[1] + 1]
    possible_move4 = [curr_node[0], curr_node[1] - 1]
    list_of_moves = [possible_move1, possible_move2, possible_move3, possible_move4]
    list_to_traverse=list_of_moves.copy()
    for x in list_to_traverse:
        if (maze[x[0]][x[1]] == 3.0 or maze[x[0]][x[1]] == 1.0):
            list_of_moves.remove(x)
    if (list_of_moves is np.empty):
            print("Bad Maze")
            exit()
    return  list_of_moves

# this returns neighbour nodes with conditions that it should not be 
# an iron wall(value 3(outer wall)), a normal wall (value 1) or ghots(value -1)
def find_neighbours_agent2(curr_node, maze):
    possible_move1 = [curr_node[0] + 1, curr_node[1]]
    possible_move2 = [curr_node[0] - 1, curr_node[1]]
    possible_move3 = [curr_node[0], curr_node[1] + 1]
    possible_move4 = [curr_node[0], curr_node[1] - 1]
    list_of_moves = [possible_move1, possible_move2, possible_move3, possible_move4]
    for x in list_of_moves:
        if (maze[x[0]][x[1]] == 3 or maze[x[0]][x[1]] == 1 or maze[x[0]][x[1]] == -1):
            list_of_moves.remove(x)
    return  list_of_moves


# here we generate a maze and make a copy of it for future use
m = (maze_creation(l, b))
mog = m.copy()    

# this is a heuristic function that calculates and returns manhatten distance
def h(to_node, end_node):
    x1,y1 = to_node
    x2,y2 = end_node

    return abs(x1-x2) + abs(y1-y2)

# this function generates and spawn ghosts at random loctaion and it should not be a block node
def ghost_creation(n):
    arr = []
    i = 0
    while(i < n):
        ind = []
        x = np.random.randint(1,l-1)
        y = np.random.randint(1,b-1)
        if (mog[x][y] != 1 or (x != 1 & y != 1)):
            ind.append(x)
            ind.append(y)
        else:
            i -= 1
        if (ind != []):
            arr.append(ind)
        i += 1
    return arr

# this function make updates to the maze during ghosts movement
def maze_update(maze, positions): 
    for x in positions:
        x_c = x[0]
        y_c = x[1]
        maze[x_c][y_c] = -1
    return maze

# this function randomly moves ghosts to any of the neighbouring cell although it does check if it is a blocked cell or not 
# if it is then it has 50% probablity of going there or it remains in its position 
# this returns new nodes of ghosts
def ghost_movement(positions):
    i = 0
    for pos in positions:
        # print(pos)
        x_c = pos[0]
        y_c = pos[1]     
        possible_cod1 = [x_c + 1, y_c]
        possible_cod2 = [x_c, y_c + 1]
        possible_cod3 = [x_c - 1, y_c]
        possible_cod4 = [x_c, y_c - 1]
        possible_pos = [possible_cod1, possible_cod2, possible_cod3, possible_cod4]
        traverse_position =  possible_pos.copy()
        for d in traverse_position:
            if (mog[d[0]][d[1]] == 3):
                possible_pos.remove(d)
        rand_pos = random.choice(possible_pos)
        x, y = rand_pos
        if (mog[x][y] == 1):
            rand_pos = random.choice([rand_pos, pos])
        positions[i] = rand_pos
        i = i + 1
    return positions

# this is astar function which finds an open path between agent and goal node without concerning about ghosts 
# it uses priority queue and hash map to keep track of the visited cell and the fvalue of that node cell
# this is only used initially to check if its a valid maze or not
def astar(maze):
    count = 0
    last_pos = {}
    g = np.zeros((int(l),int(b)))
    f = np.zeros((int(l),int(b)))
    for x in range(l):
        for y in range(b):
            g[x][y] = float("inf")
    g[start_node[0]][start_node[1]] = 0
    for x in range(l):
        for y in range(b):
            f[x][y] = float("inf") 
    f[start_node[0]][start_node[1]] = h(start_node, end_node)    
    kyu = PriorityQueue()
    kyu.put((f[start_node], count, start_node))
    h_map = {tuple(start_node)}    
    current = []
    while (current != end_node):
        if (kyu.empty()):                   #this is to check if the maze is valid or not
            maze_condition = "Bad Maze"
            print(maze_condition)
            break
        current = kyu.get()[2]
        neighbours = find_neighbours(current, maze)
        for neighbour in neighbours:
            temp_g = g[current[0]][current[1]] + 1
            if (temp_g < g[neighbour[0]][neighbour[1]]):
                last_pos[tuple(neighbour)] = tuple(current)
                g[neighbour[0]][neighbour[1]] = temp_g
                f[neighbour[0]][neighbour[1]] = temp_g + h(neighbour, end_node)
                if tuple(neighbour) not in h_map:
                    count += 1
                    kyu.put((f[neighbour[0]][neighbour[1]], count, neighbour))
                    h_map.add(tuple(neighbour))
        if (kyu.empty()):
                return []
    path = []
    last_c = (l-2,b-2)
    while(tuple(last_c) in last_pos):
        path.append(last_c)
        last_c = last_pos[tuple(last_c)]
    path = path[::-1]
    return path

# this is astar function which finds an open path between agent and goal node without concerning about ghosts 
# it uses priority queue and hash map to keep track of the visited cell and the fvalue of that node cell
# this is used for calculating the path and the maze has been made valid and the agent1 follows this path 
# this returns path length and win condition
def agent1(maze):
    agent = 4
    count = 0
    last_pos = {}
    g = np.zeros((int(l),int(b)))
    f = np.zeros((int(l),int(b)))
    for x in range(l):
        for y in range(b):
            g[x][y] = float("inf")
    g[start_node[0]][start_node[1]] = 0
    for x in range(l):
        for y in range(b):
            f[x][y] = float("inf") 
    f[start_node[0]][start_node[1]] = h(start_node, end_node)
    
    kyu = PriorityQueue()
    kyu.put((f[start_node], count, start_node))
    h_map = {tuple(start_node)}
    
    current = []

    # in this loop we are using manhatten distance as the heuristic to search for the path 
    # its also maintinging a last_pos dict that is capturing the location it is visitng to trace back to the original node
    while (current != end_node):
        current = kyu.get()[2]
        neighbours = find_neighbours(current, maze)
        for neighbour in neighbours:
            temp_g = g[current[0]][current[1]] + 1
 
            if (temp_g < g[neighbour[0]][neighbour[1]]):
                last_pos[tuple(neighbour)] = tuple(current)
                g[neighbour[0]][neighbour[1]] = temp_g
                f[neighbour[0]][neighbour[1]] = temp_g + h(neighbour, end_node)
                if tuple(neighbour) not in h_map:
                    count += 1
                    kyu.put((f[neighbour[0]][neighbour[1]], count, neighbour))
                    h_map.add(tuple(neighbour))

    path = []
    last_c = (l-2,b-2)
    while(last_c in last_pos):
        path.append(last_c)
        last_c = last_pos[last_c]
    path = path[::-1]    
    print(path)
    
    ghost_pos = ghost_creation(n)

    win_condition = "Lost"

    # agent will now follow the path and check if it encounters the ghosts it dies 
    # also we keep updating the maze and ghost position in the maze for agent to check if there is ghosts or not
    # can also visually see excatly what is happening 
    for x in path: 
        ghost_pos =  ghost_movement(ghost_pos)
        if (maze[x[0]][x[1]] == -1):
            print("ded")
            win_condition = "Lost"
            plt.close('all')
            break
        maze = mog.copy()
        maze[x[0]][x[1]] = agent
        maze = maze_update(maze, ghost_pos)
        
        if (maze[x[0]][x[1]] == -1):
            print("ded")
            win_condition = "Lost"
            plt.close('all')
            break
        if (x == (51,51)):
            win_condition = "Won"
            print("Success")
            plt.close('all')
            break
        plt.imshow(maze)
        plt.pause(0.0001)
        plt.clf()
    plt.show()
    return len(path), win_condition   

# for running it in a loop i had to create this function as we need to check it every time 
# it basically uses astar function only to confirm if the maze has a path or not
def maze_check(maze):
    path = astar(maze)
    flag_1 = 0
    while flag_1 != 1:
        if len(path) == 0:
            maze = maze_creation(l, b)
            path = astar(maze)
            if len(path) > 0:
                flag_1 = 1
        else:
            return maze
    return maze  

# im running this entire this in a loop to collect data like no opf ghosts, win condition and path length    
fields =  ["No.", "No of ghosts", "Win Condition", "Path Length"]   
data = pd.DataFrame(columns= fields)
i = 1
for x in range(20):
    for y in range(50):    
        m = (maze_creation(l, b))
        m = maze_check(m)
        mog = m.copy()
        length, reslt = agent1(m)
        values = pd.DataFrame([{"No." : i, "No of ghosts" : n, "Win Condition" : reslt, "Path Length" : length}])
        data = pd.concat([data, values], ignore_index=True)
        i = i + 1
    n = n + 10
data.to_csv('Agent1.csv', index=False)        