import numpy as np
import matplotlib.pyplot as plt
import random
from queue import PriorityQueue
import pandas as pd

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
# an iron wall(value 3(outer wall)), a normal wall (value 1) or ghots(value -1)
def find_neighbours(curr_node, maze):
    possible_move1 = [curr_node[0] + 1, curr_node[1]]
    possible_move2 = [curr_node[0] - 1, curr_node[1]]
    possible_move3 = [curr_node[0], curr_node[1] + 1]
    possible_move4 = [curr_node[0], curr_node[1] - 1]
    list_of_moves = [possible_move1, possible_move2, possible_move3, possible_move4]
    list_to_traverse=list_of_moves.copy()
    for x in list_to_traverse:
        if (maze[x[0]][x[1]] == 3.0 or maze[x[0]][x[1]] == 1.0 or maze[x[0]][x[1]] == -1.0):
            list_of_moves.remove(x)
    return  list_of_moves
    
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
# this returns the path to the goal node in the for of list with nodes
# we are using flag condition here to see if its for maze check to actually find the path 
# as in case of maze check we dont consider ghosts in the maze
# this is used for calculating the path after the maze has been made valid and the agent2 follows this path 
# this returns the path to the goal node in the for of list with nodes
def astar(maze, start, end):
    count = 0
    last_pos = {}
    g = np.zeros((int(l),int(b)))
    f = np.zeros((int(l),int(b)))
    for x in range(l):
        for y in range(b):
            g[x][y] = float("inf")
    g[start[0]][start[1]] = 0
    for x in range(l):
        for y in range(b):
            f[x][y] = float("inf") 
    f[start[0]][start[1]] = h(start, end)
    
    kyu = PriorityQueue()
    kyu.put((f[start], count, start))
    h_map = {tuple(start)}
       
    current = []

    while (current != end):
        if flag_check == 0:
            if (kyu.empty()):
                break        
        current = kyu.get()[2]
        neighbours = find_neighbours(current, maze)
        h_map.remove(tuple(current))
        for neighbour in neighbours:
            temp_g = g[current[0]][current[1]] + 1
            if (temp_g < g[neighbour[0]][neighbour[1]]):
                last_pos[tuple(neighbour)] = tuple(current)
                g[neighbour[0]][neighbour[1]] = temp_g
                f[neighbour[0]][neighbour[1]] = temp_g + h(neighbour, end)
                if tuple(neighbour) not in h_map:
                    count += 1
                    kyu.put((f[neighbour[0]][neighbour[1]], count, neighbour))
                    h_map.add(tuple(neighbour))
        if (flag_check == 1):
            if (kyu.empty()):
                return []
                  
    path = []
    last_c = (l-2,b-2)
    while(last_c in last_pos):
        path.append(last_c)
        last_c = last_pos[last_c]
    path = path[::-1]
    return path

# for running it in a loop i had to create this function as we need to check it every time 
# it basically uses astar function only to confirm if the maze has a path or not
def maze_check(maze):
    path = astar(maze, start_node, end_node)
    flag_1 = 0
    while flag_1 != 1:
        if len(path) == 0:
            maze = maze_creation(l, b)
            path = astar(maze, start_node, end_node)
            if len(path) > 0:
                flag_1 = 1
        else:
            return maze
    return maze

# this funtion is only called when th agnet in unable to find a path to the goal node
def better_node(agent_position, neighbours, ghost_position):
    h_pack = []
    curr_pos = agent_position
    for y in ghost_position:
            h_pack.append(h(curr_pos, y))
    ghost_min = h_pack.index(min(h_pack))
    ghost_min_pos = ghost_position[ghost_min]
    n_dist = []
    for x in neighbours:
        n_dist.append(h(x, ghost_min_pos))
    if (len(n_dist) == 0):
        return agent_pos
    ind_of = n_dist.index(max(n_dist))
    return neighbours[ind_of]

# Maze Dimension and number of ghosts
l = 53
b = 53
n = 10

#starting and end node
start_node = [1, 1]
end_node = [l - 2, b - 2]

# im running this entire this in a loop to collect data like no of ghosts, win condition and path length  
fields =  ["No.", "No of ghosts", "Win Condition", "Path Length"]   
data = pd.DataFrame(columns= fields)
i = 1
for x in range(20):
    for y in range(50):
        m = maze_creation(l, b)

        flag_check = 0
        m = maze_check(m)

        mog = m.copy() 



        ghost_pos = ghost_creation(n)
        curr_path = astar(m, start_node, end_node)
        status = True
        agent = 4

        flag_check = 1

        actual_path = []                  # collects the actual path to calcuate the lenght of the path
        agent_pos = [1,1]
        win_condition = "Lost"

        while (status == True):            # when agent reaches the goal node
            # print(curr_path)
            if (agent_pos == (51,51)):
                    print("Success")
                    win_condition = "won"
                    print(win_condition)
                    plt.close('all')
                    break
            if (len(curr_path) == 0):            # this is the scenario where agent is unable to find the path and uses better_node function to go the best possible node
                    neigh = find_neighbours(agent_pos, m)
                    next_node = better_node(agent_pos, neigh, ghost_pos)
                    curr_path.append(next_node)
            agent_pos = curr_path[0]
            if (agent_pos == (51,51)):
                    print("Success")
                    win_condition = "won"
                    print(win_condition)
                    plt.close('all')
                    break
            actual_path.append(agent_pos)
            curr_path.remove(curr_path[0])
            if (m[agent_pos[0]][agent_pos[1]] == -1):        #loss condition
                    print("ded")
                    plt.close('all')
                    break
            ghost_pos =  ghost_movement(ghost_pos)
            m = mog.copy()
            m[agent_pos[0]][agent_pos[1]] = agent
            m = maze_update(m, ghost_pos)
            obstruct=0
            for x in curr_path:
                if (m[x[0]][x[1]]  == -1):                   
                    # print(x)
                    ghost_obstruct = x
                    obstruct=1
            if obstruct==1:
                if h(agent_pos, ghost_obstruct) <= 15:          # only changes its path when ghosts are within 15 manhatten distance away
                    curr_path = astar(m, agent_pos, end_node)
            if (m[agent_pos[0]][agent_pos[1]] == -1):           #loss condition
                    print("ded")
                    plt.close('all')
                    break 
            if (len(curr_path) == 0):
                curr_path = astar(m, start_node, end_node)
            plt.imshow(m)
            plt.pause(0.0001)
            plt.clf()
        plt.show()
        values = pd.DataFrame([{"No." : i, "No of ghosts" : n, "Win Condition" : win_condition, "Path Length" : len(actual_path)}])
        data = pd.concat([data, values], ignore_index=True)
        i = i + 1
    n = n + 10
data.to_csv('Agent2_2.csv', index=False)