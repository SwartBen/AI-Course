import sys
import numpy as np

states_lookup = {}
sensor_lookup = {
    "0000": 0,
    "0001": 1, 
    "0010": 2, 
    "0011": 3, 
    "0100": 4, 
    "0101": 5, 
    "0110": 6, 
    "0111": 7, 
    "1000": 8, 
    "1001": 9, 
    "1010": 10, 
    "1011": 11, 
    "1100": 12, 
    "1101": 13, 
    "1110": 14, 
    "1111": 15
}

def read_input():

    path = sys.argv[-1]
    with open(path) as file: lines = file.readlines()

    # Read size of map
    R, C = lines.pop(0).split()
    R, C = int(R), int(C)

    # Read grid
    grid = []
    for _ in range(0, R):
        grid.append(lines.pop(0).split())

    # Read num Observations
    num_observations = int(lines.pop(0))

    # Read sensor observations
    observed_values = []
    for _ in range(0, num_observations):
        row = lines.pop(0)
        observed_values.append(row[0:len(row)-1])

    # Read error rate
    error_rate = float(lines.pop(0))

    return grid, observed_values, num_observations, error_rate

def viterbi(observations, states, start_probability, trans_matrix, emit_matrix, R, C):
    
    maps = [] 
    # Calc inital trellis matrix
    trellis = np.zeros((R, C))
    for state in states:
        probi = start_probability[states_lookup[state]]
        Emiy1 = emit_matrix[states_lookup[state]][sensor_lookup[observations[0]]]
        trellis[state[0]][state[1]] = probi * Emiy1
    
    maps.append(trellis)

    # Calc the trellis matrix for each obs 
    for j in range(1, len(observations)):

        # (emission_prob_dict[state_space[i]][observation_space.index(observations[j])]
        trellis = np.zeros((R, C))
        for state in states: #Next state
            prob = 0
            for p_state in states: #Prior state
                Tmki = trans_matrix[states_lookup[p_state]][states_lookup[state]]
                Emiyj = emit_matrix[states_lookup[state]][sensor_lookup[observations[j]]]
                prob = max(maps[-1][p_state[0]][p_state[1]] * Tmki * Emiyj, prob)

            trellis[state[0]][state[1]] = prob

        maps.append(trellis)

    return maps

# Transition matrix - K * K
def calculate_transition(grid, states, K):
    
    trans_matrix = np.zeros((K, K))
    # Loop over each viable pos
    for (r, c) in states:
        neighbours = find_neighbours(grid, r, c)
        n = len(neighbours)
        # Loop over neighbours of each viable pos
        for neigh in neighbours:
            trans_matrix[states_lookup[(r, c)]][states_lookup[neigh]] = 1 / n

    return trans_matrix

#Emission matrix - K * N - states * obs
def calculate_emission(grid, epsilon, K):
    
    emit_matrix = np.zeros((K, 16))

    # Loop over each state
    for (r, c) in states:
        truth = true_reading(grid, r, c)

        # Loop over each possible sensor value and calculate error
        for possible_reading in sensor_lookup:
            dit = 0
            for i in range(4):
                if truth[i] != possible_reading[i]:
                    dit += 1     
            
            error = np.power(1 - epsilon, 4 - dit) * np.power(epsilon, dit)
            emit_matrix[states_lookup[(r, c)]][sensor_lookup[possible_reading]] = error

    return emit_matrix

def find_neighbours(grid, r, c):
    
    # Find all the non blocked neighbours of pos (r, c)
    neighbours = []
    if r - 1 >= 0 and grid[r-1][c] == '0': # NORTH
        neighbours.append((r-1, c))    
    if c + 1 < len(grid[0]) and grid[r][c+1] == '0': # EAST
        neighbours.append((r, c+1)) 
    if r + 1 < len(grid) and grid[r+1][c] == '0': # SOUTH
        neighbours.append((r+1, c)) 
    if c - 1 >= 0 and grid[r][c-1] == '0': # WEST
        neighbours.append((r, c-1))

    return neighbours  

def true_reading(grid, r, c):
    
    # Determine truth value for sensor reading of pos (r, c)
    truth = ['1', '1', '1', '1']  #If no obstacle and valid pos convert 1 to 0.
    if r - 1 >= 0 and grid[r-1][c] == '0': # NORTH
        truth[0] = '0'
    if c + 1 < len(grid[0]) and grid[r][c+1] == '0': # EAST
        truth[1] = '0'   
    if r + 1 < len(grid) and grid[r+1][c] == '0': # SOUTH
        truth[2] = '0'
    if c - 1 >= 0 and grid[r][c-1] == '0': # WEST
        truth[3] = '0' 

    return truth

def init_probability(grid, K, R, C):

    # Init start probabilties for each possible starting pos in the grid
    start_probability = np.zeros(K)
    for r in range(R):
        for c in range(C):
            if grid[r][c] == '0':
                start_probability[states_lookup[(r, c)]] = 1 / K

    return start_probability

def calc_states(grid, R, C):
    
    #Calculate the states (all non blocked pos) in the grid
    K = 0
    states = []
    for r in range(R):
        for c in range(C):
            if grid[r][c] == '0':
                states.append((r, c))
                states_lookup[(r, c)] = K
                K += 1
                
    return states, K

#**********************************************MAIN**********************************************

# Read input
grid, observations, num_observations, error_rate = read_input()
R, C = len(grid), len(grid[0])

# Calc trellis matrix
states, K = calc_states(grid, R, C)
start_probability = init_probability(grid, K, R, C)
trans_matrix = calculate_transition(grid, states, K)
emit_matrix = calculate_emission(grid, error_rate, K)

maps = viterbi(observations, states, start_probability, trans_matrix, emit_matrix, R, C)

# Output
np.savez("output.npz", *maps)


