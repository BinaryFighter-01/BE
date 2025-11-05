import numpy as np
import random

# Maze: 0 = free path, 1 = wall, 2 = goal
maze = np.array([
    [0, 1, 0, 0, 0],
    [0, 1, 0, 1, 0],
    [0, 0, 0, 1, 0],
    [0, 1, 0, 0, 0],
    [0, 0, 0, 1, 2]
])

start = (0, 0)
goal = (4, 4)

# Actions: Up, Down, Left, Right
actions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

# Q-table (one value for each state × action)
Q = np.zeros((maze.size, 4))

# Parameters
alpha = 0.7     # Learning rate
gamma = 0.9     # Discount factor
epsilon = 0.2   # Exploration rate

# Get index of a position (for Q-table)
def index(pos):
    return pos[0] * maze.shape[1] + pos[1]

# Check if position is valid
def valid(pos):
    return 0 <= pos[0] < 5 and 0 <= pos[1] < 5 and maze[pos] != 1

# Take one step
def step(pos, action):
    new_pos = (pos[0] + actions[action][0], pos[1] + actions[action][1])

    if not valid(new_pos):        # Hit a wall or outside
        return pos, -1, False

    if maze[new_pos] == 2:        # Reached goal
        return new_pos, 10, True

    return new_pos, -0.01, False  # Small penalty for each move

# --- Training ---
for episode in range(2000):
    state = start

    for _ in range(100):
        # Choose explore or exploit
        if random.random() < epsilon:
            action = random.randint(0, 3)
        else:
            action = np.argmax(Q[index(state)])

        # Take action
        next_state, reward, done = step(state, action)

        # Update Q-value
        Q[index(state), action] += alpha * (reward +
                                            gamma * np.max(Q[index(next_state)]) -
                                            Q[index(state), action])

        state = next_state
        if done:
            break

# --- Find best path ---
state = start
path = [state]

for _ in range(50):  # limit steps
    action = np.argmax(Q[index(state)])
    next_state, _, done = step(state, action)
    path.append(next_state)
    state = next_state
    if done:
        break

# --- Display the maze ---
display = np.array([['█' if maze[i, j] == 1 else '.' for j in range(5)] for i in range(5)])

for r, c in path:
    if (r, c) != start and (r, c) != goal:
        display[r, c] = '*'

display[start] = 'S'
display[goal] = 'G'

for row in display:
    print(' '.join(row))
