# ai_project
Ai Project code

import heapq
import time

# Goal state
GOAL_STATE = (1,2,3,4,5,6,7,8,0)

# Directions
MOVES = {
    "UP": -3,
    "DOWN": 3,
    "LEFT": -1,
    "RIGHT": 1
}

def get_neighbors(state):
    neighbors = []
    zero_index = state.index(0)

    for move, pos_change in MOVES.items():
        new_index = zero_index + pos_change

        # invalid moves
        if move == "LEFT" and zero_index % 3 == 0:
            continue
        if move == "RIGHT" and zero_index % 3 == 2:
            continue
        if new_index < 0 or new_index >= 9:
            continue

        new_state = list(state)
        new_state[zero_index], new_state[new_index] = new_state[new_index], new_state[zero_index]
        neighbors.append(tuple(new_state))

    return neighbors

# Heuristic 1: Misplaced Tiles
def misplaced_tiles(state):
    return sum(1 for i in range(9) if state[i] != 0 and state[i] != GOAL_STATE[i])

# Heuristic 2: Manhattan Distance
def manhattan_distance(state):
    distance = 0
    for i in range(9):
        if state[i] != 0:
            x1, y1 = divmod(i, 3)
            goal_index = GOAL_STATE.index(state[i])
            x2, y2 = divmod(goal_index, 3)
            distance += abs(x1 - x2) + abs(y1 - y2)
    return distance

# A* Search
def a_star(start, heuristic):
    start_time = time.time()

    frontier = []
    heapq.heappush(frontier, (0, start))
    
    came_from = {}
    cost_so_far = {start: 0}

    nodes_expanded = 0

    while frontier:
        _, current = heapq.heappop(frontier)

        if current == GOAL_STATE:
            break

        nodes_expanded += 1

        for neighbor in get_neighbors(current):
            new_cost = cost_so_far[current] + 1

            if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                cost_so_far[neighbor] = new_cost
                priority = new_cost + heuristic(neighbor)
                heapq.heappush(frontier, (priority, neighbor))
                came_from[neighbor] = current

    end_time = time.time()

    return {
        "nodes": nodes_expanded,
        "time": end_time - start_time,
        "cost": cost_so_far.get(GOAL_STATE, -1)
    }

# Greedy Best-First Search
def greedy(start, heuristic):
    start_time = time.time()

    frontier = []
    heapq.heappush(frontier, (heuristic(start), start))

    visited = set()
    nodes_expanded = 0

    while frontier:
        _, current = heapq.heappop(frontier)

        if current == GOAL_STATE:
            break

        if current in visited:
            continue

        visited.add(current)
        nodes_expanded += 1

        for neighbor in get_neighbors(current):
            if neighbor not in visited:
                heapq.heappush(frontier, (heuristic(neighbor), neighbor))

    end_time = time.time()

    return {
        "nodes": nodes_expanded,
        "time": end_time - start_time
    }

# Test Cases (easy, medium, hard)
test_cases = {
    "Easy": (1,2,3,4,5,6,7,0,8),
    "Medium": (1,2,3,5,0,6,4,7,8),
    "Hard": (7,2,4,5,0,6,8,3,1)
}

# Run experiments
for level, start in test_cases.items():
    print(f"\n--- {level} ---")

    print("A* (Manhattan):", a_star(start, manhattan_distance))
    print("A* (Misplaced):", a_star(start, misplaced_tiles))

    print("Greedy (Manhattan):", greedy(start, manhattan_distance))
    print("Greedy (Misplaced):", greedy(start, misplaced_tiles))
