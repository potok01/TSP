import numpy as np
import python_tsp.exact
from python_tsp.exact import solve_tsp_dynamic_programming
from python_tsp.exact import solve_tsp_brute_force
from timeit import default_timer as timer
import matplotlib.pyplot as plt
import random
import time
import math
import itertools
import warnings

warnings.filterwarnings("error")


class Node:
    def __init__(self, cost, cost_matrix, path, parent):
        self.__children = []
        self.__parent = parent
        self.__path = path
        self.__cost = cost
        self.__cost_matrix = cost_matrix

    def __lt__(self, other):
        return self.get_cost() < other.get_cost()

    def __gt__(self, other):
        return self.get_cost() > other.get_cost()

    def set_children(self, children):
        self.__children = children

    def set_parent(self, parent):
        self.__parent = parent

    def set_path(self, path):
        self.__path = path

    def set_cost(self, cost):
        self.__cost = cost

    def set_cost_matrix(self, cost_matrix):
        self.__cost_matrix = cost_matrix

    def get_children(self):
        return self.__children

    def get_parent(self):
        return self.__parent

    def get_path(self):
        return self.__path

    def get_cost(self):
        return self.__cost

    def get_cost_matrix(self):
        return self.__cost_matrix

    def append_child(self, child):
        children = self.get_children()
        children.append(child)
        self.set_children(children)

    def append_town(self, town):
        path = self.get_path()
        path.append(town)
        self.set_path(path)


# This function includes the driver code for the function
def main():
    input_file_name = "input.txt"
    distances = read_input_file(input_file_name)
    tsp_print(distances.copy())
    # validation_test(n_limit=15, repetitions=100)
    # maximum_runtime_test(repetitions=5, max_time=30)
    # random_runtime_test(n_limit=15, repetitions=100)
    # animate_tsp(distances.copy())


# This function prints runtime, cost, and path for the branch and bound tsp solver and dynamic programming solver
# This function takes an adjacency matrix as an input
def tsp_print(distances):
    print(distances)

    start = timer()
    my_path, my_cost, nodes = tsp(distances.copy())
    end = timer()
    my_elapsed_time = end - start

    start = timer()
    dynamic_path, dynamic_cost = solve_tsp_dynamic_programming(distances.copy())
    end = timer()
    dynamic_elapsed_time = end - start

    print(f"Branch and bound runtime: {my_elapsed_time}")
    print(f"Dynamic programming runtime: {dynamic_elapsed_time}")

    print(f"Branch and bound minimum cost: {my_cost}")
    print(f"Dynamic programming minimum cost: {dynamic_cost}")

    print(f"Branch and bound path: {my_path}")
    print(f"Dynamic programming path: {dynamic_path}")


# This function reads the input file to generate an adjacency matrix
# This function takes a string file name as the input
# This function returns an adjacency matrix
def read_input_file(input_file_name):
    distances = []
    with open(input_file_name) as f:
        line = f.readline()
        while line:
            distances.append([float(x) for x in line.strip("\n[]").split(",")])
            line = f.readline()
    try:
        distances = np.asarray(distances)
    except:
        print("The input matrix is not square")
        quit(-1)

    i = 0
    while i < len(distances):
        j = 0
        while j < len(distances[i]):
            if i == j:
                distances[i][j] = np.inf
            j = j + 1
        i = i + 1
    return distances


# This is the main branch and bound tsp function This function takes an adjacency matrix as an input This function
# returns the path, cost, and a list of nodes to be used to show how the function explores the tree by the
# animate_tsp function
def tsp(distances):
    nodes = []
    bound = np.inf
    alive_nodes = []
    towns = [i for i in range(0, distances.shape[0])]
    start = 0
    end = 0
    cost = reduce(distances, start, end, 0)

    root = Node(cost, distances.copy(), [], None)
    root.append_town(end)

    # The nodes list is generated for use in animating the tree
    nodes.append(root)
    alive_nodes.append(root)

    answer = root
    while not alive_nodes == []:

        min_node = pop_min(alive_nodes)
        if min_node.get_cost() >= bound:
            return answer.get_path(), bound, nodes

        explore(min_node, towns)

        if min_node is not None:
            children = min_node.get_children()
            nodes += children

            if not children:
                answer = min_node
                bound = min_node.get_cost()

            for child in children:
                insert_node(alive_nodes, child)

    return answer.get_path(), bound, nodes


# This function explores the children of a give node
# This function takes the node to be explored as an input, as well as the total list of towns
def explore(node, towns):
    if node is None:
        return

    visited_towns = node.get_path()
    start = visited_towns[-1]

    remaining_towns = [i for i in towns if i not in visited_towns]
    for town in remaining_towns:
        cost_matrix = node.get_cost_matrix()
        current_cost_matrix = cost_matrix.copy()
        cost = reduce(current_cost_matrix, start, town, node.get_cost())
        node.append_child(Node(cost, current_cost_matrix, visited_towns + [town], node))


# This function reduces an adjacency matrix and returns the cost for a given node
# This function takes the cost matrix as an input, as well as the start town, end town, and the previous node cost
def reduce(cost_matrix, start, end, previous_cost):
    edge_cost = cost_matrix[start][end]
    if edge_cost == np.inf:
        edge_cost = 0

    if not end == 0:
        cost_matrix[start] = np.inf
        cost_matrix[:, end] = np.inf
        cost_matrix[end][start] = np.inf

    n = cost_matrix.shape[0]
    reduced_cost = 0
    for i in range(0, n):
        min_row = min(cost_matrix[i])
        if min_row == np.inf:
            continue
        reduced_cost += min_row
        cost_matrix[i] = cost_matrix[i] - min_row

    for i in range(0, n):
        min_column = min(cost_matrix[:, i])
        if min_column == np.inf:
            continue
        reduced_cost += min_column
        cost_matrix[:, i] = cost_matrix[:, i] - min_column

    cost = reduced_cost + edge_cost + previous_cost
    return cost


# This function is used to manage the heap which operates as a priorty queue. This function takes a node and moves it
# up in the heap to maintain the minimum heap property This function takes the heap which is a list as an input,
# as well as the index of the node to be operated on
def sift_up(heap, node_index):
    if not heap:
        return

    parent_index = int(((node_index - 1) / 2))
    if parent_index < 0:
        return

    if heap[node_index] < heap[parent_index]:
        heap[node_index], heap[parent_index] = heap[parent_index], heap[node_index]
        sift_up(heap, parent_index)


# This function is used to manage the heap. This function takes a node and move it down through the heap to maintain
# the minimum heap property This function takes the heap which is a list as an input, as well as the index of the
# node to be operated on
def sift_down(heap, node_index):
    size = len(heap)
    min_index = node_index
    left_index = 2 * node_index + 1
    right_index = 2 * node_index + 2

    if left_index < size and heap[left_index] < heap[min_index]:
        min_index = left_index

    if right_index < size and heap[right_index] < heap[min_index]:
        min_index = right_index

    if not min_index == node_index:
        heap[node_index], heap[min_index] = heap[min_index], heap[node_index]
        sift_down(heap, min_index)


# This function inserts a node into the heap and ensures the heap property is satisfied
# This function takes two inputs, the heap list and the node to be inserted
def insert_node(heap, node):
    heap.append(node)
    sift_up(heap, len(heap) - 1)


# This function returns the minimum node in the heap and removes it. It also ensures that the heap property is
# maintained after This function takes one input, the heap list
def pop_min(heap):
    minimum = heap[0]
    heap[0] = heap[-1]
    del heap[-1]
    sift_down(heap, 0)
    return minimum


# This was a function made to the presentation to display how branch and bound works The function takes in a distance
# matrix and calls tsp to get all the information needed to follow the branch and bound algorithm through its execution
def animate_tsp(distances):
    graph_delay = 0.001
    size = 300

    path, cost, nodes = tsp(distances.copy())
    depth = int(distances.shape[0])
    xlim = math.factorial(depth - 1) + 1
    ylim = depth + 1

    # Get y values, that is y values on the graph that I will plot the nodes on, for all the nodes in a breadth first
    # search order
    y = []
    for i in range(depth - 1, -1, -1):
        num = int(math.factorial(depth - 1) / (math.factorial(i)))
        y += [i] * num
    y = [x + 1 for x in y]

    # Get x values for all the nodes in a breadth first search order
    x = []
    for i in range(1, depth + 1):
        if i == 1 or i == 2:
            level = [x for x in range(math.factorial(depth - 1), 0, -1)]
        else:
            previous_level = level
            level = []
            previous_population = len(previous_level)
            current_population = previous_population / (i - 1)
            grouping = previous_population / current_population
            for j in range(0, int(current_population)):
                accumulator = 0
                for k in range(0, int(grouping)):
                    accumulator += previous_level[int(j * grouping) + k]
                level += [accumulator / grouping]
        x += level
    x = x[::-1]

    # Get the permutations of the tours starting with town zero in the same order that they would be generated in
    # breadth first search
    paths = [(0,)]
    towns = [x for x in range(0, depth)]
    for i in range(2, depth + 1):
        temp = list(itertools.permutations(towns, i))
        temp = [x for x in temp if x[0] == 0]
        paths += temp

    # Generate a diction using the permutations and coordinates of the nodes
    res = {paths[i]: (x[i], y[i]) for i in range(len(paths))}

    # Fix the graph window
    plt.xlim([0, xlim])
    plt.ylim([0, ylim])

    # Lookup each nodes path in the dictionary to determine the coordinate that the node must be placed in
    x = []
    y = []
    lines = []
    alive_nodes = [[], []]
    dead_nodes = [[], []]
    dead_nodes_tuples = []
    parent = None
    for node in nodes:
        if node.get_path() == path:
            answer_node = node

        x_temp, y_temp = res[tuple(node.get_path())]
        alive_nodes[0].append(x_temp)
        alive_nodes[1].append(y_temp)

        previous_parent = parent
        parent = node.get_parent()
        if not parent == previous_parent and previous_parent is not None:
            parent_index = nodes.index(previous_parent)
            dead_nodes[0].append(alive_nodes[0][parent_index])
            dead_nodes[1].append(alive_nodes[1][parent_index])
            dead_nodes_tuples.append((alive_nodes[0][parent_index], alive_nodes[1][parent_index]))
        if parent in nodes:
            x_parent, y_parent = res[tuple(parent.get_path())]
            line_temp = [[x_temp, x_parent], [y_temp, y_parent]]
            lines.append(line_temp)

        for line in lines:
            plt.plot(line[0], line[1], color="0")

        plt.scatter(alive_nodes[0], alive_nodes[1], color="g", s=size)
        plt.scatter(dead_nodes[0], dead_nodes[1], color="r", s=size)
        plt.draw()
        plt.xlim([0, xlim])
        plt.ylim([0, ylim])
        plt.pause(graph_delay)
        plt.clf()

    solution = [res[tuple(answer_node.get_path())]]
    parent = answer_node.get_parent()
    while parent is not None:
        solution.append(res[tuple(parent.get_path())])
        parent = parent.get_parent()

    alive_nodes[0] = alive_nodes[0][::-1]
    alive_nodes[1] = alive_nodes[1][::-1]
    i = len(alive_nodes[0]) - 1
    while i >= 0:
        x_current = alive_nodes[0][i]
        y_current = alive_nodes[1][i]
        if (x_current, y_current) not in solution and (x_current, y_current) not in dead_nodes_tuples:
            dead_nodes[0].append(x_current)
            dead_nodes[1].append(y_current)

        for line in lines:
            plt.plot(line[0], line[1], color="0")

        plt.scatter(alive_nodes[0], alive_nodes[1], color="g", s=size)
        plt.scatter(dead_nodes[0], dead_nodes[1], color="r", s=size)
        plt.draw()
        plt.xlim([0, xlim])
        plt.ylim([0, ylim])
        plt.pause(graph_delay)
        plt.clf()
        i = i - 1

    for line in lines:
        plt.plot(line[0], line[1], color="0")
    plt.scatter(alive_nodes[0], alive_nodes[1], color="g", s=size)
    plt.scatter(dead_nodes[0], dead_nodes[1], color="r", s=size)

    plt.xlim([0, xlim])
    plt.ylim([0, ylim])
    plt.show()


# This function compares the minimum costs of the branch and bound function versus the dynamic programming algorithm
# The function takes two inputs, the maximum input test size n_limit as an int, and the number of repetitions at each
# input size
def validation_test(n_limit, repetitions):
    my_times = []
    my_paths = []
    my_costs = []
    dynamic_times = []
    dynamic_paths = []
    dynamic_costs = []
    input_size = []

    for n in range(2, n_limit + 1):
        print(f"Input size = {n}")
        input_size.append(n)
        random_numbers = np.random.randint(0, 100, (n, n))
        distances = []
        for row in random_numbers:
            line = []
            for num in row:
                line.append(float(num))
            distances.append(line)
        distances = np.asarray(distances)
        np.fill_diagonal(distances, np.inf)

        test(tsp, my_costs, my_paths, my_times, distances.copy(), repetitions)
        test(solve_tsp_dynamic_programming, dynamic_costs, dynamic_paths, dynamic_times, distances.copy(), repetitions)

    my_costs = np.asarray(my_costs)
    dynamic_costs = np.asarray(dynamic_costs)

    print(f"For an input size: 2 to {n_limit}")
    print(f"Repetitions = {repetitions}")
    if np.array_equal(my_costs, dynamic_costs):
        print("The minimum costs from each algorithm are all the same")
    else:
        print("The minimum costs from each algorithm are not all the same")


# This function compares the runtime of the branch and bound algorithm versus the dynamic programming algorithm The
# function takes two inputs, the maximum input test size n_limit as an int, and the number of repetitions at each
# input size
def random_runtime_test(n_limit, repetitions):
    my_times = []
    my_paths = []
    my_costs = []
    dynamic_times = []
    dynamic_paths = []
    dynamic_costs = []
    input_size = []

    for n in range(2, n_limit + 1):
        print(f"Input size = {n}")
        input_size.append(n)
        random_numbers = np.random.randint(0, 100, (n, n))
        distances = []
        for row in random_numbers:
            line = []
            for num in row:
                line.append(float(num))
            distances.append(line)
        distances = np.asarray(distances)
        np.fill_diagonal(distances, np.inf)

        test(tsp, my_costs, my_paths, my_times, distances.copy(), repetitions)
        test(solve_tsp_dynamic_programming, dynamic_costs, dynamic_paths, dynamic_times, distances.copy(), repetitions)

    plt.plot(input_size, my_times, label="Branch and Bound")
    plt.plot(input_size, dynamic_times, label="Dynamic Programming")
    plt.title("Runtime vs Input Size")
    plt.xlabel("Input size [n]")
    plt.ylabel("Runtime [s]")
    plt.legend()
    plt.show()


# This function compares the runtime of the branch and bound algorithm versus the dynamic programming algorithm The
# function takes two inputs, the maximum input test size n_limit as an int, and the number of repetitions at each
# input size
def maximum_runtime_test(repetitions, max_time):
    my_times = []
    my_paths = []
    my_costs = []
    input_size = []

    current_average_time = 0
    n = 2
    while current_average_time <= max_time:
        print(f"Input size = {n}")
        input_size.append(n)
        random_numbers = np.random.randint(0, 100, (n, n))
        distances = []
        for row in random_numbers:
            line = []
            for num in row:
                line.append(float(num))
            distances.append(line)
        distances = np.asarray(distances)
        np.fill_diagonal(distances, np.inf)

        test(tsp, my_costs, my_paths, my_times, distances.copy(), repetitions)

        current_average_time = my_times[n - 2]
        print(current_average_time)
        n = n + 1

    plt.plot(input_size, my_times, label="Branch and Bound")
    plt.title("Runtime vs Input Size")
    plt.xlabel("Input size [n]")
    plt.ylabel("Runtime [s]")
    plt.legend()
    plt.show()


# This function is used to test the different tsp algorithms, it is called by runtime_test This function takes the
# following inputs, the tsp solving algorithm as func, the costs list to record minimum cost at each input size the
# paths list to record the path at each input size, the times list to record the average runtime at each input size,
# and the number of repetitions for the input size
def test(func, costs, paths, times, distances, repetitions):
    path = 0
    cost = 0
    elapsed_time = 0
    for i in range(0, repetitions):
        start = timer()
        try:
            path, cost, nodes = func(distances.copy())
        except:
            path, cost = func(distances.copy())

        end = timer()
        elapsed_time += end - start

    paths.append(path)
    costs.append(cost)
    times.append(elapsed_time / repetitions)


main()
