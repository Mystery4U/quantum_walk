# This script can be used to simulate a random walk on a graph and plots the
# time-averaged probability of being on a certain node vs the timesteps

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

# Define the graph
seed = 42
G = nx.erdos_renyi_graph(n=6, p=0.5, seed=seed)
# G = nx.cycle_graph(4)
# G = nx.Graph()
# nodes = [0, 1, 2, 3, 4]
# G.add_nodes_from(nodes)
#
# adj_matrix = [
#     [0, 1, 1, 0, 0],
#     [1, 0, 0, 1, 1],
#     [1, 0, 0, 1, 0],
#     [0, 1, 1, 0, 1],
#     [0, 1, 0, 1, 0]
# ]
# for i in range(len(nodes)):
#     for j in range(i + 1, len(nodes)):
#         if adj_matrix[i][j] == 1:
#             G.add_edge(nodes[i], nodes[j])
nx.draw(G, with_labels=True, node_color='lightblue', node_size=50)

plt.title("Graph Visualization")
plt.show()

# This function simulates the classic walker on the graph and outputs the probabilities to be on each node


def simulate_walker(graph, start_node, steps):

    # graph: this is the graph G
    # start_node: the walker starts on this graph
    # steps: how many steps you want the walker to make

    current_node = start_node
    node_counts = {node: 0 for node in graph.nodes()}
    node_counts[start_node] = 1
    probabilities = []

    for step in range(1, steps + 1):
        neighbors = list(graph.neighbors(current_node))
        if neighbors:
            current_node = np.random.choice(neighbors)
            node_counts[current_node] += 1
        else:
            print('There is an isolated node')

        total_steps = sum(node_counts.values())
        probabilities.append({node: count / total_steps for node, count in node_counts.items()}) # Now we can plot it against the steps taken

    return probabilities


start_node = 0
total_steps = 5000

probabilities = simulate_walker(G, start_node, total_steps)
average = {node: sum(prob[node] for prob in probabilities) / total_steps for node in G.nodes()}

for node in G.nodes():
    plt.plot(range(1, total_steps + 1), [prob[node] for prob in probabilities], label=f"Knoop {node}")
    plt.plot([1, total_steps], [average[node] for _ in range(2)], 'k--', label=f'Node {node} Average')

for node, prob in average.items():
    print(f"Node {node}: {prob}")

plt.xlabel('t')
plt.ylabel('Time-Averaged Probability')
plt.title('Average Probability vs Steps')
plt.legend()
plt.grid(True)
plt.ylim(0.05, 0.4)
plt.show()
