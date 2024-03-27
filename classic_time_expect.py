import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from tqdm import tqdm

seed = 42

G = nx.erdos_renyi_graph(n=6, p=0.5, seed=seed)
labels = {node: f"Node {node}" for node in G.nodes()}

# nx.draw(G, with_labels=True, node_color='lightblue', node_size=50)
#
# plt.title("Graph Visualization")
# plt.show()


def simulate_walker(graph, start_node, steps):
    current_node = start_node
    positions = [current_node]

    for _ in range(steps):
        neighbors = list(graph.neighbors(current_node))
        if neighbors:
            current_node = np.random.choice(neighbors)
        positions.append(current_node)
    return positions


def runtime(graph, start_node, target_node):
    current_node = start_node
    steps = 0

    while current_node != target_node:
        neighbors = list(graph.neighbors(current_node))
        if neighbors:
            current_node = np.random.choice(neighbors)
            steps += 1

    return steps


start_node = 0
target_node = 5
distance = nx.shortest_path_length(G, source=start_node, target=target_node) - 1

times = []
total_simulations = 10000

for i in tqdm(range(total_simulations)):
    times.append(runtime(G, start_node, target_node))

plt.hist(times, bins=np.arange(min(times), max(times) + 1.5) - 0.5)
plt.axvline(np.mean(times), color='red', linestyle='dashed', linewidth=1)
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('Histogram')
plt.grid(True)
plt.show()
