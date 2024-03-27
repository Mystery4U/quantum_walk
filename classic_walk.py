import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

# Creating the graph with a certain 'seed' such that it generates the same graph each time it runs.
seed = 42
G = nx.erdos_renyi_graph(n=6, p=0.5, seed=seed)

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
total_steps = 10**4

probabilities = simulate_walker(G, start_node, total_steps)
average = {node: sum(prob[node] for prob in probabilities) / total_steps for node in G.nodes()}

for node in G.nodes():
    plt.plot(range(1, total_steps + 1), [prob[node] for prob in probabilities], label=f"Node {node}")
    plt.plot([1, total_steps], [average[node] for _ in range(2)], 'k--', label=f'Node {node} Average')

for node, prob in average.items():
    print(f"Node {node}: {prob}")

plt.xlabel('Steps')
plt.ylabel('Average Probability')
plt.title('Average Probability vs Steps')
plt.legend()
plt.grid(True)
plt.ylim(0, 0.4)
plt.show()
