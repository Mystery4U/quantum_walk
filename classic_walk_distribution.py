# This script plots the distribution of a random walk on a line graph with 100 nodes.
# The walker reflects when it reaches the endpoints
import random
import matplotlib.pyplot as plt
import numpy as np
import csv
from tqdm import tqdm


def random_walk(num_nodes):
    nodes_visited = [50]
    current_node = 50
    while True:
        direction = random.choice([-1, 1])
        current_node += direction

        if current_node == 0:
            current_node = 1
        elif current_node == num_nodes:
            current_node = num_nodes - 1

        nodes_visited.append(current_node)

        if len(nodes_visited) == 10000:
            break

    return nodes_visited


num_walks = 10000
all_nodes = []
for _ in tqdm(range(num_walks)):
    nodes = random_walk(100)
    all_nodes.extend(nodes)

hist, bins = np.histogram(all_nodes, bins=range(101))
plt.hist(all_nodes, bins=range(101), edgecolor='black')
plt.xlabel('Node')
plt.ylabel('Frequency')
plt.title('Distribution of Nodes Visited in Random Walks')
plt.savefig('random_walk_histogram.png')
plt.show()
#
# with open('random_walk_data.csv', 'w', newline='') as csvfile:
#     writer = csv.writer(csvfile)
#     writer.writerow(['Node', 'Frequency'])
#     for i in range(len(hist)):
#         writer.writerow([bins[i], hist[i]])
