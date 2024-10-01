# This script simulates a quantum random walk on a graph, including a parameter epsilon to consider the incoherence
# In the limit epsilon -> 1, it becomes a classical random walk

import networkx as nx
import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib
from qutip import basis, mesolve, Qobj, ket2dm
from scipy import sparse


# Define the graph
seed = 42
G = nx.erdos_renyi_graph(n=10, p=0.5, seed=seed)
# print(G.number_of_edges())

# G = nx.path_graph(100)
# G = nx.cycle_graph(4)
#
# G = nx.Graph()
# nodes = [0, 1, 2, 3, 4]
# G.add_nodes_from(nodes)
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

nx.draw(G, with_labels=True, node_color='lightblue', font_size=12)
plt.title("Graph Visualization")
# plt.savefig('erdosrenyi8')
plt.show()

# Define the initial state
basis_states = [basis(len(G.nodes), i) for i in range(len(G.nodes))]
initial_state = sum(basis_states).unit()    # Normalized initial state 1/sqrt(N)|1>
# initial_state = basis_states[49]   # (Normalized) initial state consisting of just one of the basis states
# initial_state = basis_states[max(G.degree, key=lambda x: x[1])[0]]
# initial_state= (basis_states[24] + basis_states[74]).unit()
# state = 0
# for i in list(G.neighbors(max(G.degree, key=lambda x: x[1])[0])):
#     print(i)
#     state += basis_states[i]
# initial_state = state.unit()
# print(initial_state)
initial_density_matrix = ket2dm(initial_state)
print(initial_density_matrix)

# Define the target state
# target_state_index = random.choice(range(len(G.nodes)))
target_state_index = 0
target_state = basis_states[target_state_index]
# print("Target state:", target_state_index, target_state)

# Define the Hamiltonian
gamma = 1
epsilon = 0
H_w = -1 * Qobj(target_state * target_state.dag())
L = gamma * nx.normalized_laplacian_matrix(G)
L = sparse.csr_matrix(L)  # Converts to a SciPy CSR sparse matrix
# Two Hamiltonians are defined here, the first one 'converges' to the target state, the second one
# is the usual normalized Laplacian
H = (1-epsilon) * (Qobj(L) + H_w)
# H = (1-epsilon) * Qobj(L)
A = nx.to_numpy_array(G)
degree = np.sum(A, axis=1)
# r = 0.5
jump_operators = []

for edge in G.edges():
    n, m = edge
    gamma_nm = -1 if n == m else A[n, m] / degree[m]
    gamma_mn = -1 if n == m else A[m, n] / degree[n]
    op1 = np.sqrt(epsilon * gamma_nm) * basis(len(G.nodes), n) * basis(len(G.nodes), m).dag()
    op2 = np.sqrt(epsilon * gamma_mn) * basis(len(G.nodes), m) * basis(len(G.nodes), n).dag()
    jump_operators.append(Qobj(op1))
    jump_operators.append(Qobj(op2))

# For the reset framework:
# reset_time = np.random.exponential(scale=1/r)
# print(reset_time)
# for node in G.nodes():
#     op3 = np.sqrt(r) * initial_state * basis(len(G.nodes), node).dag()
#     jump_operators.append((Qobj(op3)))

times = np.linspace(0.0, 20.0, 20000)
result = mesolve(H, initial_density_matrix, times, jump_operators, [])
diagonal_elements = [[result.states[k].diag().real[i] for k in range(len(times))] for i in range(len(G.nodes))]
# summed_diagonal = [sum(diags) for diags in zip(*diagonal_elements[:5])]
fig, ax = plt.subplots()

for i, diag in enumerate(diagonal_elements[:len(G.nodes)]):
    ax.scatter(times, diag, label=f"Simulation Node {i+1}", s=4)  # s controls the size of the points

    # ax.plot(times, diag, linestyle='-', label=f"Simulation Node {i+1}", marker='o', markersize=2)
    # ax.plot(times, summed_diagonal)

ax.set_xlabel('t')
ax.set_ylabel('Probability for each state')
ax.legend()

# Saving the data
# lines = ax.get_lines()
# data = []
#
# for line in lines:
#     data.append(np.column_stack([line.get_xdata(), line.get_ydata()]))
# for i, d in enumerate(data):
#     np.savetxt(f"data_line_graph_5_no_decoherence.txt", d, header="Time Probability", delimiter="\t")

# plt.show()

# Plotting the time averaged probabilities for each node
time_averaged_probabilities = np.zeros(len(G.nodes))
for state in result.states:
    probabilities = state.diag().real
    time_averaged_probabilities += probabilities
time_averaged_probabilities /= len(result.states)

# with open('tap_two_super.txt', 'w') as f:
#     for tap in time_averaged_probabilities:
#         f.write(f"{tap}\n")

data = np.column_stack((range(len(G.nodes)), time_averaged_probabilities))
np.savetxt('time_averaged_probabilities_quantum_no_search.csv', data, delimiter=',', header='Node Index,Time-Averaged Probability', fmt='%d,%.6f', comments='')

# ax.plot(range(0, len(G.nodes)), time_averaged_probabilities, marker='o', linestyle='-')
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

for idx, probability in enumerate(time_averaged_probabilities[:4]):
    print(probability)
    color = colors[idx % len(colors)]
    ax.axhline(y=probability, color=color, linestyle='--', linewidth=1)
    print(probability)
    ax.text(23, probability, f'{probability:.2f}', color=color,
            verticalalignment='top', horizontalalignment='right', fontsize=12)

# plt.xlabel('Node Index')
# plt.ylabel('Time-Averaged Probability')
# plt.title('Time-Averaged Probability of Occupation for Each Node')

plt.legend()
plt.grid()
# plt.savefig('quantum_walk_cycle_graph_4_search')
plt.show()


# Visualize the density matrix over time (real part and complex part)
# time_intervals = np.arange(0, len(times), 100)
# fig, axes = plt.subplots(2, len(time_intervals), figsize=(15, 5))
# for i, idx in enumerate(time_intervals):
#     state = result.states[idx]
#     im_real = axes[0, i].imshow(state.full().real, cmap='viridis', vmin=0, vmax=0.3)
#     axes[0, i].set_title(f"Time = {times[idx]:.2f}")
#     axes[0, i].set_xlabel('Index')
#     axes[0, i].set_ylabel('Index')
#     fig.colorbar(im_real, ax=axes[0, i], fraction=0.046, pad=0.04)
#
#     im_imag = axes[1, i].imshow(state.full().imag, cmap='viridis', vmin=0, vmax=0.3)
#     axes[1, i].set_title(f"Time = {times[idx]:.2f}")
#     axes[1, i].set_xlabel('Index')
#     axes[1, i].set_ylabel('Index')
#     fig.colorbar(im_imag, ax=axes[1, i], fraction=0.046, pad=0.04)
#
# plt.tight_layout()
# plt.show()