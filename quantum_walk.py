import networkx as nx
import numpy as np
from qutip import basis, mesolve, sigmax, sigmay, sigmaz, Qobj, ket2dm
import matplotlib.pyplot as plt

seed = 42
G = nx.erdos_renyi_graph(n=6, p=0.5, seed=seed)

nx.draw(G, with_labels=True, node_color='lightblue', node_size=50)
plt.title("Graph Visualization")
plt.show()

basis_states = [basis(len(G.nodes), i) for i in range(len(G.nodes))]
initial_state = sum(basis_states).unit()
initial_density_matrix = ket2dm(initial_state)

L = nx.normalized_laplacian_matrix(G)
H = Qobj(L)

A = nx.to_numpy_array(G)
degree = np.sum(A, axis=1)

epsilon = 0.7
jump_operators = []

for edge in G.edges():
    n, m = edge
    gamma_nm = -1 if n == m else A[n, m] / degree[m]
    gamma_mn = -1 if n == m else A[m, n] / degree[n]
    op1 = np.sqrt(epsilon * gamma_nm) * basis(len(G.nodes), n) * basis(len(G.nodes), m).dag()
    op2 = np.sqrt(epsilon * gamma_mn) * basis(len(G.nodes), m) * basis(len(G.nodes), n).dag()
    # op = gamma * basis(len(G.nodes), n) * basis(len(G.nodes), m).dag() # Another possibility for collapse operators
    jump_operators.append(Qobj(op1))
    jump_operators.append(Qobj(op2))

times = np.linspace(0.0, 15.0, 500)
result = mesolve(H, initial_density_matrix, times, jump_operators, [])

diagonal_elements = [[result.states[k].diag().real[i] for k in range(len(times))] for i in range(len(G.nodes))]

fig, ax = plt.subplots()
for i, diag in enumerate(diagonal_elements):
    ax.plot(times, diag, linestyle = '-', label=f"Diagonal {i}", marker='o', markersize=2)
ax.set_xlabel('Time')
ax.set_ylabel('Density Matrix Elements')
ax.legend()
plt.show()


time_intervals = np.arange(0, len(times), 30)
fig, axes = plt.subplots(2, len(time_intervals), figsize=(15, 5))
for i, idx in enumerate(time_intervals):
    state = result.states[idx]
    im_real = axes[0, i].imshow(state.full().real, cmap='viridis', vmin=0, vmax=0.3)
    axes[0, i].set_title(f"Time = {times[idx]:.2f}")
    axes[0, i].set_xlabel('Index')
    axes[0, i].set_ylabel('Index')
    fig.colorbar(im_real, ax=axes[0, i], fraction=0.046, pad=0.04)

    im_imag = axes[1, i].imshow(state.full().imag, cmap='viridis', vmin=0, vmax=0.3)
    axes[1, i].set_title(f"Time = {times[idx]:.2f}")
    axes[1, i].set_xlabel('Index')
    axes[1, i].set_ylabel('Index')
    fig.colorbar(im_imag, ax=axes[1, i], fraction=0.046, pad=0.04)

plt.tight_layout()
plt.show()