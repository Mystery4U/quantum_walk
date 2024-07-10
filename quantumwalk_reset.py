# This script will be used to simulate (stochastic) resetting on quantum walks.
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import csv
from qutip import basis, mesolve, Qobj, ket2dm

epsilon = 0.1
rate = 0.3
seed = 42
G = nx.erdos_renyi_graph(n=10, p=0.5, seed=seed)
# G = nx.cycle_graph(50)

basis_states = [basis(len(G.nodes), i) for i in range(len(G.nodes))]

initial_density_matrix = ket2dm(sum(basis_states).unit())
target_state_index = 4
target_state = basis_states[target_state_index]

diagonals = []


def jumps(graph, eps):
    jump_operators = []
    A = nx.to_numpy_array(graph)
    degree = np.sum(A, axis=1)
    for edge in G.edges():
        n, m = edge
        gamma_nm = -1 if n == m else A[n, m] / degree[m]
        gamma_mn = -1 if n == m else A[m, n] / degree[n]
        op1 = np.sqrt(eps * gamma_nm) * basis(len(G.nodes), n) * basis(len(G.nodes), m).dag()
        op2 = np.sqrt(eps * gamma_mn) * basis(len(G.nodes), m) * basis(len(G.nodes), n).dag()
        jump_operators.append(Qobj(op1))
        jump_operators.append(Qobj(op2))
    return jump_operators


def quantumwalk(graph, initial_density, target_state, jump_operators, r):
    H_q = Qobj(nx.normalized_laplacian_matrix(G))
    H_w = -1 * Qobj(target_state * target_state.dag())
    H = (1 - epsilon) * (H_q + H_w)
    reset_time = 1/r
    times = np.linspace(0.0, reset_time, 1000)
    result = mesolve(H, initial_density, times, jump_operators, [])

    diagonal_elements = [[result.states[k].diag().real[i] for k in range(len(times))] for i in range(len(G.nodes))]
    diagonals.append(diagonal_elements)

    return result.states[-1]


def initial_condition(top_basis):
    initial_state = 0
    for i in top_basis:
        initial_state += basis_states[i]

    return ket2dm(initial_state.unit())


iterations = 5
density = initial_density_matrix
fig, ax = plt.subplots()
for i in range(iterations):
    print(i)
    rest_state = quantumwalk(G, density, target_state, jumps(G, epsilon), rate)
    # rest_basis = np.argsort(rest_state.diag())[:99 - 1*i]
    rest_basis = np.argsort(rest_state.diag())[-5:]
    print(np.sort(rest_basis))
    density = initial_condition(rest_basis)

reshaped_diagonals = np.einsum('ijk->jik', diagonals).reshape(len(G.nodes), iterations * 1000)
for i, sublist in enumerate(reshaped_diagonals):
    plt.plot(np.linspace(0, iterations * 1 / rate, iterations * 1000), sublist)
    np.savetxt(f"quantum_walk_reset_node_{i}_reset_5_03.txt", np.column_stack((np.linspace(0, iterations * 1 / rate, iterations * 1000), sublist)), header="Time\tProbability", delimiter="\t")
plt.show()
