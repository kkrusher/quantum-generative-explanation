import pennylane as qml
from pennylane import numpy as np
import pickle

from scipy.sparse import csc_matrix


def symmetry_protected_topological_phases_hamiltonian(J1, J2, N):
    hamiltonian = qml.Hamiltonian([], [])

    for i in range(N):
        hamiltonian += qml.PauliZ(i)
        hamiltonian += -J1 * qml.PauliX(i) @ qml.PauliX((i + 1) % N)
        hamiltonian += (
            -J2 * qml.PauliX((i + N - 1) % N) @ qml.PauliZ(i) @ qml.PauliX((i + 1) % N)
        )

    return hamiltonian


def generate_SPTS_data(n_qubits, n_data):
    # generate n_data J h pairs
    J1_list = np.random.uniform(-4, 4, n_data)
    J2_list = np.random.uniform(-4, 4, n_data)

    phaseTransition = dict()
    phaseTransition["states"] = []
    phaseTransition["label"] = []

    # for each pair of J h, calculate the ground state
    for i in range(n_data):
        J1 = J1_list[i]  # Interaction strength
        J2 = J2_list[i]  # Transverse field strength

        # Create the Ising Hamiltonian
        hamiltonian = symmetry_protected_topological_phases_hamiltonian(
            J1, J2, n_qubits
        )

        # Obtain the sparse matrix representation
        sparse_matrix = hamiltonian.sparse_matrix()

        # Convert the sparse matrix to a dense matrix
        dense_matrix = csc_matrix(sparse_matrix).todense()

        # print("Hamiltonian matrix:")
        # print(dense_matrix)

        # Calculate the eigenvalues
        eigenvalues, eigenvectors = np.linalg.eig(dense_matrix)

        # get the index of the mininum eigenvalue
        min_index = np.argmin(eigenvalues)

        # get the ground state
        ground_state = eigenvectors[:, min_index]
        ground_state = np.squeeze(np.array(ground_state), axis=1)

        label = 0
        if J2 > -J1 - 1 and J2 < J1 - 1:
            label = 1
        elif J2 < -J1 - 1 and J2 > J1 - 1:
            label = 2
        elif J2 > -J1 - 1 and J2 > J1 - 1 and J2 < 1:
            label = 3

        phaseTransition["states"].append([J1, J2, ground_state])
        phaseTransition["label"].append(label)
    return phaseTransition


if __name__ == "__main__":
    # Define parameters
    n_qubits = 3  # Number of spins
    n_data = 2000  # Number of samples

    phaseTransition = generate_SPTS_data(n_qubits, n_data)
    # save phaseTransition dataset
    with open(
        f"./data/SPTS_PhaseTransition/SPTS_phaseTransitionDataset_q{n_qubits}_n{n_data}.npy",
        "wb",
    ) as f:
        pickle.dump(phaseTransition, f)
