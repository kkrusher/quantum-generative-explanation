import pennylane as qml
from pennylane import numpy as np
import pickle

from scipy.sparse import csc_matrix


def transverse_field_ising_hamiltonian(J, h, N):
    hamiltonian = qml.Hamiltonian([], [])

    for i in range(N):
        hamiltonian += -J * qml.PauliZ(i) @ qml.PauliZ((i + 1) % N)
        hamiltonian += -h * qml.PauliX(i)

    return hamiltonian


def generate_TFIM_data(n_qubits, n_data):
    # generate n_data J h pairs
    J_list = np.random.uniform(-1, 1, n_data)
    h_list = np.random.uniform(-1, 1, n_data)

    phaseTransition = dict()
    phaseTransition["states"] = []
    phaseTransition["label"] = []

    # for each pair of J h, calculate the ground state
    for i in range(n_data):
        J = J_list[i]  # Interaction strength
        h = h_list[i]  # Transverse field strength

        # Create the Ising Hamiltonian
        hamiltonian = transverse_field_ising_hamiltonian(J, h, n_qubits)

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
        label = 1 if J > h else 0

        phaseTransition["states"].append([J, h, ground_state])
        phaseTransition["label"].append(label)
    return phaseTransition


if __name__ == "__main__":
    # Define parameters
    n_qubits = 3  # Number of spins
    n_data = 2000  # Number of samples

    phaseTransition = generate_TFIM_data(n_qubits, n_data)
    # save phaseTransition dataset
    with open(
        f"./data/TFIM_PhaseTransition/TFIM_phaseTransitionDataset_q{n_qubits}_n{n_data}.npy",
        "wb",
    ) as f:
        pickle.dump(phaseTransition, f)
