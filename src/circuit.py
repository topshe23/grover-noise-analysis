# src/circuit.py
# Builds Grover's algorithm circuit

from qiskit import QuantumCircuit
import numpy as np


def build_grover_circuit(n_qubits, target_state, n_iterations=None):
    """
    Builds Grover's algorithm circuit for a given target state.

    Structure:
        |0⟩^n → H^n → [Oracle + Diffusion] × k → Measure

    Args:
        n_qubits: number of qubits (search space = 2^n_qubits)
        target_state: string like '101' — the state to find
        n_iterations: number of Grover iterations
                      (default: optimal = floor(pi/4 * sqrt(2^n)))
    Returns:
        qc: QuantumCircuit
        n_iterations: number of iterations used
    """
    N = 2 ** n_qubits

    if n_iterations is None:
        n_iterations = max(1, int(np.floor(np.pi / 4 * np.sqrt(N))))

    qc = QuantumCircuit(n_qubits, n_qubits)

    # Step 1: Initialize superposition
    for i in range(n_qubits):
        qc.h(i)
    qc.barrier(label='Init')

    # Step 2: Grover iterations
    for iteration in range(n_iterations):
        # Oracle
        _apply_oracle(qc, target_state)
        qc.barrier(label=f'Oracle {iteration+1}')

        # Diffusion operator
        _apply_diffusion(qc, n_qubits)
        qc.barrier(label=f'Diffusion {iteration+1}')

    # Step 3: Measure
    qc.measure(range(n_qubits), range(n_qubits))

    return qc, n_iterations


def _apply_oracle(qc, target_state):
    """
    Phase oracle — flips the phase of the target state.
    Applies X gates to qubits where target bit is 0,
    then multi-controlled Z, then X gates again.
    """
    n = len(target_state)

    # Flip qubits where target is '0'
    for i, bit in enumerate(reversed(target_state)):
        if bit == '0':
            qc.x(i)

    # Multi-controlled Z gate
    if n == 1:
        qc.z(0)
    elif n == 2:
        qc.cz(0, 1)
    else:
        qc.h(n-1)
        qc.mcx(list(range(n-1)), n-1)
        qc.h(n-1)

    # Unflip
    for i, bit in enumerate(reversed(target_state)):
        if bit == '0':
            qc.x(i)


def _apply_diffusion(qc, n_qubits):
    """
    Grover diffusion operator (inversion about average).
    H^n → Phase flip of |0⟩ → H^n
    """
    # H on all qubits
    for i in range(n_qubits):
        qc.h(i)

    # Phase flip |0⟩ state
    for i in range(n_qubits):
        qc.x(i)

    if n_qubits == 1:
        qc.z(0)
    elif n_qubits == 2:
        qc.cz(0, 1)
    else:
        qc.h(n_qubits-1)
        qc.mcx(list(range(n_qubits-1)), n_qubits-1)
        qc.h(n_qubits-1)

    for i in range(n_qubits):
        qc.x(i)

    # H on all qubits
    for i in range(n_qubits):
        qc.h(i)


def draw_circuit(qc, save_path='images/circuit.png'):
    """Saves circuit diagram."""
    fig = qc.draw(output='mpl', fold=-1)
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Circuit saved to {save_path}")


def compute_theoretical_success_prob(n_qubits, n_iterations):
    """
    Theoretical success probability after k Grover iterations.
    P(success) = sin²((2k+1) * arcsin(1/sqrt(N)))
    """
    N = 2 ** n_qubits
    theta = np.arcsin(1 / np.sqrt(N))
    prob = np.sin((2 * n_iterations + 1) * theta) ** 2
    return prob


if __name__ == "__main__":
    # Test with 3-qubit circuit, target = '101'
    n_qubits = 3
    target = '101'

    qc, n_iter = build_grover_circuit(n_qubits, target)

    print(f"Grover Circuit:")
    print(f"  Qubits: {n_qubits}")
    print(f"  Search space: {2**n_qubits} states")
    print(f"  Target state: |{target}⟩")
    print(f"  Grover iterations: {n_iter}")
    print(f"  Circuit depth: {qc.depth()}")
    print(f"  Gate count: {qc.count_ops()}")

    theoretical_prob = compute_theoretical_success_prob(n_qubits, n_iter)
    print(f"  Theoretical success probability: {theoretical_prob:.4f}")

    draw_circuit(qc)
    print(qc.draw(output='text'))