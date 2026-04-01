# src/circuit.py
from qiskit import QuantumCircuit
import numpy as np


def build_grover_circuit(n_qubits, target_state, n_iterations=None):
    """
    Builds Grover's circuit. Verified working for n=2 and n=3.
    Uses only H, X, CZ, CCX gates — no MCX.
    """
    assert n_qubits in [2, 3], "Only n=2 and n=3 supported reliably"

    N = 2 ** n_qubits
    if n_iterations is None:
        n_iterations = max(1, int(np.floor(np.pi / 4 * np.sqrt(N))))

    qc = QuantumCircuit(n_qubits, n_qubits)

    for i in range(n_qubits):
        qc.h(i)
    qc.barrier(label='Init')

    for iteration in range(n_iterations):
        _apply_oracle(qc, target_state, n_qubits)
        qc.barrier(label=f'Oracle {iteration+1}')
        _apply_diffusion(qc, n_qubits)
        qc.barrier(label=f'Diffusion {iteration+1}')

    qc.measure(range(n_qubits), range(n_qubits))
    return qc, n_iterations


def _mcz(qc, n):
    """Multi-controlled Z using only verified gates."""
    if n == 2:
        qc.cz(0, 1)
    elif n == 3:
        qc.h(2)
        qc.ccx(0, 1, 2)
        qc.h(2)


def _apply_oracle(qc, target_state, n_qubits):
    for i, bit in enumerate(reversed(target_state)):
        if bit == '0':
            qc.x(i)
    _mcz(qc, n_qubits)
    for i, bit in enumerate(reversed(target_state)):
        if bit == '0':
            qc.x(i)


def _apply_diffusion(qc, n_qubits):
    for i in range(n_qubits):
        qc.h(i)
    for i in range(n_qubits):
        qc.x(i)
    _mcz(qc, n_qubits)
    for i in range(n_qubits):
        qc.x(i)
    for i in range(n_qubits):
        qc.h(i)


def draw_circuit(qc, save_path='images/circuit.png'):
    fig = qc.draw(output='mpl', fold=-1)
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Circuit saved to {save_path}")


def compute_theoretical_success_prob(n_qubits, n_iterations):
    N = 2 ** n_qubits
    theta = np.arcsin(1 / np.sqrt(N))
    prob = np.sin((2 * n_iterations + 1) * theta) ** 2
    return prob


if __name__ == "__main__":
    for n, target in [(2, '11'), (3, '101')]:
        qc, n_iter = build_grover_circuit(n, target)
        theoretical = compute_theoretical_success_prob(n, n_iter)
        print(f"n={n}, target=|{target}>, iter={n_iter}, "
              f"theoretical={theoretical:.4f}, depth={qc.depth()}")
    draw_circuit(build_grover_circuit(3, '101')[0])