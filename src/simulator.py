# src/simulator.py
# Runs Grover's algorithm under ideal and noisy conditions

import numpy as np
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error
from qiskit import transpile


def run_ideal(circuit, shots=4096):
    """
    Runs Grover circuit on ideal simulator.
    Returns counts dict and success probability.
    """
    sim = AerSimulator(method='statevector')
    compiled = transpile(circuit, sim)
    result = sim.run(compiled, shots=shots).result()
    counts = result.get_counts()
    return counts


def run_with_noise(circuit, noise_level, shots=4096):
    """
    Runs Grover circuit with depolarizing noise.

    Args:
        circuit: Grover QuantumCircuit
        noise_level: float 0.0 to 0.1
        shots: number of measurements
    Returns:
        counts dict
    """
    noise_model = NoiseModel()
    error_1q = depolarizing_error(noise_level, 1)
    error_2q = depolarizing_error(noise_level * 2, 2)
    error_3q = depolarizing_error(noise_level * 3, 3)

    noise_model.add_all_qubit_quantum_error(
        error_1q, ['h', 'x', 'z', 'rx', 'ry', 'rz'])
    noise_model.add_all_qubit_quantum_error(
        error_2q, ['cx', 'cz', 'ccx'])
    noise_model.add_all_qubit_quantum_error(
        error_3q, ['ccx'])

    sim = AerSimulator(noise_model=noise_model)
    compiled = transpile(circuit, sim)
    result = sim.run(compiled, shots=shots).result()
    counts = result.get_counts()
    return counts


def compute_success_probability(counts, target_state, shots=4096):
    """
    Computes probability of measuring the target state.
    Success probability = target_count / total_shots
    """
    total = sum(counts.values())
    # Qiskit returns bitstrings in reverse order
    target_reversed = target_state[::-1]
    success_count = counts.get(target_reversed, 0)
    # Also check non-reversed in case
    if success_count == 0:
        success_count = counts.get(target_state, 0)
    return success_count / total


def run_baseline(circuit, target_state, shots=4096):
    """
    Runs ideal baseline and prints results.
    Returns success probability and counts.
    """
    counts = run_ideal(circuit, shots=shots)
    success_prob = compute_success_probability(counts, target_state, shots)

    # Sort by count
    sorted_counts = sorted(counts.items(), key=lambda x: x[1], reverse=True)

    print(f"  Top 5 measured states:")
    for state, count in sorted_counts[:5]:
        prob = count / shots
        marker = " <- TARGET" if (state == target_state or
                                   state == target_state[::-1]) else ""
        print(f"    |{state}⟩: {count} ({prob:.4f}){marker}")

    print(f"  Success probability: {success_prob:.4f}")
    return success_prob, counts