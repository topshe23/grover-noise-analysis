# src/experiment.py
# Runs 3 Grover noise analysis experiments

import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.circuit import (build_grover_circuit,
                          compute_theoretical_success_prob)
from src.simulator import (run_ideal, run_with_noise,
                            compute_success_probability)
from src.utils import (plot_noise_vs_success, plot_qubits_vs_success,
                        plot_iterations_vs_success, save_results)


def experiment_noise_vs_success(n_qubits=3, target='101', shots=4096):
    """
    Experiment 1: How does depolarizing noise affect success probability?
    Sweeps noise from 0.0 to 0.10.
    """
    print("\nExperiment 1: Noise vs Success Probability...")

    noise_levels = [0.0, 0.005, 0.01, 0.02, 0.03, 0.04,
                    0.05, 0.06, 0.07, 0.08, 0.09, 0.10]
    success_probs = []

    qc, n_iter = build_grover_circuit(n_qubits, target)
    theoretical = compute_theoretical_success_prob(n_qubits, n_iter)

    for noise in noise_levels:
        if noise == 0.0:
            counts = run_ideal(qc, shots=shots)
        else:
            counts = run_with_noise(qc, noise, shots=shots)
        prob = compute_success_probability(counts, target, shots)
        success_probs.append(prob)
        print(f"  noise={noise:.3f} -> success_prob={prob:.4f}")

    return noise_levels, success_probs, theoretical, n_qubits


def experiment_qubits_vs_success(noise_level=0.02, shots=4096):
    """
    Experiment 2: Qubit count vs success probability.
    Compares n=2 (4 states) and n=3 (8 states) under ideal
    and noisy conditions. Larger circuits show more noise impact.
    Note: n>=4 excluded due to MCX compilation issue in Qiskit 2.3.
    """
    print("\nExperiment 2: Qubit Count vs Success Probability...")

    qubit_counts = [2, 3]
    targets = ['11', '101']
    success_probs_ideal = []
    success_probs_noisy = []

    for n_qubits, target in zip(qubit_counts, targets):
        qc, n_iter = build_grover_circuit(n_qubits, target)

        counts_ideal = run_ideal(qc, shots=shots)
        prob_ideal = compute_success_probability(counts_ideal, target, shots)
        success_probs_ideal.append(prob_ideal)

        counts_noisy = run_with_noise(qc, noise_level, shots=shots)
        prob_noisy = compute_success_probability(counts_noisy, target, shots)
        success_probs_noisy.append(prob_noisy)

        print(f"  n={n_qubits} ({2**n_qubits} states, depth={qc.depth()}): "
              f"ideal={prob_ideal:.4f}, noisy={prob_noisy:.4f}, "
              f"degradation={prob_ideal-prob_noisy:.4f}")

    return qubit_counts, success_probs_ideal, success_probs_noisy, noise_level

def experiment_iterations_vs_success(n_qubits=3, target='101', shots=4096):
    """
    Experiment 3: Iterations vs success probability.
    Shows sinusoidal oscillation — too many iterations overshoots.
    """
    print("\nExperiment 3: Iterations vs Success Probability...")

    iterations_list = [1, 2, 3, 4, 5, 6, 7, 8]
    success_probs = []
    theoretical_probs = []

    for n_iter in iterations_list:
        qc, _ = build_grover_circuit(n_qubits, target,
                                      n_iterations=n_iter)
        counts = run_ideal(qc, shots=shots)
        prob = compute_success_probability(counts, target, shots)
        theoretical = compute_theoretical_success_prob(n_qubits, n_iter)

        success_probs.append(prob)
        theoretical_probs.append(theoretical)
        print(f"  iterations={n_iter}: "
              f"simulated={prob:.4f}, "
              f"theoretical={theoretical:.4f}")

    return iterations_list, success_probs, theoretical_probs

if __name__ == "__main__":

    # --- Experiment 1: Noise vs Success ---
    noise_levels, success_probs, theoretical, n_qubits = \
        experiment_noise_vs_success()

    plot_noise_vs_success(
        noise_levels, success_probs, theoretical, n_qubits,
        'results/plots/noise_vs_success.png'
    )
    save_results(
        {'noise_level': noise_levels, 'success_prob': success_probs},
        'results/data/noise_results.csv'
    )

    # --- Experiment 2: Qubits vs Success ---
    qubit_counts, probs_ideal, probs_noisy, noise_level = \
        experiment_qubits_vs_success()

    plot_qubits_vs_success(
        qubit_counts, probs_ideal, probs_noisy, noise_level,
        'results/plots/qubits_vs_success.png'
    )
    save_results(
        {'n_qubits': qubit_counts,
         'success_ideal': probs_ideal,
         'success_noisy': probs_noisy},
        'results/data/qubits_results.csv'
    )

    # --- Experiment 3: Iterations vs Success ---
    iterations_list, success_probs_iter, theoretical_probs = \
        experiment_iterations_vs_success()

    plot_iterations_vs_success(
        iterations_list, success_probs_iter, theoretical_probs,
        'results/plots/iterations_vs_success.png'
    )
    save_results(
        {'iterations': iterations_list,
         'success_simulated': success_probs_iter,
         'success_theoretical': theoretical_probs},
        'results/data/iterations_results.csv'
    )

    print("\nAll experiments done!")
    print("Plots saved to results/plots/")
    print("CSVs saved to results/data/")