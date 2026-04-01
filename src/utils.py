# src/utils.py
# Plotting and saving helpers for Grover noise analysis

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import os


def plot_noise_vs_success(noise_levels, success_probs, theoretical_prob,
                          n_qubits, save_path):
    """Plots noise level vs success probability."""
    fig, ax = plt.subplots(figsize=(9, 5))

    random_guess = 1 / 2**n_qubits

    ax.plot(noise_levels, success_probs, marker='o', color='steelblue',
            linewidth=2.5, markersize=8, label="Grover's Success Probability")
    ax.axhline(y=theoretical_prob, color='green', linestyle=':',
               linewidth=1.8, label=f'Theoretical (no noise): {theoretical_prob:.4f}')
    ax.axhline(y=random_guess, color='red', linestyle='--',
               linewidth=1.5, label=f'Random guessing: {random_guess:.4f}')

    ax.fill_between(noise_levels, success_probs, random_guess,
                    where=[s > random_guess for s in success_probs],
                    alpha=0.1, color='green', label='Quantum advantage zone')

    # Find and annotate crossover point
    crossover = None
    for i in range(len(success_probs) - 1):
        if success_probs[i] > random_guess and success_probs[i+1] <= random_guess:
            crossover = noise_levels[i]
    if crossover:
        ax.axvline(x=crossover, color='purple', linestyle=':',
                   alpha=0.7, linewidth=1.5)
        ax.annotate(f'Quantum advantage\nlost at p~{crossover}',
                    xy=(crossover, random_guess),
                    xytext=(crossover + 0.01, random_guess + 0.15),
                    fontsize=9, color='purple',
                    arrowprops=dict(arrowstyle='->', color='purple', lw=1.2))

    # Annotate starting point
    ax.annotate(f'Ideal: {success_probs[0]:.4f}',
                xy=(0, success_probs[0]),
                xytext=(0.01, success_probs[0] - 0.08),
                fontsize=9, color='steelblue',
                arrowprops=dict(arrowstyle='->', color='steelblue', lw=1.2))

    ax.set_xlabel('Depolarizing Noise Level', fontsize=13)
    ax.set_ylabel('Success Probability', fontsize=13)
    ax.set_title(f"Grover's Algorithm — Noise vs Success Probability ({n_qubits} qubits)",
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Plot saved to {save_path}")

def plot_qubits_vs_success(qubit_counts, success_probs_ideal,
                           success_probs_noisy, noise_level, save_path):
    """Plots qubit count vs success probability (ideal vs noisy)."""
    fig, ax = plt.subplots(figsize=(9, 5))

    ax.plot(qubit_counts, success_probs_ideal, marker='o', color='green',
            linewidth=2.5, markersize=8, label='Ideal (no noise)')
    ax.plot(qubit_counts, success_probs_noisy, marker='s', color='steelblue',
            linewidth=2.5, markersize=8, label=f'Noisy (p={noise_level})')

    ax.set_xlabel('Number of Qubits', fontsize=13)
    ax.set_ylabel('Success Probability', fontsize=13)
    ax.set_title("Grover's Algorithm — Qubit Count vs Success Probability",
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)
    ax.set_xticks(qubit_counts)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Plot saved to {save_path}")


def plot_iterations_vs_success(iterations_list, success_probs,
                               theoretical_probs, save_path):
    """Plots number of Grover iterations vs success probability."""
    fig, ax = plt.subplots(figsize=(9, 5))

    ax.plot(iterations_list, success_probs, marker='o', color='steelblue',
            linewidth=2.5, markersize=8, label='Simulated')
    ax.plot(iterations_list, theoretical_probs, marker='^', color='darkorange',
            linewidth=2, markersize=8, linestyle='--', label='Theoretical')

    ax.set_xlabel('Number of Grover Iterations', fontsize=13)
    ax.set_ylabel('Success Probability', fontsize=13)
    ax.set_title("Grover's Algorithm — Iterations vs Success Probability",
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)
    ax.set_xticks(iterations_list)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Plot saved to {save_path}")


def save_results(data_dict, filepath):
    """Saves results to CSV."""
    df = pd.DataFrame(data_dict)
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    df.to_csv(filepath, index=False)
    print(f"Results saved to {filepath}")