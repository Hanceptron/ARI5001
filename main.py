"""
Main Runner Script for ARI 5001 HMM Project
Executes all experiments and generates the final PDF report.

Usage: python main.py

Author: Murat Emirhan Aykut
Date: December 2025
"""

import os
import sys
import numpy as np

# Ensure matplotlib uses non-interactive backend
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Import project modules
from hmm import HiddenMarkovModel, create_weather_hmm
from experiments import (
    experiment_filtering_vs_smoothing,
    experiment_emission_noise,
    experiment_transition_dynamics,
    experiment_sequence_length,
    visualize_belief_evolution,
    plot_experiment_results,
    generate_results_table
)
from generate_report import generate_report


def main():
    """Run all experiments and generate the final report."""
    
    print("=" * 70)
    print("ARI 5001 - Hidden Markov Models Project")
    print("Running experiments and generating report...")
    print("=" * 70)
    
    # Create output directory
    output_dir = "."
    
    # ==================== RUN EXPERIMENTS ====================
    print("\n[1/5] Running Experiment 1: Filtering vs Smoothing...")
    exp1_results = experiment_filtering_vs_smoothing(n_trials=100, seq_length=50, base_seed=42)
    
    print("\n[2/5] Running Experiment 2: Emission Noise Sensitivity...")
    exp2_results = experiment_emission_noise(
        noise_levels=[0.0, 0.1, 0.2, 0.3, 0.4],
        n_trials=50, 
        seq_length=50,
        base_seed=42
    )
    
    print("\n[3/5] Running Experiment 3: Transition Dynamics...")
    exp3_results = experiment_transition_dynamics(
        sticky_values=[0.5, 0.6, 0.7, 0.8, 0.9, 0.95],
        n_trials=50,
        seq_length=50,
        base_seed=42
    )
    
    print("\n[4/5] Running Experiment 4: Sequence Length Effect...")
    exp4_results = experiment_sequence_length(
        lengths=[10, 25, 50, 100, 200],
        n_trials=50,
        base_seed=42
    )
    
    # ==================== GENERATE VISUALIZATIONS ====================
    print("\n[5/5] Generating visualizations...")
    
    # Belief evolution plot
    hmm = create_weather_hmm()
    true_states, observations = hmm.generate_sequence(30, seed=123)
    fig1 = visualize_belief_evolution(
        hmm, observations, true_states,
        save_path=os.path.join(output_dir, "belief_evolution.png")
    )
    plt.close(fig1)
    
    # Summary results plot
    fig2 = plot_experiment_results(
        exp1_results, exp2_results, exp3_results, exp4_results,
        save_path=os.path.join(output_dir, "experiment_results.png")
    )
    plt.close(fig2)
    
    # Generate and save results table
    table = generate_results_table(exp1_results, exp2_results, exp3_results)
    with open(os.path.join(output_dir, "results_table.txt"), 'w') as f:
        f.write(table)
    print(f"\nResults table saved to {output_dir}/results_table.txt")
    
    # ==================== GENERATE PDF REPORT ====================
    print("\n" + "=" * 70)
    print("Generating PDF Report...")
    print("=" * 70)
    
    results_data = {
        'exp1': exp1_results,
        'exp2': exp2_results,
        'exp3': exp3_results,
        'exp4': exp4_results
    }
    
    report_path = os.path.join(output_dir, "HMM_Project_Report.pdf")
    generate_report(results_data, report_path)
    
    # ==================== SUMMARY ====================
    print("\n" + "=" * 70)
    print("PROJECT COMPLETE")
    print("=" * 70)
    print(f"\nGenerated files:")
    print(f"  - HMM_Project_Report.pdf (main report)")
    print(f"  - belief_evolution.png (Figure 1)")
    print(f"  - experiment_results.png (Figure 2)")
    print(f"  - results_table.txt (numerical results)")
    print(f"\nSource code files:")
    print(f"  - hmm.py (core HMM implementation)")
    print(f"  - experiments.py (experimental framework)")
    print(f"  - generate_report.py (PDF generation)")
    print(f"  - main.py (this runner script)")
    
    # Print key results
    print("\n" + "=" * 70)
    print("KEY RESULTS SUMMARY")
    print("=" * 70)
    print(f"\nExperiment 1 - Method Comparison (100 trials, T=50):")
    print(f"  Filtering accuracy:  {np.mean(exp1_results['filtering']):.4f} ± {np.std(exp1_results['filtering']):.4f}")
    print(f"  Smoothing accuracy:  {np.mean(exp1_results['smoothing']):.4f} ± {np.std(exp1_results['smoothing']):.4f}")
    print(f"  Viterbi accuracy:    {np.mean(exp1_results['viterbi']):.4f} ± {np.std(exp1_results['viterbi']):.4f}")
    
    improvement = np.array(exp1_results['smoothing']) - np.array(exp1_results['filtering'])
    print(f"\n  Smoothing improvement: {np.mean(improvement):.4f} ± {np.std(improvement):.4f}")
    print(f"  Smoothing better in {np.sum(improvement > 0)}/100 trials")
    
    return results_data


if __name__ == "__main__":
    results = main()
