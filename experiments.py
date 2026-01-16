"""
HMM Experiments Module
ARI 5001 - Introduction to Artificial Intelligence

This module runs all experiments for evaluating the HMM algorithms:
1. Filtering vs Smoothing accuracy comparison
2. Sensitivity to observation noise
3. Sensitivity to transition dynamics
4. Visualization of belief evolution

Author: Murat Emirhan Aykut
Date: December 2025
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
from hmm import HiddenMarkovModel, create_weather_hmm
import warnings
warnings.filterwarnings('ignore')


def compute_accuracy(predictions: np.ndarray, true_states: np.ndarray) -> float:
    """
    Compute accuracy of state predictions.
    
    Args:
        predictions: Predicted state indices (T,) or probability matrix (T, N)
        true_states: True state indices
        
    Returns:
        Accuracy as fraction of correct predictions
    """
    if predictions.ndim == 2:
        # Convert probabilities to predictions
        predictions = np.argmax(predictions, axis=1)
    
    return np.mean(predictions == true_states)


def run_single_experiment(hmm: HiddenMarkovModel, 
                          seq_length: int,
                          seed: int) -> Dict[str, float]:
    """
    Run a single experiment comparing filtering, smoothing, and Viterbi.
    
    Args:
        hmm: HMM model
        seq_length: Length of sequence
        seed: Random seed
        
    Returns:
        Dictionary with accuracy scores for each method
    """
    # Generate sequence
    true_states, observations = hmm.generate_sequence(seq_length, seed=seed)
    
    # Filtering
    filtered = hmm.filter(observations)
    filter_predictions = np.argmax(filtered, axis=1)
    filter_acc = compute_accuracy(filter_predictions, true_states)
    
    # Smoothing
    smoothed = hmm.smooth(observations)
    smooth_predictions = np.argmax(smoothed, axis=1)
    smooth_acc = compute_accuracy(smooth_predictions, true_states)
    
    # Viterbi
    viterbi_path, _ = hmm.viterbi(observations)
    viterbi_acc = compute_accuracy(viterbi_path, true_states)
    
    return {
        'filtering': filter_acc,
        'smoothing': smooth_acc,
        'viterbi': viterbi_acc
    }


def experiment_filtering_vs_smoothing(n_trials: int = 100,
                                      seq_length: int = 50,
                                      base_seed: int = 42) -> Dict[str, List[float]]:
    """
    Experiment 1: Compare filtering vs smoothing accuracy.
    
    This experiment validates the theoretical expectation that smoothing
    should outperform filtering because it uses future observations.
    
    Args:
        n_trials: Number of random sequences to generate
        seq_length: Length of each sequence
        base_seed: Base random seed
        
    Returns:
        Dictionary with lists of accuracy scores
    """
    print("Experiment 1: Filtering vs Smoothing Comparison")
    print("=" * 60)
    
    hmm = create_weather_hmm()
    
    results = {
        'filtering': [],
        'smoothing': [],
        'viterbi': []
    }
    
    for trial in range(n_trials):
        seed = base_seed + trial
        trial_results = run_single_experiment(hmm, seq_length, seed)
        
        for method in results:
            results[method].append(trial_results[method])
    
    # Print summary statistics
    print(f"\nResults over {n_trials} trials (sequence length = {seq_length}):")
    print("-" * 60)
    for method in results:
        mean = np.mean(results[method])
        std = np.std(results[method])
        print(f"{method.capitalize():12s}: Mean = {mean:.4f}, Std = {std:.4f}")
    
    # Statistical comparison
    filter_arr = np.array(results['filtering'])
    smooth_arr = np.array(results['smoothing'])
    improvement = smooth_arr - filter_arr
    print(f"\nSmoothing improvement over filtering: {np.mean(improvement):.4f} ± {np.std(improvement):.4f}")
    print(f"Smoothing better in {np.sum(improvement > 0)}/{n_trials} trials")
    
    return results


def experiment_emission_noise(noise_levels: List[float] = None,
                              n_trials: int = 50,
                              seq_length: int = 50,
                              base_seed: int = 42) -> Dict[str, Dict[str, List[float]]]:
    """
    Experiment 2: Sensitivity to observation noise.
    
    As emission noise increases, observations become less informative
    and inference should degrade.
    
    Args:
        noise_levels: List of noise values to test
        n_trials: Number of trials per noise level
        seq_length: Sequence length
        base_seed: Base random seed
        
    Returns:
        Nested dictionary: {noise_level: {method: [accuracies]}}
    """
    print("\nExperiment 2: Emission Noise Sensitivity")
    print("=" * 60)
    
    if noise_levels is None:
        noise_levels = [0.0, 0.1, 0.2, 0.3, 0.4]
    
    all_results = {}
    
    for noise in noise_levels:
        hmm = create_weather_hmm(emission_noise=noise)
        
        results = {'filtering': [], 'smoothing': [], 'viterbi': []}
        
        for trial in range(n_trials):
            seed = base_seed + trial
            trial_results = run_single_experiment(hmm, seq_length, seed)
            
            for method in results:
                results[method].append(trial_results[method])
        
        all_results[noise] = results
        
        print(f"\nEmission noise = {noise:.1f}:")
        for method in results:
            mean = np.mean(results[method])
            print(f"  {method.capitalize():12s}: {mean:.4f}")
    
    return all_results


def experiment_transition_dynamics(sticky_values: List[float] = None,
                                   n_trials: int = 50,
                                   seq_length: int = 50,
                                   base_seed: int = 42) -> Dict[str, Dict[str, List[float]]]:
    """
    Experiment 3: Sensitivity to transition matrix characteristics.
    
    Compare performance with "sticky" states (high self-transition)
    vs rapidly switching states.
    
    Args:
        sticky_values: Self-transition probabilities to test
        n_trials: Number of trials
        seq_length: Sequence length
        base_seed: Base random seed
        
    Returns:
        Nested dictionary with results
    """
    print("\nExperiment 3: Transition Dynamics Sensitivity")
    print("=" * 60)
    
    if sticky_values is None:
        sticky_values = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
    
    all_results = {}
    
    for sticky in sticky_values:
        # Create HMM with custom transition matrix
        A = np.array([
            [sticky, 1-sticky],
            [1-sticky, sticky]
        ])
        B = np.array([
            [0.9, 0.1],
            [0.2, 0.8]
        ])
        pi = np.array([0.5, 0.5])
        
        hmm = HiddenMarkovModel(A, B, pi, 
                               state_names=["Sunny", "Rainy"],
                               observation_names=["No Umbrella", "Umbrella"])
        
        results = {'filtering': [], 'smoothing': [], 'viterbi': []}
        
        for trial in range(n_trials):
            seed = base_seed + trial
            trial_results = run_single_experiment(hmm, seq_length, seed)
            
            for method in results:
                results[method].append(trial_results[method])
        
        all_results[sticky] = results
        
        print(f"\nSelf-transition probability = {sticky:.2f}:")
        for method in results:
            mean = np.mean(results[method])
            print(f"  {method.capitalize():12s}: {mean:.4f}")
    
    return all_results


def experiment_sequence_length(lengths: List[int] = None,
                               n_trials: int = 50,
                               base_seed: int = 42) -> Dict[int, Dict[str, List[float]]]:
    """
    Experiment 4: Effect of sequence length on accuracy.
    
    Args:
        lengths: List of sequence lengths to test
        n_trials: Number of trials
        base_seed: Base random seed
        
    Returns:
        Nested dictionary with results
    """
    print("\nExperiment 4: Sequence Length Effect")
    print("=" * 60)
    
    if lengths is None:
        lengths = [10, 25, 50, 100, 200]
    
    hmm = create_weather_hmm()
    all_results = {}
    
    for length in lengths:
        results = {'filtering': [], 'smoothing': [], 'viterbi': []}
        
        for trial in range(n_trials):
            seed = base_seed + trial
            trial_results = run_single_experiment(hmm, length, seed)
            
            for method in results:
                results[method].append(trial_results[method])
        
        all_results[length] = results
        
        print(f"\nSequence length = {length}:")
        for method in results:
            mean = np.mean(results[method])
            print(f"  {method.capitalize():12s}: {mean:.4f}")
    
    return all_results


def visualize_belief_evolution(hmm: HiddenMarkovModel,
                               observations: np.ndarray,
                               true_states: np.ndarray,
                               save_path: str = None) -> plt.Figure:
    """
    Visualize how beliefs evolve over time for filtering and smoothing.
    
    Args:
        hmm: HMM model
        observations: Observation sequence
        true_states: True hidden states
        save_path: Path to save figure
        
    Returns:
        Matplotlib figure
    """
    T = len(observations)
    
    # Compute beliefs
    filtered = hmm.filter(observations)
    smoothed = hmm.smooth(observations)
    viterbi_path, _ = hmm.viterbi(observations)
    
    # Create figure
    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
    
    time = np.arange(T)
    
    # Plot 1: True states and observations
    ax1 = axes[0]
    ax1.step(time, true_states, 'b-', where='mid', label='True State', linewidth=2)
    ax1.scatter(time, observations * 0.5 + 0.25, c='orange', marker='o', 
                s=50, label='Observation', zorder=5, alpha=0.7)
    ax1.set_ylabel('State')
    ax1.set_yticks([0, 1])
    ax1.set_yticklabels(['Sunny', 'Rainy'])
    ax1.legend(loc='upper right')
    ax1.set_title('True Hidden States and Observations')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Filtering probabilities
    ax2 = axes[1]
    ax2.fill_between(time, 0, filtered[:, 0], alpha=0.3, color='gold', label='P(Sunny)')
    ax2.fill_between(time, filtered[:, 0], 1, alpha=0.3, color='steelblue', label='P(Rainy)')
    ax2.plot(time, filtered[:, 0], 'k-', linewidth=1.5)
    filter_pred = np.argmax(filtered, axis=1)
    ax2.scatter(time[filter_pred != true_states], 
                filtered[filter_pred != true_states, 0],
                c='red', marker='x', s=100, zorder=5, label='Errors')
    ax2.set_ylabel('P(State)')
    ax2.set_ylim(0, 1)
    ax2.legend(loc='upper right')
    ax2.set_title(f'Filtering: P(Xₜ | e₁:ₜ) - Accuracy: {compute_accuracy(filter_pred, true_states):.2%}')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Smoothing probabilities
    ax3 = axes[2]
    ax3.fill_between(time, 0, smoothed[:, 0], alpha=0.3, color='gold', label='P(Sunny)')
    ax3.fill_between(time, smoothed[:, 0], 1, alpha=0.3, color='steelblue', label='P(Rainy)')
    ax3.plot(time, smoothed[:, 0], 'k-', linewidth=1.5)
    smooth_pred = np.argmax(smoothed, axis=1)
    ax3.scatter(time[smooth_pred != true_states], 
                smoothed[smooth_pred != true_states, 0],
                c='red', marker='x', s=100, zorder=5, label='Errors')
    ax3.set_ylabel('P(State)')
    ax3.set_ylim(0, 1)
    ax3.legend(loc='upper right')
    ax3.set_title(f'Smoothing: P(Xₜ | e₁:T) - Accuracy: {compute_accuracy(smooth_pred, true_states):.2%}')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Viterbi comparison
    ax4 = axes[3]
    ax4.step(time, true_states, 'b-', where='mid', label='True', linewidth=2, alpha=0.7)
    ax4.step(time, viterbi_path, 'r--', where='mid', label='Viterbi', linewidth=2)
    ax4.scatter(time[viterbi_path != true_states], 
                viterbi_path[viterbi_path != true_states],
                c='red', marker='x', s=150, zorder=5, label='Errors')
    ax4.set_ylabel('State')
    ax4.set_xlabel('Time Step')
    ax4.set_yticks([0, 1])
    ax4.set_yticklabels(['Sunny', 'Rainy'])
    ax4.legend(loc='upper right')
    ax4.set_title(f'Viterbi Decoding - Accuracy: {compute_accuracy(viterbi_path, true_states):.2%}')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved belief evolution plot to {save_path}")
    
    return fig


def plot_experiment_results(exp1_results: Dict,
                            exp2_results: Dict,
                            exp3_results: Dict,
                            exp4_results: Dict,
                            save_path: str = None) -> plt.Figure:
    """
    Create summary plots for all experiments.
    
    Args:
        exp1_results: Results from experiment 1
        exp2_results: Results from experiment 2
        exp3_results: Results from experiment 3
        exp4_results: Results from experiment 4
        save_path: Path to save figure
        
    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    colors = {'filtering': '#2ecc71', 'smoothing': '#3498db', 'viterbi': '#e74c3c'}
    
    # Plot 1: Filtering vs Smoothing box plot
    ax1 = axes[0, 0]
    data = [exp1_results['filtering'], exp1_results['smoothing'], exp1_results['viterbi']]
    bp = ax1.boxplot(data, labels=['Filtering', 'Smoothing', 'Viterbi'], patch_artist=True)
    for patch, color in zip(bp['boxes'], colors.values()):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax1.set_ylabel('Accuracy')
    ax1.set_title('Experiment 1: Method Comparison')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0.5, 1.0)
    
    # Plot 2: Emission noise sensitivity
    ax2 = axes[0, 1]
    noise_levels = sorted(exp2_results.keys())
    for method, color in colors.items():
        means = [np.mean(exp2_results[n][method]) for n in noise_levels]
        stds = [np.std(exp2_results[n][method]) for n in noise_levels]
        ax2.errorbar(noise_levels, means, yerr=stds, marker='o', 
                     label=method.capitalize(), color=color, capsize=5, linewidth=2)
    ax2.set_xlabel('Emission Noise Level')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Experiment 2: Emission Noise Sensitivity')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0.4, 1.0)
    
    # Plot 3: Transition dynamics
    ax3 = axes[1, 0]
    sticky_values = sorted(exp3_results.keys())
    for method, color in colors.items():
        means = [np.mean(exp3_results[s][method]) for s in sticky_values]
        stds = [np.std(exp3_results[s][method]) for s in sticky_values]
        ax3.errorbar(sticky_values, means, yerr=stds, marker='s',
                     label=method.capitalize(), color=color, capsize=5, linewidth=2)
    ax3.set_xlabel('Self-Transition Probability')
    ax3.set_ylabel('Accuracy')
    ax3.set_title('Experiment 3: Transition Dynamics Effect')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(0.5, 1.0)
    
    # Plot 4: Sequence length effect
    ax4 = axes[1, 1]
    lengths = sorted(exp4_results.keys())
    for method, color in colors.items():
        means = [np.mean(exp4_results[l][method]) for l in lengths]
        stds = [np.std(exp4_results[l][method]) for l in lengths]
        ax4.errorbar(lengths, means, yerr=stds, marker='^',
                     label=method.capitalize(), color=color, capsize=5, linewidth=2)
    ax4.set_xlabel('Sequence Length')
    ax4.set_ylabel('Accuracy')
    ax4.set_title('Experiment 4: Sequence Length Effect')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_xscale('log')
    ax4.set_ylim(0.5, 1.0)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved experiment results to {save_path}")
    
    return fig


def generate_results_table(exp1_results: Dict,
                           exp2_results: Dict,
                           exp3_results: Dict) -> str:
    """
    Generate a formatted results table for the report.
    
    Returns:
        String with formatted table
    """
    lines = []
    lines.append("=" * 80)
    lines.append("EXPERIMENTAL RESULTS SUMMARY")
    lines.append("=" * 80)
    
    # Experiment 1
    lines.append("\n--- Experiment 1: Method Comparison (100 trials, seq_length=50) ---")
    lines.append(f"{'Method':<15} {'Mean':>10} {'Std':>10} {'Min':>10} {'Max':>10}")
    lines.append("-" * 55)
    for method in ['filtering', 'smoothing', 'viterbi']:
        data = exp1_results[method]
        lines.append(f"{method.capitalize():<15} {np.mean(data):>10.4f} {np.std(data):>10.4f} "
                    f"{np.min(data):>10.4f} {np.max(data):>10.4f}")
    
    # Experiment 2
    lines.append("\n--- Experiment 2: Emission Noise Sensitivity ---")
    lines.append(f"{'Noise':<10} {'Filtering':>12} {'Smoothing':>12} {'Viterbi':>12}")
    lines.append("-" * 50)
    for noise in sorted(exp2_results.keys()):
        f_acc = np.mean(exp2_results[noise]['filtering'])
        s_acc = np.mean(exp2_results[noise]['smoothing'])
        v_acc = np.mean(exp2_results[noise]['viterbi'])
        lines.append(f"{noise:<10.2f} {f_acc:>12.4f} {s_acc:>12.4f} {v_acc:>12.4f}")
    
    # Experiment 3
    lines.append("\n--- Experiment 3: Transition Dynamics ---")
    lines.append(f"{'Sticky':<10} {'Filtering':>12} {'Smoothing':>12} {'Viterbi':>12}")
    lines.append("-" * 50)
    for sticky in sorted(exp3_results.keys()):
        f_acc = np.mean(exp3_results[sticky]['filtering'])
        s_acc = np.mean(exp3_results[sticky]['smoothing'])
        v_acc = np.mean(exp3_results[sticky]['viterbi'])
        lines.append(f"{sticky:<10.2f} {f_acc:>12.4f} {s_acc:>12.4f} {v_acc:>12.4f}")
    
    return "\n".join(lines)


def run_all_experiments(save_dir: str = ".") -> Dict:
    """
    Run all experiments and generate visualizations.
    
    Args:
        save_dir: Directory to save figures
        
    Returns:
        Dictionary with all results
    """
    print("\n" + "=" * 70)
    print("RUNNING ALL HMM EXPERIMENTS")
    print("=" * 70)
    
    # Run experiments
    exp1_results = experiment_filtering_vs_smoothing(n_trials=100, seq_length=50)
    exp2_results = experiment_emission_noise(n_trials=50, seq_length=50)
    exp3_results = experiment_transition_dynamics(n_trials=50, seq_length=50)
    exp4_results = experiment_sequence_length(n_trials=50)
    
    # Generate belief evolution visualization
    hmm = create_weather_hmm()
    true_states, observations = hmm.generate_sequence(30, seed=123)
    visualize_belief_evolution(hmm, observations, true_states, 
                               save_path=f"{save_dir}/belief_evolution.png")
    
    # Generate summary plot
    plot_experiment_results(exp1_results, exp2_results, exp3_results, exp4_results,
                           save_path=f"{save_dir}/experiment_results.png")
    
    # Print summary table
    table = generate_results_table(exp1_results, exp2_results, exp3_results)
    print("\n" + table)
    
    # Save table to file
    with open(f"{save_dir}/results_table.txt", 'w') as f:
        f.write(table)
    
    return {
        'exp1': exp1_results,
        'exp2': exp2_results,
        'exp3': exp3_results,
        'exp4': exp4_results
    }


if __name__ == "__main__":
    results = run_all_experiments(save_dir=".")
    plt.show()
