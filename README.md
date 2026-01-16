# ARI 5001 - Hidden Markov Models Project

## Project Overview

This project implements and evaluates Hidden Markov Model (HMM) inference algorithms for the ARI 5001 Introduction to Artificial Intelligence course. The implementation includes:

- **Forward Algorithm** (Filtering): Computing P(X_t | e_{1:t})
- **Forward-Backward Algorithm** (Smoothing): Computing P(X_t | e_{1:T})
- **Viterbi Algorithm** (Decoding): Finding the most likely state sequence

## Problem Domain

Weather prediction with umbrella observations:
- **Hidden States**: {Sunny, Rainy}
- **Observations**: {No Umbrella, Umbrella}

## Repository Structure

```
├── hmm.py              # Core HMM implementation
├── experiments.py      # Experimental evaluation framework
├── generate_report.py  # PDF report generator
├── main.py             # Main runner script
├── requirements.txt    # Python dependencies
└── README.md           # This file
```

## Requirements

- Python 3.8+
- NumPy
- Matplotlib
- ReportLab (for PDF generation)

## Installation

```bash
pip install -r requirements.txt
```

## Running the Experiments

To reproduce all experiments and generate the report:

```bash
python main.py
```

This will:
1. Run all four experiments (filtering vs smoothing, noise sensitivity, transition dynamics, sequence length)
2. Generate visualization figures
3. Create the PDF report

## Generated Outputs

After running `main.py`, the following files are produced:

- `HMM_Project_Report.pdf` - Complete project report
- `belief_evolution.png` - Visualization of belief updates over time
- `experiment_results.png` - Summary plots of all experiments
- `results_table.txt` - Numerical results in text format

## Individual Module Usage

### Running HMM tests only:
```bash
python hmm.py
```

### Running experiments only:
```bash
python experiments.py
```

## Key Results

- Smoothing consistently outperforms filtering by ~2-3% accuracy
- All methods degrade gracefully with increasing observation noise
- Higher state persistence ("sticky" states) improves inference accuracy

## Author

Murat Emirhan Aykut  
December 2025

## References

- Russell, S., & Norvig, P. (2020). *Artificial Intelligence: A Modern Approach* (4th ed.). Chapter 14.
- Rabiner, L. R. (1989). A tutorial on hidden Markov models. *Proceedings of the IEEE*, 77(2), 257-286.
