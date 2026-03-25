# Offline–Online Reinforcement Learning for Linear Mixture MDPs

This repository contains the simulation code accompanying the paper **Offline–Online Reinforcement Learning for Linear Mixture MDPs**. The code studies offline–online learning under environment shift and unknown offline behavior policy, and evaluates how offline coverage `tau`, environment shift `Delta`, and the design of offline value-function estimates affect regret.

## What is in this repository

The current codebase implements **synthetic tabular MDP simulations under a linear-mixture embedding**, together with plotting utilities for the main numerical results.

The experiments are organized around the following questions:

- How regret changes with environment shift `Delta`.
- How regret changes with offline data size `M_off`.
- How regret changes with effective coverage `tau`.
- How the relative scaling between `Delta` and `tau` affects whether offline data remains informative.
- How the proposed offline value-function design compares with pessimistic and online-only baselines.

## Repository structure

```text
.
├── source_codes/
│   ├── main.py                # Main entry point for reproducing figures
│   ├── hyperparam.py          # Central experiment configuration
│   ├── prep.py                # Experiment runners, sweeps, caching, summaries
│   ├── plot.py                # Plotting utilities
│   ├── MDP.py                 # Synthetic tabular MDP / linear-mixture embedding helpers
│   ├── behavior_policy.py     # Offline behavior policies used in simulation
│   ├── algo.py                # Main offline–online algorithm and online baselines
│   ├── algo_chen2022.py       # Tabular offline–online baseline inspired by Chen et al. (2022)
│   ├── Azar17.py              # Tabular UCBVI-style baseline
│   └── test.txt               # Small auxiliary file in the repo
├── README.md
├── hyperparameter_setting.md
└── requirements.txt
```

## Environment setup

### Python

- Recommended Python version: **3.10**.

### Install dependencies

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Running the code

All experiments are controlled from `source_codes/main.py`, and the configuration is centralized in `source_codes/hyperparam.py`.

A typical run from the repository root is:

```bash
python source_codes/main.py
```

The script reads `SimConfig` in `source_codes/hyperparam.py` and generates the figure selected by `plt_choice`.

## Figure selection

The code supports the following figure options through `SimConfig.plt_choice`:

- `"A"`: final regret versus `Delta`
- `"B"`: final regret versus `M_off`
- `"C"`: final regret versus `tau`
- `"D"`: final regret versus `tau` under custom `Delta(tau)` curves
- `"E"`: regret curves over online episodes `K`
- `"legend"`: legend-only export for figure assembly

### Example workflow

1. Open `source_codes/hyperparam.py`.
2. Set `plt_choice` to the panel you want.
3. Set `if_running = 1` to run simulations and cache results.
4. Run:

```bash
python source_codes/main.py
```

If cached result files already exist, setting `if_running = 0` reloads them instead of rerunning the simulations.

## Output files

The plotting code saves figures directly to the current working directory. Typical output filenames include:

- `final_regret_vs_Delta.pdf`
- `final_regret_vs_Moff.pdf`
- `final_regret_vs_tau_fixed_Delta.pdf`
- `final_regret_vs_z_multi_curves_zero_optimal.pdf`
- `regret_curves_random_offline.pdf`
- `figure2legend.pdf`

Some sweep utilities also save cached intermediate results as `.pkl` files, for example:

- `curve_A.pkl`
- `curve_B.pkl`
- `curve_C.pkl`
- `curve_D.pkl`
- `curves_E.pkl`

## Main configuration file

All main experiment settings are collected in:

```text
source_codes/hyperparam.py
```

See `hyperparameter_setting.md` for a complete description of the available fields and how they affect the experiments.

## Main baselines in the code

The simulation pipeline compares the proposed method against several baselines, including:

- `zero`: online-only / no useful offline value design
- `optimal`: proposed offline value-function design
- `pessimistic`: pessimistic offline value-function design
- `merge`: naive merge baseline that effectively assumes no environment shift
- `dp_ucb`: tabular baseline from the Chen et al. style implementation
- `ucbvi`: tabular online baseline

The exact algorithms included in a specific plot depend on the plotting routine.

## Notes on plotting

The plotting utilities use LaTeX text rendering in Matplotlib (`text.usetex = True`). On some machines, figure generation may therefore require a working LaTeX installation. If LaTeX is not available, you can either install it or modify the plotting configuration in `source_codes/plot.py`.

## Reproducibility notes

- Random seeds are controlled through `SimConfig.seed` and the sweep-level `seed_offset` values inside `main.py` and `prep.py`.
- `delta_conf` defaults to `1 / K` when not manually set.
- `z_action0` defaults to `0.3` when not manually set.
- The synthetic environment is tabular, but the algorithm is implemented through a linear-mixture representation.

## Citation

If you use this repository, please cite the associated paper.
