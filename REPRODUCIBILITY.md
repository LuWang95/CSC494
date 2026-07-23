# Reproducibility Guide

This document describes the canonical environment and commands for the final
Lotka--Volterra carrying-capacity experiments. Run commands from the directory
shown in each section; the experiment scripts use local imports.

## Environment

Use Python 3.12 and install the pinned dependencies from the repository root:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python scripts/check_reproducibility.py
```

For deterministic comparisons, use the same hardware/backend and keep the
repository clean. JAX reductions and accelerator kernels are not guaranteed to
be bitwise identical across platforms. The scripts set explicit model and data
seeds; result claims should therefore compare summary statistics rather than
requiring bitwise-identical pickle files.

Every new carrying experiment writes `run_manifest.json` into its output
directory. It records:

- all command-level experiment parameters;
- Python and core package versions;
- the Git commit;
- whether the worktree was dirty when the run started.
- SHA-256 hashes of the experiment source files and reproducibility inputs.

Do not use a result in the final report if its manifest says `"dirty": true`
unless the exact diff is archived with the result.

## Working directory

```bash
cd code/residual_neural_ode/lotka-volterra
```

Always use a new output directory for smoke tests. The sweep scripts support
resuming and will treat rows already present in the requested directory as
completed runs.

## Selected solver sweep (25 runs)

Smoke test:

```bash
python carrying_solver_sweep.py \
  --seeds 0 \
  --stage-scale 0.05 \
  --output-dir carrying_solver_smoke
```

Canonical run:

```bash
python carrying_solver_sweep.py \
  --seeds 0 1 2 3 4 \
  --stage-scale 1.0 \
  --output-dir carrying_solver_sweep_results_reproduced
```

## Big-factor sweep (192 runs)

The default grid contains 16 training configurations (three fixed-step methods
at five ratios, plus adaptive Tsit5), four regularization profiles, and three
seeds.

Smoke test:

```bash
python carrying_big_factor_sweep.py \
  --methods heun rk4 \
  --train-ratios 1 4 \
  --regularization-profiles none l2_plus_ortho \
  --seeds 0 \
  --no-diffrax \
  --stage-scale 0.05 \
  --output-dir carrying_big_factor_smoke
```

Canonical run:

```bash
python carrying_big_factor_sweep.py \
  --methods forward_euler heun rk4 \
  --train-ratios 1 2 4 8 16 \
  --regularization-profiles none l2_small ortho_small l2_plus_ortho \
  --seeds 0 1 2 \
  --stage-scale 1.0 \
  --output-dir carrying_big_factor_sweep_results_reproduced
```

## Training-interval sweep (40 runs)

Smoke test:

```bash
python carrying_train_interval_sweep.py \
  --train-intervals 1.25 2.5 \
  --seeds 0 \
  --ratio 8 \
  --noise-level 0 \
  --coverage-radius 2.5 \
  --chunk-size 100 \
  --stage-scale 0.05 \
  --output-dir carrying_train_interval_smoke
```

Canonical run:

```bash
python carrying_train_interval_sweep.py \
  --train-intervals 1.25 2.5 5.0 7.5 \
  --seeds 0 1 2 3 4 5 6 7 8 9 \
  --ratio 8 \
  --noise-level 0 \
  --coverage-radius 2.5 \
  --chunk-size 100 \
  --stage-scale 1.0 \
  --output-dir carrying_train_interval_sweep_results
```

Visualize only after checking that all 40 rows are present:

```bash
python visualize_carrying_train_interval_sweep.py \
  --input-dir carrying_train_interval_sweep_results \
  --output-dir carrying_train_interval_sweep_visualizations

python ../../../scripts/check_reproducibility.py --require-complete
```

## Protocol compatibility note

The staged training schedule is part of the experiment definition. Training
protocol version 2 removes the historical no-op stage 4 and uses the three
effective stages only. Resume keys include `chunk_size`, `stage_scale`, and the
training protocol version, so smoke-test models cannot be reused by a canonical
run. `training_manifest.json` records the non-overwritten training provenance;
`evaluation_manifest.json` records the latest evaluation invocation.

The existing 40-run interval result uses evaluation schema 2 and retains the
historical no-op stage in its saved history. Its numerical parameters remain
valid because that stage performed no updates. A new protocol-2 run should use
a fresh output directory if the historical artifacts need to be preserved.

The existing big-factor result directory predates `run_manifest.json`. Its CSV
contains the expected 192 unique configurations, but exact bitwise provenance
cannot be reconstructed retroactively. New report-quality runs should use the
`*_reproduced` directory and retain their manifest.

## Parameter-recovery importance sweep (160 runs)

This experiment tests whether physics-parameter recovery predicts fixed-horizon
extrapolation more strongly at short training intervals, after controlling for
global vector-field error and regularization condition. It crosses four
training intervals, four regularization profiles, and ten initialization seeds.

Smoke test:

```bash
python carrying_parameter_importance_sweep.py \
  --train-intervals 1.25 \
  --regularization-profiles l2_plus_ortho \
  --seeds 0 \
  --stage-scale 0.05 \
  --output-dir carrying_parameter_importance_smoke
```

Canonical run:

```bash
python carrying_parameter_importance_sweep.py \
  --train-intervals 1.25 2.5 5.0 7.5 \
  --regularization-profiles none l2_small ortho_small l2_plus_ortho \
  --seeds 0 1 2 3 4 5 6 7 8 9 \
  --ratio 8 \
  --noise-level 0 \
  --coverage-radius 2.5 \
  --chunk-size 100 \
  --stage-scale 1.0 \
  --output-dir carrying_parameter_importance_sweep_results
```

Analysis:

```bash
python analyze_carrying_parameter_importance.py \
  --input-csv carrying_parameter_importance_sweep_results/carrying_parameter_importance_sweep_summary.csv \
  --output-dir carrying_parameter_importance_analysis

python visualize_carrying_parameter_importance.py \
  --input-csv carrying_parameter_importance_sweep_results/carrying_parameter_importance_sweep_summary.csv \
  --output-dir carrying_parameter_importance_visualizations
```

The primary confirmatory statistic is the interaction between standardized
log parameter error and standardized training interval. A negative interaction
whose seed-cluster bootstrap confidence interval excludes zero supports the
short-interval-importance hypothesis. The oracle replacement metrics are a
secondary intervention: values above one mean that replacing the learned
physics parameters with the true parameters, while holding the residual network
fixed, reduces extrapolation MSE. Because the residual can compensate for the
learned physics, oracle replacement can also make a model worse and must not be
interpreted as a standalone causal estimate.
