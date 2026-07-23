"""Test whether an early training solver leaves a persistent optimization-path effect.

The experiment deliberately separates two mechanisms:

1. Path construction:
   Starting from the *same initialization* for a given seed, train the UDE with
   Euler, Heun, or RK4 for a fixed number of epochs.
2. Common-objective refinement:
   Reset all optimizer state, switch every branch to the *same* RK4 rollout
   objective and step size, run the same Adam learning-rate schedule, and then
   refine with full-batch L-BFGS.

If branches remain functionally different after they all satisfy the same
stationarity criteria, the early solver has selected different stationary
solutions/basins of the common RK4 objective. If the differences disappear,
the early-solver effect was mainly transient rather than persistent.

This script reuses the clean Lotka--Volterra carrying-capacity data, fixed
incomplete physics, and neural residual from ``solver_error_sweep.py``.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import pickle
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import optax
from jax import jit, lax, value_and_grad, vmap
from jax.flatten_util import ravel_pytree
from scipy.optimize import minimize

import solver_error_sweep as base


SOLVERS = {
    "forward_euler": base.forward_euler,
    "heun": base.heun,
    "rk4": base.rk4,
}


def progress(message: str) -> None:
    print(message, flush=True)


def params_from_nn(nn_params):
    """Insert trainable NN parameters into the fixed-physics UDE."""
    return {
        "nn_params": nn_params,
        "f_physics": jnp.array([1.0, 0.05, 1.5, 0.03]),
    }


def tree_diagnostics(tree) -> dict[str, float]:
    flat, _ = ravel_pytree(tree)
    n = max(int(flat.size), 1)
    return {
        "norm": float(jnp.linalg.norm(flat)),
        "rms": float(jnp.linalg.norm(flat) / math.sqrt(n)),
        "max_abs": float(jnp.max(jnp.abs(flat))),
        "n_parameters": n,
    }


def make_objective(method, ratio, data, residual_regularization):
    """Build one fixed-step rollout objective and its diagnostics."""
    h = base.obs_dt / ratio
    num_steps = int((base.num_observed - 1) * ratio)
    residual_states = data["train_noisy"].reshape(-1, 2)

    def sample_data_mse(nn_params, y0, target):
        params = params_from_nn(nn_params)
        prediction = base.roll_out(
            y0,
            h,
            base.model_rhs,
            params,
            num_steps,
            method,
        )[::ratio]
        return jnp.mean((prediction - target) ** 2)

    def data_mse(nn_params):
        losses = vmap(sample_data_mse, in_axes=(None, 0, 0))(
            nn_params,
            base.y0_batch,
            data["train_noisy"],
        )
        return jnp.mean(losses)

    def regularization(nn_params):
        residual = vmap(base.nn, in_axes=(0, None))(residual_states, nn_params)
        return residual_regularization * jnp.mean(residual**2)

    def objective(nn_params):
        return data_mse(nn_params) + regularization(nn_params)

    return {
        "method": method,
        "ratio": ratio,
        "h": h,
        "data_mse": jit(data_mse),
        "regularization": jit(regularization),
        "objective": jit(objective),
        "value_and_grad": jit(value_and_grad(objective)),
    }


def diagnose(nn_params, objective_bundle) -> dict[str, float]:
    objective, gradient = objective_bundle["value_and_grad"](nn_params)
    grad = tree_diagnostics(gradient)
    return {
        "objective": float(objective),
        "data_mse": float(objective_bundle["data_mse"](nn_params)),
        "regularization": float(objective_bundle["regularization"](nn_params)),
        "gradient_norm": grad["norm"],
        "gradient_rms": grad["rms"],
        "gradient_max_abs": grad["max_abs"],
        "n_parameters": grad["n_parameters"],
    }


def history_row(
    seed,
    early_solver,
    phase,
    phase_solver,
    phase_epoch,
    common_epoch,
    learning_rate,
    diagnostics,
):
    return {
        "seed": seed,
        "early_solver": early_solver,
        "phase": phase,
        "phase_solver": phase_solver,
        "phase_epoch": phase_epoch,
        "common_epoch": common_epoch,
        "learning_rate": learning_rate,
        **diagnostics,
    }


def adam_phase(
    nn_params,
    objective_bundle,
    learning_rate,
    epochs,
    chunk_size,
    seed,
    early_solver,
    phase,
    phase_solver,
    common_epoch_offset,
):
    """Run a fixed number of Adam epochs with a newly reset optimizer."""
    if epochs == 0:
        return nn_params, []
    if epochs % chunk_size != 0:
        raise ValueError(
            f"epochs={epochs} must be divisible by chunk_size={chunk_size}"
        )

    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(nn_params)

    @jit
    def train_chunk(parameters, state):
        def step(carry, _):
            current_params, current_state = carry
            loss, grads = objective_bundle["value_and_grad"](current_params)
            updates, current_state = optimizer.update(
                grads,
                current_state,
                current_params,
            )
            current_params = optax.apply_updates(current_params, updates)
            return (current_params, current_state), loss

        (parameters, state), losses = lax.scan(
            step,
            (parameters, state),
            None,
            length=chunk_size,
        )
        return parameters, state, losses

    history = []
    for phase_epoch in range(chunk_size, epochs + 1, chunk_size):
        nn_params, opt_state, _ = train_chunk(nn_params, opt_state)
        common_epoch = (
            common_epoch_offset + phase_epoch if phase == "common_rk4_adam" else 0
        )
        history.append(
            history_row(
                seed=seed,
                early_solver=early_solver,
                phase=phase,
                phase_solver=phase_solver,
                phase_epoch=phase_epoch,
                common_epoch=common_epoch,
                learning_rate=learning_rate,
                diagnostics=diagnose(nn_params, objective_bundle),
            )
        )
    return nn_params, history


def lbfgs_refine(
    nn_params,
    common_objective,
    maxiter,
    gtol,
    ftol,
    maxls,
):
    """Full-batch L-BFGS refinement of NN parameters only."""
    flat_initial, unravel = ravel_pytree(nn_params)

    @jit
    def flat_value_and_grad(flat_parameters):
        parameters = unravel(flat_parameters)
        value, grads = common_objective["value_and_grad"](parameters)
        flat_grads, _ = ravel_pytree(grads)
        return value, flat_grads

    def scipy_value_and_grad(flat_parameters):
        value, gradient = flat_value_and_grad(jnp.asarray(flat_parameters))
        return float(value), np.asarray(gradient, dtype=np.float64)

    start_diagnostics = diagnose(nn_params, common_objective)
    result = minimize(
        scipy_value_and_grad,
        np.asarray(flat_initial, dtype=np.float64),
        method="L-BFGS-B",
        jac=True,
        options={
            "maxiter": maxiter,
            "gtol": gtol,
            "ftol": ftol,
            "maxls": maxls,
            "maxcor": 20,
        },
    )
    candidate = unravel(jnp.asarray(result.x))
    candidate_diagnostics = diagnose(candidate, common_objective)

    # A failed line search can occasionally return a worse iterate. Retain the
    # lower common objective, but report the SciPy status either way.
    accepted = (
        np.isfinite(candidate_diagnostics["objective"])
        and candidate_diagnostics["objective"]
        <= start_diagnostics["objective"]
    )
    final_params = candidate if accepted else nn_params
    final_diagnostics = (
        candidate_diagnostics if accepted else start_diagnostics
    )
    info = {
        "lbfgs_success": bool(result.success),
        "lbfgs_status": int(result.status),
        "lbfgs_message": str(result.message),
        "lbfgs_iterations": int(result.nit),
        "lbfgs_function_evaluations": int(result.nfev),
        "lbfgs_accepted": bool(accepted),
        "pre_lbfgs_objective": start_diagnostics["objective"],
        "post_lbfgs_candidate_objective": candidate_diagnostics["objective"],
        **{f"final_{key}": value for key, value in final_diagnostics.items()},
    }
    return final_params, info


def matched_mse(nn_params, method, ratio, y0s, targets):
    h = base.obs_dt / ratio
    num_steps = int((targets.shape[1] - 1) * ratio)
    params = params_from_nn(nn_params)

    def one(y0, target):
        prediction = base.roll_out(
            y0,
            h,
            base.model_rhs,
            params,
            num_steps,
            method,
        )[::ratio]
        return jnp.mean((prediction - target) ** 2)

    return float(jnp.mean(vmap(one)(y0s, targets)))


def evaluate_final_model(nn_params, data, common_ratio):
    params = params_from_nn(nn_params)
    continuous_train = base.solve_model_diffrax_batch(
        base.y0_batch,
        params,
        base.t_obs,
    )
    continuous_validation = base.solve_model_diffrax_batch(
        base.y0_validation,
        params,
        base.t_obs,
    )
    continuous_extrapolation = base.solve_model_diffrax_batch(
        base.y0_batch,
        params,
        base.t_extrapolate,
    )
    extrapolation_start = base.num_observed
    parameter_flat, _ = ravel_pytree(nn_params)

    return {
        "rk4_matched_train_mse": matched_mse(
            nn_params,
            base.rk4,
            common_ratio,
            base.y0_batch,
            data["train_clean"],
        ),
        "rk4_matched_validation_mse": matched_mse(
            nn_params,
            base.rk4,
            common_ratio,
            base.y0_validation,
            data["val_clean"],
        ),
        "continuous_train_mse": float(
            jnp.mean((continuous_train - data["train_clean"]) ** 2)
        ),
        "continuous_validation_mse": float(
            jnp.mean((continuous_validation - data["val_clean"]) ** 2)
        ),
        "continuous_extrapolation_mse": float(
            jnp.mean(
                (
                    continuous_extrapolation[:, extrapolation_start:, :]
                    - data["extrap"][:, extrapolation_start:, :]
                )
                ** 2
            )
        ),
        "continuous_final_mse": float(
            jnp.mean(
                (
                    continuous_extrapolation[:, -1, :]
                    - data["extrap"][:, -1, :]
                )
                ** 2
            )
        ),
        "continuous_instability_rate": base.instability_rate(
            continuous_extrapolation
        ),
        "nn_parameter_norm": float(jnp.linalg.norm(parameter_flat)),
        **base.vector_field_metrics(params),
    }


def evaluation_grid():
    prey = jnp.linspace(5.0, 25.0, 20)
    predator = jnp.linspace(10.0, 40.0, 20)
    prey_grid, predator_grid = jnp.meshgrid(prey, predator)
    return jnp.stack(
        [prey_grid.reshape(-1), predator_grid.reshape(-1)],
        axis=1,
    )


def functional_values(nn_params):
    states = evaluation_grid()
    return vmap(base.nn, in_axes=(0, None))(states, nn_params)


def relative_distance(left, right, floor=1e-30):
    numerator = float(jnp.linalg.norm(left - right))
    denominator = 0.5 * float(jnp.linalg.norm(left) + jnp.linalg.norm(right))
    return numerator / max(denominator, floor)


def pairwise_rows(
    trained,
    summary_rows,
    stationary_grad_rms,
    stationary_grad_max,
    loss_close_rtol,
    loss_close_atol,
    functional_difference_threshold,
):
    summary_lookup = {
        (int(row["seed"]), row["early_solver"]): row for row in summary_rows
    }
    output = []
    seeds = sorted({key[0] for key in trained})
    for seed in seeds:
        solvers = sorted(name for current_seed, name in trained if current_seed == seed)
        for left_idx, left_name in enumerate(solvers):
            for right_name in solvers[left_idx + 1 :]:
                left_params = trained[(seed, left_name)]["nn_params"]
                right_params = trained[(seed, right_name)]["nn_params"]
                left_flat, _ = ravel_pytree(left_params)
                right_flat, _ = ravel_pytree(right_params)
                left_residual = functional_values(left_params)
                right_residual = functional_values(right_params)
                true_residual = vmap(
                    lambda state: base.true_rhs(state, None)
                    - base.f_physics(
                        state,
                        jnp.array([1.0, 0.05, 1.5, 0.03]),
                    )
                )(evaluation_grid())

                left_row = summary_lookup[(seed, left_name)]
                right_row = summary_lookup[(seed, right_name)]
                loss_left = float(left_row["final_common_objective"])
                loss_right = float(right_row["final_common_objective"])
                loss_close = abs(loss_left - loss_right) <= (
                    loss_close_atol
                    + loss_close_rtol * max(abs(loss_left), abs(loss_right))
                )
                both_stationary = bool(
                    float(left_row["final_gradient_rms"])
                    <= stationary_grad_rms
                    and float(right_row["final_gradient_rms"])
                    <= stationary_grad_rms
                    and float(left_row["final_gradient_max_abs"])
                    <= stationary_grad_max
                    and float(right_row["final_gradient_max_abs"])
                    <= stationary_grad_max
                )
                residual_rms_difference = float(
                    jnp.sqrt(jnp.mean((left_residual - right_residual) ** 2))
                )
                residual_relative_to_true = float(
                    jnp.linalg.norm(left_residual - right_residual)
                    / (jnp.linalg.norm(true_residual) + 1e-30)
                )
                persistent = bool(
                    both_stationary
                    and residual_relative_to_true
                    > functional_difference_threshold
                )
                output.append(
                    {
                        "seed": seed,
                        "left_early_solver": left_name,
                        "right_early_solver": right_name,
                        "both_stationary": both_stationary,
                        "common_objective_close": bool(loss_close),
                        "persistent_functional_path_effect": persistent,
                        "left_common_objective": loss_left,
                        "right_common_objective": loss_right,
                        "common_objective_abs_difference": abs(
                            loss_left - loss_right
                        ),
                        "parameter_l2_distance": float(
                            jnp.linalg.norm(left_flat - right_flat)
                        ),
                        "parameter_relative_distance": relative_distance(
                            left_flat,
                            right_flat,
                        ),
                        "residual_rms_difference": residual_rms_difference,
                        "residual_relative_distance": relative_distance(
                            left_residual,
                            right_residual,
                        ),
                        "residual_difference_relative_to_true_residual": (
                            residual_relative_to_true
                        ),
                    }
                )
    return output


def add_rk4_reference_distances(summary_rows, trained):
    lookup = {
        (int(row["seed"]), row["early_solver"]): row for row in summary_rows
    }
    for seed in sorted({key[0] for key in trained}):
        reference_key = (seed, "rk4")
        if reference_key not in trained:
            continue
        reference = trained[reference_key]["nn_params"]
        reference_flat, _ = ravel_pytree(reference)
        reference_residual = functional_values(reference)
        reference_objective = float(
            lookup[reference_key]["final_common_objective"]
        )
        for current_seed, early_solver in trained:
            if current_seed != seed:
                continue
            current = trained[(current_seed, early_solver)]["nn_params"]
            current_flat, _ = ravel_pytree(current)
            current_residual = functional_values(current)
            row = lookup[(current_seed, early_solver)]
            row["parameter_relative_distance_to_rk4_path"] = relative_distance(
                current_flat,
                reference_flat,
            )
            row["residual_rms_distance_to_rk4_path"] = float(
                jnp.sqrt(jnp.mean((current_residual - reference_residual) ** 2))
            )
            row["residual_relative_distance_to_rk4_path"] = relative_distance(
                current_residual,
                reference_residual,
            )
            row["common_objective_difference_to_rk4_path"] = abs(
                float(row["final_common_objective"]) - reference_objective
            )


def write_csv(rows, path):
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_common_history(history, output_dir):
    common = [
        row
        for row in history
        if row["phase"]
        in {"common_rk4_switch", "common_rk4_adam", "common_rk4_lbfgs"}
    ]
    if not common:
        return
    seeds = sorted({int(row["seed"]) for row in common})
    fig, axes = plt.subplots(
        len(seeds),
        1,
        figsize=(8.0, max(4.0, 3.4 * len(seeds))),
        squeeze=False,
        sharex=True,
    )
    for ax, seed in zip(axes[:, 0], seeds):
        for early_solver in SOLVERS:
            selected = [
                row
                for row in common
                if int(row["seed"]) == seed
                and row["early_solver"] == early_solver
            ]
            if not selected:
                continue
            selected.sort(
                key=lambda row: (
                    int(row["common_epoch"]),
                    row["phase"] == "common_rk4_lbfgs",
                )
            )
            ax.semilogy(
                [int(row["common_epoch"]) for row in selected],
                [max(float(row["objective"]), 1e-30) for row in selected],
                marker="o",
                markersize=2.5,
                label=early_solver,
            )
        ax.set_ylabel("RK4 objective")
        ax.set_title(f"seed={seed}")
        ax.grid(True, which="both", alpha=0.3)
    axes[-1, 0].set_xlabel("common RK4 Adam epoch")
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Do early-solver paths merge under the common RK4 objective?")
    fig.tight_layout()
    fig.savefig(output_dir / "common_rk4_refinement_history.png", dpi=200)
    plt.close(fig)


def plot_final_path_distances(summary_rows, output_dir):
    selected = [
        row
        for row in summary_rows
        if row.get("residual_rms_distance_to_rk4_path", "") != ""
    ]
    if not selected:
        return
    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    solver_names = [
        name
        for name in ["forward_euler", "heun", "rk4"]
        if any(row["early_solver"] == name for row in selected)
    ]
    for x_position, solver_name in enumerate(solver_names):
        values = [
            float(row["residual_rms_distance_to_rk4_path"])
            for row in selected
            if row["early_solver"] == solver_name
        ]
        if not values:
            continue
        jitter = np.linspace(-0.06, 0.06, len(values))
        ax.scatter(
            np.full(len(values), x_position) + jitter,
            np.maximum(values, 1e-30),
            s=38,
            alpha=0.8,
        )
        ax.plot(
            x_position,
            max(float(np.median(values)), 1e-30),
            marker="_",
            markersize=18,
            color="black",
        )
    ax.set_xticks(range(len(solver_names)))
    ax.set_xticklabels(solver_names)
    ax.set_yscale("log")
    ax.set_ylabel("final residual RMS distance to RK4-history branch")
    ax.set_xlabel("early training solver")
    ax.set_title("Persistent functional difference after common-objective refinement")
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "final_path_functional_distance.png", dpi=200)
    plt.close(fig)


def experiment_signature(args):
    return {
        "seeds": list(args.seeds),
        "early_solvers": list(args.early_solvers),
        "early_ratio": args.early_ratio,
        "early_epochs": args.early_epochs,
        "early_learning_rate": args.early_learning_rate,
        "common_ratio": args.common_ratio,
        "common_adam_learning_rates": list(args.common_adam_learning_rates),
        "common_adam_epochs": list(args.common_adam_epochs),
        "chunk_size": args.chunk_size,
        "residual_regularization": args.residual_regularization,
        "lbfgs_maxiter": args.lbfgs_maxiter,
        "lbfgs_gtol": args.lbfgs_gtol,
        "lbfgs_ftol": args.lbfgs_ftol,
        "lbfgs_maxls": args.lbfgs_maxls,
        "stationary_grad_rms": args.stationary_grad_rms,
        "stationary_grad_max": args.stationary_grad_max,
        "loss_close_rtol": args.loss_close_rtol,
        "loss_close_atol": args.loss_close_atol,
        "functional_difference_threshold": args.functional_difference_threshold,
    }


def save_checkpoint(path, signature, trained, summary_rows, history):
    temporary = path.with_suffix(".tmp")
    with temporary.open("wb") as handle:
        pickle.dump(
            {
                "signature": signature,
                "trained": trained,
                "summary_rows": summary_rows,
                "history": history,
            },
            handle,
        )
    temporary.replace(path)


def load_checkpoint(path, signature):
    if not path.exists():
        return {}, [], []
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    if payload.get("signature") != signature:
        raise ValueError(
            "Existing checkpoint was created with different experiment "
            "arguments. Use another --output-dir or remove the old checkpoint."
        )
    return (
        payload.get("trained", {}),
        payload.get("summary_rows", []),
        payload.get("history", []),
    )


def run(args):
    if len(args.common_adam_learning_rates) != len(args.common_adam_epochs):
        raise ValueError(
            "--common-adam-learning-rates and --common-adam-epochs must "
            "have the same number of entries."
        )
    unknown = set(args.early_solvers) - set(SOLVERS)
    if unknown:
        raise ValueError(f"Unknown early solvers: {sorted(unknown)}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    signature = experiment_signature(args)
    checkpoint_path = output_dir / "checkpoint.pkl"

    progress("[setup] generating clean reference trajectories")
    data = base.make_reference_data()
    common_objective = make_objective(
        base.rk4,
        args.common_ratio,
        data,
        args.residual_regularization,
    )
    trained, summary_rows, history = load_checkpoint(
        checkpoint_path,
        signature,
    )
    completed = set(trained)
    if completed:
        progress(f"[resume] found {len(completed)} completed branches")

    total = len(args.seeds) * len(args.early_solvers)
    run_index = 0
    for seed in args.seeds:
        initial_nn = base.initial_params(seed)["nn_params"]
        for early_solver_name in args.early_solvers:
            run_index += 1
            key = (int(seed), early_solver_name)
            if key in completed:
                progress(
                    f"[run {run_index}/{total}] skip seed={seed}, "
                    f"early={early_solver_name}"
                )
                continue

            progress(
                f"[run {run_index}/{total}] seed={seed}, "
                f"early={early_solver_name}"
            )
            early_objective = make_objective(
                SOLVERS[early_solver_name],
                args.early_ratio,
                data,
                args.residual_regularization,
            )
            branch_history = [
                history_row(
                    seed=seed,
                    early_solver=early_solver_name,
                    phase="initial",
                    phase_solver=early_solver_name,
                    phase_epoch=0,
                    common_epoch=0,
                    learning_rate=0.0,
                    diagnostics=diagnose(initial_nn, early_objective),
                )
            ]

            after_early, early_history = adam_phase(
                nn_params=initial_nn,
                objective_bundle=early_objective,
                learning_rate=args.early_learning_rate,
                epochs=args.early_epochs,
                chunk_size=args.chunk_size,
                seed=seed,
                early_solver=early_solver_name,
                phase="early_solver_adam",
                phase_solver=early_solver_name,
                common_epoch_offset=0,
            )
            branch_history.extend(early_history)
            early_final = diagnose(after_early, early_objective)
            switch_common = diagnose(after_early, common_objective)
            branch_history.append(
                history_row(
                    seed=seed,
                    early_solver=early_solver_name,
                    phase="common_rk4_switch",
                    phase_solver="rk4",
                    phase_epoch=0,
                    common_epoch=0,
                    learning_rate=0.0,
                    diagnostics=switch_common,
                )
            )

            current = after_early
            common_epoch_offset = 0
            for learning_rate, epochs in zip(
                args.common_adam_learning_rates,
                args.common_adam_epochs,
            ):
                # Reset optimizer state for every branch and every common
                # learning-rate stage. Only parameters carry path information.
                current, common_history = adam_phase(
                    nn_params=current,
                    objective_bundle=common_objective,
                    learning_rate=learning_rate,
                    epochs=epochs,
                    chunk_size=args.chunk_size,
                    seed=seed,
                    early_solver=early_solver_name,
                    phase="common_rk4_adam",
                    phase_solver="rk4",
                    common_epoch_offset=common_epoch_offset,
                )
                branch_history.extend(common_history)
                common_epoch_offset += epochs

            after_adam = diagnose(current, common_objective)
            final_nn, lbfgs_info = lbfgs_refine(
                nn_params=current,
                common_objective=common_objective,
                maxiter=args.lbfgs_maxiter,
                gtol=args.lbfgs_gtol,
                ftol=args.lbfgs_ftol,
                maxls=args.lbfgs_maxls,
            )
            final_common = diagnose(final_nn, common_objective)
            stationary = bool(
                final_common["gradient_rms"] <= args.stationary_grad_rms
                and final_common["gradient_max_abs"]
                <= args.stationary_grad_max
            )
            branch_history.append(
                history_row(
                    seed=seed,
                    early_solver=early_solver_name,
                    phase="common_rk4_lbfgs",
                    phase_solver="rk4",
                    phase_epoch=lbfgs_info["lbfgs_iterations"],
                    common_epoch=common_epoch_offset,
                    learning_rate="N/A",
                    diagnostics=final_common,
                )
            )
            evaluation = evaluate_final_model(
                final_nn,
                data,
                args.common_ratio,
            )
            row = {
                "seed": seed,
                "early_solver": early_solver_name,
                "early_ratio": args.early_ratio,
                "early_h": base.obs_dt / args.early_ratio,
                "early_epochs": args.early_epochs,
                "early_learning_rate": args.early_learning_rate,
                "common_solver": "rk4",
                "common_ratio": args.common_ratio,
                "common_h": base.obs_dt / args.common_ratio,
                "common_adam_total_epochs": sum(args.common_adam_epochs),
                "residual_regularization": args.residual_regularization,
                "early_final_objective": early_final["objective"],
                "early_final_data_mse": early_final["data_mse"],
                "switch_common_objective": switch_common["objective"],
                "switch_common_gradient_rms": switch_common["gradient_rms"],
                "post_adam_common_objective": after_adam["objective"],
                "post_adam_gradient_rms": after_adam["gradient_rms"],
                "final_common_objective": final_common["objective"],
                "final_common_data_mse": final_common["data_mse"],
                "final_common_regularization": final_common["regularization"],
                "final_gradient_norm": final_common["gradient_norm"],
                "final_gradient_rms": final_common["gradient_rms"],
                "final_gradient_max_abs": final_common["gradient_max_abs"],
                "stationary_grad_rms_threshold": args.stationary_grad_rms,
                "stationary_grad_max_threshold": args.stationary_grad_max,
                "stationary": stationary,
                **lbfgs_info,
                **evaluation,
                "parameter_relative_distance_to_rk4_path": "",
                "residual_rms_distance_to_rk4_path": "",
                "residual_relative_distance_to_rk4_path": "",
                "common_objective_difference_to_rk4_path": "",
            }
            trained[key] = {
                "nn_params": final_nn,
                "after_early_nn_params": after_early,
            }
            summary_rows.append(row)
            history.extend(branch_history)
            save_checkpoint(
                checkpoint_path,
                signature,
                trained,
                summary_rows,
                history,
            )
            progress(
                f"[run {run_index}/{total}] final objective="
                f"{final_common['objective']:.3e}, "
                f"grad_rms={final_common['gradient_rms']:.3e}, "
                f"stationary={stationary}"
            )

    add_rk4_reference_distances(summary_rows, trained)
    pairs = pairwise_rows(
        trained=trained,
        summary_rows=summary_rows,
        stationary_grad_rms=args.stationary_grad_rms,
        stationary_grad_max=args.stationary_grad_max,
        loss_close_rtol=args.loss_close_rtol,
        loss_close_atol=args.loss_close_atol,
        functional_difference_threshold=args.functional_difference_threshold,
    )
    summary_rows.sort(key=lambda row: (int(row["seed"]), row["early_solver"]))
    history.sort(
        key=lambda row: (
            int(row["seed"]),
            row["early_solver"],
            0 if row["phase"] == "initial" else 1,
            int(row["common_epoch"]),
            row["phase"] == "common_rk4_lbfgs",
            int(row["phase_epoch"]),
        )
    )
    write_csv(summary_rows, output_dir / "summary.csv")
    write_csv(history, output_dir / "training_history.csv")
    write_csv(pairs, output_dir / "pairwise_path_comparisons.csv")
    with (output_dir / "metadata.json").open("w") as handle:
        json.dump(
            {
                "description": __doc__,
                "signature": signature,
                "stationarity": {
                    "gradient_rms_threshold": args.stationary_grad_rms,
                    "gradient_max_abs_threshold": args.stationary_grad_max,
                },
                "path_effect_decision": {
                    "loss_close_rtol": args.loss_close_rtol,
                    "loss_close_atol": args.loss_close_atol,
                    "functional_difference_threshold_relative_to_true_residual": (
                        args.functional_difference_threshold
                    ),
                },
                "interpretation": (
                    "Parameter distance alone is not evidence because neural "
                    "networks have parameter symmetries. Persistent path "
                    "dependence should be judged from residual/vector-field "
                    "differences between branches that both pass stationarity."
                ),
                "jax_enable_x64": bool(jax.config.x64_enabled),
            },
            handle,
            indent=2,
        )
    plot_common_history(history, output_dir)
    plot_final_path_distances(summary_rows, output_dir)
    save_checkpoint(
        checkpoint_path,
        signature,
        trained,
        summary_rows,
        history,
    )
    progress(f"[done] results written to {output_dir}")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).with_name("solver_path_dependence_results"),
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument(
        "--early-solvers",
        nargs="+",
        choices=list(SOLVERS),
        default=["forward_euler", "heun", "rk4"],
    )
    parser.add_argument("--early-ratio", type=int, default=1)
    parser.add_argument("--early-epochs", type=int, default=3000)
    parser.add_argument("--early-learning-rate", type=float, default=3e-3)
    parser.add_argument("--common-ratio", type=int, default=4)
    parser.add_argument(
        "--common-adam-learning-rates",
        nargs="+",
        type=float,
        default=[1e-3, 3e-4, 1e-4],
    )
    parser.add_argument(
        "--common-adam-epochs",
        nargs="+",
        type=int,
        default=[3000, 6000, 12000],
    )
    parser.add_argument("--chunk-size", type=int, default=100)
    parser.add_argument("--residual-regularization", type=float, default=1e-5)
    parser.add_argument("--lbfgs-maxiter", type=int, default=250)
    parser.add_argument("--lbfgs-gtol", type=float, default=1e-8)
    parser.add_argument("--lbfgs-ftol", type=float, default=1e-14)
    parser.add_argument("--lbfgs-maxls", type=int, default=40)
    parser.add_argument("--stationary-grad-rms", type=float, default=1e-7)
    parser.add_argument("--stationary-grad-max", type=float, default=1e-5)
    parser.add_argument("--loss-close-rtol", type=float, default=1e-3)
    parser.add_argument("--loss-close-atol", type=float, default=1e-10)
    parser.add_argument(
        "--functional-difference-threshold",
        type=float,
        default=1e-3,
        help=(
            "Persistent path-effect threshold for the RMS residual difference "
            "relative to the RMS true missing residual."
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
