"""One-parameter experiment for isolating solver-induced training bias.

The fixed physics omits the prey carrying-capacity term. Instead of fitting a
neural network residual, this experiment fits the single parameter ``alpha`` in

    residual_alpha(x, y) = (-alpha * x**2, 0).

The data-generating system is therefore exactly recovered at ``alpha=c``. For
each fixed-step training solver and step size, we minimize the *discrete rollout
loss* J_h(alpha) with a deterministic grid search followed by bounded scalar
refinement. The fitted continuous-time model is then evaluated with a tight
Diffrax solve. This removes neural-network approximation error, Adam noise,
random seeds, and early stopping from the comparison.

Example
-------
python one_parameter_solver_bias.py --ratios 1 2 4 8 16

To make high-order effects larger, use a coarser observation interval while
keeping the original reference-data grid, for example:

python one_parameter_solver_bias.py --observation-stride 4 --ratios 1 2 4 8
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from jax import config

config.update("jax_enable_x64", True)

import diffrax
import jax
import jax.numpy as jnp
import matplotlib
import numpy as np
from scipy.optimize import minimize_scalar

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from solver import forward_euler, heun, rk4, roll_out


# Data-generating Lotka--Volterra parameters. The fixed physics uses A, B, R,
# and Z; the one-parameter residual is responsible for TRUE_ALPHA.
A = 1.0
B = 0.05
R = 1.5
Z = 0.03
TRUE_ALPHA = 0.005

T_TRAIN = 5.0
T_EXTRAPOLATE = 20.0
BASE_DT = 0.05
BASE_TRAIN_STEPS = int(round(T_TRAIN / BASE_DT))
BASE_EXTRAPOLATE_STEPS = int(round(T_EXTRAPOLATE / BASE_DT))

TRAIN_INITIAL_CONDITIONS = jnp.array(
    [
        [15.0, 25.0],
        [10.0, 20.0],
        [20.0, 30.0],
        [12.0, 22.0],
        [18.0, 28.0],
        [8.0, 18.0],
        [22.0, 26.0],
        [16.0, 15.0],
    ],
    dtype=jnp.float64,
)
VALIDATION_INITIAL_CONDITIONS = jnp.array(
    [[17.0, 25.0], [10.0, 23.0], [19.0, 24.0]], dtype=jnp.float64
)

SOLVERS = {
    "forward_euler": forward_euler,
    "heun": heun,
    "rk4": rk4,
}
NOMINAL_ORDERS = {"forward_euler": 1, "heun": 2, "rk4": 4}


def progress(message: str) -> None:
    print(message, flush=True)


def true_vector_field(t, state, args):
    del t, args
    prey, predator = state
    return jnp.array(
        [
            A * prey - B * prey * predator - TRUE_ALPHA * prey**2,
            -R * predator + Z * prey * predator,
        ]
    )


def fixed_physics(state):
    prey, predator = state
    return jnp.array(
        [A * prey - B * prey * predator, -R * predator + Z * prey * predator]
    )


def residual_alpha(state, alpha):
    prey = state[0]
    return jnp.array([-alpha * prey**2, 0.0])


def model_rhs(state, alpha):
    return fixed_physics(state) + residual_alpha(state, alpha)


def solve_diffrax_single(y0, alpha, ts, t1, rtol, atol):
    def rhs(t, state, parameter):
        del t
        return model_rhs(state, parameter)

    solution = diffrax.diffeqsolve(
        diffrax.ODETerm(rhs),
        diffrax.Tsit5(),
        t0=0.0,
        t1=t1,
        dt0=min(1e-3, t1 / 1000.0),
        y0=y0,
        args=alpha,
        saveat=diffrax.SaveAt(ts=ts),
        stepsize_controller=diffrax.PIDController(rtol=rtol, atol=atol),
        max_steps=500_000,
    )
    return solution.ys


def solve_diffrax_batch(y0s, alpha, ts, t1, rtol, atol):
    solve_one = lambda y0: solve_diffrax_single(y0, alpha, ts, t1, rtol, atol)
    return jax.vmap(solve_one)(y0s)


@dataclass(frozen=True)
class ReferenceData:
    train_ts: jax.Array
    extrapolate_ts: jax.Array
    train: jax.Array
    validation: jax.Array
    extrapolate: jax.Array
    observation_stride: int

    @property
    def observation_dt(self) -> float:
        return BASE_DT * self.observation_stride


def make_reference_data(observation_stride: int, rtol: float, atol: float):
    if BASE_TRAIN_STEPS % observation_stride != 0:
        raise ValueError(
            f"observation_stride={observation_stride} must divide "
            f"BASE_TRAIN_STEPS={BASE_TRAIN_STEPS}."
        )
    if BASE_EXTRAPOLATE_STEPS % observation_stride != 0:
        raise ValueError(
            f"observation_stride={observation_stride} must divide "
            f"BASE_EXTRAPOLATE_STEPS={BASE_EXTRAPOLATE_STEPS}."
        )

    full_extrapolate_ts = jnp.linspace(
        0.0, T_EXTRAPOLATE, BASE_EXTRAPOLATE_STEPS + 1
    )
    full_train_ts = full_extrapolate_ts[: BASE_TRAIN_STEPS + 1]
    progress("[setup] generating tight-tolerance reference trajectories")
    full_extrapolate = solve_diffrax_batch(
        TRAIN_INITIAL_CONDITIONS,
        jnp.asarray(TRUE_ALPHA),
        full_extrapolate_ts,
        T_EXTRAPOLATE,
        rtol,
        atol,
    )
    full_validation = solve_diffrax_batch(
        VALIDATION_INITIAL_CONDITIONS,
        jnp.asarray(TRUE_ALPHA),
        full_train_ts,
        T_TRAIN,
        rtol,
        atol,
    )

    selection = slice(None, None, observation_stride)
    train_ts = full_train_ts[selection]
    extrapolate_ts = full_extrapolate_ts[selection]
    return ReferenceData(
        train_ts=train_ts,
        extrapolate_ts=extrapolate_ts,
        train=full_extrapolate[:, : BASE_TRAIN_STEPS + 1, :][:, selection, :],
        validation=full_validation[:, selection, :],
        extrapolate=full_extrapolate[:, selection, :],
        observation_stride=observation_stride,
    )


def rollout_batch(y0s, alpha, h, num_steps, method):
    solve_one = lambda y0: roll_out(y0, h, model_rhs, alpha, num_steps, method)
    return jax.vmap(solve_one)(y0s)


def make_discrete_objective(method, ratio: int, data: ReferenceData):
    h = data.observation_dt / ratio
    num_steps = (data.train.shape[1] - 1) * ratio

    @jax.jit
    def objective(alpha):
        prediction = rollout_batch(
            TRAIN_INITIAL_CONDITIONS, alpha, h, num_steps, method
        )[:, ::ratio, :]
        return jnp.mean((prediction - data.train) ** 2)

    return objective


def make_diffrax_objective(data: ReferenceData, rtol: float, atol: float):
    @jax.jit
    def objective(alpha):
        prediction = solve_diffrax_batch(
            TRAIN_INITIAL_CONDITIONS, alpha, data.train_ts, T_TRAIN, rtol, atol
        )
        return jnp.mean((prediction - data.train) ** 2)

    return objective


def safe_float_objective(objective: Callable, alpha: float) -> float:
    value = float(objective(jnp.asarray(alpha, dtype=jnp.float64)))
    return value if math.isfinite(value) else float(np.finfo(np.float64).max)


def globally_refined_scalar_search(
    objective: Callable,
    alpha_min: float,
    alpha_max: float,
    grid_size: int,
    xatol: float,
):
    """Deterministic global grid search followed by local bounded refinement."""
    if grid_size < 3:
        raise ValueError("grid_size must be at least 3")
    grid = np.linspace(alpha_min, alpha_max, grid_size)
    losses = np.array([safe_float_objective(objective, x) for x in grid])
    finite = np.isfinite(losses) & (losses < np.finfo(np.float64).max)
    if not finite.any():
        raise RuntimeError("The objective was non-finite over the entire alpha grid.")

    grid_index = int(np.argmin(losses))
    left_index = max(0, grid_index - 1)
    right_index = min(grid_size - 1, grid_index + 1)
    left, right = float(grid[left_index]), float(grid[right_index])

    refinement_trace: list[tuple[float, float]] = []

    def traced_objective(x):
        loss = safe_float_objective(objective, float(x))
        refinement_trace.append((float(x), loss))
        return loss

    if right > left:
        refined = minimize_scalar(
            traced_objective,
            bounds=(left, right),
            method="bounded",
            options={"xatol": xatol, "maxiter": 500},
        )
        candidates = [
            (float(grid[grid_index]), float(losses[grid_index])),
            (float(refined.x), float(refined.fun)),
            (alpha_min, safe_float_objective(objective, alpha_min)),
            (alpha_max, safe_float_objective(objective, alpha_max)),
        ]
        alpha_hat, minimum_loss = min(candidates, key=lambda item: item[1])
        success = bool(refined.success)
        optimizer_message = str(refined.message)
        optimizer_nfev = int(refined.nfev)
    else:
        alpha_hat = float(grid[grid_index])
        minimum_loss = float(losses[grid_index])
        success = True
        optimizer_message = "Grid boundary optimum; no refinement interval."
        optimizer_nfev = 0

    boundary_tolerance = max(xatol * 10.0, (alpha_max - alpha_min) * 1e-8)
    at_boundary = bool(
        alpha_hat <= alpha_min + boundary_tolerance
        or alpha_hat >= alpha_max - boundary_tolerance
    )
    return {
        "alpha_hat": alpha_hat,
        "minimum_loss": minimum_loss,
        "grid": grid,
        "grid_losses": losses,
        "refinement_trace": refinement_trace,
        "success": success,
        "message": optimizer_message,
        "nfev": optimizer_nfev,
        "at_boundary": at_boundary,
    }


def instability_rate(prediction, threshold=1e6) -> float:
    prediction = np.asarray(prediction)
    bad = (
        ~np.isfinite(prediction).all(axis=(1, 2))
        | (prediction < 0).any(axis=(1, 2))
        | (np.abs(prediction) > threshold).any(axis=(1, 2))
    )
    return float(np.mean(bad))


def vector_field_metrics(alpha: float):
    prey_values = jnp.linspace(5.0, 25.0, 20)
    predator_values = jnp.linspace(10.0, 40.0, 20)
    prey_grid, predator_grid = jnp.meshgrid(prey_values, predator_values)
    states = jnp.stack([prey_grid.reshape(-1), predator_grid.reshape(-1)], axis=1)
    true_field = jax.vmap(lambda state: model_rhs(state, TRUE_ALPHA))(states)
    model_field = jax.vmap(lambda state: model_rhs(state, alpha))(states)
    true_residual = jax.vmap(lambda state: residual_alpha(state, TRUE_ALPHA))(states)
    model_residual = jax.vmap(lambda state: residual_alpha(state, alpha))(states)
    return {
        "residual_mse": float(jnp.mean((model_residual - true_residual) ** 2)),
        "residual_relative_error": float(
            jnp.linalg.norm(model_residual - true_residual)
            / (jnp.linalg.norm(true_residual) + 1e-30)
        ),
        "model_vector_field_mse": float(jnp.mean((model_field - true_field) ** 2)),
        "model_vector_field_relative_error": float(
            jnp.linalg.norm(model_field - true_field)
            / (jnp.linalg.norm(true_field) + 1e-30)
        ),
    }


def evaluate_fit(
    solver_name: str,
    method,
    ratio,
    alpha_hat: float,
    search_result,
    objective,
    data: ReferenceData,
    evaluation_rtol: float,
    evaluation_atol: float,
):
    continuous_train = solve_diffrax_batch(
        TRAIN_INITIAL_CONDITIONS,
        jnp.asarray(alpha_hat),
        data.train_ts,
        T_TRAIN,
        evaluation_rtol,
        evaluation_atol,
    )
    continuous_validation = solve_diffrax_batch(
        VALIDATION_INITIAL_CONDITIONS,
        jnp.asarray(alpha_hat),
        data.train_ts,
        T_TRAIN,
        evaluation_rtol,
        evaluation_atol,
    )
    continuous_extrapolate = solve_diffrax_batch(
        TRAIN_INITIAL_CONDITIONS,
        jnp.asarray(alpha_hat),
        data.extrapolate_ts,
        T_EXTRAPOLATE,
        evaluation_rtol,
        evaluation_atol,
    )

    if method is None:
        h = float("nan")
        matched_train = continuous_train
        matched_extrapolate = continuous_extrapolate
        matched_loss_at_true = safe_float_objective(objective, TRUE_ALPHA)
    else:
        h = data.observation_dt / ratio
        train_steps = (data.train.shape[1] - 1) * ratio
        extrapolate_steps = (data.extrapolate.shape[1] - 1) * ratio
        matched_train = rollout_batch(
            TRAIN_INITIAL_CONDITIONS,
            jnp.asarray(alpha_hat),
            h,
            train_steps,
            method,
        )[:, ::ratio, :]
        matched_extrapolate = rollout_batch(
            TRAIN_INITIAL_CONDITIONS,
            jnp.asarray(alpha_hat),
            h,
            extrapolate_steps,
            method,
        )[:, ::ratio, :]
        matched_loss_at_true = safe_float_objective(objective, TRUE_ALPHA)

    extrapolation_mask = np.asarray(data.extrapolate_ts) > T_TRAIN
    continuous_train_mse = float(jnp.mean((continuous_train - data.train) ** 2))
    continuous_validation_mse = float(
        jnp.mean((continuous_validation - data.validation) ** 2)
    )
    continuous_extrapolate_mse = float(
        jnp.mean(
            (
                continuous_extrapolate[:, extrapolation_mask, :]
                - data.extrapolate[:, extrapolation_mask, :]
            )
            ** 2
        )
    )
    matched_train_mse = float(jnp.mean((matched_train - data.train) ** 2))
    matched_extrapolate_mse = float(
        jnp.mean(
            (
                matched_extrapolate[:, extrapolation_mask, :]
                - data.extrapolate[:, extrapolation_mask, :]
            )
            ** 2
        )
    )
    continuous_final_mse = float(
        jnp.mean((continuous_extrapolate[:, -1, :] - data.extrapolate[:, -1, :]) ** 2)
    )
    matched_final_mse = float(
        jnp.mean((matched_extrapolate[:, -1, :] - data.extrapolate[:, -1, :]) ** 2)
    )
    tiny = np.finfo(np.float64).tiny
    signed_bias = alpha_hat - TRUE_ALPHA

    row = {
        "experiment_kind": "one_parameter_rollout_supervision",
        "training_solver": solver_name,
        "nominal_order": NOMINAL_ORDERS.get(solver_name, "adaptive"),
        "ratio": ratio,
        "observation_dt": data.observation_dt,
        "h": h,
        "true_alpha": TRUE_ALPHA,
        "alpha_hat": alpha_hat,
        "signed_alpha_bias": signed_bias,
        "absolute_alpha_bias": abs(signed_bias),
        "relative_alpha_bias": abs(signed_bias) / TRUE_ALPHA,
        "matched_train_loss": search_result["minimum_loss"],
        "matched_train_mse_recomputed": matched_train_mse,
        "matched_loss_at_true_alpha": matched_loss_at_true,
        "discrete_loss_reduction_vs_true_alpha": (
            matched_loss_at_true - search_result["minimum_loss"]
        ),
        "continuous_train_mse": continuous_train_mse,
        "continuous_validation_mse": continuous_validation_mse,
        "continuous_extrapolation_mse": continuous_extrapolate_mse,
        "continuous_final_mse": continuous_final_mse,
        "matched_extrapolation_mse": matched_extrapolate_mse,
        "matched_final_mse": matched_final_mse,
        "train_compensation_ratio_continuous_over_matched": (
            continuous_train_mse / max(matched_train_mse, tiny)
        ),
        "train_compensation_gap_log10": math.log10(
            max(continuous_train_mse, tiny) / max(matched_train_mse, tiny)
        ),
        "extrapolation_compensation_ratio_continuous_over_matched": (
            continuous_extrapolate_mse / max(matched_extrapolate_mse, tiny)
        ),
        "extrapolation_compensation_gap_log10": math.log10(
            max(continuous_extrapolate_mse, tiny)
            / max(matched_extrapolate_mse, tiny)
        ),
        "continuous_instability_rate": instability_rate(continuous_extrapolate),
        "matched_instability_rate": instability_rate(matched_extrapolate),
        "optimizer_success": search_result["success"],
        "optimizer_nfev": search_result["nfev"],
        "optimum_at_search_boundary": search_result["at_boundary"],
        "local_parameter_bias_order": "",
        **vector_field_metrics(alpha_hat),
    }
    return row


def add_local_orders(rows):
    for solver_name in SOLVERS:
        solver_rows = [row for row in rows if row["training_solver"] == solver_name]
        solver_rows.sort(key=lambda row: float(row["h"]), reverse=True)
        for coarse, fine in zip(solver_rows[:-1], solver_rows[1:]):
            coarse_bias = float(coarse["absolute_alpha_bias"])
            fine_bias = float(fine["absolute_alpha_bias"])
            if coarse_bias > 0 and fine_bias > 0:
                fine["local_parameter_bias_order"] = math.log(
                    coarse_bias / fine_bias
                ) / math.log(float(coarse["h"]) / float(fine["h"]))


def empirical_order_rows(rows, bias_floor: float):
    output = []
    solver_names = [
        name
        for name in SOLVERS
        if any(row["training_solver"] == name for row in rows)
    ]
    for solver_name in solver_names:
        selected = [
            row
            for row in rows
            if row["training_solver"] == solver_name
            and float(row["absolute_alpha_bias"]) > bias_floor
        ]
        selected.sort(key=lambda row: float(row["h"]))
        hs = np.asarray([float(row["h"]) for row in selected])
        biases = np.asarray([float(row["absolute_alpha_bias"]) for row in selected])
        if len(selected) >= 2:
            coefficients = np.polyfit(np.log(hs), np.log(biases), 1)
            predicted = np.polyval(coefficients, np.log(hs))
            residual_sum = float(np.sum((np.log(biases) - predicted) ** 2))
            total_sum = float(
                np.sum((np.log(biases) - np.mean(np.log(biases))) ** 2)
            )
            r_squared = 1.0 - residual_sum / total_sum if total_sum > 0 else float("nan")
            slope = float(coefficients[0])
        else:
            slope = float("nan")
            r_squared = float("nan")
        output.append(
            {
                "training_solver": solver_name,
                "nominal_order": NOMINAL_ORDERS[solver_name],
                "empirical_parameter_bias_order": slope,
                "r_squared": r_squared,
                "points_used": len(selected),
                "bias_floor": bias_floor,
                "minimum_h_used": float(np.min(hs)) if len(hs) else float("nan"),
                "maximum_h_used": float(np.max(hs)) if len(hs) else float("nan"),
            }
        )
    return output


def write_csv(rows, path: Path):
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_parameter_bias(rows, output_dir: Path):
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    for solver_name in SOLVERS:
        selected = [row for row in rows if row["training_solver"] == solver_name]
        selected.sort(key=lambda row: float(row["h"]))
        if not selected:
            continue
        hs = np.asarray([float(row["h"]) for row in selected])
        biases = np.asarray([float(row["absolute_alpha_bias"]) for row in selected])
        ax.loglog(hs, np.maximum(biases, 1e-18), marker="o", label=solver_name)
    ax.set_xlabel("training solver step size h")
    ax.set_ylabel(r"absolute parameter bias $|\hat{\alpha}-c|$")
    ax.set_title("Solver-induced parameter bias")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "parameter_bias_vs_stepsize_loglog.png", dpi=200)
    plt.close(fig)


def plot_alpha_hat(rows, output_dir: Path):
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    for solver_name in SOLVERS:
        selected = [row for row in rows if row["training_solver"] == solver_name]
        selected.sort(key=lambda row: float(row["h"]))
        if not selected:
            continue
        ax.semilogx(
            [float(row["h"]) for row in selected],
            [float(row["alpha_hat"]) for row in selected],
            marker="o",
            label=solver_name,
        )
    ax.axhline(TRUE_ALPHA, color="black", linestyle="--", label="true c")
    ax.set_xlabel("training solver step size h")
    ax.set_ylabel(r"fitted $\hat{\alpha}$")
    ax.set_title("Location of the discrete-objective optimum")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "alpha_hat_vs_stepsize.png", dpi=200)
    plt.close(fig)


def plot_objective_landscapes(curve_rows, output_dir: Path):
    solver_names = [
        name
        for name in SOLVERS
        if any(row["training_solver"] == name for row in curve_rows)
    ]
    if not solver_names:
        return
    fig, axes = plt.subplots(
        1, len(solver_names), figsize=(5.4 * len(solver_names), 4.8), sharey=False
    )
    axes = np.atleast_1d(axes)
    for ax, solver_name in zip(axes, solver_names):
        solver_rows = [
            row
            for row in curve_rows
            if row["training_solver"] == solver_name
            and row["search_stage"] == "global_grid"
        ]
        ratios = sorted({int(row["ratio"]) for row in solver_rows})
        for ratio in ratios:
            selected = [row for row in solver_rows if int(row["ratio"]) == ratio]
            selected.sort(key=lambda row: float(row["alpha"]))
            ax.semilogy(
                [float(row["alpha"]) for row in selected],
                [max(float(row["loss"]), 1e-30) for row in selected],
                label=f"ratio={ratio}",
            )
        ax.axvline(TRUE_ALPHA, color="black", linestyle="--", linewidth=1)
        ax.set_title(solver_name)
        ax.set_xlabel(r"$\alpha$")
        ax.grid(True, alpha=0.25)
    axes[0].set_ylabel(r"discrete rollout loss $J_h(\alpha)$")
    axes[-1].legend(fontsize=8)
    fig.suptitle("Solver-dependent objective landscapes")
    fig.tight_layout()
    fig.savefig(output_dir / "objective_landscapes.png", dpi=200)
    plt.close(fig)


def parse_args():
    default_output = Path(__file__).with_name("one_parameter_solver_bias_results")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path, default=default_output, help="Result directory."
    )
    parser.add_argument(
        "--solvers",
        nargs="+",
        choices=list(SOLVERS),
        default=list(SOLVERS),
    )
    parser.add_argument("--ratios", nargs="+", type=int, default=[1, 2, 4, 8, 16])
    parser.add_argument(
        "--observation-stride",
        type=int,
        default=1,
        help=(
            "Use every k-th point from the original dt=0.05 grid. Increasing k "
            "makes high-order solver bias easier to resolve."
        ),
    )
    parser.add_argument("--alpha-min", type=float, default=0.0)
    parser.add_argument("--alpha-max", type=float, default=0.02)
    parser.add_argument("--grid-size", type=int, default=201)
    parser.add_argument("--xatol", type=float, default=1e-12)
    parser.add_argument("--bias-floor", type=float, default=1e-11)
    parser.add_argument("--reference-rtol", type=float, default=1e-12)
    parser.add_argument("--reference-atol", type=float, default=1e-12)
    parser.add_argument("--evaluation-rtol", type=float, default=1e-10)
    parser.add_argument("--evaluation-atol", type=float, default=1e-10)
    parser.add_argument(
        "--include-diffrax-reference",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Fit the same scalar objective with tight adaptive Tsit5 as a control.",
    )
    parser.add_argument(
        "--plots", action=argparse.BooleanOptionalAction, default=True
    )
    return parser.parse_args()


def validate_args(args):
    if any(ratio < 1 for ratio in args.ratios):
        raise ValueError("All ratios must be positive integers.")
    if args.observation_stride < 1:
        raise ValueError("observation_stride must be positive.")
    if not args.alpha_min < TRUE_ALPHA < args.alpha_max:
        raise ValueError(
            "The search interval must contain TRUE_ALPHA so the unbiased control "
            "can be tested."
        )
    if args.xatol <= 0:
        raise ValueError("xatol must be positive.")


def main():
    args = parse_args()
    validate_args(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    data = make_reference_data(
        args.observation_stride, args.reference_rtol, args.reference_atol
    )
    progress(
        f"[setup] observation_dt={data.observation_dt:g}, "
        f"train_points={data.train.shape[1]}"
    )

    summary_rows = []
    curve_rows = []
    total = len(args.solvers) * len(args.ratios)
    run_index = 0
    for solver_name in args.solvers:
        method = SOLVERS[solver_name]
        for ratio in args.ratios:
            run_index += 1
            h = data.observation_dt / ratio
            progress(
                f"[fit {run_index}/{total}] solver={solver_name}, "
                f"ratio={ratio}, h={h:g}"
            )
            objective = make_discrete_objective(method, ratio, data)
            search = globally_refined_scalar_search(
                objective,
                args.alpha_min,
                args.alpha_max,
                args.grid_size,
                args.xatol,
            )
            for alpha, loss in zip(search["grid"], search["grid_losses"]):
                curve_rows.append(
                    {
                        "training_solver": solver_name,
                        "ratio": ratio,
                        "h": h,
                        "search_stage": "global_grid",
                        "alpha": float(alpha),
                        "loss": float(loss),
                    }
                )
            for alpha, loss in search["refinement_trace"]:
                curve_rows.append(
                    {
                        "training_solver": solver_name,
                        "ratio": ratio,
                        "h": h,
                        "search_stage": "local_refinement",
                        "alpha": alpha,
                        "loss": loss,
                    }
                )
            row = evaluate_fit(
                solver_name,
                method,
                ratio,
                search["alpha_hat"],
                search,
                objective,
                data,
                args.evaluation_rtol,
                args.evaluation_atol,
            )
            summary_rows.append(row)
            progress(
                f"[fit {run_index}/{total}] alpha_hat={row['alpha_hat']:.10g}, "
                f"|bias|={row['absolute_alpha_bias']:.3e}, "
                f"J_h={row['matched_train_loss']:.3e}, "
                f"continuous_train_mse={row['continuous_train_mse']:.3e}"
            )
            if search["at_boundary"]:
                progress(
                    "[warning] optimum is at the alpha search boundary; widen "
                    "--alpha-min/--alpha-max before interpreting this row."
                )

    if args.include_diffrax_reference:
        progress("[control] fitting alpha with tight adaptive Diffrax")
        objective = make_diffrax_objective(
            data, args.evaluation_rtol, args.evaluation_atol
        )
        diffrax_grid_size = min(args.grid_size, 51)
        search = globally_refined_scalar_search(
            objective,
            args.alpha_min,
            args.alpha_max,
            diffrax_grid_size,
            args.xatol,
        )
        row = evaluate_fit(
            "diffrax_tsit5",
            None,
            "adaptive",
            search["alpha_hat"],
            search,
            objective,
            data,
            args.evaluation_rtol,
            args.evaluation_atol,
        )
        summary_rows.append(row)
        progress(
            f"[control] alpha_hat={row['alpha_hat']:.10g}, "
            f"|bias|={row['absolute_alpha_bias']:.3e}"
        )

    add_local_orders(summary_rows)
    order_rows = empirical_order_rows(summary_rows, args.bias_floor)
    write_csv(summary_rows, args.output_dir / "summary.csv")
    write_csv(curve_rows, args.output_dir / "objective_curves.csv")
    write_csv(order_rows, args.output_dir / "empirical_orders.csv")

    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "description": "One-parameter residual rollout-supervision experiment",
        "fixed_physics": [A, B, R, Z],
        "true_alpha": TRUE_ALPHA,
        "base_dt": BASE_DT,
        "train_horizon": T_TRAIN,
        "extrapolation_horizon": T_EXTRAPOLATE,
        "jax_enable_x64": bool(jax.config.x64_enabled),
        "arguments": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
    }
    with (args.output_dir / "metadata.json").open("w") as handle:
        json.dump(metadata, handle, indent=2)

    if args.plots:
        plot_parameter_bias(summary_rows, args.output_dir)
        plot_alpha_hat(summary_rows, args.output_dir)
        plot_objective_landscapes(curve_rows, args.output_dir)

    progress(f"[done] results written to {args.output_dir}")


if __name__ == "__main__":
    main()
