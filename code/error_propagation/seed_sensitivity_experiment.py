import argparse
import csv
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import optax
from jax import jit, lax, random, value_and_grad, vmap
import jax.numpy as jnp

from solver_error_experiment import (
    evaluate_model,
    heun,
    init_network_params,
    make_optimizer,
    make_reference_data,
    model_rhs,
    nn,
    num_observed,
    obs_dt,
    rk4,
    roll_out,
    y0_batch,
)


METHODS = {
    "heun": heun,
    "rk4": rk4,
}


def parse_float(value, default=np.nan):
    if value is None:
        return default
    if isinstance(value, (float, int, np.floating, np.integer)):
        return float(value)
    value = str(value).strip()
    if value == "" or value.upper() in {"N/A", "NULL", "NONE", "NAN"}:
        return default
    try:
        return float(value)
    except ValueError:
        return default


def parse_int(value, default=None):
    number = parse_float(value)
    if not np.isfinite(number):
        return default
    return int(number)


def initial_params(seed):
    return {
        "nn_params": init_network_params([2, 64, 64, 2], random.key(seed)),
        "f_physics": jnp.array([1.0, 0.05, 1.5, 0.03]),
    }


def train_model_with_seed(method, ratio, seed, data, num_epochs, step_size, chunk_size):
    h = obs_dt / ratio
    num_steps = int((num_observed - 1) * ratio)
    params = initial_params(seed)
    optimizer = make_optimizer(step_size)
    opt_state = optimizer.init(params)

    def sample_loss(parameters, y0, target):
        pred = roll_out(y0, h, model_rhs, parameters, num_steps, method)[::ratio]
        return jnp.mean((target - pred) ** 2)

    def batch_loss(parameters):
        losses = vmap(sample_loss, in_axes=(None, 0, 0))(
            parameters, y0_batch, data["train_noisy"]
        )
        residuals = vmap(nn, in_axes=(0, None))(
            data["train_noisy"].reshape(-1, 2), parameters["nn_params"]
        )
        return jnp.mean(losses) + 1e-5 * jnp.mean(residuals**2)

    @jit
    def train_step(carry, _):
        parameters, opt_state = carry
        loss_val, grads = value_and_grad(batch_loss)(parameters)
        updates, opt_state = optimizer.update(grads, opt_state, parameters)
        return (optax.apply_updates(parameters, updates), opt_state), loss_val

    @jit
    def train_chunk(parameters, opt_state):
        (parameters, opt_state), losses = lax.scan(
            train_step, (parameters, opt_state), None, length=chunk_size
        )
        return parameters, opt_state, losses

    best_loss = float("inf")
    best_params = params
    best_epoch = 0
    patience = 500
    counter = 0

    for i in range(num_epochs // chunk_size):
        params, opt_state, losses = train_chunk(params, opt_state)
        loss_val = float(losses[-1])
        current_epoch = (i + 1) * chunk_size
        if best_loss - loss_val > 1e-7:
            best_loss = loss_val
            best_params = params
            best_epoch = current_epoch
            counter = 0
        else:
            counter += chunk_size
        if counter >= patience:
            break

    return {
        "params": best_params,
        "best_loss": best_loss,
        "best_epoch": best_epoch,
        "training_method": method.__name__,
        "ratio": ratio,
        "h_model": h,
        "seed": seed,
    }


def residual_diagnostics(trained, data):
    residual = vmap(nn, in_axes=(0, None))(
        data["train_noisy"].reshape(-1, 2),
        trained["params"]["nn_params"],
    )
    return {
        "residual_mean_abs": float(jnp.mean(jnp.abs(residual))),
        "residual_l2_norm": float(jnp.linalg.norm(residual)),
        "residual_rms": float(jnp.sqrt(jnp.mean(residual**2))),
    }


def physics_param_row(trained):
    f_physics = np.asarray(trained["params"]["f_physics"], dtype=float)
    return {
        "f_physics_a": float(f_physics[0]),
        "f_physics_b": float(f_physics[1]),
        "f_physics_r": float(f_physics[2]),
        "f_physics_z": float(f_physics[3]),
    }


def write_summary_csv(rows, path):
    fieldnames = [
        "experiment_kind",
        "training_method",
        "prediction_method",
        "ratio",
        "predict_ratio",
        "seed",
        "best_epoch",
        "h_model",
        "h_predict",
        "noise_level",
        "best_loss",
        "train_mse",
        "validation_mse",
        "extrapolate_mse",
        "residual_mse",
        "residual_rel_error",
        "model_vf_mse",
        "model_vf_rel_error",
        "instability_rate",
        "final_mse",
        "final_l2_error",
        "residual_mean_abs",
        "residual_l2_norm",
        "residual_rms",
        "f_physics_a",
        "f_physics_b",
        "f_physics_r",
        "f_physics_z",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def read_csv_rows(path):
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def choose_prediction_method(train_method_name, prediction_method):
    if prediction_method == "same":
        return METHODS[train_method_name]
    return METHODS[prediction_method]


def metric_values(rows, metric):
    values = np.array([parse_float(row.get(metric)) for row in rows], dtype=float)
    return values[np.isfinite(values)]


def log10_values(values):
    values = np.asarray(values, dtype=float)
    return np.log10(values[np.isfinite(values) & (values > 0.0)])


def summarize_values(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    logs = log10_values(values)
    if values.size == 0:
        return {
            "mean": np.nan,
            "std": np.nan,
            "range": np.nan,
            "cv": np.nan,
            "log10_std": np.nan,
            "log10_range": np.nan,
        }

    mean = float(np.mean(values))
    std = float(np.std(values, ddof=1)) if values.size >= 2 else 0.0
    value_range = float(np.max(values) - np.min(values))
    cv = std / mean if mean != 0 else np.nan
    log_std = float(np.std(logs, ddof=1)) if logs.size >= 2 else 0.0
    log_range = float(np.max(logs) - np.min(logs)) if logs.size else np.nan
    return {
        "mean": mean,
        "std": std,
        "range": value_range,
        "cv": cv,
        "log10_std": log_std,
        "log10_range": log_range,
    }


def matching_ratio_rows(
    rows,
    train_method_name,
    prediction_method_name,
    predict_ratio,
):
    return [
        row
        for row in rows
        if row.get("training_method") == train_method_name
        and row.get("prediction_method") == prediction_method_name
        and parse_int(row.get("predict_ratio")) == predict_ratio
    ]


def write_comparison_csv(
    seed_rows,
    sweep_rows,
    output_dir,
    metric,
    fixed_train_ratio,
    predict_ratio,
):
    comparison_rows = []
    for train_method_name in sorted(METHODS):
        method_seed_rows = [
            row for row in seed_rows if row.get("training_method") == train_method_name
        ]
        if not method_seed_rows:
            continue

        prediction_method_name = method_seed_rows[0]["prediction_method"]
        method_ratio_rows = matching_ratio_rows(
            sweep_rows,
            train_method_name,
            prediction_method_name,
            predict_ratio,
        )
        seed_values = metric_values(method_seed_rows, metric)
        ratio_values = metric_values(method_ratio_rows, metric)
        seed_stats = summarize_values(seed_values)
        ratio_stats = summarize_values(ratio_values)

        ratio_train_values = sorted(
            {
                parse_int(row.get("ratio"))
                for row in method_ratio_rows
                if parse_int(row.get("ratio")) is not None
            }
        )
        denominator = ratio_stats["log10_range"]
        log_range_ratio = (
            seed_stats["log10_range"] / denominator
            if np.isfinite(denominator) and denominator != 0
            else np.nan
        )

        comparison_rows.append(
            {
                "training_method": train_method_name,
                "prediction_method": prediction_method_name,
                "metric": metric,
                "fixed_train_ratio": fixed_train_ratio,
                "predict_ratio": predict_ratio,
                "seeds": ";".join(
                    str(parse_int(row.get("seed"))) for row in method_seed_rows
                ),
                "seed_values": ";".join(f"{value:.12g}" for value in seed_values),
                "seed_mean": seed_stats["mean"],
                "seed_std": seed_stats["std"],
                "seed_range": seed_stats["range"],
                "seed_cv": seed_stats["cv"],
                "seed_log10_std": seed_stats["log10_std"],
                "seed_log10_range": seed_stats["log10_range"],
                "train_ratios": ";".join(str(value) for value in ratio_train_values),
                "train_ratio_values": ";".join(
                    f"{value:.12g}" for value in ratio_values
                ),
                "train_ratio_mean": ratio_stats["mean"],
                "train_ratio_std": ratio_stats["std"],
                "train_ratio_range": ratio_stats["range"],
                "train_ratio_cv": ratio_stats["cv"],
                "train_ratio_log10_std": ratio_stats["log10_std"],
                "train_ratio_log10_range": ratio_stats["log10_range"],
                "seed_to_train_ratio_log10_range": log_range_ratio,
            }
        )

    path = output_dir / f"seed_vs_train_ratio_variability_{metric}.csv"
    fieldnames = list(comparison_rows[0].keys()) if comparison_rows else []
    if not fieldnames:
        return comparison_rows
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(comparison_rows)
    return comparison_rows


def write_mini_sweep_summary(rows, output_dir):
    summary_rows = []
    for train_method_name in sorted(METHODS):
        method_rows = [
            row for row in rows if row.get("training_method") == train_method_name
        ]
        ratios = sorted(
            {
                parse_int(row.get("ratio"))
                for row in method_rows
                if parse_int(row.get("ratio")) is not None
            }
        )
        for ratio in ratios:
            ratio_rows = [
                row for row in method_rows if parse_int(row.get("ratio")) == ratio
            ]
            row_summary = {
                "training_method": train_method_name,
                "ratio": ratio,
                "prediction_method": ratio_rows[0].get("prediction_method"),
                "predict_ratio": parse_int(ratio_rows[0].get("predict_ratio")),
                "n_seeds": len(ratio_rows),
            }
            for metric in [
                "best_loss",
                "best_epoch",
                "extrapolate_mse",
                "final_mse",
                "residual_mean_abs",
                "residual_l2_norm",
                "residual_rms",
            ]:
                stats = summarize_values(metric_values(ratio_rows, metric))
                row_summary[f"{metric}_mean"] = stats["mean"]
                row_summary[f"{metric}_std"] = stats["std"]
                row_summary[f"{metric}_range"] = stats["range"]
                row_summary[f"{metric}_log10_range"] = stats["log10_range"]
            summary_rows.append(row_summary)

    if not summary_rows:
        return

    path = output_dir / "mini_sweep_group_summary.csv"
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)


def plot_seed_vs_ratio(seed_rows, sweep_rows, output_dir, metric, predict_ratio):
    methods = [method for method in sorted(METHODS) if any(
        row.get("training_method") == method for row in seed_rows
    )]
    if not methods:
        return

    fig, axes = plt.subplots(1, len(methods), figsize=(6 * len(methods), 5), squeeze=False)
    for ax, train_method_name in zip(axes[0], methods):
        method_seed_rows = [
            row for row in seed_rows if row.get("training_method") == train_method_name
        ]
        prediction_method_name = method_seed_rows[0]["prediction_method"]
        method_ratio_rows = matching_ratio_rows(
            sweep_rows,
            train_method_name,
            prediction_method_name,
            predict_ratio,
        )

        seed_values = metric_values(method_seed_rows, metric)
        ratio_values = metric_values(method_ratio_rows, metric)
        seed_x = np.full(seed_values.shape, 0.0)
        ratio_x = np.full(ratio_values.shape, 1.0)

        if seed_values.size:
            ax.scatter(seed_x, seed_values, label="different seeds", alpha=0.85)
            ax.hlines(
                np.mean(seed_values),
                -0.18,
                0.18,
                color="tab:blue",
                linewidth=2,
            )
        if ratio_values.size:
            ax.scatter(ratio_x, ratio_values, label="different train ratios", alpha=0.85)
            ax.hlines(
                np.mean(ratio_values),
                0.82,
                1.18,
                color="tab:orange",
                linewidth=2,
            )

        for row in method_seed_rows:
            y = parse_float(row.get(metric))
            if np.isfinite(y):
                ax.text(
                    0.04,
                    y,
                    f"seed={parse_int(row.get('seed'))}",
                    fontsize=7,
                    va="center",
                )
        for row in method_ratio_rows:
            y = parse_float(row.get(metric))
            if np.isfinite(y):
                ax.text(
                    1.04,
                    y,
                    f"ratio={parse_int(row.get('ratio'))}",
                    fontsize=7,
                    va="center",
                )

        ax.set_xticks([0, 1])
        ax.set_xticklabels(["seed sweep\nfixed train_ratio", "train_ratio sweep"])
        ax.set_yscale("log")
        ax.set_ylabel(metric)
        ax.set_title(f"train={train_method_name}, pred={prediction_method_name}")
        ax.grid(True, which="both", axis="y", alpha=0.3)
        ax.legend(fontsize=8)

    fig.suptitle(
        f"Optimization Seed Variability vs Train-Ratio Variability ({metric})",
        fontsize=14,
    )
    fig.tight_layout()
    fig.savefig(output_dir / f"seed_vs_train_ratio_variability_{metric}.png", dpi=220)
    plt.close(fig)


def plot_seed_error_curves(details, output_dir, t_final, extrap_start, metric_name):
    if not details:
        return
    first_curves = next(iter(details.values()))
    n_times = len(first_curves["extrapolate_mse_by_time"])
    time_grid = np.linspace(0.0, t_final, n_times)

    methods = sorted({key[0] for key in details})
    fig, axes = plt.subplots(len(methods), 1, figsize=(9, 4 * len(methods)), squeeze=False)
    for ax, train_method_name in zip(axes[:, 0], methods):
        method_items = [
            (key, curves) for key, curves in details.items() if key[0] == train_method_name
        ]
        method_items.sort(key=lambda item: (item[0][1], item[0][2]))
        for key, curves in method_items:
            train_ratio = key[1]
            seed = key[2]
            mse = np.asarray(curves["extrapolate_mse_by_time"], dtype=float)
            ax.semilogy(time_grid, mse, label=f"ratio={train_ratio}, seed={seed}")
        ax.axvline(extrap_start, color="tab:red", linestyle=":", linewidth=1.2)
        ax.set_xlabel("t")
        ax.set_ylabel(metric_name)
        ax.set_title(f"train={train_method_name}")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    fig.suptitle("Seed Sweep: Extrapolation Error Growth", fontsize=14)
    fig.tight_layout()
    fig.savefig(output_dir / "seed_sweep_error_over_time.png", dpi=220)
    plt.close(fig)


def finite_pair_values(rows, x_metric, y_metric):
    x_values = np.array([parse_float(row.get(x_metric)) for row in rows], dtype=float)
    y_values = np.array([parse_float(row.get(y_metric)) for row in rows], dtype=float)
    mask = np.isfinite(x_values) & np.isfinite(y_values)
    return x_values[mask], y_values[mask]


def correlation(x_values, y_values):
    if x_values.size < 2 or y_values.size < 2:
        return np.nan
    if np.std(x_values) == 0.0 or np.std(y_values) == 0.0:
        return np.nan
    return float(np.corrcoef(x_values, y_values)[0, 1])


def analyze_seed_results(rows, output_dir):
    analyses = []
    groups = [("all", rows)]
    for method in sorted(METHODS):
        method_rows = [row for row in rows if row.get("training_method") == method]
        if method_rows:
            groups.append((method, method_rows))
        ratios = sorted(
            {
                parse_int(row.get("ratio"))
                for row in method_rows
                if parse_int(row.get("ratio")) is not None
            }
        )
        for ratio in ratios:
            ratio_rows = [
                row for row in method_rows if parse_int(row.get("ratio")) == ratio
            ]
            if ratio_rows:
                groups.append((f"{method}:ratio={ratio}", ratio_rows))

    metric_pairs = [
        ("best_loss", "extrapolate_mse"),
        ("best_loss", "final_mse"),
        ("residual_mean_abs", "extrapolate_mse"),
        ("residual_l2_norm", "extrapolate_mse"),
        ("residual_rms", "extrapolate_mse"),
        ("best_epoch", "extrapolate_mse"),
    ]

    print()
    print("Seed diagnostics correlations")
    for group_name, group_rows in groups:
        for x_metric, y_metric in metric_pairs:
            x_values, y_values = finite_pair_values(group_rows, x_metric, y_metric)
            corr = correlation(x_values, y_values)
            analyses.append(
                {
                    "group": group_name,
                    "x_metric": x_metric,
                    "y_metric": y_metric,
                    "n": int(x_values.size),
                    "correlation": corr,
                }
            )
            print(f"{group_name}: corr({x_metric}, {y_metric}) = {corr}")

    path = output_dir / "seed_correlation_analysis.csv"
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["group", "x_metric", "y_metric", "n", "correlation"],
        )
        writer.writeheader()
        writer.writerows(analyses)


def plot_seed_diagnostic_scatter(rows, output_dir):
    if not rows:
        return

    plot_specs = [
        ("best_loss", "extrapolate_mse", "Best Loss vs Extrapolation MSE"),
        ("residual_mean_abs", "extrapolate_mse", "Residual Mean Abs vs Extrapolation MSE"),
        ("residual_l2_norm", "extrapolate_mse", "Residual L2 Norm vs Extrapolation MSE"),
        ("best_epoch", "extrapolate_mse", "Best Epoch vs Extrapolation MSE"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), squeeze=False)

    for ax, (x_metric, y_metric, title) in zip(axes.ravel(), plot_specs):
        for method in sorted(METHODS):
            method_rows = [row for row in rows if row.get("training_method") == method]
            x_values, y_values = finite_pair_values(method_rows, x_metric, y_metric)
            if x_values.size == 0:
                continue
            ax.scatter(x_values, y_values, label=method, alpha=0.85)
            corr = correlation(x_values, y_values)
            if np.isfinite(corr):
                ax.text(
                    0.03,
                    0.92 - 0.08 * list(sorted(METHODS)).index(method),
                    f"{method}: r={corr:.2f}",
                    transform=ax.transAxes,
                    fontsize=8,
                )

        if x_metric != "best_epoch":
            ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(x_metric)
        ax.set_ylabel(y_metric)
        ax.set_title(title)
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=8)

    fig.suptitle("Seed Diagnostics", fontsize=14)
    fig.tight_layout()
    fig.savefig(output_dir / "seed_diagnostic_scatter.png", dpi=220)
    plt.close(fig)


def run_seed_sensitivity(args):
    script_dir = Path(__file__).resolve().parent
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = script_dir / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    prediction_summary = Path(args.prediction_summary)
    if not prediction_summary.is_absolute():
        prediction_summary = script_dir / prediction_summary
    sweep_rows = read_csv_rows(prediction_summary)

    data = make_reference_data()
    rows = []
    details = {}
    train_ratios = (
        args.train_ratios
        if args.train_ratios is not None
        else [args.fixed_train_ratio]
    )

    for train_method_name in args.methods:
        train_method = METHODS[train_method_name]
        pred_method = choose_prediction_method(train_method_name, args.prediction_method)
        for train_ratio in train_ratios:
            for seed in args.seeds:
                trained = train_model_with_seed(
                    train_method,
                    train_ratio,
                    seed,
                    data,
                    num_epochs=args.num_epochs,
                    step_size=args.step_size,
                    chunk_size=args.chunk_size,
                )
                row, curves = evaluate_model(
                    trained=trained,
                    method=pred_method,
                    predict_ratio=args.predict_ratio,
                    data=data,
                    experiment_type="seed_sensitivity",
                )
                row["seed"] = seed
                row["best_epoch"] = trained["best_epoch"]
                row.update(residual_diagnostics(trained, data))
                row.update(physics_param_row(trained))
                rows.append(row)
                key = (
                    train_method_name,
                    train_ratio,
                    seed,
                    pred_method.__name__,
                    args.predict_ratio,
                )
                details[key] = curves
                print(
                    "train =",
                    train_method_name,
                    "train_ratio =",
                    train_ratio,
                    "seed =",
                    seed,
                    "pred =",
                    pred_method.__name__,
                    "predict_ratio =",
                    args.predict_ratio,
                    "extrapolate_mse =",
                    row["extrapolate_mse"],
                    "final_mse =",
                    row["final_mse"],
                    "best_loss =",
                    row["best_loss"],
                    "best_epoch =",
                    row["best_epoch"],
                    "residual_mean_abs =",
                    row["residual_mean_abs"],
                    "residual_l2_norm =",
                    row["residual_l2_norm"],
                    "f_physics =",
                    trained["params"]["f_physics"],
                )

    write_summary_csv(rows, output_dir / "seed_sensitivity_summary.csv")
    with (output_dir / "seed_sensitivity_details.pkl").open("wb") as f:
        pickle.dump({"rows": rows, "details": details}, f)

    analyze_seed_results(rows, output_dir)
    plot_seed_diagnostic_scatter(rows, output_dir)
    write_mini_sweep_summary(rows, output_dir)

    if len(train_ratios) == 1:
        for metric in args.metrics:
            write_comparison_csv(
                rows,
                sweep_rows,
                output_dir,
                metric=metric,
                fixed_train_ratio=train_ratios[0],
                predict_ratio=args.predict_ratio,
            )
            plot_seed_vs_ratio(
                rows,
                sweep_rows,
                output_dir,
                metric=metric,
                predict_ratio=args.predict_ratio,
            )

    plot_seed_error_curves(
        details,
        output_dir,
        t_final=args.t_final,
        extrap_start=args.extrap_start,
        metric_name="MSE by time",
    )
    print(f"Saved seed sensitivity results to {output_dir}")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Repeat Heun/RK4 training across random seeds and compare seed "
            "variability with train-ratio variability from prediction_sweep."
        )
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["heun", "rk4"],
        choices=sorted(METHODS),
        help="Training solvers to repeat across seeds.",
    )
    parser.add_argument(
        "--fixed-train-ratio",
        type=int,
        default=4,
        help="Train ratio used for the seed sweep when --train-ratios is not set.",
    )
    parser.add_argument(
        "--train-ratios",
        nargs="+",
        type=int,
        default=None,
        help=(
            "Optional mini sweep over train ratios. Example: "
            "--train-ratios 2 4 8."
        ),
    )
    parser.add_argument(
        "--prediction-method",
        default="rk4",
        choices=["same", "heun", "rk4"],
        help="Prediction solver used to evaluate seed-sweep models.",
    )
    parser.add_argument(
        "--predict-ratio",
        type=int,
        default=16,
        help="Prediction ratio used for evaluation and existing-sweep comparison.",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[0, 1, 2, 3, 4],
        help="Network initialization seeds to test.",
    )
    parser.add_argument("--num-epochs", type=int, default=30000)
    parser.add_argument("--step-size", type=float, default=3e-3)
    parser.add_argument("--chunk-size", type=int, default=100)
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=["extrapolate_mse", "final_mse"],
        help="Metrics used in seed-vs-train-ratio comparison plots.",
    )
    parser.add_argument(
        "--prediction-summary",
        default="prediction_sweep_results/prediction_sweep_summary.csv",
        help="Existing prediction sweep CSV used for train-ratio variability.",
    )
    parser.add_argument(
        "--output-dir",
        default="seed_sensitivity_results",
        help="Directory where seed sensitivity outputs are written.",
    )
    parser.add_argument("--t-final", type=float, default=20.0)
    parser.add_argument("--extrap-start", type=float, default=5.0)
    return parser.parse_args()


if __name__ == "__main__":
    run_seed_sensitivity(parse_args())
